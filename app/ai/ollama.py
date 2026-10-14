from __future__ import annotations

import base64
import copy
import io
import json
import threading
import time
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from PIL import Image

from app.ai.json_util import parse_model_json
from app.config import Settings, get_settings, running_in_docker

_COMPOSE_OLLAMA = "http://ollama:11434"
_DOCKER_HOST = "host.docker.internal"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_FALLBACK_CONNECT_TIMEOUT = 3
_RESTORE_CHECK_INTERVAL = 60.0
_MEMORY_ERROR_TOKENS = (
    "out of memory",
    "insufficient memory",
    "requires more system memory",
    "cudamalloc failed",
)

# Oldest Ollama release that can load each model family.
MIN_OLLAMA_VERSION: dict[str, tuple[int, ...]] = {"qwen3-vl": (0, 12, 7)}


class OllamaError(RuntimeError):
    pass


class OllamaConnectionError(OllamaError):
    """Ollama could not be reached, so further calls in the same job will fail too."""


class OllamaTruncatedError(OllamaError):
    """The reply hit the num_predict limit before the JSON was complete."""


class OllamaPullError(OllamaError):
    """Ollama reported an error while downloading a model."""


def _with_host(root: str, host: str) -> str:
    parts = urlsplit(root)
    netloc = host if parts.port is None else f"{host}:{parts.port}"
    return urlunsplit((parts.scheme or "http", netloc, parts.path, "", ""))


def fallback_roots(root: str) -> list[str]:
    """Other addresses that usually reach the same Ollama when the configured one is unreachable."""
    host = (urlsplit(root).hostname or "").lower()
    if host == _DOCKER_HOST:
        return [_with_host(root, "127.0.0.1")]
    if host in _LOOPBACK_HOSTS and running_in_docker():
        return [_with_host(root, _DOCKER_HOST), _COMPOSE_OLLAMA]
    return []


def version_tuple(version: str | None) -> tuple[int, ...] | None:
    numbers: list[int] = []
    for part in str(version or "").split("-", 1)[0].split("."):
        if not part.isdigit():
            break
        numbers.append(int(part))
    if not numbers or not any(numbers):
        return None
    return tuple(numbers)


def version_problem(version: str | None, model: str) -> str | None:
    family = (model or "").split(":", 1)[0].lower()
    required = MIN_OLLAMA_VERSION.get(family)
    current = version_tuple(version)
    if not required or current is None or current >= required:
        return None
    needed = ".".join(str(part) for part in required)
    return f"Ollama {version} is too old for {model}. Update Ollama to {needed} or newer."


def instruct_variant(model: str) -> str | None:
    """The non-thinking tag of a Qwen3-VL model. The bare qwen3-vl:2b/4b/8b/30b tags are the
    thinking versions, which do not turn reasoning off when asked."""
    family, _, tag = (model or "").partition(":")
    if family.lower() != "qwen3-vl":
        return None
    size = tag.lower().removesuffix("-thinking") or "8b"
    if size == "latest":
        size = "8b"
    if size in {"30b", "30b-a3b"}:
        return "qwen3-vl:30b-a3b-instruct"
    if size in {"2b", "4b", "8b", "32b"}:
        return f"qwen3-vl:{size}-instruct"
    return None


def thinking_vision_warning(model: str) -> str:
    suggestion = instruct_variant(model)
    choice = f"Install and choose {suggestion}" if suggestion else "Choose an instruct (non-thinking) vision model"
    return (
        f"{model} is a thinking model. It spends its reply on hidden reasoning, so frames get no AI "
        f"score and each scene takes much longer. {choice} for visual analysis."
    )


def predict_budget(configured: int, items: int, per_item: int, num_ctx: int) -> int:
    """num_predict for a batched JSON reply: room for every item, but never more than half the
    context window so the prompt still fits. Never below the configured value."""
    wanted = max(configured, per_item * max(1, items) + 64)
    return min(wanted, max(configured, num_ctx // 2))


def is_memory_error(message: str) -> bool:
    text = (message or "").lower()
    return any(token in text for token in _MEMORY_ERROR_TOKENS)


_USAGE_FIELDS = (
    "calls",
    "prompt_tokens",
    "output_tokens",
    "prompt_sec",
    "output_sec",
    "load_sec",
    "cut_off",
    "near_context_limit",
)


def usage_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: after.get(key, 0) - before.get(key, 0) for key in _USAGE_FIELDS}


def describe_usage(usage: dict[str, float]) -> str:
    """One log line about the model calls of a stage, or "" when there were none."""
    calls = int(usage.get("calls") or 0)
    if not calls:
        return ""
    output_tokens = int(usage.get("output_tokens") or 0)
    output_sec = float(usage.get("output_sec") or 0)
    speed = f" ({output_tokens / output_sec:.1f} tok/s)" if output_sec > 0 else ""
    text = (
        f"{calls} call{'s' if calls != 1 else ''}, "
        f"{int(usage.get('prompt_tokens') or 0):,} prompt tokens in {float(usage.get('prompt_sec') or 0):.1f}s, "
        f"{output_tokens:,} reply tokens in {output_sec:.1f}s{speed}"
    )
    load = float(usage.get("load_sec") or 0)
    if load >= 1:
        text += f", model load {load:.1f}s"
    return text


def _extract_message_text(body: dict[str, Any]) -> str:
    message = body.get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    response = body.get("response")
    if isinstance(response, str) and response.strip():
        return response.strip()
    thinking = message.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        return thinking.strip()
    return ""


class OllamaClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        configured = self.settings.ollama_base_url.rstrip("/")
        if configured.endswith("/api"):
            configured = configured[: -len("/api")]
        self.root = configured
        self._configured_root = configured
        self.timeout = self.settings.ollama_timeout
        self.retries = self.settings.ollama_max_retries
        self._session = requests.Session()
        self.apply_auth()
        self._capabilities: dict[tuple[str, str, str], list[str] | None] = {}
        self._usage = threading.local()
        self._restore_checked_at = 0.0

    def apply_auth(self) -> None:
        """Send the optional API key, for Ollama behind an authenticating proxy on another machine."""
        key = str(getattr(self.settings, "ollama_api_key", "") or "").strip()
        if key:
            self._session.headers["Authorization"] = f"Bearer {key}"
        else:
            self._session.headers.pop("Authorization", None)

    @staticmethod
    def _join(root: str, path: str) -> str:
        return urljoin(root.rstrip("/") + "/", path.lstrip("/"))

    def _url(self, path: str) -> str:
        return self._join(self.root, path)

    def is_local(self) -> bool:
        """True when Ollama runs on this machine, so its disk and GPU are the ones detected here."""
        host = (urlsplit(self.root).hostname or "").lower()
        return host in _LOOPBACK_HOSTS and not running_in_docker()

    def shares_cpu(self) -> bool:
        """True when Ollama runs on this computer (directly, on the Docker host, or in the Compose
        service), so the thread count chosen for this CPU applies to it. A remote server keeps
        its own default, which matches its own CPU or GPU."""
        host = (urlsplit(self.root).hostname or "").lower()
        return host in _LOOPBACK_HOSTS or host in {_DOCKER_HOST, urlsplit(_COMPOSE_OLLAMA).hostname}

    def _thread_option(self) -> dict[str, Any]:
        return {"num_thread": self.settings.ollama_num_thread} if self.shares_cpu() else {}

    def _default_options(self) -> dict[str, Any]:
        return {
            "temperature": 0.1,
            "num_ctx": self.settings.ollama_num_ctx,
            **self._thread_option(),
        }

    def role_options(self, role: str) -> dict[str, Any]:
        """Runner options for one role. Warm-up, probes, and real calls share them because
        Ollama reloads a model whenever num_ctx or num_thread change."""
        num_ctx = self.settings.ollama_vision_num_ctx if role == "vision" else self.settings.ollama_text_num_ctx
        return {"num_ctx": num_ctx, **self._thread_option()}

    def usage_snapshot(self) -> dict[str, float]:
        """Running totals of model work done by chat() on the calling thread."""
        return dict(self._thread_usage())

    def _thread_usage(self) -> dict[str, float]:
        usage = getattr(self._usage, "totals", None)
        if usage is None:
            usage = {key: 0 for key in _USAGE_FIELDS}
            self._usage.totals = usage
        return usage

    def _record_usage(self, body: dict[str, Any], num_ctx: int | None) -> None:
        usage = self._thread_usage()
        prompt_tokens = int(body.get("prompt_eval_count") or 0)
        output_tokens = int(body.get("eval_count") or 0)
        usage["calls"] += 1
        usage["prompt_tokens"] += prompt_tokens
        usage["output_tokens"] += output_tokens
        usage["prompt_sec"] += int(body.get("prompt_eval_duration") or 0) / 1e9
        usage["output_sec"] += int(body.get("eval_duration") or 0) / 1e9
        usage["load_sec"] += int(body.get("load_duration") or 0) / 1e9
        if body.get("done_reason") == "length":
            usage["cut_off"] += 1
        if num_ctx and prompt_tokens + output_tokens >= num_ctx * 0.95:
            usage["near_context_limit"] += 1

    def _extra_roots(self) -> list[str]:
        extra = str(getattr(self.settings, "ollama_fallback_url", "") or "").strip().rstrip("/")
        if extra.endswith("/api"):
            extra = extra[: -len("/api")]
        return [extra] if extra else []

    def _role_for(self, model: str) -> str:
        return "vision" if model == self.settings.ollama_vision_model else "text"

    def _try_fallback_roots(self, method: str, path: str, kwargs: dict[str, Any]) -> requests.Response | None:
        timeout = kwargs.get("timeout")
        read_timeout = timeout if isinstance(timeout, (int, float)) else self.timeout
        trial = {**kwargs, "timeout": (min(_FALLBACK_CONNECT_TIMEOUT, read_timeout), read_timeout)}
        candidates = [self._configured_root, *fallback_roots(self._configured_root), *self._extra_roots()]
        for candidate in dict.fromkeys(candidates):
            if candidate == self.root:
                continue
            try:
                response = self._session.request(method, self._join(candidate, path), **trial)
            except requests.RequestException:
                continue
            self.root = candidate
            return response
        return None

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        quick = path.rstrip("/").endswith(("tags", "version"))
        kwargs.setdefault("timeout", 2 if quick else self.timeout)
        attempts = 1 if quick else max(1, self.retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._session.request(method, self._url(path), **kwargs)
            except requests.ConnectionError as exc:
                last_error = exc
                response = self._try_fallback_roots(method, path, kwargs)
                if response is not None:
                    return response
            except requests.Timeout as exc:
                # The server is up but slow. Sending the same request again only doubles the wait.
                raise OllamaError(f"Ollama timed out after {kwargs['timeout']}s on {path}") from exc
            except requests.RequestException as exc:
                last_error = exc
            if quick or attempt + 1 >= attempts:
                break
            time.sleep(min(2**attempt, 8))
        if isinstance(last_error, requests.ConnectionError):
            raise OllamaConnectionError(str(last_error))
        raise OllamaError(str(last_error))

    def health(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "connected": False,
            "version": None,
            "models": [],
            "model_details": [],
            "running": [],
            "vision_model": self.settings.ollama_vision_model,
            "text_model": self.settings.ollama_text_model,
            "vision_present": False,
            "text_present": False,
            "vision_capable": None,
            "vision_ready": False,
            "warnings": [],
            "base_url": self.root,
            "error": None,
        }
        self._restore_configured_root()
        try:
            version_resp = self._request("GET", "/api/version")
            if version_resp.ok:
                info["version"] = (version_resp.json() or {}).get("version")
            tags_resp = self._request("GET", "/api/tags")
            tags_resp.raise_for_status()
            raw_models = (tags_resp.json() or {}).get("models", [])
            details = []
            for item in raw_models:
                name = item.get("name")
                if not name:
                    continue
                details.append({"name": name, "size": item.get("size") or 0, "digest": item.get("digest")})
            models = [item["name"] for item in details]
            info["models"] = models
            info["model_details"] = details
            info["connected"] = True
            info["base_url"] = self.root
            info["vision_present"] = _model_present(self.settings.ollama_vision_model, models)
            info["text_present"] = _model_present(self.settings.ollama_text_model, models)
            info["running"] = self.running_models()
            self._check_readiness(info, details)
        except Exception as exc:  # noqa: BLE001
            info["error"] = str(exc)
        return info

    def _restore_configured_root(self) -> None:
        """Go back to the configured server once it answers again after a fallback, for
        example a remote GPU server that was briefly offline. Checked at most once a minute
        so status polling stays fast while it is down."""
        if self.root == self._configured_root:
            return
        now = time.monotonic()
        if now - self._restore_checked_at < _RESTORE_CHECK_INTERVAL:
            return
        self._restore_checked_at = now
        try:
            response = self._session.get(
                self._join(self._configured_root, "/api/version"), timeout=_FALLBACK_CONNECT_TIMEOUT
            )
        except requests.RequestException:
            return
        if response.ok:
            self.root = self._configured_root

    def _check_readiness(self, info: dict[str, Any], details: list[dict[str, Any]]) -> None:
        vision = self.settings.ollama_vision_model
        warnings: list[str] = info["warnings"]
        if info["vision_present"]:
            capabilities = self.model_capabilities(vision, _digest_for(vision, details))
            if capabilities is not None:
                info["vision_capable"] = "vision" in capabilities
                if not info["vision_capable"]:
                    warnings.append(
                        f"{vision} cannot read images, so it cannot score video frames. "
                        "Choose a vision model such as qwen3-vl:4b-instruct."
                    )
                elif "thinking" in capabilities:
                    warnings.append(thinking_vision_warning(vision))
        vision_version = version_problem(info.get("version"), vision)
        for model in (vision, self.settings.ollama_text_model):
            problem = version_problem(info.get("version"), model)
            if problem and problem not in warnings:
                warnings.append(problem)
        info["vision_ready"] = bool(
            info["vision_present"] and info["vision_capable"] is not False and not vision_version
        )

    def model_capabilities(self, name: str, digest: str | None = None) -> list[str] | None:
        """Capabilities reported by /api/show, or None when this Ollama does not report them."""
        key = (self.root, name, digest or "")
        if key in self._capabilities:
            return self._capabilities[key]
        try:
            response = self._request("POST", "/api/show", json={"model": name}, timeout=10)
            if not response.ok:
                return None
            body = response.json()
        except (OllamaError, ValueError):
            return None
        raw = body.get("capabilities") if isinstance(body, dict) else None
        capabilities = [str(item) for item in raw] if isinstance(raw, list) else None
        self._capabilities[key] = capabilities
        return capabilities

    def running_models(self) -> list[dict[str, Any]]:
        try:
            response = self._request("GET", "/api/ps")
            if not response.ok:
                return []
            models = (response.json() or {}).get("models") or []
            return [item for item in models if isinstance(item, dict)]
        except Exception:
            return []

    def probe_chat(
        self,
        model: str,
        prompt: str,
        images: list[str] | None = None,
        num_predict: int = 32,
        role: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        user: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            user["images"] = images
        payload = {
            "model": model,
            "messages": [user],
            "stream": False,
            "think": False,
            "keep_alive": self.settings.ollama_keep_alive,
            "options": {
                "temperature": 0,
                "num_predict": num_predict,
                **self.role_options(role or self._role_for(model)),
            },
        }
        response = self._request("POST", "/api/chat", json=payload)
        elapsed = time.perf_counter() - started
        if not response.ok:
            raise OllamaError(_response_error(response))
        body = response.json()
        eval_count = int(body.get("eval_count") or 0)
        eval_ns = int(body.get("eval_duration") or 0)
        load_ns = int(body.get("load_duration") or 0)
        tokens_per_sec = round(eval_count / (eval_ns / 1e9), 2) if eval_ns else None
        return {
            "ok": True,
            "text": _extract_message_text(body),
            "elapsed_sec": round(elapsed, 3),
            "load_sec": round(load_ns / 1e9, 3) if load_ns else None,
            "eval_count": eval_count,
            "tokens_per_sec": tokens_per_sec,
            "total_sec": round(int(body.get("total_duration") or 0) / 1e9, 3),
        }

    def warm_model(self, model: str, role: str | None = None) -> None:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "think": False,
            "keep_alive": self.settings.ollama_keep_alive,
            "options": {
                "num_predict": 1,
                **self.role_options(role or self._role_for(model)),
            },
        }
        response = self._request("POST", "/api/chat", json=payload)
        if not response.ok:
            raise OllamaError(_response_error(response))

    def chat(
        self,
        model: str,
        prompt: str,
        images: list[str] | None = None,
        schema: dict | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        schema_first: bool = True,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        user: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            user["images"] = images
        messages.append(user)

        attempts: list[dict[str, Any]] = []
        if schema and schema_first:
            attempts.append({"format": schema, "messages": copy.deepcopy(messages), "images": images})
        attempts.append({"format": schema or "json", "messages": copy.deepcopy(messages), "images": images})

        chat_options = self._default_options()
        if options:
            chat_options.update(options)

        # Transport failures propagate: _request has already retried them. Only replies that
        # a second try can fix (empty, malformed, or a runner crash) are retried here.
        last_error = None
        for attempt in attempts:
            payload: dict[str, Any] = {
                "model": model,
                "messages": attempt["messages"],
                "stream": False,
                "think": False,
                "keep_alive": self.settings.ollama_keep_alive,
                "format": attempt["format"],
                "options": dict(chat_options),
            }
            if attempt.get("images"):
                payload["messages"][-1]["images"] = attempt["images"]
            for _retry in range(max(1, self.retries + 1)):
                response = self._request("POST", "/api/chat", json=payload)
                if not response.ok:
                    message = _response_error(response)
                    last_error = f"HTTP {response.status_code}: {message}"
                    if response.status_code < 500 or is_memory_error(message):
                        # Missing model, bad request, or out of memory: the same request fails again.
                        raise OllamaError(last_error)
                    continue
                body = response.json()
                if isinstance(body, dict):
                    self._record_usage(body, payload["options"].get("num_ctx"))
                content = _extract_message_text(body)
                done_reason = body.get("done_reason") or "unknown"
                if not content:
                    if done_reason == "length":
                        raise OllamaTruncatedError("The model used its whole output budget without replying.")
                    last_error = f"Empty model response (done_reason={done_reason})"
                    payload["messages"] = attempt["messages"] + [
                        {
                            "role": "user",
                            "content": "Reply with strict JSON only. Do not leave the response empty.",
                        }
                    ]
                    continue
                try:
                    return parse_model_json(content)
                except ValueError as exc:
                    if done_reason == "length":
                        raise OllamaTruncatedError(
                            "The model reply was cut off before the JSON was complete."
                        ) from exc
                    last_error = str(exc)
                    payload["messages"] = attempt["messages"] + [
                        {
                            "role": "user",
                            "content": "Your previous reply was invalid. Reply with strict JSON only matching the schema.",
                        }
                    ]
        raise OllamaError(last_error or "Ollama chat failed")

    def encode_image(self, path: str, max_pixels: int | None = None) -> str:
        limit = max_pixels or self.settings.ollama_vision_max_pixels
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            longest = max(width, height)
            if longest > limit:
                scale = limit / longest
                image = image.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    Image.Resampling.BILINEAR,
                )
            buffer = io.BytesIO()
            quality = getattr(self.settings, "ollama_vision_jpeg_quality", 75)
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            data = buffer.getvalue()
        return base64.b64encode(data).decode("ascii")

    def pull_model(self, name: str, on_progress=None) -> None:
        timeout = max(3600, self.timeout)
        response = self._request(
            "POST",
            "/api/pull",
            json={"name": name, "stream": True},
            timeout=timeout,
            stream=True,
        )
        if not response.ok:
            raise OllamaPullError(_response_error(response, limit=1000))
        finished = False
        for line in response.iter_lines():
            if not line:
                continue
            try:
                body = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(body, dict):
                continue
            # Ollama reports download failures inside the stream after an HTTP 200.
            if body.get("error"):
                raise OllamaPullError(str(body["error"]))
            status = body.get("status") or ""
            if status == "success":
                finished = True
            if on_progress is not None:
                on_progress(
                    {
                        "status": status,
                        "completed": body.get("completed") or 0,
                        "total": body.get("total") or 0,
                    }
                )
        if not finished:
            raise OllamaPullError(f"The download of {name} stopped before Ollama confirmed it.")


def _response_error(response: requests.Response, limit: int = 500) -> str:
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and body.get("error"):
        return str(body["error"])[:limit]
    return (response.text or "")[:limit]


def _digest_for(name: str, details: list[dict[str, Any]]) -> str | None:
    for item in details:
        if item.get("name") == name or str(item.get("name") or "").startswith(name + ":"):
            return item.get("digest")
    return None


def describe_placement(models: list[dict[str, Any]], prefer: str | None = None) -> dict[str, Any]:
    empty = {
        "measured": False,
        "mode": None,
        "label": "Not measured",
        "model": None,
        "gpu_fraction": None,
        "size": None,
        "size_vram": None,
    }
    if not models:
        return empty
    chosen = None
    if prefer:
        for item in models:
            name = str(item.get("name") or item.get("model") or "")
            if name == prefer or name.startswith(prefer + ":"):
                chosen = item
                break
    if chosen is None:
        chosen = max(models, key=lambda item: int(item.get("size") or 0))
    size = int(chosen.get("size") or 0)
    size_vram = int(chosen.get("size_vram") or 0)
    fraction = (size_vram / size) if size else 0.0
    if fraction >= 0.9:
        mode, label = "gpu", "GPU"
    elif fraction <= 0.05:
        mode, label = "cpu", "CPU"
    else:
        cpu_pct = round((1 - fraction) * 100)
        gpu_pct = round(fraction * 100)
        mode, label = "split", f"{cpu_pct}% CPU / {gpu_pct}% GPU"
    return {
        "measured": True,
        "mode": mode,
        "label": label,
        "model": chosen.get("name") or chosen.get("model"),
        "gpu_fraction": round(fraction, 3),
        "size": size,
        "size_vram": size_vram,
    }


def _model_present(name: str, models: list[str]) -> bool:
    if name in models:
        return True
    return any(m == name or m.startswith(name + ":") for m in models)
