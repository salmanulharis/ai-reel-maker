from __future__ import annotations

import base64
import copy
import io
import json
import time
from typing import Any
from urllib.parse import urljoin

import requests
from PIL import Image

from app.ai.json_util import parse_model_json
from app.config import Settings, get_settings

_LOCAL_OLLAMA = "http://127.0.0.1:11434"


class OllamaError(RuntimeError):
    pass


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
        self._fallback_root = (
            _LOCAL_OLLAMA if "host.docker.internal" in self.root and self.root != _LOCAL_OLLAMA else None
        )
        self.timeout = self.settings.ollama_timeout
        self.retries = self.settings.ollama_max_retries
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        return urljoin(self.root.rstrip("/") + "/", path.lstrip("/"))

    def _switch_to_fallback_root(self) -> bool:
        if self._fallback_root and self.root != self._fallback_root:
            self.root = self._fallback_root
            return True
        return False

    def _default_options(self) -> dict[str, Any]:
        return {
            "temperature": 0.1,
            "num_ctx": self.settings.ollama_num_ctx,
            "num_thread": self.settings.ollama_num_thread,
        }

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        kwargs.setdefault(
            "timeout",
            2 if path.rstrip("/").endswith(("tags", "version")) else self.timeout,
        )
        last_error = None
        quick = path.rstrip("/").endswith(("tags", "version"))
        attempts = 1 if quick else max(1, self.retries + 1)
        for attempt in range(attempts):
            try:
                response = self._session.request(method, self._url(path), **kwargs)
                return response
            except requests.RequestException as exc:
                last_error = exc
                if self._switch_to_fallback_root():
                    try:
                        return self._session.request(method, self._url(path), **kwargs)
                    except requests.RequestException as retry_exc:
                        last_error = retry_exc
                if quick:
                    break
                time.sleep(min(2**attempt, 8))
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
            "base_url": self.root,
            "error": None,
        }
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
                details.append({"name": name, "size": item.get("size") or 0})
            models = [item["name"] for item in details]
            info["models"] = models
            info["model_details"] = details
            info["connected"] = True
            info["base_url"] = self.root
            info["vision_present"] = _model_present(self.settings.ollama_vision_model, models)
            info["text_present"] = _model_present(self.settings.ollama_text_model, models)
            info["running"] = self.running_models()
        except Exception as exc:  # noqa: BLE001
            info["error"] = str(exc)
        return info

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
                "num_ctx": 1024,
                "num_thread": self.settings.ollama_num_thread,
            },
        }
        response = self._request("POST", "/api/chat", json=payload)
        elapsed = time.perf_counter() - started
        if not response.ok:
            raise OllamaError(response.text[:500])
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

    def warm_model(self, model: str) -> None:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "think": False,
            "keep_alive": self.settings.ollama_keep_alive,
            "options": {
                "num_predict": 1,
                "num_ctx": 512,
                "num_thread": self.settings.ollama_num_thread,
            },
        }
        response = self._request("POST", "/api/chat", json=payload)
        if not response.ok:
            raise OllamaError(response.text[:500])

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
        if images and len(images) > 1:
            attempts.append(
                {
                    "format": "json",
                    "messages": copy.deepcopy(messages),
                    "images": [images[len(images) // 2]],
                }
            )

        chat_options = self._default_options()
        if options:
            chat_options.update(options)

        last_error = None
        for attempt_index, attempt in enumerate(attempts):
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
            for retry in range(max(1, self.retries + 1)):
                try:
                    response = self._request("POST", "/api/chat", json=payload)
                    if not response.ok:
                        last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                        continue
                    body = response.json()
                    content = _extract_message_text(body)
                    if not content:
                        done_reason = body.get("done_reason") or "unknown"
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
                        last_error = str(exc)
                        payload["messages"] = attempt["messages"] + [
                            {
                                "role": "user",
                                "content": "Your previous reply was invalid. Reply with strict JSON only matching the schema.",
                            }
                        ]
                except OllamaError as exc:
                    last_error = str(exc)
            if attempt_index + 1 < len(attempts):
                continue
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
            raise OllamaError(response.text[:1000])
        for line in response.iter_lines():
            if not line or on_progress is None:
                continue
            try:
                body = json.loads(line)
            except json.JSONDecodeError:
                continue
            on_progress(
                {
                    "status": body.get("status") or "",
                    "completed": body.get("completed") or 0,
                    "total": body.get("total") or 0,
                }
            )


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
