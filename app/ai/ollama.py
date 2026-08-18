from __future__ import annotations

import base64
import json
import time
from typing import Any
from urllib.parse import urljoin

import requests

from app.ai.json_util import parse_model_json
from app.config import Settings, get_settings


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.ollama_base_url.rstrip("/")
        if self.base_url.endswith("/api"):
            self.root = self.base_url[: -len("/api")]
        else:
            self.root = self.base_url
        self.timeout = self.settings.ollama_timeout
        self.retries = self.settings.ollama_max_retries

    def _url(self, path: str) -> str:
        return urljoin(self.root.rstrip("/") + "/", path.lstrip("/"))

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
                response = requests.request(method, self._url(path), **kwargs)
                return response
            except requests.RequestException as exc:
                last_error = exc
                if quick:
                    break
                time.sleep(min(2 ** attempt, 8))
        raise OllamaError(str(last_error))

    def health(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "connected": False,
            "version": None,
            "models": [],
            "vision_model": self.settings.ollama_vision_model,
            "text_model": self.settings.ollama_text_model,
            "vision_present": False,
            "text_present": False,
            "base_url": self.settings.ollama_base_url,
            "error": None,
        }
        try:
            version_resp = self._request("GET", "/api/version")
            if version_resp.ok:
                info["version"] = (version_resp.json() or {}).get("version")
            tags_resp = self._request("GET", "/api/tags")
            tags_resp.raise_for_status()
            models = [m.get("name") for m in (tags_resp.json() or {}).get("models", []) if m.get("name")]
            info["models"] = models
            info["connected"] = True
            info["vision_present"] = _model_present(self.settings.ollama_vision_model, models)
            info["text_present"] = _model_present(self.settings.ollama_text_model, models)
        except Exception as exc:  # noqa: BLE001
            info["error"] = str(exc)
        return info

    def chat(
        self,
        model: str,
        prompt: str,
        images: list[str] | None = None,
        schema: dict | None = None,
        system: str | None = None,
    ) -> dict[str, Any]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        user: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            user["images"] = images
        messages.append(user)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        if schema:
            payload["format"] = schema
        else:
            payload["format"] = "json"
        last_error = None
        for attempt in range(max(1, self.retries + 1)):
            try:
                response = self._request("POST", "/api/chat", json=payload)
                if not response.ok:
                    last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                    continue
                body = response.json()
                content = ((body.get("message") or {}).get("content")) or body.get("response") or ""
                try:
                    return parse_model_json(content if isinstance(content, str) else json.dumps(content))
                except ValueError as exc:
                    last_error = str(exc)
                    payload["messages"] = messages + [
                        {
                            "role": "user",
                            "content": "Your previous reply was invalid. Reply with strict JSON only matching the schema.",
                        }
                    ]
            except OllamaError as exc:
                last_error = str(exc)
        raise OllamaError(last_error or "Ollama chat failed")

    def encode_image(self, path: str) -> str:
        data = open(path, "rb").read()
        return base64.b64encode(data).decode("ascii")

    def pull_model(self, name: str) -> None:
        response = self._request("POST", "/api/pull", json={"name": name, "stream": False}, timeout=self.timeout)
        if not response.ok:
            raise OllamaError(response.text[:1000])


def _model_present(name: str, models: list[str]) -> bool:
    if name in models:
        return True
    base = name.split(":")[0]
    return any(m == name or m.startswith(base + ":") or m.startswith(name) for m in models)
