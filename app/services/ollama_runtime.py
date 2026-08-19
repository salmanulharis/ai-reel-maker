from __future__ import annotations

import logging
import threading

from app.ai.ollama import OllamaClient
from app.config import Settings, get_settings

_log = logging.getLogger(__name__)
_client: OllamaClient | None = None
_lock = threading.Lock()
_warm_started = False


def get_ollama_client(settings: Settings | None = None) -> OllamaClient:
    global _client
    settings = settings or get_settings()
    with _lock:
        if _client is None:
            _client = OllamaClient(settings)
        return _client


def warm_ollama_models(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    if not settings.ollama_warm_on_start:
        return
    client = get_ollama_client(settings)
    health = client.health()
    if not health.get("connected"):
        return
    for model in (settings.ollama_text_model, settings.ollama_vision_model):
        if not model:
            continue
        try:
            client.warm_model(model)
            _log.info("Warmed Ollama model %s", model)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Failed to warm Ollama model %s: %s", model, exc)


def schedule_ollama_warm(settings: Settings | None = None) -> None:
    global _warm_started
    settings = settings or get_settings()
    if not settings.ollama_warm_on_start:
        return
    with _lock:
        if _warm_started:
            return
        _warm_started = True

    thread = threading.Thread(
        target=warm_ollama_models,
        args=(settings,),
        daemon=True,
        name="ollama-warm",
    )
    thread.start()
