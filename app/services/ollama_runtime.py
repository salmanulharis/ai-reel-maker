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
    configured = settings.ollama_base_url.rstrip("/")
    with _lock:
        if _client is None or getattr(_client, "configured_url", None) != configured:
            _client = OllamaClient(settings)
            _client.configured_url = configured
        else:
            _client.settings = settings
            _client.timeout = settings.ollama_timeout
            _client.retries = settings.ollama_max_retries
        return _client


def warm_ollama_models(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    if not settings.ollama_warm_on_start:
        return
    client = get_ollama_client(settings)
    health = client.health()
    if not health.get("connected"):
        return
    models = _models_to_warm(settings, health)
    for model in models:
        try:
            client.warm_model(model)
            _log.info("Warmed Ollama model %s", model)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Failed to warm Ollama model %s: %s", model, exc)


def _models_to_warm(settings: Settings, health: dict) -> list[str]:
    from app.services.hardware import detect_hardware

    hardware = detect_hardware()
    recommendation = hardware.get("recommendation") or {}
    installed = set(health.get("models") or [])
    candidates = [settings.ollama_text_model]
    if recommendation.get("warm_both"):
        candidates.append(settings.ollama_vision_model)
    elif settings.ollama_text_model not in installed:
        candidates.append(settings.ollama_vision_model)
    warmed: list[str] = []
    for model in candidates:
        if model and model not in warmed and _installed(model, installed):
            warmed.append(model)
    return warmed


def _installed(model: str, installed: set[str]) -> bool:
    if model in installed:
        return True
    return any(name == model or name.startswith(model + ":") for name in installed)


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
