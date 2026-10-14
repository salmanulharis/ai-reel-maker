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
            _client.apply_auth()
        return _client


def warm_ollama_models(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    if not settings.ollama_warm_on_start:
        return
    client = get_ollama_client(settings)
    health = client.health()
    if not health.get("connected"):
        return
    from app.services.hardware import detect_hardware

    recommendation = detect_hardware().get("recommendation") or {}
    models = _models_to_warm(settings, health, warm_both=bool(recommendation.get("warm_both")))
    for model, role in models:
        try:
            client.warm_model(model, role=role)
            _log.info("Warmed Ollama model %s", model)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Failed to warm Ollama model %s: %s", model, exc)


def _models_to_warm(settings: Settings, health: dict, *, warm_both: bool) -> list[tuple[str, str]]:
    """Models to load ahead of the first job, in the order a job uses them.

    Transcript scoring runs before visual analysis (it decides which scenes the vision model
    looks at), so the text model goes first. Without room for both, only the first one that is
    installed is loaded.
    """
    installed = set(health.get("models") or [])
    order = [(settings.ollama_text_model, "text")] if settings.enable_whisper else []
    if settings.enable_visual_analysis and health.get("vision_ready", True):
        order.append((settings.ollama_vision_model, "vision"))
    warmed: list[tuple[str, str]] = []
    for model, role in order:
        if not model or any(model == name for name, _ in warmed) or not _installed(model, installed):
            continue
        warmed.append((model, role))
        if not warm_both:
            break
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
