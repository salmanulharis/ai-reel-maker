from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from app.ai.ollama import describe_placement
from app.config import Settings, get_settings
from app.services.hardware import detect_hardware
from app.services.model_catalog import CATALOG
from app.services.ollama_runtime import get_ollama_client
from app.services.settings_store import apply_values, effective_settings, read_setting, write_setting

_log = logging.getLogger(__name__)
_lock = threading.Lock()
CACHE_TTL_SECONDS = 3 * 60 * 60
# Part of the fingerprint, so automatic mode applies a changed recommendation once even when the
# hardware is the same. v2: Qwen3-VL instruct tags (the bare tags are thinking models).
RECOMMENDATION_VERSION = "v2"

APPLIED_FIELDS = (
    "ollama_vision_model",
    "ollama_text_model",
    "whisper_model",
    "whisper_device",
    "ollama_vision_batch_size",
    "ollama_vision_num_ctx",
    "ollama_text_num_ctx",
    "ollama_num_thread",
    "whisper_cpu_threads",
)


def fingerprint(hardware: dict[str, Any], ollama_version: str | None) -> str:
    ram = float(hardware.get("ram_gb") or 0)
    bucket = int(ram // 4) * 4
    vram = hardware.get("vram_gb")
    vram_text = f"{float(vram):.1f}" if vram is not None else ""
    return "|".join(
        [
            RECOMMENDATION_VERSION,
            str(hardware.get("cpu") or ""),
            str(bucket),
            str(hardware.get("gpu") or ""),
            vram_text,
            str(ollama_version or ""),
        ]
    )


def resolve_runtime(hardware: dict[str, Any], running: list[dict[str, Any]]) -> dict[str, Any]:
    placement = describe_placement(running)
    expected_gpu = hardware.get("inference_mode") == "gpu"
    if placement["measured"]:
        mode = placement["mode"]
        if mode == "gpu":
            label = "GPU-accelerated"
            hint = ""
        elif mode == "split":
            label = "Partial GPU"
            hint = "Only part of the model is in GPU memory. A smaller vision model is usually faster and more stable."
        elif expected_gpu:
            label = "GPU detected, running on CPU"
            hint = (
                "A GPU is visible, but the model loaded in system memory. Check the driver, "
                "run Ollama on the machine with the GPU, or choose a smaller model. "
                "On AMD, restart Ollama with OLLAMA_VULKAN=1 if ROCm cannot load the model."
            )
        else:
            label = "CPU-only"
            hint = "This model is running on the CPU."
            if _named_gpu(hardware):
                hint = "Graphics were found, but there is not enough dedicated memory to accelerate models. This setup runs on the CPU."
        return {
            "mode": mode,
            "label": label,
            "measured": True,
            "hint": hint,
            "placement": placement,
        }
    if hardware.get("inference_mode") == "gpu":
        return {
            "mode": "gpu",
            "label": "GPU-accelerated",
            "measured": False,
            "hint": "GPU detected. Run Test local AI to confirm the model is using it.",
            "placement": placement,
        }
    hint = "No supported GPU was detected. Models run on the CPU."
    if hardware.get("gpu") and "unavailable" in str(hardware.get("gpu")).lower():
        hint = "A display adapter may be present, but its memory could not be read."
    elif _named_gpu(hardware):
        hint = "A GPU name was found, but its memory was not read. Run Test local AI to see where the model loads."
    return {
        "mode": "cpu",
        "label": "CPU-only",
        "measured": False,
        "hint": hint,
        "placement": placement,
    }


def sync_hardware_profile(
    settings: Settings | None = None,
    *,
    refresh: bool = False,
    include_encoders: bool = False,
) -> dict[str, Any]:
    settings = settings or get_settings()
    with _lock:
        try:
            effective_settings(settings)
        except Exception as exc:  # noqa: BLE001
            _log.debug("Settings overlay skipped: %s", exc)
        cached = _read_cache()
        fresh = _cache_is_fresh(cached)
        encoders_ready = bool(cached and cached.get("hardware", {}).get("encoders_checked"))
        if cached and fresh and not refresh and (not include_encoders or encoders_ready):
            hardware = cached["hardware"]
            ollama = _safe_health(settings)
            runtime = resolve_runtime(hardware, ollama.get("running") or [])
            return _payload(settings, hardware, ollama, runtime, cached, from_cache=True)

        hardware = detect_hardware(include_encoders=include_encoders)
        if cached and not include_encoders and cached.get("hardware", {}).get("encoders_checked"):
            previous = cached["hardware"]
            hardware["encoders"] = previous.get("encoders") or []
            hardware["encoders_checked"] = True
            hardware["encode_acceleration"] = previous.get("encode_acceleration")
            hardware["acceleration"] = previous.get("acceleration")
        ollama = _safe_health(settings)
        current_fp = fingerprint(hardware, ollama.get("version"))
        stored_fp = (cached or {}).get("fingerprint") or read_setting("hw_fingerprint") or ""
        mode = _config_mode(settings)
        auto_summary = (cached or {}).get("auto_summary") or read_setting("hw_auto_summary") or ""
        applied = False
        if mode == "auto" and current_fp != stored_fp:
            auto_summary = apply_recommendation(settings, hardware.get("recommendation") or {})
            applied = True
            if ollama.get("connected"):
                # Presence, capabilities, and version checks depend on the models just selected.
                ollama = _safe_health(settings)
        snapshot = {
            "fingerprint": current_fp,
            "saved_at": time.time(),
            "auto_summary": auto_summary,
            "hardware": hardware,
        }
        _write_cache(snapshot)
        runtime = resolve_runtime(hardware, ollama.get("running") or [])
        payload = _payload(settings, hardware, ollama, runtime, snapshot, from_cache=False)
        payload["profile"]["applied_now"] = applied
        return payload


def apply_recommendation(settings: Settings, recommendation: dict[str, Any]) -> str:
    if not recommendation:
        return ""
    values = {
        "ai_config_mode": "auto",
        "ollama_vision_model": recommendation["vision_model"],
        "ollama_text_model": recommendation["text_model"],
        "whisper_model": recommendation["whisper_model"],
        "whisper_device": recommendation["whisper_device"],
        "ollama_vision_batch_size": recommendation["vision_batch_size"],
        "ollama_vision_num_ctx": recommendation["vision_num_ctx"],
        "ollama_text_num_ctx": recommendation["text_num_ctx"],
        "ollama_num_thread": recommendation["num_thread"],
        "whisper_cpu_threads": recommendation["whisper_threads"],
    }
    apply_values(values, settings)
    label = "GPU" if recommendation.get("inference_mode") == "gpu" else "CPU-only"
    summary = (
        f"Automatic setup applied for {label}: {recommendation['vision_model']}, "
        f"{recommendation['text_model']}, Whisper {recommendation['whisper_model']} "
        f"on {recommendation['whisper_device']}."
    )
    write_setting("hw_auto_summary", summary)
    cached = _read_cache()
    if cached is not None:
        cached["auto_summary"] = summary
        _write_cache(cached)
    _log.info(summary)
    return summary


def schedule_startup_ai(settings: Settings | None = None) -> None:
    settings = settings or get_settings()

    def _run() -> None:
        try:
            sync_hardware_profile(settings)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Hardware profile sync failed: %s", exc)
        try:
            from app.services.ollama_runtime import warm_ollama_models

            warm_ollama_models(settings)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Ollama warm failed: %s", exc)

    thread = threading.Thread(target=_run, daemon=True, name="local-ai-startup")
    thread.start()


def _payload(
    settings: Settings,
    hardware: dict[str, Any],
    ollama: dict[str, Any],
    runtime: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    from_cache: bool,
) -> dict[str, Any]:
    recommendation = hardware.get("recommendation") or {}
    return {
        "hardware": {key: value for key, value in hardware.items() if key != "recommendation"},
        "recommendation": recommendation,
        "catalog": CATALOG,
        "runtime": runtime,
        "ollama": {
            "connected": bool(ollama.get("connected")),
            "version": ollama.get("version"),
            "models": ollama.get("models") or [],
            "model_details": ollama.get("model_details") or [],
            "vision_present": bool(ollama.get("vision_present")),
            "text_present": bool(ollama.get("text_present")),
            "vision_capable": ollama.get("vision_capable"),
            "vision_ready": bool(ollama.get("vision_ready", ollama.get("vision_present"))),
            "warnings": list(ollama.get("warnings") or []),
            "error": ollama.get("error"),
            "base_url": ollama.get("base_url") or settings.ollama_base_url,
        },
        "profile": {
            "mode": _config_mode(settings),
            "fingerprint": snapshot.get("fingerprint") or "",
            "auto_summary": snapshot.get("auto_summary") or "",
            "cached": from_cache,
            "applied_now": False,
        },
    }


def _config_mode(settings: Settings) -> str:
    mode = str(getattr(settings, "ai_config_mode", "auto") or "auto").lower()
    return mode if mode in {"auto", "custom"} else "auto"


def _safe_health(settings: Settings) -> dict[str, Any]:
    try:
        return get_ollama_client(settings).health()
    except Exception as exc:  # noqa: BLE001
        return {"connected": False, "version": None, "models": [], "running": [], "error": str(exc)}


def _named_gpu(hardware: dict[str, Any]) -> bool:
    name = str(hardware.get("gpu") or "")
    if not name or "unavailable" in name.lower():
        return False
    return bool(hardware.get("gpu_vendor") or hardware.get("cuda") or (hardware.get("vram_gb") or 0) >= 2)


def _read_cache() -> dict[str, Any] | None:
    raw = read_setting("hw_profile_json")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _cache_is_fresh(cached: dict[str, Any] | None) -> bool:
    if not cached:
        return False
    if not str(cached.get("fingerprint") or "").startswith(RECOMMENDATION_VERSION + "|"):
        return False
    saved = float(cached.get("saved_at") or 0)
    return (time.time() - saved) < CACHE_TTL_SECONDS


def _write_cache(snapshot: dict[str, Any]) -> None:
    write_setting("hw_profile_json", json.dumps(snapshot))
    write_setting("hw_fingerprint", str(snapshot.get("fingerprint") or ""))
    if snapshot.get("auto_summary"):
        write_setting("hw_auto_summary", str(snapshot["auto_summary"]))
