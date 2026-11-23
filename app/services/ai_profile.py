from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from app.ai.ollama import describe_placement, model_present
from app.config import Settings, get_settings
from app.services.hardware import GPU_CONFIRMED_KEY, detect_hardware
from app.services.model_catalog import CATALOG, normalize_goal
from app.services.ollama_runtime import get_ollama_client
from app.services.settings_store import apply_values, effective_settings, read_setting, write_setting

_log = logging.getLogger(__name__)
_lock = threading.Lock()
CACHE_TTL_SECONDS = 3 * 60 * 60
# Part of the fingerprint, so automatic mode applies a changed recommendation once even when the
# hardware is the same. v2: Qwen3-VL instruct tags (the bare tags are thinking models).
# v3: Qwen3 4B instruct text tag, for the same reason.
# v4: memory sizes read as their label (15.8 GB is 16 GB), batch sizes that fit their context
# window, and goals.
# v5: the AI editor's context window and whether it reasons first (prompt mode).
RECOMMENDATION_VERSION = "v5"

# The settings automatic optimization manages, and where each comes from in a recommendation.
# Nothing else (video quality, format, visual analysis, connection, frame rate) is ever changed.
FIELD_SOURCES: dict[str, str] = {
    "ollama_vision_model": "vision_model",
    "ollama_text_model": "text_model",
    "whisper_model": "whisper_model",
    "whisper_device": "whisper_device",
    "ollama_vision_batch_size": "vision_batch_size",
    "ollama_vision_num_ctx": "vision_num_ctx",
    "ollama_text_num_ctx": "text_num_ctx",
    "ollama_num_thread": "num_thread",
    "whisper_cpu_threads": "whisper_threads",
    "ollama_director_num_ctx": "director_num_ctx",
    "director_think": "director_think",
}
APPLIED_FIELDS = tuple(FIELD_SOURCES)
# Ollama models: applied only once they are installed, so a job never loses its working model.
MODEL_FIELDS = ("ollama_vision_model", "ollama_text_model")
# Settings that describe the computer Ollama runs on. They are left alone when Ollama runs
# elsewhere, because this computer's hardware says nothing about that one.
OLLAMA_HOST_FIELDS = (
    *MODEL_FIELDS,
    "ollama_vision_batch_size",
    "ollama_vision_num_ctx",
    "ollama_text_num_ctx",
    "ollama_director_num_ctx",
    "director_think",
)
_HELD_KEY = "hw_auto_held"


def goal_of(settings: Settings) -> str:
    return normalize_goal(getattr(settings, "ai_goal", None))


def pinned_fields(settings: Settings) -> set[str]:
    """Managed settings the user chose by hand while automatic optimization is on."""
    raw = str(getattr(settings, "ai_auto_pinned", "") or "")
    return {key for key in (part.strip() for part in raw.split(",")) if key in FIELD_SOURCES}


def save_pins(settings: Settings, keys: set[str]) -> None:
    apply_values({"ai_auto_pinned": ",".join(key for key in FIELD_SOURCES if key in keys)}, settings)


def held_fields() -> list[str]:
    """Model settings waiting for their recommended model to be downloaded."""
    try:
        data = json.loads(read_setting(_HELD_KEY) or "[]")
    except json.JSONDecodeError:
        return []
    return [key for key in data if key in MODEL_FIELDS] if isinstance(data, list) else []


def ollama_on_this_computer(settings: Settings) -> bool:
    check = getattr(get_ollama_client(settings), "shares_cpu", None)
    return bool(check()) if callable(check) else True


def installed_models(ollama: dict[str, Any] | None) -> set[str] | None:
    """Installed Ollama models, or None when Ollama did not answer (so nothing is known)."""
    if not ollama or not ollama.get("connected"):
        return None
    return set(ollama.get("models") or [])


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
            str(hardware.get("inference_mode") or ""),
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
        goal = goal_of(settings)
        cached = _read_cache()
        fresh = _cache_is_fresh(cached) and (cached or {}).get("goal", goal) == goal
        encoders_ready = bool(cached and cached.get("hardware", {}).get("encoders_checked"))
        ollama = _safe_health(settings)
        usable = cached and fresh and not refresh and (not include_encoders or encoders_ready)
        if usable and not _confirm_gpu(cached["hardware"], ollama, settings):
            hardware = cached["hardware"]
            if _apply_held(settings, hardware.get("recommendation") or {}, ollama):
                ollama = _safe_health(settings)
            runtime = resolve_runtime(hardware, ollama.get("running") or [])
            return _payload(settings, hardware, ollama, runtime, cached, from_cache=True)

        hardware = detect_hardware(include_encoders=include_encoders, goal=goal)
        if _confirm_gpu(hardware, ollama, settings):
            hardware = detect_hardware(include_encoders=include_encoders, goal=goal)
        if cached and not include_encoders and cached.get("hardware", {}).get("encoders_checked"):
            previous = cached["hardware"]
            hardware["encoders"] = previous.get("encoders") or []
            hardware["encoders_checked"] = True
            hardware["encode_acceleration"] = previous.get("encode_acceleration")
            hardware["acceleration"] = previous.get("acceleration")
        current_fp = fingerprint(hardware, ollama.get("version"))
        stored_fp = (cached or {}).get("fingerprint") or read_setting("hw_fingerprint") or ""
        mode = _config_mode(settings)
        auto_summary = (cached or {}).get("auto_summary") or read_setting("hw_auto_summary") or ""
        applied = False
        if mode == "auto" and current_fp != stored_fp:
            auto_summary = apply_recommendation(settings, hardware.get("recommendation") or {}, ollama)
            applied = True
        elif _apply_held(settings, hardware.get("recommendation") or {}, ollama):
            applied = True
        if applied and ollama.get("connected"):
            # Presence, capabilities, and version checks depend on the models just selected.
            ollama = _safe_health(settings)
        snapshot = {
            "fingerprint": current_fp,
            "goal": goal,
            "saved_at": time.time(),
            "auto_summary": auto_summary,
            "hardware": hardware,
        }
        _write_cache(snapshot)
        runtime = resolve_runtime(hardware, ollama.get("running") or [])
        payload = _payload(settings, hardware, ollama, runtime, snapshot, from_cache=False)
        payload["profile"]["applied_now"] = applied
        return payload


def managed_now(settings: Settings) -> set[str]:
    """The settings automatic optimization may change right now: everything it manages, minus
    the user's own choices, minus the Ollama settings when Ollama runs on another computer."""
    skip = pinned_fields(settings)
    if not ollama_on_this_computer(settings):
        skip |= set(OLLAMA_HOST_FIELDS)
    return {key for key in FIELD_SOURCES if key not in skip}


def waits_for_download(settings: Settings, key: str, recommended: Any, installed: set[str] | None) -> bool:
    """True when switching to the recommended model would replace a working model with one
    that is not downloaded yet. Unknown (Ollama offline) or nothing installed: do not wait."""
    if key not in MODEL_FIELDS or installed is None:
        return False
    current = str(getattr(settings, key, "") or "")
    return (
        str(recommended) != current
        and not model_present(str(recommended), installed)
        and model_present(current, installed)
    )


def apply_recommendation(
    settings: Settings, recommendation: dict[str, Any], ollama: dict[str, Any] | None = None
) -> str:
    """Turn automatic mode on and apply the recommendation to the settings it may change.
    ``ollama`` is a health report; with it, a recommended model that is not installed waits
    for its download while the current model keeps working."""
    if not recommendation:
        return ""
    installed = installed_models(ollama)
    values: dict[str, Any] = {"ai_config_mode": "auto"}
    held = []
    allowed = managed_now(settings)
    for key in FIELD_SOURCES:
        if key not in allowed:
            continue
        value = recommendation[FIELD_SOURCES[key]]
        if waits_for_download(settings, key, value, installed):
            held.append(key)
            continue
        values[key] = value
    apply_values(values, settings)
    write_setting(_HELD_KEY, json.dumps(held))
    label = "GPU" if recommendation.get("inference_mode") == "gpu" else "CPU-only"
    summary = (
        f"Automatic setup applied for {label}: {settings.ollama_vision_model}, "
        f"{settings.ollama_text_model}, Whisper {settings.whisper_model} "
        f"on {settings.whisper_device}."
    )
    if held:
        waiting = ", ".join(str(recommendation[FIELD_SOURCES[key]]) for key in held)
        summary += f" Waiting for download: {waiting}."
    write_setting("hw_auto_summary", summary)
    cached = _read_cache()
    if cached is not None:
        cached["auto_summary"] = summary
        _write_cache(cached)
    _log.info(summary)
    return summary


def _apply_held(settings: Settings, recommendation: dict[str, Any], ollama: dict[str, Any]) -> bool:
    """Switch to recommended models that were waiting for their download. True when any changed."""
    held = held_fields()
    installed = installed_models(ollama)
    if not held or installed is None or _config_mode(settings) != "auto" or not recommendation:
        return False
    allowed = managed_now(settings)
    ready = {
        key: recommendation[FIELD_SOURCES[key]]
        for key in held
        if key in allowed and model_present(str(recommendation[FIELD_SOURCES[key]]), installed)
    }
    # A setting the user has since chosen by hand no longer waits.
    remaining = [key for key in held if key in allowed and key not in ready]
    if remaining != held:
        write_setting(_HELD_KEY, json.dumps(remaining))
    if not ready:
        return False
    apply_values(ready, settings)
    _log.info("Downloaded models are now in use: %s", ", ".join(str(value) for value in ready.values()))
    return True


def _confirm_gpu(hardware: dict[str, Any], ollama: dict[str, Any], settings: Settings) -> bool:
    """Remember a separate graphics card once Ollama on this computer has a model loaded on it.
    True when that was just learned, so hardware should be read again."""
    if not hardware.get("gpu_unconfirmed") or not hardware.get("gpu"):
        return False
    placement = describe_placement(ollama.get("running") or [])
    if placement.get("mode") not in {"gpu", "split"}:
        return False
    is_local = getattr(get_ollama_client(settings), "is_local", None)
    if not callable(is_local) or not is_local():
        return False
    write_setting(GPU_CONFIRMED_KEY, str(hardware["gpu"]))
    _log.info("Ollama is using %s; larger models can now be recommended", hardware["gpu"])
    return True


def cached_hardware() -> dict[str, Any]:
    """The last detected hardware, or a fresh quick detection when none is stored yet."""
    hardware = (_read_cache() or {}).get("hardware")
    return hardware if isinstance(hardware, dict) and hardware else detect_hardware()


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
            "goal": goal_of(settings),
            "pinned": sorted(pinned_fields(settings)),
            "held": held_fields(),
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
