"""Automatic optimization: tune the AI settings to this computer, explain each choice, and undo.

Hardware limits are decided by deterministic detection and rules (app.services.hardware and
model_catalog), never by asking a model. This module adds what people need around those rules:
a review of what would change and why, trade-offs stated plainly, safety warnings, an undo
snapshot, and suggestions from recent runs. Suggestions are never applied automatically, so
settings do not drift from job to job.
"""

from __future__ import annotations

import json
import time
from typing import Any

from app.ai.context_fit import vision_scene_cost
from app.config import Settings, get_settings
from app.services import calibration
from app.services.ai_profile import (
    FIELD_SOURCES,
    apply_recommendation,
    goal_of,
    installed_models,
    managed_now,
    pinned_fields,
    save_pins,
    sync_hardware_profile,
    waits_for_download,
)
from app.services.model_catalog import GOALS, model_memory_gb
from app.services.ollama_runtime import get_ollama_client
from app.services.run_plan import BASE_MEMORY_GB
from app.services.settings_store import apply_values, read_setting, write_setting

# The settings automatic optimization manages (see ai_profile.FIELD_SOURCES).
MANAGED = FIELD_SOURCES

LABELS = {
    "ollama_vision_model": "Video understanding model",
    "ollama_text_model": "Speech understanding model",
    "whisper_model": "Speech to text model",
    "whisper_device": "Speech to text runs on",
    "ollama_vision_batch_size": "Pictures per AI request",
    "ollama_vision_num_ctx": "Video model memory",
    "ollama_text_num_ctx": "Speech model memory",
    "ollama_num_thread": "Processor threads for AI",
    "whisper_cpu_threads": "Processor threads for speech to text",
    "ollama_director_num_ctx": "AI editor memory",
    "director_think": "AI editor thinks first",
}

_SNAPSHOT_KEY = "auto_tune_previous"
_HISTORY_KEY = "perf_history"
_HISTORY_SIZE = 5
_LOW_DISK_GB = 10.0
# Thresholds for suggestions from recent runs.
_SLOW_VISION_SECONDS = 20.0
_SLOW_TEXT_TOKENS_PER_SEC = 3.0


def build_plan(settings: Settings | None = None, *, refresh: bool = False) -> dict[str, Any]:
    """Everything the Settings page shows: hardware, each managed setting with its reason,
    pending changes, trade-offs, warnings, undo availability, and tips from recent runs."""
    settings = settings or get_settings()
    profile = sync_hardware_profile(settings, refresh=refresh)
    hardware = profile.get("hardware") or {}
    recommendation = profile.get("recommendation") or {}
    mode = (profile.get("profile") or {}).get("mode") or "auto"
    runtime = profile.get("runtime") or {}
    installed = installed_models(profile.get("ollama"))
    pinned = pinned_fields(settings) if mode == "auto" else set()
    managed = managed_now(settings)

    items = []
    for key, rec_key in MANAGED.items():
        if rec_key not in recommendation:
            continue
        current = getattr(settings, key, None)
        recommended = recommendation[rec_key]
        items.append(
            {
                "key": key,
                "label": LABELS[key],
                "current": current,
                "recommended": recommended,
                "changed": str(current) != str(recommended),
                "reason": _reason(key, recommendation, hardware),
                # The user's own choice, kept while the other settings stay automatic.
                "pinned": key in pinned,
                # The recommended model is not downloaded yet; the current one stays in use.
                "needs_download": waits_for_download(settings, key, recommended, installed),
                # False for Ollama settings when Ollama runs on another computer.
                "managed": key in managed or key in pinned,
            }
        )
    snapshot = read_snapshot()
    notes = list(recommendation.get("notes") or [])
    graphics = builtin_graphics_note(hardware, runtime, settings)
    if graphics:
        notes.append(graphics)
    if hardware.get("gpu_unconfirmed") and hardware.get("vram_note"):
        notes.append(f"{hardware.get('gpu')} ({hardware.get('vram_gb')} GB): {hardware['vram_note']}")
    history = read_history()
    goal = goal_of(settings)
    speeds = calibration.current_speeds(settings, history)
    return {
        "enabled": mode == "auto",
        "goal": goal,
        "hardware": hardware_summary(hardware),
        "items": items,
        "changes": [item for item in items if item["changed"] and item["managed"] and not item["pinned"]],
        "tradeoffs": tradeoffs(recommendation, hardware),
        "warnings": warnings(recommendation, hardware, runtime, settings),
        "notes": notes,
        "can_undo": bool(snapshot and snapshot.get("values")),
        "undo_saved_at": (snapshot or {}).get("saved_at"),
        "history": history,
        "tips": tips(history, settings, hardware) + calibration.suggestions(settings, hardware, goal, speeds),
        "estimate": calibration.estimate(speeds, settings),
        "measurement": calibration.status(),
    }


def enable(settings: Settings | None = None) -> dict[str, Any]:
    """Turn automatic optimization on: remember the current values, then apply the recommendation."""
    settings = settings or get_settings()
    if settings.ai_config_mode != "auto":
        # Choices made by hand while it was off are replaced; pins only exist while it is on.
        save_pins(settings, set())
    profile = sync_hardware_profile(settings, refresh=True)
    _apply(settings, profile)
    return build_plan(settings)


def _apply(settings: Settings, profile: dict[str, Any]) -> None:
    """Apply the profile's recommendation, keeping an undo point when anything will change."""
    recommendation = profile.get("recommendation") or {}
    current = {key: getattr(settings, key, None) for key in MANAGED}
    allowed = managed_now(settings)
    if any(str(current[key]) != str(recommendation.get(MANAGED[key])) for key in allowed):
        _save_snapshot(current)
    apply_recommendation(settings, recommendation, profile.get("ollama"))


def disable(settings: Settings | None = None) -> dict[str, Any]:
    """Turn it off. Current values stay as they are; from now on they are only changed by hand."""
    settings = settings or get_settings()
    apply_values({"ai_config_mode": "custom"}, settings)
    save_pins(settings, set())
    return build_plan(settings)


def set_goal(goal: str, settings: Settings | None = None) -> dict[str, Any]:
    """Choose what automatic optimization aims for. While it is on, the new recommendation is
    applied at once (with an undo point); while it is off, only the recommendation changes."""
    settings = settings or get_settings()
    if goal not in GOALS:
        raise ValueError("goal must be quality, balanced or fast")
    apply_values({"ai_goal": goal}, settings)
    profile = sync_hardware_profile(settings, refresh=True)
    if settings.ai_config_mode == "auto":
        _apply(settings, profile)
    return build_plan(settings)


def set_pin(key: str, pinned: bool, settings: Settings | None = None) -> dict[str, Any]:
    """Keep one managed setting as the user's own choice, or hand it back to automatic
    optimization (which then applies the recommended value)."""
    settings = settings or get_settings()
    if key not in MANAGED:
        raise ValueError("unknown setting")
    pins = pinned_fields(settings)
    pins = pins | {key} if pinned else pins - {key}
    save_pins(settings, pins)
    if not pinned and settings.ai_config_mode == "auto":
        profile = sync_hardware_profile(settings)
        apply_recommendation(settings, profile.get("recommendation") or {}, profile.get("ollama"))
    return build_plan(settings)


def undo(settings: Settings | None = None) -> dict[str, Any]:
    """Restore the values from before the last automatic optimization and switch to manual."""
    settings = settings or get_settings()
    snapshot = read_snapshot()
    values = dict((snapshot or {}).get("values") or {})
    restore = {key: value for key, value in values.items() if key in MANAGED and value is not None}
    apply_values({**restore, "ai_config_mode": "custom"}, settings)
    save_pins(settings, set())
    write_setting(_SNAPSHOT_KEY, "")
    return build_plan(settings)


def read_snapshot() -> dict[str, Any] | None:
    return _read_json(_SNAPSHOT_KEY, dict)


def _save_snapshot(values: dict[str, Any]) -> None:
    write_setting(_SNAPSHOT_KEY, json.dumps({"values": values, "saved_at": time.time()}))


# Explanations


def hardware_summary(hardware: dict[str, Any]) -> dict[str, Any]:
    gpu = hardware.get("gpu")
    vram = hardware.get("vram_gb")
    uses_gpu = hardware.get("inference_mode") == "gpu"
    parts = []
    if hardware.get("cpu_cores"):
        threads = hardware.get("cpu_threads")
        parts.append(f"{hardware['cpu_cores']}-core processor" + (f" ({threads} threads)" if threads else ""))
    if hardware.get("ram_gb"):
        parts.append(f"{hardware['ram_gb']} GB memory")
    if uses_gpu and gpu:
        parts.append(f"{gpu}" + (f" with {vram} GB graphics memory" if vram else ""))
    else:
        parts.append("no graphics card the AI can use")
    return {
        "text": ", ".join(parts),
        "details": hardware_details(hardware),
        "system": hardware.get("system"),
        "cpu": hardware.get("cpu"),
        "cores": hardware.get("cpu_cores"),
        "threads": hardware.get("cpu_threads"),
        "ram_gb": hardware.get("ram_gb"),
        "ram_available_gb": hardware.get("ram_available_gb"),
        "disk_free_gb": hardware.get("disk_free_gb"),
        "gpu": gpu,
        "vram_gb": vram,
        "uses_gpu": uses_gpu,
    }


def hardware_details(hardware: dict[str, Any]) -> list[str]:
    """Finer points of what was detected, for people who want to check it."""
    found = []
    if hardware.get("cpu"):
        found.append(str(hardware["cpu"]))
    fast, slow = hardware.get("cpu_performance_cores"), hardware.get("cpu_efficiency_cores")
    if fast and slow:
        found.append(f"{fast} performance cores and {slow} efficiency cores; AI uses the performance cores")
    features = [name.upper().replace("AVX512", "AVX-512") for name in hardware.get("cpu_features") or [] if name != "avx"]
    if features:
        found.append("Fast math instructions: " + ", ".join(features))
    for gpu in hardware.get("gpus") or []:
        if gpu.get("gpu_integrated"):
            found.append(f"{gpu.get('gpu')}: built-in graphics, shares system memory")
        elif gpu.get("vram_gb"):
            found.append(f"{gpu.get('gpu')}: {gpu['vram_gb']} GB graphics memory")
        elif gpu.get("gpu"):
            found.append(f"{gpu['gpu']}: memory could not be read")
    if hardware.get("gpu_temp_c") is not None:
        found.append(f"Graphics card temperature: {hardware['gpu_temp_c']:.0f} °C")
    if hardware.get("on_battery"):
        found.append("Running on battery")
    return found


def _reason(key: str, rec: dict[str, Any], hw: dict[str, Any]) -> str:
    gpu = rec.get("inference_mode") == "gpu"
    ram = hw.get("ram_gb")
    vram = hw.get("vram_free_gb") or hw.get("vram_gb")
    cores = hw.get("cpu_cores") or hw.get("cpu_threads")
    if key in {"ollama_vision_model", "ollama_text_model"}:
        if gpu:
            return f"The largest model that fits in the {vram} GB of free graphics memory, so it runs fully on the graphics card."
        return f"Runs on the processor. The most capable model that stays practical with {ram} GB of memory."
    if key == "whisper_model":
        if rec.get("whisper_device") == "cuda":
            return "An accurate model that the graphics card handles quickly."
        return "Accurate for clear speech and quick on the processor."
    if key == "whisper_device":
        if rec.get("whisper_device") == "cuda":
            return "The NVIDIA graphics card has memory to spare, so speech to text uses it."
        return "No NVIDIA graphics card with spare memory, so the processor is used. This is the reliable choice."
    if key == "ollama_vision_batch_size":
        return "How many pictures fit in one request without running out of memory. It is lowered automatically if needed."
    if key in {"ollama_vision_num_ctx", "ollama_text_num_ctx"}:
        return "Enough working memory for each request without using more RAM than this computer can spare."
    if key == "ollama_director_num_ctx":
        return "Room for the AI editor of prompt mode to read every candidate moment of a video at once."
    if key == "director_think":
        if rec.get("director_think"):
            return "The graphics card is fast enough for the AI editor to reason before choosing, which gives better plans."
        return "Off: reasoning first takes many minutes on the processor, or the text model cannot do it."
    if key in {"ollama_num_thread", "whisper_cpu_threads"}:
        if cores:
            return f"Uses {rec.get('num_thread')} of {cores} processor cores and leaves the rest free, so your computer stays responsive."
        return "Leaves part of the processor free so your computer stays responsive."
    return ""


def tradeoffs(rec: dict[str, Any], hw: dict[str, Any]) -> list[str]:
    """Places where the recommendation gives up some quality to fit the hardware, said plainly."""
    notes = []
    ram = hw.get("ram_gb")
    vision = str(rec.get("vision_model") or "")
    if rec.get("inference_mode") != "gpu" and ":2b" in vision:
        notes.append(
            f"A smaller video model is used because this computer has {ram} GB of memory and no graphics card. "
            "It is less accurate at reading small on-screen text. With 16 GB of memory or a graphics card, a larger model is chosen."
        )
    if rec.get("whisper_model") == "tiny":
        notes.append("The smallest speech-to-text model is used because memory is very limited. Transcripts may contain more mistakes.")
    return notes


def warnings(rec: dict[str, Any], hw: dict[str, Any], runtime: dict[str, Any], settings: Settings) -> list[str]:
    """Conditions that could make processing slow or unstable right now."""
    found = []
    available = hw.get("ram_available_gb")
    need = _model_memory_gb(rec) + BASE_MEMORY_GB
    if available is not None and float(available) < need:
        found.append(
            f"Only {available} GB of memory is free right now, and analysis needs about {need:.0f} GB. "
            "Close other programs before creating a highlight to avoid slowdowns."
        )
    disk = hw.get("disk_free_gb")
    if disk is not None and float(disk) < _LOW_DISK_GB:
        found.append(f"Only {disk} GB of disk space is free. Videos, models, and analysis files need room; free up space if you can.")
    placement = (runtime.get("placement") or {}).get("mode")
    if hw.get("inference_mode") != "gpu" and placement == "gpu" and not _uses_builtin_graphics(hw, runtime, settings):
        found.append(
            "The AI is running on a graphics card that isn't in this computer (another computer or a tunnel). "
            "Recommendations are based on this computer. To use larger models there, turn automatic optimization off and choose them yourself."
        )
    return found


def _uses_builtin_graphics(hw: dict[str, Any], runtime: dict[str, Any], settings: Settings) -> bool:
    """Ollama on this computer reports GPU use although no dedicated graphics memory was detected:
    it is running on the built-in graphics, which share system memory."""
    placement = (runtime.get("placement") or {}).get("mode")
    if hw.get("inference_mode") == "gpu" or placement != "gpu" or not hw.get("gpu"):
        return False
    return get_ollama_client(settings).is_local()


def builtin_graphics_note(hw: dict[str, Any], runtime: dict[str, Any], settings: Settings) -> str | None:
    if not _uses_builtin_graphics(hw, runtime, settings):
        return None
    return (
        f"Ollama runs the models on the built-in graphics ({hw['gpu']}), which share system memory. "
        "That speeds them up; model sizes stay chosen for this computer's memory."
    )


def _model_memory_gb(rec: dict[str, Any]) -> float:
    """Approximate memory for the largest of the two models, since only one is loaded at a time on CPU."""
    return max((model_memory_gb(rec.get(key)) for key in ("vision_model", "text_model")), default=0.0)


# Learning from real runs (suggestions only)


def record_run(progress: Any, cache: dict[str, Any], settings: Settings | None = None) -> None:
    """Remember how fast the last analyses were, so Settings can suggest improvements."""
    settings = settings or get_settings()
    usage = dict(getattr(progress, "usage", {}) or {})
    timings = dict(getattr(progress, "timings", {}) or {})
    if not usage and not timings:
        return
    entry = {
        "ts": time.time(),
        "video_seconds": float((cache.get("metadata") or {}).get("duration") or 0),
        "total_seconds": round(sum(timings.values()), 1),
        "stages": timings,
        "visual_enabled": bool(settings.enable_visual_analysis),
        "vision_model": settings.ollama_vision_model,
        "text_model": settings.ollama_text_model,
        "vision": _usage_summary(usage.get("vision")),
        "text": _usage_summary(usage.get("text")),
        "adjustments": list(getattr(progress, "adjustments", []) or [])[:3],
    }
    if entry["vision"]:
        vision = usage["vision"]
        busy = float(vision.get("prompt_sec") or 0) + float(vision.get("output_sec") or 0)
        # Kept as the time of a one-frame request: a picture of several scenes is read once but
        # answered for each scene (see context_fit.vision_scene_cost).
        tiles = max(1, int((getattr(progress, "speeds", None) or {}).get("vision_scenes_per_image") or 1))
        per_picture = busy / calibration.images_in(vision)
        entry["vision"]["seconds_per_image"] = round(per_picture / (tiles * vision_scene_cost(tiles)), 1)
    history = [entry, *read_history()][:_HISTORY_SIZE]
    write_setting(_HISTORY_KEY, json.dumps(history))


def _usage_summary(usage: dict[str, float] | None) -> dict[str, float] | None:
    if not usage or not usage.get("calls"):
        return None
    calls = float(usage["calls"])
    busy = float(usage.get("prompt_sec") or 0) + float(usage.get("output_sec") or 0)
    output_sec = float(usage.get("output_sec") or 0)
    return {
        "calls": int(calls),
        "seconds_per_call": round(busy / calls, 1),
        "tokens_per_sec": round(float(usage.get("output_tokens") or 0) / output_sec, 1) if output_sec else 0.0,
        "near_context_limit": int(usage.get("near_context_limit") or 0),
        "cut_off": int(usage.get("cut_off") or 0),
    }


def read_history() -> list[dict[str, Any]]:
    return _read_json(_HISTORY_KEY, list) or []


def tips(history: list[dict[str, Any]], settings: Settings, hw: dict[str, Any]) -> list[str]:
    """Suggestions from the most recent run. Shown to the user, never applied automatically."""
    if not history:
        return []
    last = history[0]
    found = []
    if last.get("adjustments"):
        found.append(
            "The last analysis ran out of memory and had to free other models and retry. "
            "Closing other programs, or a smaller model in AI models, avoids the delay."
        )
    vision = last.get("vision") or {}
    text = last.get("text") or {}
    if (
        settings.enable_visual_analysis
        and hw.get("inference_mode") != "gpu"
        and float(vision.get("seconds_per_call") or 0) > _SLOW_VISION_SECONDS
    ):
        found.append(
            f"Understanding what is on screen took about {vision['seconds_per_call']:.0f} seconds per request on the processor. "
            "For talks, interviews, and lectures, turning it off in AI models makes analysis much faster; speech is still analyzed."
        )
    if (vision.get("near_context_limit") or vision.get("cut_off")) and settings.ollama_vision_batch_size > 1:
        found.append("Some video requests ran out of working memory. Lowering \"Pictures per AI request\" in Speed and memory avoids this.")
    if text.get("near_context_limit") or text.get("cut_off"):
        if float(hw.get("ram_gb") or 0) >= 16:
            found.append(
                f"Some speech requests ran out of working memory. Raising \"Speech model memory\" to {settings.ollama_text_num_ctx * 2} "
                "in Speed and memory gives them more room."
            )
        else:
            found.append("Some speech requests ran out of working memory. Closing other programs leaves more memory for the AI.")
    tokens = float(text.get("tokens_per_sec") or 0)
    if tokens and tokens < _SLOW_TEXT_TOKENS_PER_SEC:
        found.append("Speech understanding ran slowly. Other programs may be using the processor; closing them helps.")
    return found[:3]


def _read_json(key: str, kind: type) -> Any:
    raw = read_setting(key)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, kind) else None
