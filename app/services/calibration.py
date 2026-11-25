"""Measured speed of the AI setup on this computer, and time estimates built from it.

"Measure this computer" runs each model once on a small fixed task and stores the result.
Real analyses refine the numbers afterwards. Estimates only ever feed suggestions: nothing
here changes a setting.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
import wave
from typing import Any

from app.ai.context_fit import VISION_PROMPT_TOKENS_PER_IMAGE, vision_scene_cost
from app.ai.ollama import describe_placement
from app.config import Settings, get_settings
from app.services.model_catalog import VISION_FIT_GB, catalog_entry, model_memory_gb, next_larger, nominal_ram_gb
from app.services.ollama_runtime import get_ollama_client
from app.services.settings_store import read_setting, write_setting

_KEY = "perf_profile"
# Estimates describe this video: ten minutes, with the usual number of scenes and spoken
# segments the AI models handle (the video model only looks at scenes that could be chosen).
REFERENCE_VIDEO_SECONDS = 600.0
_REFERENCE_SCENES = 30
_REFERENCE_SEGMENTS = 120
# Reply tokens per transcript segment (see pipeline._TEXT_TOKENS_PER_SEGMENT).
_TOKENS_PER_SEGMENT = 48
# Balanced aims to analyze a video in at most twice its length.
BALANCED_BUDGET_SECONDS = REFERENCE_VIDEO_SECONDS * 2
_SLOW_FACTOR = 3.0
_WHISPER_SAMPLE_SECONDS = 20.0
_ACTIVE_JOB_STATES = {"queued", "analyzing", "rendering"}

_lock = threading.Lock()
_state: dict[str, Any] = {"state": "idle", "step": None, "error": None}


def status() -> dict[str, Any]:
    with _lock:
        return {**_state, "result": read_profile()}


def start(settings: Settings | None = None) -> dict[str, Any]:
    """Measure in the background. Refused while a video is being processed, because both
    would compete for the same processor and the numbers would be wrong."""
    settings = settings or get_settings()
    if _job_running():
        raise RuntimeError("A video is being processed. Measure again when it has finished.")
    with _lock:
        if _state["state"] == "running":
            return dict(_state)
        _state.update(state="running", step="text", error=None)
    threading.Thread(target=_run, args=(settings,), daemon=True, name="ai-calibration").start()
    return status()


def _job_running() -> bool:
    try:
        from app.services.job_manager import get_job_manager

        return any(job.status in _ACTIVE_JOB_STATES for job in get_job_manager().list_jobs(limit=20))
    except Exception:  # noqa: BLE001 - never block measuring on a bookkeeping error
        return False


def _run(settings: Settings) -> None:
    try:
        profile = measure(settings, on_step=_set_step)
        error = None if any(profile.get(key) for key in ("text", "vision", "whisper")) else (
            profile.get("error") or "Nothing could be measured. Check that Ollama is running and the models are installed."
        )
        if not error:
            write_setting(_KEY, json.dumps(profile))
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    with _lock:
        _state.update(state="error" if error else "done", step=None, error=error)


def _set_step(step: str) -> None:
    with _lock:
        _state["step"] = step


def measure(settings: Settings, on_step=None) -> dict[str, Any]:
    """Run each model once and return its speed. A part that fails is left out."""
    notify = on_step or (lambda _step: None)
    client = get_ollama_client(settings)
    health = client.health()
    profile: dict[str, Any] = {
        "ts": time.time(),
        "fingerprint": read_setting("hw_fingerprint") or "",
        "text": None,
        "vision": None,
        "whisper": None,
        "error": None,
    }
    if health.get("connected"):
        notify("text")
        if settings.enable_whisper and health.get("text_present"):
            profile["text"] = _measure_text(client, settings.ollama_text_model)
        notify("vision")
        if settings.enable_visual_analysis and health.get("vision_ready"):
            profile["vision"] = _measure_vision(client, settings.ollama_vision_model)
    else:
        profile["error"] = str(health.get("error") or "Ollama is not reachable")
    notify("whisper")
    if settings.enable_whisper:
        profile["whisper"] = _measure_whisper(settings)
    return profile


def _measure_text(client, model: str) -> dict[str, Any] | None:
    try:
        result = client.probe_chat(
            model, "Write four short sentences about the sea.", num_predict=96, role="text"
        )
    except Exception:  # noqa: BLE001
        return None
    speed = result.get("tokens_per_sec")
    if not speed:
        return None
    return {"model": model, "tokens_per_sec": float(speed)}


def _measure_vision(client, model: str) -> dict[str, Any] | None:
    try:
        result = client.probe_chat(
            model, "Describe this picture in one sentence.", images=[_sample_image()], num_predict=80, role="vision"
        )
        placement = describe_placement(client.running_models(), model)
    except Exception:  # noqa: BLE001
        return None
    seconds = float(result.get("total_sec") or result.get("elapsed_sec") or 0) - float(result.get("load_sec") or 0)
    if seconds <= 0:
        return None
    return {"model": model, "seconds_per_image": round(seconds, 1), "placement": placement.get("mode")}


def _measure_whisper(settings: Settings) -> dict[str, Any] | None:
    from app.analysis.audio import transcribe_detailed

    folder = settings.tmp_dir
    folder.mkdir(parents=True, exist_ok=True)
    warm, sample = folder / "ai-measure-warm.wav", folder / "ai-measure.wav"
    arguments = dict(
        model_size=settings.whisper_model,
        device=settings.whisper_device or "auto",
        cpu_threads=settings.whisper_cpu_threads,
        num_workers=settings.whisper_num_workers,
        batch_size=settings.whisper_batch_size,
    )
    try:
        _write_sample_audio(warm, 1.0)
        _write_sample_audio(sample, _WHISPER_SAMPLE_SECONDS)
        # The first call loads the model, so only the second is timed.
        _transcript, info = transcribe_detailed(warm, **arguments)
        if not info.get("ok"):
            return None
        started = time.perf_counter()
        _transcript, info = transcribe_detailed(sample, **arguments)
        elapsed = time.perf_counter() - started
    except Exception:  # noqa: BLE001
        return None
    finally:
        warm.unlink(missing_ok=True)
        sample.unlink(missing_ok=True)
    if not info.get("ok") or elapsed <= 0:
        return None
    return {
        "model": settings.whisper_model,
        "device": info.get("device"),
        "realtime_factor": round(_WHISPER_SAMPLE_SECONDS / elapsed, 1),
    }


def _sample_image() -> str:
    """A 512 px picture with some detail, the size of the frames a real analysis sends."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(7)
    ramp = np.linspace(0, 255, 512, dtype=np.float32)
    pixels = np.stack([np.tile(ramp, (512, 1)), np.tile(ramp[:, None], (1, 512)), np.full((512, 512), 96.0)], axis=2)
    pixels += rng.normal(0, 12, pixels.shape)
    image = Image.fromarray(np.clip(pixels, 0, 255).astype("uint8"), "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=75)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _write_sample_audio(path, seconds: float, sample_rate: int = 16000) -> None:
    """Voice-like sound (a wavering tone with noise), so speech detection does not skip it."""
    import numpy as np

    count = int(seconds * sample_rate)
    t = np.arange(count) / sample_rate
    rng = np.random.default_rng(3)
    tone = np.sin(2 * np.pi * (160 + 40 * np.sin(2 * np.pi * 3 * t)) * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    samples = np.clip(tone * 0.5 + rng.normal(0, 0.05, count), -1, 1)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((samples * 32767).astype("<i2").tobytes())


# Reading the numbers back


def read_profile() -> dict[str, Any] | None:
    raw = read_setting(_KEY)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def current_speeds(settings: Settings, history: list[dict[str, Any]] | None = None) -> dict[str, float]:
    """Speed of the models in use now: from the latest real analysis when it used the same
    models, otherwise from the last measurement. Missing parts are simply absent."""
    speeds: dict[str, float] = {}
    profile = read_profile() or {}
    text, vision, whisper = (profile.get(key) or {} for key in ("text", "vision", "whisper"))
    if text.get("model") == settings.ollama_text_model and text.get("tokens_per_sec"):
        speeds["text_tokens_per_sec"] = float(text["tokens_per_sec"])
    if vision.get("model") == settings.ollama_vision_model and vision.get("seconds_per_image"):
        speeds["vision_seconds_per_image"] = float(vision["seconds_per_image"])
    if whisper.get("model") == settings.whisper_model and whisper.get("realtime_factor"):
        speeds["whisper_realtime_factor"] = float(whisper["realtime_factor"])
    last = (history or [None])[0] or {}
    run_text, run_vision = last.get("text") or {}, last.get("vision") or {}
    if last.get("text_model") == settings.ollama_text_model and run_text.get("tokens_per_sec"):
        speeds["text_tokens_per_sec"] = float(run_text["tokens_per_sec"])
    if last.get("vision_model") == settings.ollama_vision_model and run_vision.get("seconds_per_image"):
        speeds["vision_seconds_per_image"] = float(run_vision["seconds_per_image"])
    return speeds


def images_in(usage: dict[str, Any]) -> int:
    """Roughly how many frames a vision stage read, from its prompt tokens."""
    return max(1, round(float(usage.get("prompt_tokens") or 0) / (VISION_PROMPT_TOKENS_PER_IMAGE + 50)))


def estimate(speeds: dict[str, float], settings: Settings) -> dict[str, Any] | None:
    """Estimated time to analyze the ten-minute reference video with the current setup.
    None until at least one model has been measured; ``complete`` is False when a part that
    is turned on has no measurement yet."""
    parts: dict[str, float] = {}
    missing = False
    if settings.enable_whisper:
        if speeds.get("whisper_realtime_factor"):
            parts["speech_to_text"] = REFERENCE_VIDEO_SECONDS / speeds["whisper_realtime_factor"]
        else:
            missing = True
        if speeds.get("text_tokens_per_sec"):
            parts["speech_understanding"] = _REFERENCE_SEGMENTS * _TOKENS_PER_SEGMENT / speeds["text_tokens_per_sec"]
        else:
            missing = True
    if settings.enable_visual_analysis:
        if speeds.get("vision_seconds_per_image"):
            # The measured speed is for one frame per request; tiled scenes share the reading.
            share = vision_scene_cost(getattr(settings, "ollama_vision_scenes_per_image", 1))
            parts["video_understanding"] = _REFERENCE_SCENES * speeds["vision_seconds_per_image"] * share
        else:
            missing = True
    if not parts:
        return None
    return {
        "video_seconds": REFERENCE_VIDEO_SECONDS,
        "seconds": round(sum(parts.values())),
        "parts": {key: round(value) for key, value in parts.items()},
        "complete": not missing,
    }


def suggestions(settings: Settings, hardware: dict[str, Any], goal: str, speeds: dict[str, float]) -> list[str]:
    """Suggestions from measured speed: a larger video model this computer has time and memory
    for, or a faster goal when analysis is far over the balanced time budget."""
    result = estimate(speeds, settings)
    vision_seconds = (result or {}).get("parts", {}).get("video_understanding")
    if not result or not result["complete"] or not vision_seconds:
        return []
    minutes = max(1, round(result["seconds"] / 60))
    if goal != "fast" and result["seconds"] > BALANCED_BUDGET_SECONDS * _SLOW_FACTOR:
        return [
            f"A 10-minute video takes about {minutes} minutes to analyze on this computer. "
            "Choosing Fastest uses a smaller video model, and turning off \"Understand what is on screen\" "
            "is quicker still for talks and interviews."
        ]
    larger = next_larger(settings.ollama_vision_model)
    current, bigger = catalog_entry(settings.ollama_vision_model), catalog_entry(larger or "")
    if goal != "balanced" or not current or not bigger or not _fits(larger, hardware):
        return []
    # A model's time per frame grows roughly with its size.
    scaled = vision_seconds * float(bigger["disk_gb"]) / float(current["disk_gb"])
    total = result["seconds"] - vision_seconds + scaled
    if total > BALANCED_BUDGET_SECONDS:
        return []
    return [
        f"This computer measured fast enough for {larger}: a 10-minute video would take about "
        f"{max(1, round(total / 60))} minutes instead of {minutes}. Choose Best quality, or pick the model in AI models."
    ]


def _fits(model: str | None, hardware: dict[str, Any]) -> bool:
    if not model or "30b" in model:
        return False
    if hardware.get("inference_mode") == "gpu":
        budget = hardware.get("vram_free_gb") or hardware.get("vram_gb") or 0
        return float(budget) >= VISION_FIT_GB.get(model, 99)
    # On the processor the model shares memory with everything else that is running.
    return model_memory_gb(model) <= nominal_ram_gb(hardware.get("ram_gb")) * 0.45
