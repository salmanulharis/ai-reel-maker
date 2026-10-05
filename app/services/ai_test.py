from __future__ import annotations

import base64
import io
import threading
import time
import wave
from typing import Any

from PIL import Image

from app.ai.ollama import OllamaError, describe_placement
from app.config import Settings, get_settings
from app.services.ai_profile import resolve_runtime, sync_hardware_profile
from app.services.ollama_runtime import get_ollama_client

_lock = threading.Lock()
_state: dict[str, Any] = {"state": "idle", "result": None}
_whisper_state: dict[str, Any] = {"state": "idle", "model": None, "error": None}


def explain_ai_error(message: str, *, gpu_present: bool = False, placement_mode: str | None = None) -> str:
    text = (message or "").lower()
    if any(token in text for token in ("connection", "refused", "failed to resolve", "name resolution")):
        return "Ollama is not reachable. Install it, start it, and check the Ollama URL in Advanced settings."
    if "timeout" in text or "timed out" in text:
        return "The model took too long. Pick a smaller model, or check that Ollama is running."
    if "not found" in text or "try pulling" in text:
        return "That model is not installed. Use Install on the model card."
    if any(token in text for token in ("out of memory", "cuda out", "insufficient memory", "requires more system memory")):
        return "The model does not fit in memory. Pick a smaller model, or close other programs using the GPU."
    if gpu_present and placement_mode == "cpu":
        return (
            "A GPU is detected, but this model loaded on CPU. Update the GPU driver, run Ollama on the "
            "computer with the GPU, or pick a smaller model. On AMD, set OLLAMA_VULKAN=1 and restart Ollama "
            "if ROCm cannot load the model."
        )
    if "faster-whisper" in text or "no module named" in text:
        return "Whisper is not installed in this Python environment. Install the app requirements and try again."
    return message or "The test failed. Check that Ollama is running and the selected model is installed."


def test_status() -> dict[str, Any]:
    with _lock:
        return dict(_state)


def whisper_download_status() -> dict[str, Any]:
    with _lock:
        return dict(_whisper_state)


def start_ai_test(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    with _lock:
        if _state.get("state") == "running":
            return dict(_state)
        _state["state"] = "running"
        _state["result"] = None
    thread = threading.Thread(target=_run_test, args=(settings,), daemon=True, name="ai-test")
    thread.start()
    return test_status()


def start_whisper_download(model: str, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    with _lock:
        if _whisper_state.get("state") == "running":
            return dict(_whisper_state)
        _whisper_state.update({"state": "running", "model": model, "error": None})
    thread = threading.Thread(
        target=_download_whisper,
        args=(model, settings),
        daemon=True,
        name="whisper-download",
    )
    thread.start()
    return whisper_download_status()


def _run_test(settings: Settings) -> None:
    try:
        profile = sync_hardware_profile(settings)
        result = _execute(settings, profile)
        state = "done" if result.get("ok") else "error"
    except Exception as exc:  # noqa: BLE001
        result = {"ok": False, "error": explain_ai_error(str(exc))}
        state = "error"
    with _lock:
        _state["state"] = state
        _state["result"] = result


def _execute(settings: Settings, profile: dict[str, Any]) -> dict[str, Any]:
    hardware = profile.get("hardware") or {}
    ollama = profile.get("ollama") or {}
    gpu_present = hardware.get("inference_mode") == "gpu" or bool(hardware.get("gpu"))
    client = get_ollama_client(settings)
    report: dict[str, Any] = {
        "ok": True,
        "error": None,
        "warning": None,
        "text": None,
        "vision": None,
        "whisper": None,
        "runtime": profile.get("runtime"),
    }
    if not ollama.get("connected"):
        report["ok"] = False
        report["error"] = explain_ai_error(str(ollama.get("error") or "connection refused"), gpu_present=gpu_present)
        return report

    report["text"] = _probe_model(
        client,
        settings.ollama_text_model,
        ollama.get("text_present"),
        "Reply with the single word ready.",
        gpu_present=gpu_present,
    )
    report["vision"] = _probe_model(
        client,
        settings.ollama_vision_model,
        ollama.get("vision_present"),
        "What color is this image? Reply with one word.",
        images=[_sample_image()],
        gpu_present=gpu_present,
    )
    health = client.health()
    runtime = resolve_runtime(hardware, health.get("running") or [])
    report["runtime"] = runtime
    placement = (runtime.get("placement") or {}) if isinstance(runtime, dict) else {}
    if runtime.get("mode") == "split":
        report["warning"] = runtime.get("hint")
    elif gpu_present and placement.get("mode") == "cpu":
        report["warning"] = runtime.get("hint") or explain_ai_error("", gpu_present=True, placement_mode="cpu")
    report["whisper"] = _probe_whisper(settings)
    failed = [item for item in (report["text"], report["vision"], report["whisper"]) if item and not item.get("ok")]
    if failed:
        report["ok"] = False
        report["error"] = failed[0].get("error")
    return report


def _probe_model(client, model: str, present: bool, prompt: str, images: list[str] | None = None, gpu_present: bool = False) -> dict[str, Any]:
    if not present:
        return {
            "ok": False,
            "model": model,
            "error": explain_ai_error(f"model '{model}' not found, try pulling it"),
        }
    try:
        result = client.probe_chat(model, prompt, images=images, num_predict=16)
    except OllamaError as exc:
        return {"ok": False, "model": model, "error": explain_ai_error(str(exc), gpu_present=gpu_present)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "model": model, "error": explain_ai_error(str(exc), gpu_present=gpu_present)}
    running = client.running_models()
    placement = describe_placement(running, model)
    result["model"] = model
    result["placement"] = placement
    if not result.get("text"):
        result["ok"] = False
        result["error"] = "The model returned an empty response. Try the test again, or pick a smaller model."
    return result


def _probe_whisper(settings: Settings) -> dict[str, Any]:
    if not settings.enable_whisper:
        return {"ok": True, "skipped": True, "message": "Whisper is turned off."}
    from app.analysis.audio import transcribe_detailed

    path = settings.tmp_dir / "ai-whisper-test.wav"
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    _write_silence(path, seconds=2.0)
    started = time.perf_counter()
    try:
        _transcript, info = transcribe_detailed(
            path,
            model_size=settings.whisper_model,
            device=settings.whisper_device or "auto",
            cpu_threads=settings.whisper_cpu_threads,
            num_workers=settings.whisper_num_workers,
        )
    finally:
        path.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    info = dict(info)
    info["elapsed_sec"] = round(elapsed, 3)
    info["audio_sec"] = 2.0
    if elapsed > 0:
        info["realtime_factor"] = round(2.0 / elapsed, 2)
    if not info.get("ok"):
        info["error"] = explain_ai_error(str(info.get("error") or "Whisper failed"))
    return info


def _download_whisper(model: str, settings: Settings) -> None:
    try:
        from app.analysis.audio import preload_whisper

        preload_whisper(
            model,
            cpu_threads=settings.whisper_cpu_threads,
            num_workers=settings.whisper_num_workers,
        )
        state = {"state": "done", "model": model, "error": None}
    except Exception as exc:  # noqa: BLE001
        state = {"state": "error", "model": model, "error": explain_ai_error(str(exc))}
    with _lock:
        _whisper_state.update(state)


def _sample_image() -> str:
    image = Image.new("RGB", (64, 64), (32, 96, 180))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _write_silence(path, seconds: float, sample_rate: int = 16000) -> None:
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)


def whisper_installed(model: str) -> bool:
    import os
    from pathlib import Path

    cache = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HUB_CACHE")
    root = Path(cache) if cache else Path.home() / ".cache" / "huggingface" / "hub"
    folder = root / f"models--Systran--faster-whisper-{model}"
    return folder.is_dir()
