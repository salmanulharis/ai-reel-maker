from __future__ import annotations

import platform
import shutil
import subprocess
import threading
from pathlib import Path

from flask import Blueprint, jsonify, request

from app.ai.ollama import OllamaError, OllamaPullError
from app.ai.registry_pull import models_root
from app.config import get_settings
from app.analysis import image_match
from app.services import auto_tune
from app.services.ai_profile import apply_recommendation, resolve_runtime, sync_hardware_profile
from app.services.ai_test import (
    explain_ai_error,
    start_ai_test,
    start_whisper_download,
    test_status,
    whisper_download_status,
    whisper_installed,
)
from app.services.hardware import detect_hardware
from app.services.job_manager import get_job_manager
from app.services.model_catalog import CATALOG, catalog_entry
from app.services.ollama_runtime import get_ollama_client

system_bp = Blueprint("system", __name__)
_pull_lock = threading.Lock()
_pull_status: dict[str, dict] = {}


@system_bp.get("/health")
def health():
    return jsonify({"ok": True, "service": "ai-video-highlights"})


@system_bp.get("/system/info")
def system_info():
    settings = get_settings()
    refresh = request.args.get("refresh") in {"1", "true", "yes"}
    include_encoders = request.args.get("encoders") in {"1", "true", "yes"}
    profile = sync_hardware_profile(settings, refresh=refresh, include_encoders=include_encoders)
    hardware = profile["hardware"]
    ollama = profile["ollama"]
    checks = {
        "python": True,
        "ffmpeg": hardware.get("ffmpeg"),
        "ffprobe": hardware.get("ffprobe"),
        "storage": settings.data_dir.exists() and settings.data_dir.is_dir(),
        "ollama": bool(ollama.get("connected")),
        "vision_model": bool(ollama.get("vision_ready")),
        "text_model": bool(ollama.get("text_present")),
    }
    whisper_cache = {item["name"]: whisper_installed(item["name"]) for item in CATALOG["whisper"]}
    return jsonify(
        {
            **profile,
            "checks": checks,
            "whisper_cache": whisper_cache,
            "privacy": "Everything runs locally. Your videos are not uploaded to third-party AI services.",
            "settings": {
                "ollama_base_url": settings.ollama_base_url,
                "vision_model": settings.ollama_vision_model,
                "text_model": settings.ollama_text_model,
                "whisper_model": settings.whisper_model,
                "whisper_device": settings.whisper_device,
                "ai_config_mode": settings.ai_config_mode,
            },
        }
    )


@system_bp.post("/system/ai/apply-recommended")
def apply_recommended():
    settings = get_settings()
    profile = sync_hardware_profile(settings, refresh=True)
    summary = apply_recommendation(settings, profile.get("recommendation") or {})
    profile["profile"]["auto_summary"] = summary
    profile["profile"]["mode"] = "auto"
    profile["profile"]["applied_now"] = True
    profile["settings"] = {
        "ollama_base_url": settings.ollama_base_url,
        "vision_model": settings.ollama_vision_model,
        "text_model": settings.ollama_text_model,
        "whisper_model": settings.whisper_model,
        "whisper_device": settings.whisper_device,
        "ai_config_mode": settings.ai_config_mode,
    }
    return jsonify(profile)


@system_bp.get("/system/ai/optimization")
def optimization_plan():
    refresh = request.args.get("refresh") in {"1", "true", "yes"}
    return jsonify(auto_tune.build_plan(get_settings(), refresh=refresh))


@system_bp.post("/system/ai/optimization")
def set_optimization():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload.get("enabled"), bool):
        return jsonify({"error": "enabled must be true or false"}), 400
    settings = get_settings()
    plan = auto_tune.enable(settings) if payload["enabled"] else auto_tune.disable(settings)
    return jsonify(plan)


@system_bp.post("/system/ai/optimization/undo")
def undo_optimization():
    if not auto_tune.read_snapshot():
        return jsonify({"error": "There is nothing to undo."}), 400
    return jsonify(auto_tune.undo(get_settings()))


@system_bp.post("/system/ai/test")
def start_test():
    return jsonify(start_ai_test(get_settings()))


@system_bp.get("/system/ai/test")
def read_test():
    return jsonify(test_status())


@system_bp.post("/system/image-model/download")
def download_image_model():
    if not image_match.runtime_available():
        return jsonify({"error": "Image matching needs onnxruntime and tokenizers. Install the app requirements."}), 400
    return jsonify(image_match.start_download())


@system_bp.get("/system/image-model")
def image_model_status():
    return jsonify(image_match.download_status())


@system_bp.post("/system/whisper/download")
def download_whisper():
    data = request.get_json(silent=True) or {}
    name = data.get("model") or get_settings().whisper_model
    if not _valid_model_name(name):
        return jsonify({"error": "invalid model name"}), 400
    return jsonify(start_whisper_download(name, get_settings()))


@system_bp.get("/system/whisper/download")
def whisper_download():
    return jsonify(whisper_download_status())


@system_bp.get("/system/ollama/status")
def ollama_status():
    settings = get_settings()
    client = get_ollama_client(settings)
    health = client.health()
    hardware = _cached_or_fast_hardware()
    health["runtime"] = resolve_runtime(hardware, health.get("running") or [])
    health["pull"] = {key: dict(value) for key, value in _pull_status.items()}
    health["test"] = test_status()
    health["whisper_download"] = whisper_download_status()
    health["image_model"] = image_match.download_status()
    return jsonify(health)


@system_bp.post("/system/ollama/pull")
def ollama_pull():
    data = request.get_json(silent=True) or {}
    names = data.get("models") or [data.get("model") or get_settings().ollama_vision_model]
    names = [name for name in names if name]
    if not names or any(not _valid_model_name(name) for name in names):
        return jsonify({"error": "invalid model name"}), 400
    for name in names:
        problem = _disk_problem(name)
        if problem:
            return jsonify({"error": problem}), 400
    if not _pull_lock.acquire(blocking=False):
        return jsonify({"ok": True, "models": names, "status": "running", "pull": dict(_pull_status)})

    def _run():
        try:
            for name in names:
                _pull_one(name)
        finally:
            _pull_lock.release()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return jsonify({"ok": True, "models": names, "status": "started"})


def _pull_one(name: str) -> None:
    _pull_status[name] = {"state": "running", "status": "starting", "completed": 0, "total": 0, "error": None}

    def _progress(progress: dict) -> None:
        _pull_status[name] = {
            "state": "running",
            "status": progress.get("status") or "",
            "completed": progress.get("completed") or 0,
            "total": progress.get("total") or 0,
            "error": None,
        }

    try:
        client = get_ollama_client()
        try:
            client.pull_model(name, on_progress=_progress)
        except OllamaPullError:
            # Ollama answered and refused the download; the CLI would hit the same error.
            raise
        except OllamaError:
            ollama_bin = shutil.which("ollama")
            if not ollama_bin or not client.is_local():
                raise
            _pull_status[name]["status"] = "downloading"
            subprocess.run([ollama_bin, "pull", name], check=True, timeout=3600)
        _pull_status[name] = {"state": "done", "status": "success", "completed": 1, "total": 1, "error": None}
    except Exception as exc:  # noqa: BLE001
        _pull_status[name] = {
            "state": "error",
            "status": "error",
            "completed": 0,
            "total": 0,
            "error": explain_ai_error(str(exc)),
        }


def _valid_model_name(name: str) -> bool:
    return bool(name) and not any(char in name for char in ";&|$`\n\r")


def _disk_problem(name: str) -> str | None:
    entry = catalog_entry(name)
    need_gb = float(entry["disk_gb"]) if entry else 4.0
    free = _ollama_disk_free_gb()
    if free is None:
        return None
    if float(free) < need_gb * 1.15:
        return (
            f"Not enough free disk space for {name}. It needs about {need_gb:g} GB and "
            f"{float(free):.1f} GB is free."
        )
    return None


def _ollama_disk_free_gb() -> float | None:
    """Free space where Ollama stores models, or None when that disk is not on this machine."""
    if not get_ollama_client().is_local():
        return None
    candidates = [models_root()]
    if platform.system() == "Linux":
        candidates.append(Path("/usr/share/ollama/.ollama/models"))
    folder = next((path for path in candidates if path.is_dir()), Path.home())
    try:
        return round(shutil.disk_usage(folder).free / (1024**3), 1)
    except OSError:
        return None


def _cached_or_fast_hardware() -> dict:
    try:
        profile = sync_hardware_profile(refresh=False)
        return profile.get("hardware") or {}
    except Exception:
        return detect_hardware()


@system_bp.get("/system/queue")
def queue():
    jobs = get_job_manager().list_jobs(limit=20)
    active = [
        get_job_manager().serialize(j)
        for j in jobs
        if j.status in {"queued", "analyzing", "planning", "rendering"}
    ]
    return jsonify({"jobs": active})
