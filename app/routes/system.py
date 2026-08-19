from __future__ import annotations

import shutil
import subprocess
import threading

from flask import Blueprint, jsonify, request

from app.ai.ollama import OllamaError
from app.config import get_settings
from app.services.hardware import detect_hardware
from app.services.job_manager import get_job_manager
from app.services.ollama_runtime import get_ollama_client

system_bp = Blueprint("system", __name__)
_pull_lock = threading.Lock()
_pull_status: dict[str, str] = {}


@system_bp.get("/health")
def health():
    return jsonify({"ok": True, "service": "ai-video-highlights"})


@system_bp.get("/system/info")
def system_info():
    settings = get_settings()
    hw = detect_hardware()
    checks = {
        "python": True,
        "ffmpeg": hw["ffmpeg"],
        "ffprobe": hw["ffprobe"],
        "storage": settings.data_dir.exists() and settings.data_dir.is_dir(),
    }
    ollama = get_ollama_client(settings).health()
    checks["ollama"] = bool(ollama.get("connected"))
    checks["vision_model"] = bool(ollama.get("vision_present"))
    checks["text_model"] = bool(ollama.get("text_present"))
    return jsonify(
        {
            "hardware": hw,
            "checks": checks,
            "privacy": "Everything runs locally. Your videos are not uploaded to third-party AI services.",
            "settings": {
                "ollama_base_url": settings.ollama_base_url,
                "vision_model": settings.ollama_vision_model,
                "text_model": settings.ollama_text_model,
            },
        }
    )


@system_bp.get("/system/ollama/status")
def ollama_status():
    client = get_ollama_client()
    health = client.health()
    health["pull"] = dict(_pull_status)
    return jsonify(health)


@system_bp.post("/system/ollama/pull")
def ollama_pull():
    data = request.get_json(silent=True) or {}
    name = data.get("model") or get_settings().ollama_vision_model
    if not name or any(c in name for c in ";&|$`\n"):
        return jsonify({"error": "invalid model name"}), 400

    def _run():
        _pull_status[name] = "running"
        try:
            client = get_ollama_client()
            try:
                client.pull_model(name)
            except OllamaError:
                ollama_bin = shutil.which("ollama")
                if not ollama_bin:
                    raise
                subprocess.run([ollama_bin, "pull", name], check=True, timeout=3600)
            _pull_status[name] = "done"
        except Exception as exc:  # noqa: BLE001
            _pull_status[name] = f"error: {exc}"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return jsonify({"ok": True, "model": name, "status": "started"})


@system_bp.get("/system/queue")
def queue():
    jobs = get_job_manager().list_jobs(limit=20)
    active = [
        get_job_manager().serialize(j)
        for j in jobs
        if j.status in {"queued", "analyzing", "planning", "rendering"}
    ]
    return jsonify({"jobs": active})
