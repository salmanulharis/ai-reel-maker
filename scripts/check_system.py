#!/usr/bin/env python3
"""Print local dependency readiness for AI Video Highlights."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.services.hardware import detect_hardware  # noqa: E402
from app.ai.ollama import OllamaClient  # noqa: E402


def _narrated_videos(settings) -> dict:
    """Readiness of the brief engines, the narration voice and the Remotion tools."""
    from app.brief.engines import engines_status
    from app.compose import remotion
    from app.voice import kokoro

    tools = remotion.status(settings)
    return {
        "brief_engines": {item["name"]: {"ready": item["ready"], "problems": item["problems"]} for item in engines_status(settings)},
        "voice_model": {"installed": kokoro.installed(), "runtime": kokoro.runtime_available()},
        "remotion": {key: tools[key] for key in ("node", "node_ok", "installed", "bundle", "browser", "ready")},
    }


def main() -> None:
    settings = get_settings()
    hw = detect_hardware(include_encoders=True)
    ollama = OllamaClient(settings).health()
    report = {
        "ffmpeg": hw["ffmpeg"],
        "ffprobe": hw["ffprobe"],
        "cpu": hw["cpu"],
        "ram_gb": hw["ram_gb"],
        "gpu": hw["gpu"],
        "inference_mode": hw["inference_mode"],
        "encode_acceleration": hw["encode_acceleration"],
        "acceleration": hw["acceleration"],
        "data_dir": str(settings.data_dir),
        "ollama": ollama,
        "recommended_vision": hw["recommended_vision_model"],
        "recommended_text": hw["recommended_text_model"],
        "narrated_videos": _narrated_videos(settings),
    }
    print(json.dumps(report, indent=2))
    if not hw["ffmpeg"] or not hw["ffprobe"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
