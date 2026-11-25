"""``run.py --self-test``: check that this copy of the app can run, without starting it.

Used by the installer build (packaging/build.ps1) on the staged folder and on a real silent
install. Each check imports or runs one thing the app needs: every package with compiled parts,
the bundled programs, the data folder. Nothing is downloaded and no model is loaded.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from app import paths

# Packages the app imports, with compiled parts first (they fail on a broken install).
MODULES = (
    "numpy",
    "cv2",
    "av",
    "ctranslate2",
    "onnxruntime",
    "tokenizers",
    "faster_whisper",
    "sklearn.cluster",
    "sklearn.feature_extraction.text",
    "scenedetect",
    "soundfile",
    "kokoro_onnx",
    "psutil",
    "PIL.Image",
    "sqlalchemy",
    "flask",
    "flask_socketio",
    "engineio.async_drivers.threading",
    "pydantic",
    "pydantic_settings",
    "huggingface_hub",
    "requests",
)


def _modules() -> str:
    missing = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            missing.append(f"{name} ({type(exc).__name__}: {exc})")
    if missing:
        raise RuntimeError("; ".join(missing))
    return f"{len(MODULES)} packages"


def _app() -> str:
    from app import create_app
    from app.config import Settings, set_settings

    # A throwaway data folder (its database stays open until the process ends, so it is left for
    # Windows to clean up), and the real settings put back afterwards.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        try:
            settings = Settings(data_dir=Path(folder) / "data", enable_whisper=False)
            settings.ensure_directories()
            app = create_app(settings, start_background=False)
            with app.test_client() as client:
                reply = client.get("/api/health")
                if reply.status_code != 200:
                    raise RuntimeError(f"health check answered {reply.status_code}")
                version = reply.get_json().get("version")
        finally:
            set_settings(None)
    return f"version {version}"


def _program(name: str, args: list[str], first_line: Callable[[str], str] = lambda line: line) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} was not found")
    result = subprocess.run([path, *args], capture_output=True, text=True, timeout=60, check=False)  # noqa: S603
    if result.returncode != 0:
        raise RuntimeError(f"{name} exited with {result.returncode}")
    line = (result.stdout or result.stderr).strip().splitlines()[0] if (result.stdout or result.stderr).strip() else ""
    return f"{first_line(line)} ({path})"


def _face_model() -> str:
    import cv2

    cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade.is_file() or cv2.CascadeClassifier(str(cascade)).empty():
        raise RuntimeError("the face detection file is missing")
    return "face detection file found"


def _voice() -> str:
    import espeakng_loader

    library = Path(espeakng_loader.get_library_path())
    data = Path(espeakng_loader.get_data_path())
    if not library.is_file() or not data.is_dir():
        raise RuntimeError("espeak-ng (used to read the narration) is missing")
    return "espeak-ng found"


def _whisper_assets() -> str:
    import faster_whisper

    folder = Path(faster_whisper.__file__).parent / "assets"
    if not folder.is_dir() or not any(folder.glob("*.onnx")):
        raise RuntimeError("faster-whisper's voice detection file is missing")
    return "voice detection file found"


def _data_folder() -> str:
    from app.config import get_settings

    settings = get_settings()
    probe = settings.tmp_dir / ".self-test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    return str(settings.data_dir)


def run(out=None) -> int:
    """Run every check, print one line each; 0 when all passed."""
    out = out or sys.stdout
    paths.put_bundled_tools_first()
    print(
        f"AI Reel Maker self-test: Python {platform.python_version()}, "
        f"{'installed' if paths.is_installed() else 'repository'} copy at {paths.program_dir()}",
        file=out,
    )
    checks: list[tuple[str, Callable[[], str]]] = [
        ("packages", _modules),
        ("data folder", _data_folder),
        ("app", _app),
        ("ffmpeg", lambda: _program("ffmpeg", ["-hide_banner", "-version"])),
        ("ffprobe", lambda: _program("ffprobe", ["-hide_banner", "-version"])),
        ("node", lambda: _program("node", ["--version"])),
        ("face framing", _face_model),
        ("narration voice", _voice),
        ("speech", _whisper_assets),
    ]
    failed = 0
    for name, check in checks:
        try:
            detail = check()
            print(f"  ok    {name}: {detail}", file=out)
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            failed += 1
            print(f"  FAIL  {name}: {exc}", file=out)
    print("All checks passed." if not failed else f"{failed} check(s) failed.", file=out)
    return 0 if not failed else 1
