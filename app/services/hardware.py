from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any


def detect_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cpu": platform.processor() or platform.machine(),
        "system": platform.system(),
        "ram_gb": None,
        "gpu": None,
        "vram_gb": None,
        "cuda": False,
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "ffprobe": bool(shutil.which("ffprobe")),
        "encoders": [],
        "acceleration": "CPU fallback",
        "recommended_vision_model": "qwen3-vl:8b",
        "recommended_text_model": "qwen3:8b",
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        info["cpu"] = platform.processor() or info["cpu"]
    except Exception:
        pass
    smi = _nvidia_smi_name()
    if smi:
        info["gpu"] = smi.split(",")[0].strip()
        info["cuda"] = True
        try:
            mem = smi.split(",")[1].strip()
            if "MiB" in mem:
                info["vram_gb"] = round(float(mem.replace("MiB", "").strip()) / 1024, 1)
        except Exception:
            pass
    if not info["gpu"]:
        info["gpu"] = _sysfs_gpu()
    info["encoders"] = detect_ffmpeg_encoders()
    hw = preferred_hw_encoder()
    labels = {
        "h264_nvenc": "NVIDIA NVENC detected",
        "h264_qsv": "Intel Quick Sync detected",
        "h264_amf": "AMD hardware encoding detected",
        "h264_videotoolbox": "Apple VideoToolbox detected",
    }
    if hw:
        info["acceleration"] = labels.get(hw, hw)
    elif any("nvenc" in e for e in info["encoders"]):
        info["acceleration"] = "CPU fallback (NVENC listed but CUDA is unavailable)"
    elif any(e in {"h264_qsv", "h264_amf", "h264_videotoolbox"} for e in info["encoders"]):
        info["acceleration"] = "CPU fallback (hardware encoder not usable)"
    info["recommended_vision_model"], info["recommended_text_model"] = recommend_models(
        info.get("vram_gb"), info.get("ram_gb")
    )
    return info


def detect_ffmpeg_encoders() -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    found = []
    for name in ("h264_nvenc", "hevc_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox", "libx264"):
        if name in (result.stdout or ""):
            found.append(name)
    return found


def cuda_runtime_available() -> bool:
    visible = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    if visible is not None and visible.strip().lower() in {"void", "none"}:
        return False
    if _nvidia_smi_name():
        return True
    return _load_cuda_library()


def _load_cuda_library() -> bool:
    try:
        import ctypes

        for lib in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
            try:
                ctypes.CDLL(lib)
                return True
            except OSError:
                continue
    except Exception:
        return False
    return False


_probe_cache: dict[str, bool] = {}


def probe_encoder(name: str) -> bool:
    if name in _probe_cache:
        return _probe_cache[name]
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        _probe_cache[name] = False
        return False
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:d=0.04",
        "-frames:v",
        "1",
        "-an",
        "-c:v",
        name,
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
        ok = result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    _probe_cache[name] = ok
    return ok


def encoder_is_usable(name: str) -> bool:
    if name in {"libx264", "libx265", "libvpx-vp9", "libsvtav1"}:
        return True
    if name.endswith("_nvenc") and not cuda_runtime_available():
        return False
    return probe_encoder(name)


def preferred_hw_encoder() -> str | None:
    encoders = detect_ffmpeg_encoders()
    for name in ("h264_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox"):
        if name in encoders and encoder_is_usable(name):
            return name
    return None


def recommend_models(vram_gb: float | None, ram_gb: float | None) -> tuple[str, str]:
    vram = vram_gb or 0
    ram = ram_gb or 0
    if vram >= 20 or ram >= 64:
        return "qwen3-vl:30b", "qwen3:30b"
    if vram >= 8 or ram >= 24:
        return "qwen3-vl:8b", "qwen3:8b"
    if vram >= 4 or ram >= 12:
        return "qwen3-vl:4b", "qwen3:4b"
    return "qwen3-vl:2b", "qwen3:4b"


def _nvidia_smi_name() -> str | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        result = subprocess.run(
            [smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        line = (result.stdout or "").strip().splitlines()
        return line[0] if line else None
    except Exception:
        return None


def _sysfs_gpu() -> str | None:
    path = "/sys/class/drm/card0/device/vendor"
    if os.path.exists(path):
        return "GPU present (details unavailable)"
    return None
