from __future__ import annotations

import csv
import io
import os
import platform
import shutil
import subprocess
from typing import Any

from app.services.model_catalog import recommend_setup


def detect_hardware(*, include_encoders: bool = False) -> dict[str, Any]:
    """Fast hardware snapshot. Encoder probes stay off unless include_encoders is set."""
    memory = _memory()
    info: dict[str, Any] = {
        "cpu": memory.get("cpu") or platform.processor() or platform.machine(),
        "cpu_cores": memory.get("cores"),
        "cpu_threads": memory.get("threads"),
        "system": platform.system(),
        "ram_gb": memory.get("ram_gb"),
        "ram_available_gb": memory.get("ram_available_gb"),
        "gpu": None,
        "gpu_vendor": None,
        "vram_gb": None,
        "vram_free_gb": None,
        "vram_note": None,
        "driver": None,
        "cuda": False,
        "disk_free_gb": _disk_free_gb(),
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "ffprobe": bool(shutil.which("ffprobe")),
        "encoders": [],
        "encoders_checked": False,
        "encode_acceleration": "Not checked",
        "acceleration": "Not checked",
        "inference_mode": "cpu",
        "inference_label": "CPU-only",
        "inference_measured": False,
    }
    gpu = _best_gpu(info.get("ram_gb"))
    if gpu:
        info.update(gpu)
    has_vram = (info.get("vram_gb") or 0) >= 2
    if has_vram and info.get("gpu_vendor") in {"nvidia", "amd", "intel", "apple"}:
        info["inference_mode"] = "gpu"
    info["inference_label"] = "GPU-accelerated" if info["inference_mode"] == "gpu" else "CPU-only"
    if include_encoders:
        encoders, encode_label = _encode_status()
        info["encoders"] = encoders
        info["encoders_checked"] = True
        info["encode_acceleration"] = encode_label
        info["acceleration"] = encode_label
    recommendation = recommend_setup(
        info.get("vram_gb"),
        info.get("ram_gb"),
        info.get("vram_free_gb"),
        info.get("cpu_cores") or info.get("cpu_threads"),
    )
    info["recommendation"] = recommendation
    info["recommended_vision_model"] = recommendation["vision_model"]
    info["recommended_text_model"] = recommendation["text_model"]
    info["recommended_whisper_model"] = recommendation["whisper_model"]
    info["recommended_whisper_device"] = recommendation["whisper_device"]
    return info


def _encode_status() -> tuple[list[str], str]:
    encoders = detect_ffmpeg_encoders()
    hw = preferred_hw_encoder()
    labels = {
        "h264_nvenc": "NVIDIA NVENC",
        "h264_qsv": "Intel Quick Sync",
        "h264_amf": "AMD hardware encoding",
        "h264_videotoolbox": "Apple VideoToolbox",
    }
    if hw:
        return encoders, labels.get(hw, hw)
    if any("nvenc" in encoder for encoder in encoders):
        return encoders, "CPU encoding (NVENC listed but CUDA is unavailable)"
    if any(encoder in {"h264_qsv", "h264_amf", "h264_videotoolbox"} for encoder in encoders):
        return encoders, "CPU encoding (hardware encoder not usable)"
    return encoders, "CPU encoding (libx264)"


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


def recommend_models(
    vram_gb: float | None,
    ram_gb: float | None,
    free_vram_gb: float | None = None,
) -> tuple[str, str]:
    setup = recommend_setup(vram_gb, ram_gb, free_vram_gb)
    return setup["vision_model"], setup["text_model"]


def _memory() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": None,
        "ram_available_gb": None,
        "cores": None,
        "threads": None,
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        info["ram_gb"] = round(memory.total / (1024**3), 1)
        info["ram_available_gb"] = round(memory.available / (1024**3), 1)
        info["cores"] = psutil.cpu_count(logical=False) or psutil.cpu_count()
        info["threads"] = psutil.cpu_count(logical=True)
        info["cpu"] = platform.processor() or info["cpu"]
    except Exception:
        pass
    return info


def _disk_free_gb() -> float | None:
    try:
        from app.config import get_settings

        path = get_settings().data_dir
    except Exception:
        path = os.getcwd()
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    return round(usage.free / (1024**3), 1)


def _best_gpu(ram_gb: float | None) -> dict[str, Any] | None:
    nvidia = _nvidia_gpus()
    if nvidia:
        best = max(nvidia, key=lambda item: item.get("vram_gb") or 0)
        best["cuda"] = True
        best["gpu_vendor"] = "nvidia"
        return best
    apple = _apple_gpu(ram_gb)
    if apple:
        return apple
    rocm = _rocm_gpu()
    if rocm:
        return rocm
    windows = _windows_gpu()
    if windows:
        return windows
    if _sysfs_gpu():
        return {
            "gpu": "GPU details unavailable",
            "gpu_vendor": None,
            "vram_gb": None,
            "vram_free_gb": None,
            "cuda": False,
        }
    return None


def _nvidia_gpus() -> list[dict[str, Any]]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        result = subprocess.run(
            [
                smi,
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    found = []
    for row in csv.reader(io.StringIO(result.stdout or "")):
        if len(row) < 2:
            continue
        name = row[0].strip()
        if not name:
            continue
        entry: dict[str, Any] = {
            "gpu": name,
            "vram_gb": _mib_to_gb(row[1]) if len(row) > 1 else None,
            "vram_free_gb": _mib_to_gb(row[2]) if len(row) > 2 else None,
            "driver": row[3].strip() if len(row) > 3 else None,
        }
        found.append(entry)
    return found


def _mib_to_gb(value: str) -> float | None:
    text = (value or "").strip().replace("MiB", "")
    if not text or text.upper() == "[N/A]":
        return None
    try:
        return round(float(text) / 1024, 1)
    except ValueError:
        return None


def _apple_gpu(ram_gb: float | None) -> dict[str, Any] | None:
    if platform.system() != "Darwin":
        return None
    if platform.machine().lower() not in {"arm64", "aarch64"}:
        return None
    if not ram_gb:
        budget = None
    else:
        budget = round(min(ram_gb * 0.5, max(ram_gb - 4, 0)), 1)
    return {
        "gpu": "Apple Silicon",
        "gpu_vendor": "apple",
        "vram_gb": budget,
        "vram_free_gb": None,
        "vram_note": "Unified memory estimate, not discrete VRAM.",
        "cuda": False,
        "driver": None,
    }


def _rocm_gpu() -> dict[str, Any] | None:
    smi = shutil.which("rocm-smi")
    if not smi:
        return None
    try:
        result = subprocess.run(
            [smi, "--showproductname"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 and not (result.stdout or "").strip():
        return None
    name = None
    for line in (result.stdout or "").splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        if "card" in label.lower() or "gpu" in label.lower() or "series" in label.lower() or "name" in label.lower():
            candidate = value.strip()
            if candidate and candidate.lower() not in {"n/a", "none"}:
                name = candidate
                break
    return {
        "gpu": name or "AMD GPU",
        "gpu_vendor": "amd",
        "vram_gb": None,
        "vram_free_gb": None,
        "cuda": False,
        "driver": None,
    }


def _windows_gpu() -> dict[str, Any] | None:
    if platform.system() != "Windows":
        return None
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    names = []
    for line in (result.stdout or "").splitlines():
        name = line.strip()
        if not name or "basic display" in name.lower():
            continue
        names.append(name)
    if not names:
        return None
    name = names[0]
    lowered = name.lower()
    if "nvidia" in lowered or "geforce" in lowered or "quadro" in lowered:
        vendor = "nvidia"
    elif "amd" in lowered or "radeon" in lowered:
        vendor = "amd"
    elif "intel" in lowered or "arc" in lowered:
        vendor = "intel"
    else:
        vendor = "unknown"
    return {
        "gpu": name,
        "gpu_vendor": vendor,
        "vram_gb": None,
        "vram_free_gb": None,
        "cuda": False,
        "driver": None,
        "vram_note": "Memory was not read. Install the vendor tool or run Test local AI to confirm acceleration.",
    }


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
