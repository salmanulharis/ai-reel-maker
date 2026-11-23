from __future__ import annotations

import csv
import io
import os
import platform
import json
import re
import shutil
import struct
import subprocess
from typing import Any

from app.services.model_catalog import DEFAULT_GOAL, recommend_setup

# Settings-table key naming the graphics card Ollama was last seen using.
GPU_CONFIRMED_KEY = "gpu_confirmed"
_GB = 1024**3


def detect_hardware(*, include_encoders: bool = False, goal: str = DEFAULT_GOAL) -> dict[str, Any]:
    """Fast hardware snapshot. Encoder probes stay off unless include_encoders is set."""
    memory = _memory()
    info: dict[str, Any] = {
        "cpu": memory.get("cpu") or platform.processor() or platform.machine(),
        "cpu_cores": memory.get("cores"),
        "cpu_threads": memory.get("threads"),
        # Performance and efficiency cores, on processors that mix both (None otherwise).
        "cpu_performance_cores": memory.get("performance_cores"),
        "cpu_efficiency_cores": memory.get("efficiency_cores"),
        "cpu_features": memory.get("features") or [],
        "on_battery": memory.get("on_battery"),
        "battery_percent": memory.get("battery_percent"),
        "system": platform.system(),
        "ram_gb": memory.get("ram_gb"),
        "ram_available_gb": memory.get("ram_available_gb"),
        "gpu": None,
        "gpu_vendor": None,
        "gpu_integrated": False,
        # A separate graphics card whose memory was read, but that Ollama has not been seen using.
        "gpu_unconfirmed": False,
        "gpu_temp_c": None,
        "gpus": [],
        "vram_gb": None,
        "vram_free_gb": None,
        "vram_note": None,
        "driver": None,
        "cuda": False,
        "whisper_cuda": None,
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
    gpus = _all_gpus(info.get("ram_gb"))
    info["gpus"] = [
        {key: item.get(key) for key in ("gpu", "gpu_vendor", "vram_gb", "gpu_integrated")} for item in gpus
    ]
    gpu = _pick_gpu(gpus)
    if gpu:
        confirmed = bool(gpu.pop("gpu_confirmed", False))
        info.update(gpu)
        has_vram = (info.get("vram_gb") or 0) >= 2
        vendor = info.get("gpu_vendor")
        if has_vram and not info.get("gpu_integrated"):
            if vendor in {"nvidia", "apple"} or confirmed or _gpu_confirmed(info.get("gpu")):
                info["inference_mode"] = "gpu"
            elif vendor in {"amd", "intel"}:
                # Ollama supports only some of these cards. Larger models are recommended once
                # a loaded model has been seen in this card's memory (Test AI, or any highlight).
                info["gpu_unconfirmed"] = True
                info["vram_note"] = (
                    "This card's memory was read, but Ollama has not been seen using it yet. "
                    "Run Test AI: when the model loads on the card, larger models are recommended."
                )
    uses_gpu = info["inference_mode"] == "gpu"
    if info.get("cuda"):
        info["whisper_cuda"] = whisper_cuda_available()
    info["inference_label"] = "GPU-accelerated" if uses_gpu else "CPU-only"
    if include_encoders:
        encoders, encode_label = _encode_status()
        info["encoders"] = encoders
        info["encoders_checked"] = True
        info["encode_acceleration"] = encode_label
        info["acceleration"] = encode_label
    info["recommendation"] = recommend_for(info, goal)
    recommendation = info["recommendation"]
    info["recommended_vision_model"] = recommendation["vision_model"]
    info["recommended_text_model"] = recommendation["text_model"]
    info["recommended_whisper_model"] = recommendation["whisper_model"]
    info["recommended_whisper_device"] = recommendation["whisper_device"]
    return info


def recommend_for(hardware: dict[str, Any], goal: str = DEFAULT_GOAL) -> dict[str, Any]:
    """The recommendation for a hardware snapshot. Graphics memory counts only when the AI can
    use the card, and threads follow the performance cores on processors that mix core types."""
    uses_gpu = hardware.get("inference_mode") == "gpu"
    return recommend_setup(
        hardware.get("vram_gb") if uses_gpu else None,
        hardware.get("ram_gb"),
        hardware.get("vram_free_gb") if uses_gpu else None,
        hardware.get("cpu_performance_cores") or hardware.get("cpu_cores") or hardware.get("cpu_threads"),
        goal=goal,
        whisper_cuda=hardware.get("whisper_cuda") if uses_gpu else None,
    )


_whisper_cuda: bool | None = None


def whisper_cuda_available() -> bool:
    """True when faster-whisper's engine (CTranslate2) can see an NVIDIA card. Checked once."""
    global _whisper_cuda
    if _whisper_cuda is None:
        try:
            import ctranslate2

            _whisper_cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:  # noqa: BLE001 - not installed, or no CUDA runtime
            _whisper_cuda = False
    return _whisper_cuda


def _gpu_confirmed(name: str | None) -> bool:
    """True when Ollama was seen running a model on this card (stored by ai_profile)."""
    if not name:
        return False
    try:
        from app.services.settings_store import read_setting

        return read_setting(GPU_CONFIRMED_KEY) == name
    except Exception:  # noqa: BLE001 - no database yet
        return False


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
            encoding="utf-8",
            errors="replace",
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
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20, check=False)
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
        "on_battery": None,
        "battery_percent": None,
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        info["ram_gb"] = round(memory.total / (1024**3), 1)
        info["ram_available_gb"] = round(memory.available / (1024**3), 1)
        info["cores"] = psutil.cpu_count(logical=False) or psutil.cpu_count()
        info["threads"] = psutil.cpu_count(logical=True)
        info["cpu"] = platform.processor() or info["cpu"]
        battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
        if battery is not None:
            info["on_battery"] = battery.power_plugged is False
            info["battery_percent"] = round(float(battery.percent))
    except Exception:
        pass
    info["cpu"] = _cpu_name() or info["cpu"]
    info["performance_cores"], info["efficiency_cores"] = _core_types(info["cores"])
    info["features"] = _cpu_features()
    return info


def _cpu_name() -> str | None:
    """The processor's marketing name (platform.processor() gives only a family code on Windows)."""
    system = platform.system()
    try:
        if system == "Windows":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip() or None
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.lower().startswith("model name") and ":" in line:
                        return line.split(":", 1)[1].strip() or None
        if system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=4, check=False
            )
            return (result.stdout or "").strip() or None
    except Exception:  # noqa: BLE001 - the name is only shown, never required
        return None
    return None


def parse_core_classes(buffer: bytes) -> list[int]:
    """Efficiency class of every physical core, from GetLogicalProcessorInformationEx records
    (RelationProcessorCore). Higher means a faster core."""
    classes: list[int] = []
    offset = 0
    while offset + 12 <= len(buffer):
        relationship, size = struct.unpack_from("<II", buffer, offset)
        if size <= 0:
            break
        if relationship == 0:
            classes.append(buffer[offset + 9])
        offset += size
    return classes


def split_core_classes(classes: list[int]) -> tuple[int | None, int | None]:
    """(performance cores, efficiency cores), or (None, None) when all cores are the same kind."""
    if len(set(classes)) < 2:
        return None, None
    fastest = max(classes)
    performance = sum(1 for value in classes if value == fastest)
    return performance, len(classes) - performance


def count_cpu_list(text: str) -> int:
    """Number of processors in a Linux list such as "0-11,16"."""
    total = 0
    for part in (text or "").strip().split(","):
        if not part.strip():
            continue
        low, _, high = part.partition("-")
        try:
            total += int(high or low) - int(low) + 1
        except ValueError:
            return 0
    return total


def _core_types(physical: int | None) -> tuple[int | None, int | None]:
    system = platform.system()
    try:
        if system == "Windows":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            length = ctypes.c_ulong(0)
            kernel32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(length))
            if not length.value:
                return None, None
            buffer = ctypes.create_string_buffer(length.value)
            if not kernel32.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(length)):
                return None, None
            return split_core_classes(parse_core_classes(buffer.raw[: length.value]))
        if system == "Linux" and physical and os.path.exists("/sys/devices/cpu_atom/cpus"):
            # Efficiency cores have one thread each, so their thread count is their core count.
            with open("/sys/devices/cpu_atom/cpus", encoding="utf-8") as handle:
                efficiency = count_cpu_list(handle.read())
            if 0 < efficiency < physical:
                return physical - efficiency, efficiency
    except Exception:  # noqa: BLE001
        return None, None
    return None, None


def _cpu_features() -> list[str]:
    """Instruction sets that speed up local AI, when the system reports them."""
    system = platform.system()
    found: list[str] = []
    try:
        if system == "Windows":
            import ctypes

            present = ctypes.windll.kernel32.IsProcessorFeaturePresent
            for code, name in ((39, "avx"), (40, "avx2"), (41, "avx512")):
                if present(code):
                    found.append(name)
        elif system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith("flags"):
                        flags = set(line.split(":", 1)[1].split())
                        found = [name for name, flag in (("avx", "avx"), ("avx2", "avx2"), ("avx512", "avx512f")) if flag in flags]
                        break
    except Exception:  # noqa: BLE001
        return []
    return found


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


def _all_gpus(ram_gb: float | None) -> list[dict[str, Any]]:
    """Every graphics adapter found, each with its vendor and memory when it could be read."""
    nvidia = _nvidia_gpus()
    for item in nvidia:
        item.update(cuda=True, gpu_vendor="nvidia", gpu_integrated=False)
    if nvidia:
        # nvidia-smi reports memory exactly; other adapters in the same computer are listed after.
        others = [item for item in _windows_gpus() if item.get("gpu_vendor") != "nvidia"]
        return nvidia + others
    apple = _apple_gpu(ram_gb)
    if apple:
        return [apple]
    rocm = _rocm_gpu()
    if rocm:
        return [rocm]
    windows = _windows_gpus()
    if windows:
        return windows
    if _sysfs_gpu():
        return [
            {
                "gpu": "GPU details unavailable",
                "gpu_vendor": None,
                "vram_gb": None,
                "vram_free_gb": None,
                "cuda": False,
            }
        ]
    return []


def _pick_gpu(gpus: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The adapter the AI would use: a separate card over built-in graphics, then the most memory."""
    if not gpus:
        return None
    best = max(gpus, key=lambda item: (not item.get("gpu_integrated"), item.get("vram_gb") or 0))
    return dict(best)


def _nvidia_gpus() -> list[dict[str, Any]]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        result = subprocess.run(
            [
                smi,
                "--query-gpu=name,memory.total,memory.free,driver_version,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return parse_nvidia_smi(result.stdout or "")


def parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    found = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 2:
            continue
        name = row[0].strip()
        if not name:
            continue
        found.append(
            {
                "gpu": name,
                "vram_gb": _mib_to_gb(row[1]),
                "vram_free_gb": _mib_to_gb(row[2]) if len(row) > 2 else None,
                "driver": row[3].strip() if len(row) > 3 else None,
                "gpu_temp_c": _number(row[4]) if len(row) > 4 else None,
            }
        )
    return found


def _number(value: str) -> float | None:
    try:
        return float((value or "").strip())
    except ValueError:
        return None


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
        "gpu_integrated": False,
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
            encoding="utf-8",
            errors="replace",
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
    total, free = _rocm_vram(smi)
    return {
        "gpu": name or "AMD GPU",
        "gpu_vendor": "amd",
        "gpu_integrated": False,
        "vram_gb": total,
        "vram_free_gb": free,
        "cuda": False,
        "driver": None,
        # rocm-smi is part of the ROCm stack Ollama uses, so the card counts as usable.
        "gpu_confirmed": total is not None,
    }


def _rocm_vram(smi: str) -> tuple[float | None, float | None]:
    try:
        result = subprocess.run(
            [smi, "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=4,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    return parse_rocm_vram(result.stdout or "")


def parse_rocm_vram(text: str) -> tuple[float | None, float | None]:
    """(total GB, free GB) of the card with the most memory, from `rocm-smi --showmeminfo vram --json`."""
    start = text.find("{")
    if start < 0:
        return None, None
    try:
        data = json.loads(text[start:])
    except ValueError:
        return None, None
    best: tuple[float | None, float | None] = (None, None)
    for card in data.values() if isinstance(data, dict) else []:
        if not isinstance(card, dict):
            continue
        total = used = None
        for key, value in card.items():
            label = key.lower()
            if "vram" not in label or "(b)" not in label:
                continue
            number = _number(str(value))
            if "used" in label:
                used = number
            elif "total" in label:
                total = number
        if total and total / _GB > (best[0] or 0):
            free = round((total - used) / _GB, 1) if used is not None else None
            best = (round(total / _GB, 1), free)
    return best


_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
# Separate Intel cards carry a model number (Arc A770, Arc B580); built-in ones do not.
_INTEL_DISCRETE = re.compile(r"\barc\b.*\b[ab]\d{3}", re.IGNORECASE)
_AMD_DISCRETE = re.compile(r"\brx\s*\d|\bpro\b|firepro", re.IGNORECASE)


def gpu_vendor(name: str) -> str:
    lowered = name.lower()
    if "nvidia" in lowered or "geforce" in lowered or "quadro" in lowered or "rtx" in lowered:
        return "nvidia"
    if "amd" in lowered or "radeon" in lowered:
        return "amd"
    if "intel" in lowered or "arc" in lowered:
        return "intel"
    return "unknown"


def is_integrated(name: str, vendor: str) -> bool:
    """True for graphics built into the processor, which share system memory."""
    if vendor == "intel":
        return not _INTEL_DISCRETE.search(name)
    if vendor == "amd":
        return not _AMD_DISCRETE.search(name)
    return False


def adapters_to_gpus(adapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Windows display adapters ({"name", "memory_bytes"}) as GPU entries."""
    found = []
    for adapter in adapters:
        name = str(adapter.get("name") or "").strip()
        if not name or "basic display" in name.lower():
            continue
        vendor = gpu_vendor(name)
        integrated = is_integrated(name, vendor)
        memory = adapter.get("memory_bytes")
        vram = round(float(memory) / _GB, 1) if memory and not integrated else None
        note = None
        if integrated:
            note = "Built-in graphics share system memory, so model sizes follow this computer's memory."
        elif vram is None:
            note = "Memory was not read. Install the vendor tool or run Test local AI to confirm acceleration."
        found.append(
            {
                "gpu": name,
                "gpu_vendor": vendor,
                "gpu_integrated": integrated,
                "vram_gb": vram,
                "vram_free_gb": None,
                "cuda": False,
                "driver": None,
                "vram_note": note,
            }
        )
    return found


def _windows_gpus() -> list[dict[str, Any]]:
    if platform.system() != "Windows":
        return []
    return adapters_to_gpus(present_adapters(_registry_adapters(), _powershell_adapters()))


def present_adapters(registry: list[dict[str, Any]], present: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Registry entries (which have the memory size) for adapters that are actually installed.
    The registry keeps entries for cards that were removed, so it is checked against the list
    of present adapters; without that list, the registry is used as it is."""
    if not present:
        return registry
    names = {str(item.get("name") or "").strip() for item in present}
    matched = [item for item in registry if str(item.get("name") or "").strip() in names]
    return matched or present


def _registry_adapters() -> list[dict[str, Any]]:
    """Display adapters with their dedicated memory. The registry holds the full 64-bit size;
    the WMI AdapterRAM value stops at 4 GB."""
    try:
        import winreg

        found = []
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS) as root:
            for index in range(64):
                try:
                    name = winreg.EnumKey(root, index)
                except OSError:
                    break
                if not name.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, name) as key:
                        description = _registry_value(winreg, key, "DriverDesc")
                        memory = _registry_value(winreg, key, "HardwareInformation.qwMemorySize")
                        if memory is None:
                            memory = _registry_value(winreg, key, "HardwareInformation.MemorySize")
                except OSError:
                    continue
                if description:
                    found.append({"name": str(description), "memory_bytes": _as_bytes_count(memory)})
        return found
    except Exception:  # noqa: BLE001
        return []


def _registry_value(winreg, key, name: str):
    try:
        return winreg.QueryValueEx(key, name)[0]
    except OSError:
        return None


def _as_bytes_count(value) -> int | None:
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, (bytes, bytearray)) and value:
        return int.from_bytes(bytes(value[:8]), "little") or None
    return None


def _powershell_adapters() -> list[dict[str, Any]]:
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
            errors="replace",
            timeout=4,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [{"name": line.strip(), "memory_bytes": None} for line in (result.stdout or "").splitlines() if line.strip()]


def _nvidia_smi_name() -> str | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        result = subprocess.run(
            [smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
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
