"""Adjustments for one analysis, decided from the memory and load at the moment it starts.

Saved settings are never changed here: a run plan lives for one job and is written to its log.
It decides whether models are loaded one at a time (so they do not compete for memory), and
notes conditions that will slow the job down. Models themselves are never swapped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.services.model_catalog import model_memory_gb, nominal_ram_gb, whisper_memory_gb

# Memory the app, FFmpeg and the operating system need besides the loaded models.
BASE_MEMORY_GB = 2.0
# Assumed size of a model the catalog does not know (a user's own choice).
_UNKNOWN_MODEL_GB = 5.0
# Above these shares of memory, models are loaded one at a time.
_RAM_SHARE = 0.7
_VRAM_SHARE = 0.9
_BUSY_CPU_PERCENT = 75.0
_SMALL_CARD_GB = 8.0


@dataclass
class RunPlan:
    # Free each model as soon as its stage is done instead of keeping it loaded.
    one_at_a_time: bool = False
    # Unload the AI models before speech to text, which needs the graphics card to itself.
    free_gpu_for_speech: bool = False
    # Precision for speech to text on a small graphics card (None: the default).
    whisper_gpu_compute: str | None = None
    notes: list[str] = field(default_factory=list)


def live_readings() -> dict[str, Any]:
    """Memory, processor load and power state right now (cached hardware data can be hours old)."""
    readings: dict[str, Any] = {"ram_available_gb": None, "cpu_percent": None, "on_battery": None}
    try:
        import psutil

        readings["ram_available_gb"] = round(psutil.virtual_memory().available / (1024**3), 1)
        readings["cpu_percent"] = float(psutil.cpu_percent(interval=0.2))
        battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
        if battery is not None:
            readings["on_battery"] = battery.power_plugged is False
    except Exception:  # noqa: BLE001 - readings are advisory
        pass
    return readings


def _model_gb(name: str) -> float:
    return model_memory_gb(name) or _UNKNOWN_MODEL_GB


def build_run_plan(
    settings: Settings,
    hardware: dict[str, Any],
    *,
    ollama_here: bool,
    whisper_on_gpu: bool = False,
    readings: dict[str, Any] | None = None,
) -> RunPlan:
    """``hardware`` is the detected profile; ``ollama_here`` says whether Ollama shares this
    computer's memory; ``whisper_on_gpu`` whether speech to text will use the graphics card."""
    readings = live_readings() if readings is None else readings
    plan = RunPlan()
    speech = bool(settings.enable_whisper)
    vision = bool(settings.enable_visual_analysis)
    text_gb = _model_gb(settings.ollama_text_model) if speech else 0.0
    vision_gb = _model_gb(settings.ollama_vision_model) if vision else 0.0
    whisper_gb = whisper_memory_gb(settings.whisper_model) if speech else 0.0
    largest = max(text_gb, vision_gb)
    uses_gpu = hardware.get("inference_mode") == "gpu"
    available = readings.get("ram_available_gb")

    if ollama_here and uses_gpu:
        vram = float(hardware.get("vram_gb") or 0)
        on_card = text_gb + vision_gb + (whisper_gb if whisper_on_gpu else 0.0)
        if vram and on_card > vram * _VRAM_SHARE:
            plan.one_at_a_time = True
            plan.notes.append(
                f"The models need about {on_card:.0f} GB together and the graphics card has {vram:g} GB, "
                "so they are loaded one at a time."
            )
        if whisper_on_gpu and vram:
            plan.free_gpu_for_speech = whisper_gb + largest > vram * _VRAM_SHARE
            if vram <= _SMALL_CARD_GB:
                plan.whisper_gpu_compute = "int8_float16"
    elif ollama_here:
        together = text_gb + vision_gb + whisper_gb + BASE_MEMORY_GB
        total = nominal_ram_gb(hardware.get("ram_gb"))
        short_now = available is not None and float(available) < together
        if (total and together > total * _RAM_SHARE) or short_now:
            plan.one_at_a_time = True
            free = f" ({available} GB is free now)" if short_now else ""
            plan.notes.append(
                f"The models need about {together:.0f} GB together{free}, so they are loaded one at a time."
            )
        if available is not None and float(available) < largest + 1.0:
            plan.notes.append(
                f"Only {available} GB of memory is free and the largest model needs about {largest:.0f} GB. "
                "Closing other programs avoids slowdowns."
            )
    elif speech and available is not None and float(available) < whisper_gb + 1.0:
        plan.notes.append(f"Only {available} GB of memory is free. Closing other programs helps speech to text.")

    load = readings.get("cpu_percent")
    if load is not None and float(load) >= _BUSY_CPU_PERCENT and not uses_gpu:
        plan.notes.append(
            f"Other programs are using {float(load):.0f}% of the processor, so analysis will be slower than usual."
        )
    if readings.get("on_battery"):
        plan.notes.append("This computer is running on battery, which usually limits speed. Plug it in for faster analysis.")
    return plan
