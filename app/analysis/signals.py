"""Cheap, model-free signals that run the same on any computer.

Scene picture quality comes from the frames already sampled for each scene; loudness comes
from the 16 kHz mono WAV already extracted for transcription. Neither needs a model, a GPU, or
an extra decode of the video. Every function returns neutral values on failure, so a missing
or unreadable input never stops an analysis.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

# Sharpness: variance of the Laplacian on a 256-px grey frame, on a log scale. Around 30 is
# visibly soft, several hundred is crisp.
_SHARP_LOW, _SHARP_HIGH = math.log10(30), math.log10(600)
_BLACK_BRIGHTNESS = 0.07
_BLACK_CONTRAST = 0.035
# Below this the picture is dark, blurry, or black enough that a viewer would notice.
POOR_QUALITY = 0.3

ENERGY_HOP = 0.25  # seconds per loudness value


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def frame_quality(path: str | Path) -> dict | None:
    """Sharpness, brightness, and contrast of one frame (0-1 each), or None if unreadable."""
    try:
        import cv2
    except ImportError:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    scale = 256 / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    pixels = image.astype(np.float32) / 255.0
    brightness = float(pixels.mean())
    contrast = float(pixels.std())
    variance = float(cv2.Laplacian(image, cv2.CV_64F).var())
    sharpness = _clamp((math.log10(variance + 1.0) - _SHARP_LOW) / (_SHARP_HIGH - _SHARP_LOW))
    return {"sharpness": sharpness, "brightness": brightness, "contrast": contrast}


def scene_quality(frame_paths: list[str | Path]) -> dict:
    """Picture quality of a scene from its sampled frames.

    quality: 0-1 blend of sharpness, exposure, and contrast; black frames score near 0.
    black: every sampled frame is (nearly) black, e.g. a fade or an empty slate."""
    measured = [item for item in (frame_quality(path) for path in frame_paths) if item]
    if not measured:
        return {"quality": None, "black": False, "frames": 0}
    sharpness = float(np.median([m["sharpness"] for m in measured]))
    brightness = float(np.median([m["brightness"] for m in measured]))
    contrast = float(np.median([m["contrast"] for m in measured]))
    black = all(m["brightness"] < _BLACK_BRIGHTNESS and m["contrast"] < _BLACK_CONTRAST for m in measured)
    exposure = _clamp(1.0 - abs(brightness - 0.5) * 1.6)
    detail = _clamp(contrast / 0.18)
    quality = 0.03 if black else _clamp(0.5 * sharpness + 0.3 * exposure + 0.2 * detail)
    return {
        "quality": round(quality, 3),
        "sharpness": round(sharpness, 3),
        "brightness": round(brightness, 3),
        "black": black,
        "frames": len(measured),
    }


def audio_energy(wav_path: str | Path, hop: float = ENERGY_HOP) -> list[float]:
    """Loudness of each `hop` seconds, scaled 0-1 within this recording (5th to 95th
    percentile), so "loud" means loud for this video. Empty when the file cannot be read."""
    try:
        with wave.open(str(wav_path), "rb") as handle:
            if handle.getsampwidth() != 2:
                return []
            channels = handle.getnchannels()
            rate = handle.getframerate()
            step = max(1, int(rate * hop))
            levels: list[float] = []
            while True:
                raw = handle.readframes(step * 64)
                if not raw:
                    break
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1)
                usable = len(samples) - len(samples) % step
                if usable:
                    blocks = samples[:usable].reshape(-1, step)
                    levels.extend((20 * np.log10(np.sqrt((blocks**2).mean(axis=1)) + 1.0)).tolist())
                if len(samples) > usable:
                    tail = samples[usable:]
                    levels.append(float(20 * np.log10(np.sqrt((tail**2).mean()) + 1.0)))
    except (OSError, EOFError, wave.Error, ValueError):
        return []
    if not levels:
        return []
    values = np.asarray(levels, dtype=np.float32)
    low, high = np.percentile(values, 5), np.percentile(values, 95)
    if high - low < 1e-6:
        return [0.5] * len(levels)
    return [round(float(v), 3) for v in np.clip((values - low) / (high - low), 0.0, 1.0)]


def loudness_peak(energy: list[float], start: float, end: float, hop: float = ENERGY_HOP) -> float | None:
    """Loudest part of a span: the 90th percentile, so one click or cough does not count."""
    if not energy or end <= start:
        return None
    first = max(0, int(start / hop))
    last = min(len(energy), max(first + 1, int(math.ceil(end / hop))))
    window = energy[first:last]
    if not window:
        return None
    return float(np.percentile(window, 90))


def quiet_point(energy: list[float], t: float, radius: float = 0.5, hop: float = ENERGY_HOP) -> float:
    """The quietest moment within `radius` seconds of t, for cutting between sounds instead of
    through them. Returns t unchanged when there is nothing clearly quieter."""
    if not energy:
        return t
    center = int(t / hop)
    reach = max(1, int(radius / hop))
    lo, hi = max(0, center - reach), min(len(energy) - 1, center + reach)
    if lo > hi or center >= len(energy):
        return t
    best = min(range(lo, hi + 1), key=lambda i: (energy[i], abs(i - center)))
    if energy[best] >= energy[min(center, len(energy) - 1)] - 0.1:
        return t
    return round(best * hop + hop / 2, 3)
