from __future__ import annotations

import re
from typing import Union

Number = Union[int, float, str]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def seconds_to_timecode(seconds: float) -> str:
    total_ms = int(round(float(seconds) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def parse_timestamp(value: Number) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    if not parts or len(parts) > 3:
        raise ValueError(f"Invalid timestamp: {value}")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def clamp_timestamp(timestamp: float, duration: float | None, pad: float = 0.15) -> float:
    t = max(0.0, float(timestamp))
    if duration is None or duration <= 0:
        return t
    limit = max(0.0, float(duration) - pad)
    return min(t, limit)
