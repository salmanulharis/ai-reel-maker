"""Finding a job's finished videos: the highlight, or one of several shorts."""

from __future__ import annotations

from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".webm"}
MAX_SHORTS = 10


class ShortParamError(ValueError):
    pass


def parse_short(value: str | None) -> int | None:
    """The ?short= query value: a short's number (1-10), or None for the highlight."""
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except ValueError:
        raise ShortParamError("invalid short") from None
    if not 1 <= number <= MAX_SHORTS:
        raise ShortParamError("invalid short")
    return number


def output_stem(short: int | None) -> str:
    return "highlight" if short is None else f"short_{short:02d}"


def find_output(job_dir: Path, short: int | None = None) -> Path | None:
    """The finished video: highlight.mp4/.webm, or short_NN.mp4/.webm for short number N."""
    outputs = job_dir / "outputs"
    if not outputs.is_dir():
        return None
    stem = output_stem(short)
    for path in sorted(outputs.glob(f"{stem}.*")):
        if path.suffix.lower() in VIDEO_SUFFIXES and path.is_file():
            return path
    return None
