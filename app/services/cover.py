"""Cover image: a frame from the finished highlight, as a JPEG to download or preview."""

from __future__ import annotations

import math
from pathlib import Path

from app.video.ffmpeg_runner import FFmpegError, FFmpegRunner
from app.video.processor import VideoProcessor

MIN_WIDTH = 64
MAX_PREVIEW_WIDTH = 1920
COVER_DIR = "covers"


class CoverError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def output_video(job_dir: Path) -> Path | None:
    outputs = job_dir / "outputs"
    files = sorted(outputs.glob("highlight.*")) if outputs.is_dir() else []
    return next((f for f in files if f.suffix.lower() in {".mp4", ".webm"} and f.is_file()), None)


def parse_request(time_text: str | None, width_text: str | None) -> tuple[float, int | None]:
    """Seconds into the highlight and an optional preview width, from query text."""
    try:
        seconds = float(time_text if time_text not in (None, "") else "0")
    except ValueError:
        raise CoverError("invalid time") from None
    if not math.isfinite(seconds) or seconds < 0:
        raise CoverError("invalid time")
    width = None
    if width_text not in (None, ""):
        try:
            width = int(width_text)
        except ValueError:
            raise CoverError("invalid width") from None
        if width < MIN_WIDTH:
            raise CoverError("invalid width")
        width = min(width, MAX_PREVIEW_WIDTH)
    return seconds, width


def cover_image(job_dir: Path, seconds: float, width: int | None = None, runner: FFmpegRunner | None = None) -> Path:
    """
    The frame `seconds` into the finished highlight (clamped to its length), full size or
    `width` pixels wide. Frames are kept per video version: covers from an earlier render of the
    highlight are removed the first time a frame of the new one is asked for.
    """
    video = output_video(job_dir)
    if video is None:
        raise CoverError("no output", status=404)
    runner = runner or FFmpegRunner()
    try:
        duration = float(runner.probe(video).duration or 0)
    except Exception:  # noqa: BLE001 - without a length, only negative times are clamped
        duration = 0.0
    if duration > 0:
        # The very last frame can be missing in a cut file; stay a little before the end.
        seconds = min(seconds, max(0.0, duration - 0.1))
    version = str(video.stat().st_mtime_ns)
    folder = job_dir / "outputs" / COVER_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.jpg"):
        if not old.name.startswith(f"{version}-"):
            old.unlink(missing_ok=True)
    dest = folder / f"{version}-{int(round(seconds * 1000))}-{width or 'full'}.jpg"
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    try:
        runner.run(VideoProcessor.extract_frame_command(video, dest, seconds, width=width))
    except FFmpegError as exc:
        raise CoverError("could not read a frame", status=500) from exc
    if not dest.is_file() or dest.stat().st_size == 0:
        raise CoverError("could not read a frame", status=500)
    return dest
