"""Download for video editors: the kept clips as editor timelines in one zip file."""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Callable

from app.analysis.sequence_limits import clamp_plan_clips, sequence_cap
from app.config import resolve_project_path
from app.models.db_models import Job
from app.models.schemas import ClipPlanItem, HighlightConfig, VideoMetadata
from app.video.quality import source_frame_rate
from app.video.sources import fallback_source, load_sources
from app.video.timeline_export import (
    Timeline,
    TimelineClip,
    TimelineSource,
    build_timeline,
    edl,
    fcpxml,
    srt,
    xmeml,
)

Probe = Callable[[Path], VideoMetadata]


class ExportError(Exception):
    """A reason the timeline cannot be made; the message is shown to the user."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _default_probe(path: Path) -> VideoMetadata:
    from app.video.ffmpeg_runner import FFmpegRunner

    return FFmpegRunner().probe(path)


def _original_path(source: dict) -> Path:
    """The uploaded file itself (not the cropped working copy)."""
    path = Path(str(source.get("path") or ""))
    if path.is_file():
        return path.resolve()
    return resolve_project_path(path)


def _segments(job_dir: Path, source_id: str, multi: bool) -> list[dict]:
    path = job_dir / "work" / source_id / "transcript.json" if multi else job_dir / "transcript.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    segments = data.get("segments") if isinstance(data, dict) else None
    return [s for s in segments or [] if isinstance(s, dict)]


def _timeline_source(source: dict, job_dir: Path, multi: bool, use_speech: bool, probe: Probe) -> TimelineSource:
    path = _original_path(source)
    if not path.is_file():
        raise ExportError("source video missing", status=404)
    try:
        meta = probe(path)
        fps = source_frame_rate(meta) or float(source.get("fps") or 0)
        width, height = int(meta.width or 0), int(meta.height or 0)
        duration, has_audio, channels = float(meta.duration or 0), bool(meta.has_audio), int(meta.channels or 0)
    except Exception:  # noqa: BLE001 - FFprobe missing or failing: use what the upload recorded
        fps = float(source.get("fps") or 0)
        width, height = int(source.get("width") or 0), int(source.get("height") or 0)
        duration, has_audio, channels = float(source.get("duration") or 0), True, 2
    source_id = str(source.get("id") or "v0")
    return TimelineSource(
        id=source_id,
        name=str(source.get("filename") or path.name),
        path=path,
        fps=fps,
        width=width,
        height=height,
        duration=duration,
        has_audio=has_audio,
        channels=channels or 2,
        segments=_segments(job_dir, source_id, multi) if use_speech else [],
    )


def _clock(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _readme(timeline: Timeline, has_subtitles: bool) -> str:
    rate = timeline.rate
    fps = f"{rate.fps:.3f}".rstrip("0").rstrip(".")
    paths = "\n".join(f"  {source.path}" for source in timeline.sources)
    subtitles = "  highlight.srt     Subtitles timed to the edit\n" if has_subtitles else ""
    return (
        "AI Video Highlights - timeline for video editors\n"
        "\n"
        f"{timeline.name}: {len(timeline.clips)} clips, {_clock(rate.seconds(timeline.duration))} long, "
        f"{fps} fps\n"
        "\n"
        "Files\n"
        "  highlight.xml     Premiere Pro and DaVinci Resolve (Final Cut Pro 7 XML)\n"
        "  highlight.fcpxml  Final Cut Pro and DaVinci Resolve\n"
        "  highlight.edl     Other editors (CMX 3600 EDL)\n"
        f"{subtitles}"
        "\n"
        "How to open it\n"
        "  Premiere Pro:    File > Import, then choose highlight.xml.\n"
        "  DaVinci Resolve: File > Import > Timeline, then choose highlight.xml or highlight.fcpxml.\n"
        "  Final Cut Pro:   File > Import > XML, then choose highlight.fcpxml.\n"
        "  Other editors:   import highlight.edl, then link each clip to the file in its SOURCE FILE note.\n"
        "\n"
        "The timeline uses your original video files, where the app keeps them:\n"
        f"{paths}\n"
        "Leave them there while you edit, or relink the clips if you move them. Deleting this highlight\n"
        "in the app also deletes these files.\n"
        "\n"
        "Not included: the crop, the aspect ratio (such as 9:16), fades and burned-in subtitles.\n"
        "Set these in your editor.\n"
    )


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")
    return slug[:60] or "highlight"


def build_editor_package(job: Job, job_dir: Path, clips: list[dict], probe: Probe | None = None) -> tuple[bytes, str]:
    """
    A zip with the kept clips as timelines (FCPXML, Final Cut Pro 7 XML, EDL), subtitles when
    there is speech, and a README. Returns the zip bytes and a download file name.

    Clips are limited to the longest-clip setting exactly as when the video is created, so the
    timeline matches the rendered highlight (apart from fades, crop and aspect ratio).
    """
    probe = probe or _default_probe
    config = HighlightConfig.model_validate(json.loads(job.config_json or "{}"))
    kept = [c for c in clips if c.get("kept", True)]
    if not kept:
        raise ExportError("no clips to export")
    items = [
        ClipPlanItem(
            start=float(c["start"]),
            end=float(c["end"]),
            reason="",
            source_id=str(c.get("source_id") or ""),
        )
        for c in kept
        if float(c["end"]) > float(c["start"])
    ]
    clamp_plan_clips(items, sequence_cap(config))

    sources = load_sources(job_dir) or [fallback_source(job.input_path, job.original_filename)]
    multi = len(sources) > 1
    known = {str(s.get("id") or "v0"): s for s in sources}
    first_id = str(sources[0].get("id") or "v0")
    # Clips without a known video id come from the first video.
    for item in items:
        if item.source_id not in known:
            item.source_id = first_id
    # Only the videos the clips use are probed.
    by_id = {
        source_id: _timeline_source(known[source_id], job_dir, multi, config.use_speech, probe)
        for source_id in dict.fromkeys(item.source_id for item in items)
    }
    timeline_clips = [
        TimelineClip(source=by_id[item.source_id], start=item.start, end=item.end) for item in items
    ]

    stem = Path(sources[0].get("filename") or job.original_filename or "video").stem
    name = f"{stem} highlight" if not multi else "Highlight"
    timeline = build_timeline(name, timeline_clips)
    if not timeline.clips:
        raise ExportError("no clips to export")

    subtitles = srt(timeline)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("highlight.xml", xmeml(timeline))
        archive.writestr("highlight.fcpxml", fcpxml(timeline))
        archive.writestr("highlight.edl", edl(timeline))
        if subtitles:
            archive.writestr("highlight.srt", subtitles)
        archive.writestr("README.txt", _readme(timeline, bool(subtitles)))
    return buffer.getvalue(), f"{_slug(stem if not multi else 'highlight')}-for-editors.zip"
