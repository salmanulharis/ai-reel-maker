"""The Review page's preview of a narrated video: the brief's lines spoken in the background (the
same per-line cache the render uses, so nothing is spoken twice), and the timeline built from
the kept clips with the same functions as the render (app.compose.render_spec). The browser
plays it with Remotion's Player: source clips, the voice, texts and captions, without rendering.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from app.brief.service import load_brief
from app.compose.compose import fit_clips
from app.compose.render_spec import build_spec, preview_video
from app.models.schemas import ClipPlanItem, HighlightConfig
from app.utils.atomic import read_json
from app.video.sources import load_sources
from app.voice import kokoro

_lock = threading.Lock()
_state: dict[str, dict[str, Any]] = {}
_CANVAS = {"9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350), "16:9": (1920, 1080)}


def status(job_id: str) -> dict[str, Any]:
    with _lock:
        return dict(_state.get(job_id) or {"state": "idle", "done": 0, "total": 0, "error": None})


def start(job_id: str, job_dir: Path, settings) -> dict[str, Any]:
    """Speak the brief's lines for the preview, in the background. Lines already spoken are reused."""
    with _lock:
        if (_state.get(job_id) or {}).get("state") == "running":
            return dict(_state[job_id])
    if not kokoro.runtime_available():
        return _set(job_id, state="error", error="Narration needs kokoro-onnx. Install the app requirements.")
    if not kokoro.installed():
        return _set(job_id, state="error", error="Download the narration voice in Settings → Video tools first.")
    brief = load_brief(job_dir)
    if brief is None:
        return _set(job_id, state="error", error="This job has no brief yet.")
    lines = [(beat, text) for beat, text in brief.narration_lines() if kokoro.speakable(text)]
    _set(job_id, state="running", done=0, total=len(lines), error=None)

    def run() -> None:
        try:
            kokoro.narrate(
                job_dir,
                lines,
                voice=kokoro.valid_voice(str(getattr(settings, "tts_voice", kokoro.DEFAULT_VOICE))),
                speed=float(getattr(settings, "tts_speed", 1.0) or 1.0),
                settings=settings,
                on_line=lambda done, total: _set(job_id, done=done, total=total),
            )
            _set(job_id, state="done")
        except Exception as exc:  # noqa: BLE001
            _set(job_id, state="error", error=f"The narration could not be spoken: {exc}"[:300])

    threading.Thread(target=run, daemon=True, name=f"narration-{job_id}").start()
    return status(job_id)


def spoken_lines(job_dir: Path, settings) -> dict[str, dict]:
    """The brief's lines that are already spoken with the current voice and speed."""
    brief = load_brief(job_dir)
    if brief is None:
        return {}
    voice = kokoro.valid_voice(str(getattr(settings, "tts_voice", kokoro.DEFAULT_VOICE)))
    speed = float(getattr(settings, "tts_speed", 1.0) or 1.0)
    folder = Path(job_dir) / kokoro.NARRATION_DIR
    manifest = read_json(folder / kokoro.MANIFEST, {})
    manifest = manifest if isinstance(manifest, dict) else {}
    result = {}
    for beat, text in brief.narration_lines():
        key = kokoro.line_key(text, voice, speed)
        entry = manifest.get(key)
        if isinstance(entry, dict) and (folder / f"{key}.wav").is_file():
            result[beat] = {**entry, "key": key}
    return result


def preview_spec(job_id: str, job_dir: Path, job: dict, settings) -> dict[str, Any] | None:
    """The timeline the Review player plays (None without a brief)."""
    brief = load_brief(job_dir)
    if brief is None:
        return None
    config = HighlightConfig.model_validate(job.get("config") or {})
    sources = load_sources(job_dir)
    first = str(sources[0].get("id") or "v0") if sources else "v0"
    clips = []
    for item in job.get("clips") or []:
        if item.get("kept", True) is False:
            continue
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start >= 0:
            clips.append(ClipPlanItem(start=start, end=end, reason="", source_id=str(item.get("source_id") or first)))
    if not clips:
        return None
    narration = spoken_lines(job_dir, settings) if config.narration else {}
    if narration:
        clips, _notes = fit_clips(clips, brief, narration, sources)
    meta = job.get("metadata") or {}
    width, height = _CANVAS.get(config.aspect_ratio, (int(meta.get("width") or 1920), int(meta.get("height") or 1080)))
    spec = build_spec(
        clips,
        brief,
        narration,
        width=width,
        height=height,
        fps=30.0,
        caption_style=config.caption_style if config.subtitles == "burned" else "none",
        video=preview_video(clips, lambda clip: f"/api/media/{job_id}/source/{clip.source_id or first}"),
        narration_src=lambda line: f"/api/media/{job_id}/narration/{line.get('file')}",
    )
    return spec.model_dump()


def _set(job_id: str, **fields) -> dict[str, Any]:
    with _lock:
        current = _state.setdefault(job_id, {"state": "idle", "done": 0, "total": 0, "error": None})
        current.update(fields)
        return dict(current)
