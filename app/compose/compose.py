"""Turning a finished highlight into a narrated video.

Called by the render worker for every video it makes, after the usual render (which already
did the cuts, framing, loudness, logo, progress bar and transitions):

1. ``prepare_narration`` (once per job, before the clips are rendered): the brief's lines are
   spoken (cached per line), and the clips of narrated beats are lengthened where a line needs
   more time (``render_spec.fit_narration``).
2. ``compose_video``: the timeline (render_spec) is built for the finished video; titles, beat
   texts and captions are drawn by Remotion, or burned in as ASS subtitles by FFmpeg when
   Remotion is not prepared or fails; then FFmpeg mixes the voice in, lowering the video's own
   sound under it. The music pass (if any) runs after this, as before.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from app.brief.schema import VideoBrief
from app.brief.service import load_brief
from app.compose import remotion
from app.compose.render_spec import RenderSpec, SpecVideo, build_spec, fit_narration, narration_srt, spec_ass
from app.models.schemas import ClipPlanItem, HighlightConfig
from app.utils.atomic import atomic_write_json, part_path, replace_file
from app.utils.security import safe_path_argument
from app.video.audio_mix import narration_mix_command, subtitle_filter
from app.video.ffmpeg_runner import CancelledError, CommandError, FFmpegError, report_progress
from app.video.quality import audio_encode_args, video_encode_args
from app.voice import kokoro

_log = logging.getLogger(__name__)
SPEC_FILE = "render_spec.json"


def wants_composition(config: HighlightConfig) -> bool:
    return bool(config.video_brief)


def prepare_narration(
    job_dir: Path,
    config: HighlightConfig,
    settings,
    *,
    log: Callable[[str, str], None],
    check_cancel: Callable[[], None] | None = None,
    on_line: Callable[[int, int], None] | None = None,
) -> tuple[VideoBrief | None, dict[str, dict]]:
    """The job's brief and its spoken lines ({beat id: line}); no lines when narration is off,
    the voice model is missing, or the text is not something the English voices can read."""
    brief = load_brief(job_dir)
    if brief is None:
        log("This job has no brief; the video is made without narration or titles.", "WARNING")
        return None, {}
    if not config.narration:
        return brief, {}
    lines = []
    for beat_id, text in brief.narration_lines():
        if kokoro.speakable(text):
            lines.append((beat_id, text))
        else:
            log(f"Beat {beat_id[1:]}: the narration is not in English, which the voices speak; it is left silent.", "WARNING")
    if not lines:
        return brief, {}
    if not kokoro.runtime_available():
        log("Narration needs kokoro-onnx. Install the app requirements; the video is made without a voice.", "WARNING")
        return brief, {}
    if not kokoro.installed():
        log("The narration voice is not downloaded (Settings → Video tools); the video is made without a voice.", "WARNING")
        return brief, {}
    voice = kokoro.valid_voice(str(getattr(settings, "tts_voice", kokoro.DEFAULT_VOICE)))
    speed = float(getattr(settings, "tts_speed", 1.0) or 1.0)
    try:
        narration = kokoro.narrate(
            job_dir, lines, voice=voice, speed=speed, settings=settings, check_cancel=check_cancel, on_line=on_line
        )
    except CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - a video without a voice beats no video
        log(f"The narration could not be spoken ({exc}); the video is made without a voice.", "WARNING")
        return brief, {}
    seconds = sum(float(item["duration"]) for item in narration.values())
    log(f"Narration: {len(narration)} line(s), {seconds:.1f}s, voice {voice}", "INFO")
    return brief, narration


def fit_clips(
    clips: list[ClipPlanItem], brief: VideoBrief | None, narration: dict[str, dict], sources: list[dict]
) -> tuple[list[ClipPlanItem], list[str]]:
    durations = {str(s.get("id") or "v0"): float(s.get("duration") or 0) for s in sources or []}
    if len(durations) == 1:
        durations[""] = next(iter(durations.values()))
    return fit_narration(clips, brief, narration, durations)


def compose_video(
    runner,
    output: Path,
    clips: list[ClipPlanItem],
    brief: VideoBrief | None,
    narration: dict[str, dict],
    config: HighlightConfig,
    settings,
    work_dir: Path,
    job_dir: Path,
    *,
    log: Callable[[str, str], None],
    on_progress: Callable[[float], None] | None = None,
) -> RenderSpec | None:
    """Draw the brief's texts on ``output`` and mix its narration in (in place). Returns the
    timeline used, or None when there was nothing to add."""
    progress = on_progress or (lambda share: None)
    meta = runner.probe(output)
    duration = float(meta.duration or sum(c.duration for c in clips))
    fps = float(meta.fps or 30.0)
    folder = work_dir / "compose"
    folder.mkdir(parents=True, exist_ok=True)
    drawn = config.caption_style if config.subtitles == "burned" else "none"
    spec = build_spec(
        clips,
        brief,
        narration,
        width=int(meta.width or 1920),
        height=int(meta.height or 1080),
        fps=fps,
        caption_style=drawn,
        video=SpecVideo(kind="base", src="base.mp4"),
        narration_src=lambda line: str(line.get("file") or ""),
        duration=duration,
    )
    atomic_write_json(folder / SPEC_FILE, spec.model_dump())
    atomic_write_json(job_dir / SPEC_FILE, spec.model_dump())
    voices = [(Path(job_dir) / kokoro.NARRATION_DIR / item.src, item.start) for item in spec.narration]
    voices = [(path, start) for path, start in voices if path.is_file()]
    needs_drawing = bool(spec.texts) or (spec.caption_style != "none" and bool(spec.captions))
    if not needs_drawing and not voices:
        return None
    webm = output.suffix.lower() == ".webm"
    audio_args = ["-c:a", "libopus", "-b:a", "160k"] if webm else audio_encode_args(config.quality, settings.output_audio_bitrate)
    encode_args = (
        ["-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0", "-row-mt", "1", "-pix_fmt", "yuv420p"]
        if webm
        else video_encode_args("libx264", config.quality, settings.output_crf)
    )
    picture: Path | None = None
    if needs_drawing and str(getattr(settings, "render_engine", "remotion")) == "remotion":
        if remotion.ready(settings):
            picture = _remotion_picture(runner, output, spec, folder, fps, settings, log, progress)
        else:
            log("The Remotion video tools are not prepared (Settings → Video tools); texts are drawn with FFmpeg.", "WARNING")
    subtitles = None
    if needs_drawing and picture is None:
        subtitles = folder / "overlay.ass"
        subtitles.write_text(spec_ass(spec), encoding="utf-8")
    temp = part_path(output)
    if voices:
        command = narration_mix_command(
            output,
            voices,
            temp,
            duration,
            bool(meta.has_audio),
            audio_args=audio_args,
            picture=picture,
            subtitles=subtitles,
            video_args=encode_args if (subtitles is not None or (picture is not None and webm)) else None,
        )
    else:
        command = _overlay_command(output, temp, duration, picture, subtitles, encode_args, webm, bool(meta.has_audio))
    progress(0.9 if picture is not None else 0.0)
    with report_progress(runner, duration, lambda share: progress((0.9 + 0.1 * share) if picture is not None else share)):
        runner.run(command)
    replace_file(temp, output)
    progress(1.0)
    how = "Remotion" if picture is not None else ("FFmpeg" if subtitles is not None else "")
    parts = []
    if voices:
        parts.append(f"{len(voices)} narration line(s)")
    if spec.texts:
        parts.append(f"{len(spec.texts)} text(s)")
    if spec.caption_style != "none" and spec.captions:
        parts.append(f"{len(spec.captions)} caption line(s)")
    log(f"Narrated video: {', '.join(parts)}" + (f", drawn with {how}" if how else ""), "INFO")
    return spec


def write_srt(spec: RenderSpec, dest: Path) -> bool:
    text = narration_srt(spec)
    if not text.strip():
        return False
    dest.write_text(text, encoding="utf-8")
    return True


def _remotion_picture(runner, base: Path, spec: RenderSpec, folder: Path, fps: float, settings, log, progress) -> Path | None:
    """The drawn picture from Remotion, or None (with a warning) when it cannot draw it."""
    _free_memory(settings)
    prepared = folder / "base.mp4"
    try:
        # Frequent keyframes: the renderer jumps to any frame quickly and exactly.
        runner.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i",
                safe_path_argument(base),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "12",
                "-g",
                str(max(1, round(fps / 2))),
                "-pix_fmt",
                "yuv420p",
                safe_path_argument(prepared),
            ]
        )
        progress(0.05)
        return remotion.render_picture(
            json.loads(spec.model_dump_json()),
            folder,
            settings,
            runner,
            on_progress=lambda share: progress(0.05 + 0.85 * share),
            log=log,
        )
    except CancelledError:
        raise
    except (CommandError, FFmpegError, OSError, ValueError) as exc:
        log(f"Remotion could not draw the texts ({str(exc)[:300]}); they are drawn with FFmpeg instead.", "WARNING")
        return None


def _overlay_command(
    base: Path, output: Path, duration: float, picture: Path | None, subtitles: Path | None, encode_args, webm: bool, has_audio: bool
) -> list[str]:
    """Texts without narration: the drawn picture with the video's own sound."""
    command = ["ffmpeg", "-hide_banner", "-y", "-i", safe_path_argument(base)]
    if picture is not None:
        command += ["-i", safe_path_argument(picture), "-map", "1:v:0"]
        command += encode_args if webm else ["-c:v", "copy"]
    else:
        command += ["-map", "0:v:0", "-vf", subtitle_filter(subtitles), *encode_args]
    if has_audio:
        command += ["-map", "0:a:0", "-c:a", "copy"]
    command += ["-t", f"{duration:.3f}"]
    if not webm:
        command += ["-movflags", "+faststart"]
    return [*command, safe_path_argument(output)]


def _free_memory(settings) -> None:
    """Rendering with a browser needs memory: free the voice, Whisper and Ollama models first."""
    kokoro.release_narrator()
    try:
        from app.analysis.audio import release_whisper_models

        release_whisper_models()
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.services.ollama_runtime import get_ollama_client

        get_ollama_client(settings).unload_others(None)
    except Exception:  # noqa: BLE001 - freeing memory early is an optimization
        _log.debug("Could not unload Ollama models", exc_info=True)
