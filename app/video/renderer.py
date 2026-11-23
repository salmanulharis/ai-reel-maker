from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from app.utils.atomic import atomic_write_json, read_json, replace_file
from app.models.schemas import ClipPlan, ClipPlanItem, RenderSettings, VideoMetadata
from app.video.ffmpeg_runner import CancelledError, FFmpegError, FFmpegRunner, report_progress
from app.video.processor import VideoProcessor
from app.video.reframe import aspect_crop_filter, clip_frames, subject_center, to_canvas
from app.video.audio_mix import leveling_filters, leveling_gain, measure_command, parse_loudness
from app.video.overlays import Extras, build_graph, output_size, punch_in_filters
from app.video.quality import (
    OutputCanvas,
    choose_canvas,
    clip_normalize_filters,
    describe_frame_rate,
    encode_crf,
    encode_preset,
    quality_software_first,
)


CODEC_MAP = {
    "auto": "libx264",
    "h264": "libx264",
    "h.264": "libx264",
    "h265": "libx265",
    "h.265": "libx265",
    "hevc": "libx265",
    "vp9": "libvpx-vp9",
    "av1": "libsvtav1",
}

HW_ENCODERS = {
    "h264_nvenc": "NVIDIA NVENC",
    "hevc_nvenc": "NVIDIA NVENC (HEVC)",
    "h264_qsv": "Intel Quick Sync",
    "h264_amf": "AMD hardware encoding",
    "h264_videotoolbox": "Apple VideoToolbox",
}


# Length of the sound fade at a plain cut, in seconds: too short to hear, long enough to avoid a
# click where the waveform is cut mid-swing.
_CLICK_FADE = 0.02
# Clips this close together in the same video play on as one: no fade between them ("auto").
_FLOW_GAP = 0.5


def _flows(previous, clip) -> bool:
    """Whether ``clip`` continues right where ``previous`` ended in the same video."""
    if previous is None or str(previous.source_id or "") != str(clip.source_id or ""):
        return False
    return 0 <= float(clip.start) - float(previous.end) < _FLOW_GAP


def _edge_fades(transition: str, clips: list, index: int, edge_in: bool, edge_out: bool) -> tuple[bool, bool]:
    """Whether a clip fades in and out. "fade": at every clip edge. "auto": where the video
    jumps to another moment (and at the very start and end), not where it flows on."""
    if transition == "fade":
        return edge_in, edge_out
    if transition == "auto":
        previous = clips[index - 1] if index > 0 else None
        following = clips[index + 1] if index + 1 < len(clips) else None
        return (
            edge_in and not _flows(previous, clips[index]),
            edge_out and (following is None or not _flows(clips[index], following)),
        )
    return False, False


# Fingerprints of the clips a render has finished, kept in its work folder.
CLIP_MANIFEST = "clips_done.json"


def _command_fingerprint(cmd: list[str], subtitle_path: Path | None) -> str:
    """Identifies the exact clip a command makes, including the subtitle text it burns in."""
    digest = hashlib.sha256("\0".join(str(part) for part in cmd).encode("utf-8"))
    if subtitle_path is not None:
        try:
            digest.update(Path(subtitle_path).read_bytes())
        except OSError:
            digest.update(b"<no subtitles>")
    return digest.hexdigest()


def _usable_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


class Renderer:
    def __init__(self, runner: FFmpegRunner):
        self.runner = runner

    def render(
        self,
        source: Path,
        plan: ClipPlan,
        output: Path,
        settings: RenderSettings,
        work_dir: Path,
        metadata: VideoMetadata | None = None,
        clip_subtitles: Callable[[int, ClipPlanItem, OutputCanvas], Path | None] | None = None,
        hardware_encoder: str | None = None,
        source_lookup: dict[str, Path] | None = None,
        extras: Extras | None = None,
        subtitle_segments: dict[str, list[dict]] | list[dict] | None = None,
        on_clip: Callable[[int, float], None] | None = None,
    ) -> dict:
        """
        Cut, normalize and join the clips into ``output``. Each clip is cut on its own and
        starts at 0, so burned subtitles are written per clip, timed from that clip's start.

        `clip_subtitles(index, clip, canvas)` returns the subtitle file to burn into one clip, or
        None for no subtitles on it. Without that callback, ``subtitle_segments`` (transcript
        segments, per source id or for every clip) are burned in as plain subtitles when
        ``settings.subtitles == "burned"``.

        ``on_clip(index, share)`` receives how far each clip is encoded (1.0 when it is finished
        or reused from an earlier run).
        """
        work_dir.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        clips = [c for c in plan.clips if c.end > c.start]
        if not clips:
            raise FFmpegError("No clips to render")

        source_lookup = {str(k): Path(v) for k, v in (source_lookup or {}).items()}
        clip_sources = [self._clip_source(source, clip, source_lookup) for clip in clips]
        source_metas = self._probe_sources(clip_sources, metadata)
        canvas = choose_canvas(source_metas, settings)
        video_codec = self._codec(settings, hardware_encoder)
        crf = encode_crf(settings.quality, settings.crf)
        aspect = self._aspect_filter(settings)
        reframed = 0
        leveled = 0
        out_w, out_h = output_size(canvas.width, canvas.height, settings.aspect_ratio)
        total = sum(max(0.0, c.end - c.start) for c in clips)
        offset = 0.0
        moment = -1
        probes: dict[str, VideoMetadata] = {}
        failed_probes: set[str] = set()
        any_audio = any(meta.has_audio for meta in source_metas)
        burn = settings.subtitles == "burned" and subtitle_segments is not None
        # Filters, subtitle file and silent-track flag per clip. Worked out once, so a second
        # pass in software does not measure, reframe or count the clips again.
        prepared: dict[int, tuple[list[str], Path | None, bool]] = {}

        def prepare(i: int, clip) -> tuple[list[str], Path | None, bool]:
            nonlocal reframed, leveled, offset, moment
            clip_source = clip_sources[i]
            src_meta = self._cached_probe(probes, clip_source, metadata, failed_probes)
            clip_aspect = aspect
            if aspect and settings.reframe == "subject":
                center = self._subject_center(clip_source, clip, src_meta, canvas, work_dir, i)
                if center is not None:
                    clip_aspect = aspect_crop_filter(settings.aspect_ratio, center)
                    reframed += 1
            filters = clip_normalize_filters(
                src_meta, canvas, extra=[clip_aspect] if clip_aspect else None, smooth_motion=settings.smooth_motion
            )
            audio_filters = ["asetpts=PTS-STARTPTS", "aresample=48000:async=1"]
            if not getattr(clip, "join_previous", False):
                moment += 1
            if settings.level_audio:
                gain = leveling_gain(self._measure(clip_source, clip))
                if gain is not None:
                    audio_filters.extend(leveling_filters(gain))
                    leveled += 1
            dur = max(0.05, clip.end - clip.start)
            # Pieces of one moment (pauses cut out) join without any fade between them.
            edge_in = not getattr(clip, "join_previous", False)
            edge_out = not (i + 1 < len(clips) and getattr(clips[i + 1], "join_previous", False))
            fade_in, fade_out = _edge_fades(settings.transition, clips, i, edge_in, edge_out)
            fade = min(0.12, dur / 6)
            video_fades = []
            if fade_in:
                video_fades.append(f"fade=t=in:st=0:d={fade}")
                audio_filters.append(f"afade=t=in:st=0:d={fade}")
            elif edge_in:
                # A very short sound fade at a plain cut avoids a click; the picture is untouched.
                audio_filters.append(f"afade=t=in:st=0:d={min(_CLICK_FADE, dur / 10):.3f}")
            if fade_out:
                video_fades.append(f"fade=t=out:st={dur - fade}:d={fade}")
                audio_filters.append(f"afade=t=out:st={dur - fade}:d={fade}")
            elif edge_out:
                click = min(_CLICK_FADE, dur / 10)
                audio_filters.append(f"afade=t=out:st={dur - click:.3f}:d={click:.3f}")
            if video_fades:
                filters.append(",".join(video_fades))
            if extras and extras.punch_in and moment % 2 == 1:
                filters.extend(punch_in_filters(out_w, out_h))
            video_graph = ",".join(filters)
            if extras and extras.active:
                video_graph = build_graph(video_graph, extras, out_w, out_h, canvas.fps, offset, total)
            offset += max(0.0, clip.end - clip.start)
            extra = ["-vf", video_graph, "-af", ",".join(audio_filters)]
            subtitle_path = None
            if clip_subtitles:
                subtitle_path = clip_subtitles(i, clip, canvas)
            elif burn:
                clip_srt = work_dir / f"clip_{i:03d}.srt"
                if build_clip_srt(clip, _segments_for(clip, subtitle_segments), clip_srt):
                    subtitle_path = clip_srt
            silent = any_audio and not src_meta.has_audio and str(clip_source) not in failed_probes
            return extra, subtitle_path, silent

        def clip_command(i: int, clip, codec: str, output_path: Path) -> list[str]:
            if i not in prepared:
                prepared[i] = prepare(i, clip)
            extra, subtitle_path, silent = prepared[i]
            cmd = VideoProcessor.cut_clip_command(
                clip_sources[i],
                output_path,
                clip.start,
                clip.end,
                stream_copy=False,
                video_codec=codec,
                crf=crf,
                quality=settings.quality,
                audio_bitrate=settings.audio_bitrate,
                extra=list(extra),
                silent_audio=silent,
            )
            if subtitle_path:
                cmd = self._inject_subtitles(cmd, subtitle_path)
            return cmd

        # Clips finished by an earlier run that was paused or cut off (a power cut, a crash),
        # by the fingerprint of the exact command that made them. A clip is reused only when
        # its command is unchanged, so edited clips and changed settings are encoded again.
        manifest_path = work_dir / CLIP_MANIFEST
        encoding_dir = work_dir / "encoding"
        encoding_dir.mkdir(exist_ok=True)
        finished = read_json(manifest_path, {})
        if not isinstance(finished, dict):
            finished = {}

        def encode_all(codec: str) -> list[Path]:
            files = []
            for i, clip in enumerate(clips):
                target = work_dir / f"clip_{i:03d}.mp4"
                # Encoded in a side folder and moved into place when complete, so a stopped
                # command never leaves a partial clip that looks finished.
                pending_clip = encoding_dir / target.name
                cmd = clip_command(i, clip, codec, pending_clip)
                fingerprint = _command_fingerprint(cmd, prepared[i][1])
                if finished.get(target.name) != fingerprint or not _usable_file(target):
                    report = None
                    if on_clip is not None:
                        # Below 1.0 until the clip is moved into place.
                        report = lambda share, index=i: on_clip(index, min(share, 0.99))  # noqa: E731
                    try:
                        with report_progress(self.runner, max(0.05, clip.end - clip.start), report):
                            self.runner.run(cmd)
                        if pending_clip.exists():
                            replace_file(pending_clip, target)
                    except BaseException:
                        pending_clip.unlink(missing_ok=True)
                        raise
                    finished[target.name] = fingerprint
                    atomic_write_json(manifest_path, finished)
                if on_clip is not None:
                    on_clip(i, 1.0)
                files.append(target)
            return files

        try:
            clip_files = encode_all(video_codec)
        except FFmpegError:
            if video_codec not in HW_ENCODERS:
                raise
            # Start over in software, so every clip shares one encoder and the stream-copied
            # join does not mix differently encoded pieces.
            video_codec = "libx264"
            clip_files = encode_all(video_codec)

        # Write next to the final file and move it into place only once it is verified, so a
        # failed re-render never replaces a good earlier video. The leading dot keeps it out of
        # the "highlight.*" lookups that serve the finished video.
        pending = output.with_name(f".{output.stem}.rendering{output.suffix}")
        list_path = work_dir / "concat.txt"
        VideoProcessor.write_concat_list(list_path, clip_files)
        concat = VideoProcessor.concat_command(list_path, pending, stream_copy=True)
        try:
            try:
                self.runner.run(concat)
            except FFmpegError:
                self.runner.run(
                    VideoProcessor.concat_command(
                        list_path,
                        pending,
                        stream_copy=False,
                        video_codec=video_codec,
                        crf=crf,
                        quality=settings.quality,
                        audio_bitrate=settings.audio_bitrate,
                    )
                )
            probe = self.runner.probe(pending)
            validation = validate_output(probe, metadata, pending, canvas=canvas, settings=settings)
            if not validation["ok"]:
                raise FFmpegError(validation["error"])
            replace_file(pending, output)
        except BaseException:
            # Failed, cancelled, or could not be moved into place: never leave a partial file.
            pending.unlink(missing_ok=True)
            raise
        probe = probe.model_copy(update={"filename": output.name})
        return {
            "metadata": probe.model_dump(),
            "validation": validation,
            "canvas": {"width": canvas.width, "height": canvas.height, "fps": canvas.fps},
            "frame_rate": describe_frame_rate(source_metas, canvas.fps, settings.smooth_motion),
            "reframe": {"followed": reframed, "clips": len(clips)} if aspect and settings.reframe == "subject" else None,
            "leveled": {"clips": leveled, "of": len(clips)} if settings.level_audio else None,
            "encoder": {
                "codec": video_codec,
                "preset": encode_preset(video_codec, settings.quality),
                "crf": crf,
            },
        }

    def _codec(self, settings: RenderSettings, hardware_encoder: str | None) -> str:
        requested = (settings.codec or "auto").lower()
        if requested in {"auto"}:
            if quality_software_first(settings.quality):
                return "libx264"
            if hardware_encoder:
                return hardware_encoder
        return CODEC_MAP.get(requested, "libx264")

    def _measure(self, source: Path, clip) -> float | None:
        """A clip's loudness in LUFS, or None when it cannot be measured."""
        try:
            result = self.runner.run(measure_command(source, clip.start, clip.end))
        except Exception:  # noqa: BLE001 - leveling is a nicety; the clip keeps its sound
            return None
        return parse_loudness(getattr(result, "stderr", "") or "")

    def _aspect_filter(self, settings: RenderSettings) -> str | None:
        return aspect_crop_filter(settings.aspect_ratio)

    def _subject_center(self, source: Path, clip, meta: VideoMetadata, canvas: OutputCanvas, work_dir: Path, index: int):
        """Where the people in a clip are, in the output canvas, or None (keep the middle)."""
        try:
            center = subject_center(clip_frames(self.runner, source, clip.start, clip.end, work_dir, index))
        except Exception:  # noqa: BLE001 - framing must never fail a render
            return None
        if center is None:
            return None
        width, height = int(meta.width or 0), int(meta.height or 0)
        if abs(int(meta.rotation or 0)) % 180 == 90:
            # Frames come out turned upright; the stored size is before turning.
            width, height = height, width
        return to_canvas(center, width, height, canvas.width, canvas.height)

    def _clip_source(self, source: Path, clip, source_lookup: dict[str, Path]) -> Path:
        if source_lookup and clip.source_id and clip.source_id in source_lookup:
            return source_lookup[clip.source_id]
        if clip.source_path:
            return Path(clip.source_path)
        return source

    def _probe_sources(self, paths: list[Path], fallback: VideoMetadata | None) -> list[VideoMetadata]:
        unique: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        metas = []
        for path in unique:
            try:
                metas.append(self.runner.probe(path))
            except CancelledError:
                raise
            except Exception:
                if fallback:
                    metas.append(fallback)
        if not metas and fallback:
            metas.append(fallback)
        return metas

    def _cached_probe(
        self,
        cache: dict[str, VideoMetadata],
        path: Path,
        fallback: VideoMetadata | None,
        failed: set[str] | None = None,
    ) -> VideoMetadata:
        key = str(path)
        if key not in cache:
            try:
                cache[key] = self.runner.probe(path)
            except CancelledError:
                raise
            except Exception:
                cache[key] = fallback or VideoMetadata()
                if failed is not None:
                    failed.add(key)
        return cache[key]

    def _inject_subtitles(self, cmd: list[str], srt_path: Path) -> list[str]:
        srt = srt_path.as_posix().replace("\\", "/").replace(":", r"\:")
        if "-vf" in cmd:
            idx = cmd.index("-vf")
            cmd[idx + 1] = cmd[idx + 1] + f",subtitles='{srt}'"
            return cmd
        cmd = list(cmd)
        cmd[-1:-1] = ["-vf", f"subtitles='{srt}'"]
        return cmd


def validate_output(
    probe: VideoMetadata,
    source: VideoMetadata | None,
    path: Path,
    canvas: OutputCanvas | None = None,
    settings: RenderSettings | None = None,
) -> dict:
    if not path.exists() or path.stat().st_size < 1000:
        return {"ok": False, "error": "Output file missing or too small"}
    if probe.duration <= 0.05:
        return {"ok": False, "error": "Output duration is invalid"}
    if probe.width <= 0 or probe.height <= 0:
        return {"ok": False, "error": "Output has no video stream"}
    if source and source.has_audio and not probe.has_audio:
        return {"ok": False, "error": "Output is missing audio"}
    if canvas:
        if abs(int(probe.width) - canvas.width) > 2 or abs(int(probe.height) - canvas.height) > 2:
            return {
                "ok": False,
                "error": (
                    f"Output resolution {probe.width}×{probe.height} does not match "
                    f"target {canvas.width}×{canvas.height}"
                ),
            }
        # The average rate reflects what was encoded; the declared rate of a stream-copied
        # concat can read double (e.g. 47.952 for 23.976 clips) because of gaps at the joins.
        measured = float(probe.avg_fps or probe.fps or 0)
        if canvas.fps and measured and abs(measured - canvas.fps) > 1.5:
            return {
                "ok": False,
                "error": f"Output frame rate {measured:.3f} fps does not match target {canvas.fps:.3f} fps",
            }
    return {"ok": True, "error": None}


def _segments_for(clip, transcripts: dict[str, list[dict]] | list[dict] | None) -> list[dict]:
    """The transcript segments of the video a clip comes from. A plain list applies to every
    clip; a mapping is keyed by source id, and a clip without one uses the only transcript."""
    if not transcripts:
        return []
    if isinstance(transcripts, list):
        return transcripts
    sid = str(getattr(clip, "source_id", "") or "")
    if sid in transcripts:
        return transcripts[sid]
    if not sid and len(transcripts) == 1:
        return next(iter(transcripts.values()))
    return []


def _clip_entries(clip, segments: list[dict], offset: float) -> list[tuple[float, float, str]]:
    entries = []
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if not text or seg["end"] <= clip.start or seg["start"] >= clip.end:
            continue
        start = max(clip.start, seg["start"]) + offset
        end = min(clip.end, seg["end"]) + offset
        if end > start:
            entries.append((start, end, text))
    return entries


def _write_srt(entries: list[tuple[float, float, str]], dest: Path) -> None:
    lines = []
    for idx, (start, end, text) in enumerate(entries, start=1):
        lines += [str(idx), f"{_srt_ts(start)} --> {_srt_ts(end)}", text, ""]
    dest.write_text("\n".join(lines), encoding="utf-8")


def build_srt(plan: ClipPlan, transcript_segments: dict[str, list[dict]] | list[dict], dest: Path) -> Path:
    """Subtitles for the joined highlight, timed on its own timeline."""
    entries = []
    cursor = 0.0
    for clip in plan.clips:
        if clip.end <= clip.start:
            continue
        entries += _clip_entries(clip, _segments_for(clip, transcript_segments), cursor - clip.start)
        cursor += clip.end - clip.start
    _write_srt(entries, dest)
    return dest


def build_clip_srt(clip, segments: list[dict], dest: Path) -> bool:
    """Subtitles for one clip, timed from the clip's own start. Returns False (and writes
    nothing) when no speech falls inside the clip, because FFmpeg cannot open an empty file."""
    entries = _clip_entries(clip, segments, -clip.start)
    if not entries:
        dest.unlink(missing_ok=True)
        return False
    _write_srt(entries, dest)
    return True


def _srt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
