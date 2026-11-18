from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.models.schemas import ClipPlan, ClipPlanItem, RenderSettings, VideoMetadata
from app.video.ffmpeg_runner import FFmpegError, FFmpegRunner
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
    ) -> dict:
        """
        Cut, normalize and join the clips. `clip_subtitles(index, clip, canvas)` returns a
        subtitle file to burn into one clip, timed from that clip's start (each clip is cut on
        its own and starts at 0), or None for no subtitles on it.
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

        clip_files = []
        for i, clip in enumerate(clips):
            clip_path = work_dir / f"clip_{i:03d}.mp4"
            clip_source = clip_sources[i]
            src_meta = self._cached_probe(probes, clip_source, metadata)
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
            if settings.transition == "fade":
                dur = max(0.05, clip.end - clip.start)
                fade = min(0.12, dur / 6)
                # Pieces of one moment (pauses cut out) join without a fade between them.
                fade_in = not getattr(clip, "join_previous", False)
                fade_out = not (i + 1 < len(clips) and getattr(clips[i + 1], "join_previous", False))
                video_fades = []
                if fade_in:
                    video_fades.append(f"fade=t=in:st=0:d={fade}")
                    audio_filters.append(f"afade=t=in:st=0:d={fade}")
                if fade_out:
                    video_fades.append(f"fade=t=out:st={dur - fade}:d={fade}")
                    audio_filters.append(f"afade=t=out:st={dur - fade}:d={fade}")
                if video_fades:
                    filters.append(",".join(video_fades))
            if extras and extras.punch_in and moment % 2 == 1:
                filters.extend(punch_in_filters(out_w, out_h))
            video_graph = ",".join(filters)
            if extras and extras.active:
                video_graph = build_graph(video_graph, extras, out_w, out_h, canvas.fps, offset, total)
            offset += max(0.0, clip.end - clip.start)
            extra = ["-vf", video_graph, "-af", ",".join(audio_filters)]
            subtitle_path = clip_subtitles(i, clip, canvas) if clip_subtitles else None
            cmd = VideoProcessor.cut_clip_command(
                clip_source,
                clip_path,
                clip.start,
                clip.end,
                stream_copy=False,
                video_codec=video_codec,
                crf=crf,
                quality=settings.quality,
                audio_bitrate=settings.audio_bitrate,
                extra=extra,
            )
            if subtitle_path:
                cmd = self._inject_subtitles(cmd, subtitle_path)
            try:
                self.runner.run(cmd)
            except FFmpegError:
                if video_codec not in HW_ENCODERS:
                    raise
                video_codec = "libx264"
                cmd = VideoProcessor.cut_clip_command(
                    clip_source,
                    clip_path,
                    clip.start,
                    clip.end,
                    stream_copy=False,
                    video_codec=video_codec,
                    crf=crf,
                    quality=settings.quality,
                    audio_bitrate=settings.audio_bitrate,
                    extra=extra,
                )
                if subtitle_path:
                    cmd = self._inject_subtitles(cmd, subtitle_path)
                self.runner.run(cmd)
            clip_files.append(clip_path)

        list_path = work_dir / "concat.txt"
        VideoProcessor.write_concat_list(list_path, clip_files)
        concat = VideoProcessor.concat_command(list_path, output, stream_copy=True)
        try:
            self.runner.run(concat)
        except FFmpegError:
            self.runner.run(
                VideoProcessor.concat_command(
                    list_path,
                    output,
                    stream_copy=False,
                    video_codec=video_codec,
                    crf=crf,
                    quality=settings.quality,
                    audio_bitrate=settings.audio_bitrate,
                )
            )
        probe = self.runner.probe(output)
        validation = validate_output(probe, metadata, output, canvas=canvas, settings=settings)
        if not validation["ok"]:
            raise FFmpegError(validation["error"])
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
            except Exception:
                if fallback:
                    metas.append(fallback)
        if not metas and fallback:
            metas.append(fallback)
        return metas

    def _cached_probe(
        self, cache: dict[str, VideoMetadata], path: Path, fallback: VideoMetadata | None
    ) -> VideoMetadata:
        key = str(path)
        if key not in cache:
            try:
                cache[key] = self.runner.probe(path)
            except Exception:
                cache[key] = fallback or VideoMetadata()
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
