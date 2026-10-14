from __future__ import annotations

from pathlib import Path

from app.models.schemas import ClipPlan, RenderSettings, VideoMetadata
from app.video.ffmpeg_runner import FFmpegError, FFmpegRunner
from app.video.processor import VideoProcessor
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
        srt_path: Path | None = None,
        hardware_encoder: str | None = None,
        source_lookup: dict[str, Path] | None = None,
    ) -> dict:
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
        probes: dict[str, VideoMetadata] = {}

        clip_files = []
        for i, clip in enumerate(clips):
            clip_path = work_dir / f"clip_{i:03d}.mp4"
            clip_source = clip_sources[i]
            src_meta = self._cached_probe(probes, clip_source, metadata)
            filters = clip_normalize_filters(
                src_meta, canvas, extra=[aspect] if aspect else None, smooth_motion=settings.smooth_motion
            )
            if settings.transition == "fade":
                dur = max(0.05, clip.end - clip.start)
                fade = min(0.12, dur / 6)
                filters.append(f"fade=t=in:st=0:d={fade},fade=t=out:st={dur - fade}:d={fade}")
            audio_filters = ["asetpts=PTS-STARTPTS", "aresample=48000:async=1"]
            if settings.transition == "fade":
                audio_filters.extend(
                    [f"afade=t=in:st=0:d={fade}", f"afade=t=out:st={dur - fade}:d={fade}"]
                )
            extra = ["-vf", ",".join(filters), "-af", ",".join(audio_filters)]
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
            if srt_path and settings.subtitles == "burned":
                cmd = self._inject_subtitles(cmd, srt_path)
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
                if srt_path and settings.subtitles == "burned":
                    cmd = self._inject_subtitles(cmd, srt_path)
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

    def _aspect_filter(self, settings: RenderSettings) -> str | None:
        if settings.aspect_ratio in {"", "original"}:
            return None
        ratio = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1, "4:5": 4 / 5}[settings.aspect_ratio]
        return (
            f"crop='if(gte(iw/ih,{ratio}),ih*{ratio},iw)':"
            f"'if(gte(iw/ih,{ratio}),ih,iw/{ratio})'"
        )

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


def build_srt(plan: ClipPlan, transcript_segments: list[dict], dest: Path) -> Path:
    lines = []
    cursor = 0.0
    idx = 1
    for clip in plan.clips:
        offset = cursor - clip.start
        for seg in transcript_segments:
            if seg["end"] < clip.start or seg["start"] > clip.end:
                continue
            start = max(clip.start, seg["start"]) + offset
            end = min(clip.end, seg["end"]) + offset
            lines.append(str(idx))
            lines.append(f"{_srt_ts(start)} --> {_srt_ts(end)}")
            lines.append(seg["text"].strip())
            lines.append("")
            idx += 1
        cursor += clip.end - clip.start
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def _srt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
