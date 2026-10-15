from __future__ import annotations

import os
import time
from pathlib import Path

from app.models.schemas import ClipPlan, RenderSettings, VideoMetadata
from app.video.ffmpeg_runner import CancelledError, FFmpegError, FFmpegRunner
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
        hardware_encoder: str | None = None,
        source_lookup: dict[str, Path] | None = None,
        subtitle_segments: dict[str, list[dict]] | list[dict] | None = None,
    ) -> dict:
        """Cut, normalize and join the clips into ``output``.

        ``subtitle_segments`` (transcript segments, per source id or for every clip) are burned
        into the picture when ``settings.subtitles == "burned"``. Each clip gets its own subtitle
        file timed from its own start, because clips are encoded separately and joined after."""
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
        probes: dict[str, VideoMetadata] = {}
        failed_probes: set[str] = set()
        any_audio = any(meta.has_audio for meta in source_metas)
        burn = settings.subtitles == "burned" and subtitle_segments is not None

        def clip_command(i: int, clip, codec: str) -> list[str]:
            clip_path = work_dir / f"clip_{i:03d}.mp4"
            clip_source = clip_sources[i]
            src_meta = self._cached_probe(probes, clip_source, metadata, failed_probes)
            filters = clip_normalize_filters(src_meta, canvas, smooth_motion=settings.smooth_motion)
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
            silent = any_audio and not src_meta.has_audio and str(clip_source) not in failed_probes
            cmd = VideoProcessor.cut_clip_command(
                clip_source,
                clip_path,
                clip.start,
                clip.end,
                stream_copy=False,
                video_codec=codec,
                crf=crf,
                quality=settings.quality,
                audio_bitrate=settings.audio_bitrate,
                extra=extra,
                silent_audio=silent,
            )
            if burn:
                clip_srt = work_dir / f"clip_{i:03d}.srt"
                if build_clip_srt(clip, _segments_for(clip, subtitle_segments), clip_srt):
                    cmd = self._inject_subtitles(cmd, clip_srt)
            return cmd

        def encode_all(codec: str) -> list[Path]:
            files = []
            for i, clip in enumerate(clips):
                self.runner.run(clip_command(i, clip, codec))
                files.append(work_dir / f"clip_{i:03d}.mp4")
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
            _replace(pending, output)
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


def _replace(pending: Path, output: Path) -> None:
    """Move the verified file into place. On Windows the old file can be briefly held open
    (for example while it is being streamed), so a few retries are allowed."""
    for attempt in range(10):
        try:
            os.replace(pending, output)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5)


def _srt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
