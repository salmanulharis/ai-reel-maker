from __future__ import annotations

from pathlib import Path
from typing import Sequence

from app.utils.security import safe_path_argument
from app.video.quality import audio_encode_args, video_encode_args


class VideoProcessor:
    """Centralized FFmpeg/FFprobe command builders. Execute via subprocess argument arrays."""

    @staticmethod
    def probe_command(input_path: str | Path) -> list[str]:
        source = safe_path_argument(input_path)
        return [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-print_format",
            "json",
            "-i",
            source,
        ]

    @staticmethod
    def extract_audio_command(input_path: str | Path, output_path: str | Path) -> list[str]:
        return [
            "ffmpeg",
            "-y",
            "-i",
            safe_path_argument(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            safe_path_argument(output_path),
        ]

    @staticmethod
    def extract_frame_command(
        input_path: str | Path,
        output_path: str | Path,
        timestamp: float,
    ) -> list[str]:
        return [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-ss",
            f"{max(0.0, timestamp):.3f}",
            "-i",
            safe_path_argument(input_path),
            "-an",
            "-frames:v",
            "1",
            "-update",
            "1",
            "-strict",
            "unofficial",
            "-q:v",
            "2",
            "-f",
            "image2",
            safe_path_argument(output_path),
        ]

    @staticmethod
    def cut_clip_command(
        input_path: str | Path,
        output_path: str | Path,
        start: float,
        end: float,
        stream_copy: bool = False,
        video_codec: str = "libx264",
        crf: int = 18,
        quality: str = "high",
        audio_bitrate: str = "192k",
        extra: Sequence[str] | None = None,
        silent_audio: bool = False,
    ) -> list[str]:
        duration = max(0.01, float(end) - float(start))
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{max(0.0, float(start)):.3f}",
            "-i",
            safe_path_argument(input_path),
        ]
        if silent_audio:
            # A source without sound still gets a silent stereo track, so clips from several
            # videos join with continuous audio.
            cmd += [
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=48000:cl=stereo",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
            ]
        cmd += ["-t", f"{duration:.3f}"]
        if stream_copy:
            cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        else:
            cmd += video_encode_args(video_codec, quality, crf)
            cmd += audio_encode_args(quality, audio_bitrate)
            cmd += ["-avoid_negative_ts", "make_zero", "-movflags", "+faststart"]
        if extra:
            cmd += [str(part) for part in extra]
        cmd.append(safe_path_argument(output_path))
        return cmd

    @staticmethod
    def concat_command(
        list_path: str | Path,
        output_path: str | Path,
        stream_copy: bool = True,
        video_codec: str = "libx264",
        crf: int = 18,
        quality: str = "high",
        audio_bitrate: str = "192k",
    ) -> list[str]:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            safe_path_argument(list_path),
        ]
        if stream_copy:
            cmd += ["-c", "copy", "-movflags", "+faststart"]
        else:
            cmd += video_encode_args(video_codec, quality, crf)
            cmd += audio_encode_args(quality, audio_bitrate)
            cmd += ["-movflags", "+faststart"]
        cmd.append(safe_path_argument(output_path))
        return cmd

    @staticmethod
    def silencedetect_command(input_path: str | Path, noise_db: float, min_duration: float) -> list[str]:
        return [
            "ffmpeg",
            "-i",
            safe_path_argument(input_path),
            "-af",
            f"silencedetect=noise={float(noise_db)}dB:d={float(min_duration)}",
            "-f",
            "null",
            "-",
        ]

    @staticmethod
    def write_concat_list(list_path: str | Path, files: Sequence[str | Path]) -> None:
        lines = []
        for item in files:
            path = Path(item).resolve().as_posix().replace("'", r"'\''")
            lines.append(f"file '{path}'")
        Path(list_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def crop_command(
        input_path: str | Path,
        output_path: str | Path,
        width: int,
        height: int,
        x: int,
        y: int,
        video_codec: str = "libx264",
        crf: int = 18,
    ) -> list[str]:
        return [
            "ffmpeg",
            "-y",
            "-i",
            safe_path_argument(input_path),
            "-vf",
            f"crop={int(width)}:{int(height)}:{int(x)}:{int(y)}",
            "-c:v",
            video_codec,
            "-preset",
            "veryfast",
            "-crf",
            str(int(crf)),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            safe_path_argument(output_path),
        ]
