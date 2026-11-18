"""
Sound for the finished video: even loudness across clips, and optional background music that
drops (ducks) while someone speaks. Commands are FFmpeg argument lists, never shell strings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.utils.security import safe_path_argument

# Social platforms play back at about -14 LUFS; quieter clips are raised, louder ones lowered.
TARGET_LUFS = -14.0
MAX_BOOST_DB = 12.0
MAX_CUT_DB = 20.0
# Below this a clip is (nearly) silent: raising it would only raise noise.
SILENCE_LUFS = -60.0
LIMIT = 0.891  # about -1 dBTP, so raising a clip never clips it
MUSIC_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus"}


def measure_command(source: Path, start: float, end: float) -> list[str]:
    """Measure one clip's loudness (EBU R128). The result is printed as JSON on stderr."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-ss",
        f"{max(0.0, float(start)):.3f}",
        "-i",
        safe_path_argument(source),
        "-t",
        f"{max(0.01, float(end) - float(start)):.3f}",
        "-vn",
        "-af",
        f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11:print_format=json",
        "-f",
        "null",
        "-",
    ]


def parse_loudness(stderr: str) -> float | None:
    """Integrated loudness (LUFS) from loudnorm's JSON report, or None."""
    blocks = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr or "")
    if not blocks:
        return None
    try:
        value = float(json.loads(blocks[-1])["input_i"])
    except (ValueError, KeyError, TypeError):
        return None
    return value if value == value and value not in (float("inf"), float("-inf")) else None


def leveling_gain(loudness: float | None) -> float | None:
    """dB to apply so a clip plays at the target, or None to leave it (unknown or silent)."""
    if loudness is None or loudness <= SILENCE_LUFS:
        return None
    return round(max(-MAX_CUT_DB, min(MAX_BOOST_DB, TARGET_LUFS - loudness)), 2)


def leveling_filters(gain_db: float | None) -> list[str]:
    if gain_db is None or abs(gain_db) < 0.1:
        return []
    return [f"volume={gain_db}dB", f"alimiter=limit={LIMIT}:level=disabled"]


def music_mix_command(
    video: Path,
    music: Path,
    output: Path,
    duration: float,
    volume: float,
    duck: bool,
    video_has_audio: bool,
    audio_bitrate: str = "192k",
) -> list[str]:
    """
    Mix background music under the finished video: looped to its length, faded in and out,
    at `volume` (0-1), and ducked under the video's own sound when `duck` is on. The picture is
    copied untouched.
    """
    duration = max(0.5, float(duration))
    fade = min(1.5, duration / 4)
    music_chain = (
        f"[1:a]aresample=48000,aformat=channel_layouts=stereo,volume={max(0.0, min(1.0, volume)):.3f},"
        f"atrim=0:{duration:.3f},afade=t=in:st=0:d={fade:.2f},afade=t=out:st={duration - fade:.3f}:d={fade:.2f}[m]"
    )
    if video_has_audio:
        voice = "[0:a]aresample=48000,aformat=channel_layouts=stereo,asplit=2[v][sc]"
        if duck:
            mix = "[m][sc]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[md];[v][md]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]"
        else:
            mix = "[sc]anullsink;[v][m]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]"
        graph = f"{voice};{music_chain};{mix}"
    else:
        graph = music_chain.replace("[m]", "[a]")
    return [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        safe_path_argument(video),
        "-stream_loop",
        "-1",
        "-i",
        safe_path_argument(music),
        "-filter_complex",
        graph,
        "-map",
        "0:v:0",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-t",
        f"{duration:.3f}",
        "-movflags",
        "+faststart",
        safe_path_argument(output),
    ]


def find_music(job_dir: Path) -> Path | None:
    folder = job_dir / "music"
    if not folder.is_dir():
        return None
    return next((p for p in sorted(folder.iterdir()) if p.suffix.lower() in MUSIC_SUFFIXES and p.is_file()), None)
