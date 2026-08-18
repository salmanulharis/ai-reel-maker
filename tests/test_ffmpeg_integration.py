import shutil
import subprocess
from pathlib import Path

import pytest

from app.video.ffmpeg_runner import FFmpegRunner
from app.video.processor import VideoProcessor

ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")


@pytest.mark.skipif(not ffmpeg or not ffprobe, reason="ffmpeg/ffprobe not installed")
def test_probe_and_audio_extract_on_generated_clip(tmp_path: Path):
    source = tmp_path / "sample.mp4"
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.skip(result.stderr[-500:])
    runner = FFmpegRunner()
    meta = runner.probe(source)
    assert meta.duration >= 1.5
    assert meta.width == 320
    assert meta.has_audio is True
    wav = tmp_path / "audio.wav"
    runner.run(VideoProcessor.extract_audio_command(source, wav))
    assert wav.exists() and wav.stat().st_size > 1000
