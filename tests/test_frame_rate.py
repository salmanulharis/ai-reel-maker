"""Output frame rate: keep the source rate or convert to a chosen one, without needless work."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models.schemas import ClipPlan, ClipPlanItem, HighlightConfig, RenderSettings, VideoMetadata
from app.video.ffmpeg_runner import FFmpegRunner
from app.video.quality import (
    choose_canvas,
    clip_normalize_filters,
    describe_frame_rate,
    frame_rate_filters,
    frame_rate_plan,
    is_variable_rate,
    source_frame_rate,
    target_frame_rate,
)
from app.video.renderer import Renderer


def meta(fps: float, avg: float | None = None, width: int = 1920, height: int = 1080) -> VideoMetadata:
    return VideoMetadata(width=width, height=height, fps=fps, avg_fps=fps if avg is None else avg, duration=10)


# Detection


def test_constant_rates_are_read_exactly():
    assert source_frame_rate(meta(24)) == 24.0
    assert source_frame_rate(meta(23.98)) == pytest.approx(24000 / 1001)
    assert source_frame_rate(meta(29.97)) == pytest.approx(30000 / 1001)
    assert source_frame_rate(VideoMetadata(fps=25)) == 25.0  # containers without an average rate


def test_variable_rate_sources_use_their_real_average():
    phone = meta(60, 29.6)
    assert is_variable_rate(phone)
    assert source_frame_rate(phone) == pytest.approx(30000 / 1001, abs=0.05)
    screen = meta(60, 12.3)
    assert source_frame_rate(screen) == 12.3
    doubled = meta(47.952, 23.748)  # stream-copied concat: declared rate reads double
    assert source_frame_rate(doubled) == pytest.approx(24000 / 1001)


def test_unknown_rate_is_zero_not_a_guess():
    assert source_frame_rate(VideoMetadata()) == 0.0


# Choosing the output rate


def test_source_choice_keeps_the_main_videos_rate():
    sources = [meta(30, width=1280, height=720), meta(25)]
    assert target_frame_rate(sources[1], sources, "source") == 25.0
    assert choose_canvas(sources, RenderSettings()).fps == 25.0


def test_custom_choice_overrides_the_source_rate():
    assert choose_canvas([meta(30)], RenderSettings(frame_rate="24")).fps == 24.0
    assert choose_canvas([meta(30)], RenderSettings(frame_rate="60")).fps == 60.0


def test_source_without_a_known_rate_falls_back_to_30():
    assert choose_canvas([VideoMetadata(width=640, height=360)]).fps == 30.0


# Conversion plan


@pytest.mark.parametrize(
    ("source", "target", "smooth", "plan"),
    [
        (meta(24), 24.0, False, "keep"),
        (meta(24), 24.0, True, "keep"),
        (meta(60), 30.0, False, "convert"),
        (meta(60), 30.0, True, "convert"),  # even reduction: smoothing adds nothing
        (meta(50), 25.0, True, "convert"),
        (meta(30), 24.0, False, "convert"),
        (meta(30), 24.0, True, "smooth"),  # uneven: would judder
        (meta(24), 60.0, True, "smooth"),  # raising the rate
        (meta(60, 29.6), 30000 / 1001, False, "convert"),  # variable rate is always made constant
        (VideoMetadata(), 30.0, True, "convert"),  # unknown rate: plain conversion, as before
    ],
)
def test_frame_rate_plan(source, target, smooth, plan):
    assert frame_rate_plan(source, target, smooth) == plan


def test_matching_source_gets_no_frame_rate_filter():
    canvas = choose_canvas([meta(24)])
    filters = clip_normalize_filters(meta(24), canvas)
    assert not any(item.startswith(("fps=", "minterpolate")) for item in filters)
    assert filters[-1] == "setpts=PTS-STARTPTS"


def test_conversion_filters():
    assert frame_rate_filters(meta(30), 24.0) == ["fps=24"]
    assert frame_rate_filters(meta(24), 23.976) == ["fps=24000/1001"]
    smooth = frame_rate_filters(meta(30), 24.0, smooth=True)
    assert len(smooth) == 1 and smooth[0].startswith("minterpolate=fps=24:")


def test_frame_rate_filter_runs_before_aspect_and_pixel_format():
    canvas = choose_canvas([meta(30)], RenderSettings(frame_rate="24"))
    filters = clip_normalize_filters(meta(30), canvas, extra=["crop=ih*9/16:ih"])
    assert filters.index("fps=24") < filters.index("crop=ih*9/16:ih") < filters.index("format=yuv420p")


def test_describe_frame_rate():
    assert "no conversion" in describe_frame_rate([meta(24)], 24.0)
    text = describe_frame_rate([meta(60, 29.6)], 24.0, smooth=True)
    assert "variable" in text and "motion smoothing" in text


# Settings


def test_frame_rate_options_and_defaults():
    config = HighlightConfig()
    assert config.frame_rate == "source" and config.smooth_motion is False
    assert HighlightConfig(frame_rate="60", smooth_motion=True).frame_rate == "60"
    with pytest.raises(ValidationError):
        HighlightConfig(frame_rate="47")


def test_older_saved_jobs_without_frame_rate_still_load():
    old = {"min_duration": 20, "max_duration": 30, "style": "auto", "aspect_ratio": "original"}
    assert HighlightConfig.model_validate(old).frame_rate == "source"


# Real encodes

ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")


def _make_source(path: Path, rate: str, seconds: int = 3) -> Path:
    result = subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc2=s=320x240:r={rate}:d={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(path),
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(result.stderr[-500:])
    return path


def _render(tmp_path: Path, source: Path, settings: RenderSettings) -> VideoMetadata:
    plan = ClipPlan(
        clips=[
            ClipPlanItem(start=0.0, end=1.2, score=0.9, reason="a"),
            ClipPlanItem(start=1.5, end=2.7, score=0.8, reason="b"),
        ],
        target_duration=2.4,
        estimated_duration=2.4,
    )
    output = tmp_path / f"out_{settings.frame_rate}.mp4"
    runner = FFmpegRunner()
    Renderer(runner).render(source, plan, output, settings, tmp_path / f"work_{settings.frame_rate}", metadata=runner.probe(source))
    return runner.probe(output)


@pytest.mark.skipif(not ffmpeg or not ffprobe, reason="ffmpeg/ffprobe not installed")
@pytest.mark.parametrize(("source_rate", "choice", "expected"), [("30", "source", 30.0), ("30", "24", 24.0), ("25", "50", 50.0)])
def test_render_hits_the_target_rate_and_keeps_duration_and_audio(tmp_path, source_rate, choice, expected):
    source = _make_source(tmp_path / f"src_{source_rate}.mp4", source_rate)
    out = _render(tmp_path, source, RenderSettings(frame_rate=choice, quality="high", crf=28))
    assert out.avg_fps == pytest.approx(expected, abs=0.6)
    assert out.duration == pytest.approx(2.4, abs=0.25)
    assert out.has_audio


@pytest.mark.skipif(not ffmpeg or not ffprobe, reason="ffmpeg/ffprobe not installed")
def test_smooth_motion_render(tmp_path):
    source = _make_source(tmp_path / "src.mp4", "24", seconds=3)
    out = _render(tmp_path, source, RenderSettings(frame_rate="30", smooth_motion=True, quality="high", crf=28))
    assert out.avg_fps == pytest.approx(30.0, abs=0.6)
    assert out.duration == pytest.approx(2.4, abs=0.25)
    assert out.has_audio
