from pathlib import Path

from app.models.schemas import ClipPlan, ClipPlanItem, RenderSettings, VideoMetadata
from app.video.processor import VideoProcessor
from app.video.quality import choose_canvas, clip_normalize_filters, encode_crf, video_encode_args
from app.video.renderer import Renderer, validate_output


def test_single_source_keeps_native_resolution_and_fps():
    canvas = choose_canvas(
        [VideoMetadata(width=1920, height=1080, fps=23.976)],
        RenderSettings(resolution="original"),
    )
    assert canvas.width == 1920
    assert canvas.height == 1080
    assert abs(canvas.fps - 23.976) < 0.01


def test_multi_source_uses_highest_resolution_without_upscale_filter():
    canvas = choose_canvas(
        [
            VideoMetadata(width=1280, height=720, fps=30),
            VideoMetadata(width=1920, height=1080, fps=24),
        ],
        RenderSettings(resolution="original"),
    )
    assert canvas.width == 1920
    assert canvas.height == 1080
    assert canvas.fps == 24
    filters = clip_normalize_filters(VideoMetadata(width=1280, height=720, fps=30), canvas)
    joined = ",".join(filters)
    assert "force_original_aspect_ratio=decrease" in joined
    assert "flags=lanczos" in joined
    assert "pad=1920:1080" in joined
    same = clip_normalize_filters(VideoMetadata(width=1920, height=1080, fps=24), canvas)
    assert not any(item.startswith("scale=") for item in same)


def test_user_1080p_cap_does_not_upscale_720p():
    canvas = choose_canvas(
        [VideoMetadata(width=1280, height=720, fps=30)],
        RenderSettings(resolution="1080p"),
    )
    assert canvas.width == 1280
    assert canvas.height == 720


def test_source_quality_uses_slow_crf14():
    args = video_encode_args("libx264", "source", crf=18)
    assert args[args.index("-preset") + 1] == "slow"
    assert args[args.index("-crf") + 1] == "14"
    assert encode_crf("source", 23) == 14


def test_cut_command_source_quality_is_high():
    cmd = VideoProcessor.cut_clip_command(
        "/tmp/in.mp4",
        "/tmp/clip.mp4",
        start=1,
        end=4,
        quality="source",
        video_codec="libx264",
    )
    assert cmd[cmd.index("-preset") + 1] == "slow"
    assert cmd[cmd.index("-crf") + 1] == "14"
    assert "-pix_fmt" in cmd
    assert "yuv420p" in cmd


def test_validate_output_rejects_resolution_mismatch(tmp_path: Path):
    path = tmp_path / "out.mp4"
    path.write_bytes(b"x" * 2000)
    probe = VideoMetadata(duration=5, width=1280, height=720, fps=24, has_audio=True)
    canvas = choose_canvas([VideoMetadata(width=1920, height=1080, fps=24)])
    result = validate_output(probe, None, path, canvas=canvas)
    assert result["ok"] is False
    assert "1920" in result["error"]


def test_source_quality_ignores_nvenc(tmp_path: Path):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x" * 2000)
    output = tmp_path / "out.mp4"
    commands: list[list[str]] = []

    class FakeRunner:
        def run(self, cmd):
            commands.append(cmd)
            Path(cmd[-1]).write_bytes(b"x" * 2000)

        def probe(self, path):
            return VideoMetadata(duration=5, width=1920, height=1080, fps=24, has_audio=True, size_bytes=2000)

    Renderer(FakeRunner()).render(
        source,
        ClipPlan(
            target_duration=5,
            estimated_duration=5,
            clips=[ClipPlanItem(start=0, end=5, reason="moment", score=0.9)],
        ),
        output,
        RenderSettings(codec="auto", quality="source"),
        tmp_path / "work",
        hardware_encoder="h264_nvenc",
    )
    codecs = [cmd[cmd.index("-c:v") + 1] for cmd in commands if "-c:v" in cmd]
    assert codecs
    assert "h264_nvenc" not in codecs
    assert "libx264" in codecs
