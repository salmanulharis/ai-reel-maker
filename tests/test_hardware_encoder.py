from pathlib import Path

from app.models.schemas import ClipPlan, ClipPlanItem, RenderSettings, VideoMetadata
from app.services.hardware import preferred_hw_encoder
from app.video.ffmpeg_runner import FFmpegError
from app.video.renderer import Renderer


def test_preferred_hw_encoder_skips_nvenc_without_cuda(monkeypatch):
    monkeypatch.setattr(
        "app.services.hardware.detect_ffmpeg_encoders",
        lambda: ["h264_nvenc", "libx264"],
    )
    monkeypatch.setattr("app.services.hardware.cuda_runtime_available", lambda: False)
    monkeypatch.setattr("app.services.hardware.probe_encoder", lambda _name: True)
    assert preferred_hw_encoder() is None


def test_preferred_hw_encoder_uses_nvenc_when_runtime_works(monkeypatch):
    monkeypatch.setattr(
        "app.services.hardware.detect_ffmpeg_encoders",
        lambda: ["h264_nvenc", "libx264"],
    )
    monkeypatch.setattr("app.services.hardware.cuda_runtime_available", lambda: True)
    monkeypatch.setattr("app.services.hardware.probe_encoder", lambda name: name == "h264_nvenc")
    assert preferred_hw_encoder() == "h264_nvenc"


def test_renderer_retries_with_libx264_when_nvenc_cannot_load_cuda(tmp_path: Path):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"fake")
    output = tmp_path / "out.mp4"
    work = tmp_path / "work"
    commands: list[list[str]] = []

    class FakeRunner:
        def run(self, cmd):
            commands.append(cmd)
            if "h264_nvenc" in cmd:
                raise FFmpegError("Cannot load libcuda.so.1")
            Path(cmd[-1]).write_bytes(b"x" * 2000)

        def probe(self, path):
            return VideoMetadata(
                duration=5.0,
                width=1920,
                height=1080,
                has_audio=True,
                size_bytes=2000,
            )

    plan = ClipPlan(
        target_duration=5,
        estimated_duration=5,
        clips=[ClipPlanItem(start=0, end=5, reason="moment", score=0.9)],
    )
    renderer = Renderer(FakeRunner())
    result = renderer.render(
        source,
        plan,
        output,
        RenderSettings(codec="auto", quality="high"),
        work,
        hardware_encoder="h264_nvenc",
    )
    codecs = [cmd[cmd.index("-c:v") + 1] for cmd in commands if "-c:v" in cmd]
    assert "h264_nvenc" in codecs
    assert "libx264" in codecs
    assert result["validation"]["ok"] is True
