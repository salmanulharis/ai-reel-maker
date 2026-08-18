from app.models.schemas import HighlightConfig
from app.video.crop import to_crop_pixels
from app.video.processor import VideoProcessor


def test_full_frame_crop_is_skipped():
    assert to_crop_pixels(1920, 1080, {"x": 0, "y": 0, "width": 1, "height": 1}) is None
    assert to_crop_pixels(1920, 1080, None) is None


def test_center_crop_uses_even_ffmpeg_pixels():
    box = to_crop_pixels(1920, 1080, {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5})
    assert box is not None
    assert box.width == 960
    assert box.height == 540
    assert box.x == 480
    assert box.y == 270
    assert box.width % 2 == 0
    assert box.height % 2 == 0
    assert "crop=960:540:480:270" == box.ffmpeg_filter()


def test_crop_command_uses_filter_and_keeps_audio():
    cmd = VideoProcessor.crop_command(
        "/tmp/in.mp4",
        "/tmp/out.mp4",
        width=960,
        height=540,
        x=480,
        y=270,
    )
    assert cmd[0] == "ffmpeg"
    assert "-vf" in cmd
    assert cmd[cmd.index("-vf") + 1] == "crop=960:540:480:270"
    assert "/tmp/in.mp4" in cmd
    assert "/tmp/out.mp4" in cmd
    assert not any(";" in part for part in cmd)


def test_highlight_config_accepts_crop_region():
    config = HighlightConfig.model_validate(
        {
            "min_duration": 20,
            "max_duration": 30,
            "crop": {"x": 0.2, "y": 0.1, "width": 0.5, "height": 0.6},
        }
    )
    assert config.crop is not None
    assert config.crop.x == 0.2
    assert config.crop.width == 0.5
    dumped = config.model_dump()
    assert dumped["crop"]["height"] == 0.6
