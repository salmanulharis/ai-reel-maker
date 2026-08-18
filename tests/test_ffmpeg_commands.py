from app.video.processor import VideoProcessor


def test_probe_command_uses_argument_array():
    cmd = VideoProcessor.probe_command("/tmp/video.mp4")
    assert cmd[0] == "ffprobe"
    assert "-i" in cmd
    assert cmd[cmd.index("-i") + 1] == "/tmp/video.mp4"
    assert all(isinstance(part, str) for part in cmd)


def test_extract_audio_command_never_uses_shell_interpolation():
    cmd = VideoProcessor.extract_audio_command("/tmp/in.mp4", "/tmp/out.wav")
    assert cmd[0] == "ffmpeg"
    assert "-i" in cmd
    assert "/tmp/in.mp4" in cmd
    assert "/tmp/out.wav" in cmd
    assert not any(";" in part or "|" in part for part in cmd)


def test_extract_frame_command_uses_png_and_avoids_mjpeg():
    cmd = VideoProcessor.extract_frame_command("/tmp/in.mp4", "/tmp/frame.png", timestamp=12.5)
    assert cmd[0] == "ffmpeg"
    assert "-hide_banner" in cmd
    assert "-an" in cmd
    assert "/tmp/frame.png" in cmd
    assert "-strict" in cmd
    assert "unofficial" in cmd
    assert cmd[cmd.index("-ss") + 1] == "12.500"


def test_cut_clip_stream_copy_when_safe():
    cmd = VideoProcessor.cut_clip_command(
        "/tmp/in.mp4", "/tmp/clip.mp4", start=10.0, end=20.0, stream_copy=True
    )
    assert "-c" in cmd
    assert "copy" in cmd
    assert "-ss" in cmd


def test_cut_clip_reencodes_with_crf_when_frame_accurate():
    cmd = VideoProcessor.cut_clip_command(
        "/tmp/in.mp4",
        "/tmp/clip.mp4",
        start=10.12,
        end=20.44,
        stream_copy=False,
        video_codec="libx264",
        crf=18,
    )
    assert "libx264" in cmd
    assert "-crf" in cmd
    assert "18" in cmd
    assert "copy" not in cmd or cmd[cmd.index("-c:a") + 1] == "aac"


def test_concat_uses_concat_demuxer_file():
    cmd = VideoProcessor.concat_command("/tmp/list.txt", "/tmp/out.mp4", stream_copy=True)
    assert "-f" in cmd
    assert "concat" in cmd
    assert "-safe" in cmd
    assert "0" in cmd


def test_user_filename_cannot_inject_arguments():
    sneaky = "/tmp/video.mp4; rm -rf /"
    cmd = VideoProcessor.probe_command(sneaky)
    assert sneaky in cmd
    assert cmd.count("-i") == 1
    joined = " ".join(cmd)
    assert "rm -rf" in joined  # present as a single path argument, not extra argv
    assert cmd[-1] == sneaky or sneaky in cmd
