from app.video.ffmpeg_runner import parse_probe, parse_silence


def test_parse_probe_extracts_video_and_audio_fields():
    payload = {
        "format": {"duration": "12.5", "bit_rate": "2500000", "size": "1000"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
                "pix_fmt": "yuv420p",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
            },
        ],
    }
    from pathlib import Path

    meta = parse_probe(payload, Path("talk.mp4"))
    assert meta.duration == 12.5
    assert meta.width == 1920
    assert meta.fps == 30
    assert meta.has_audio is True
    assert meta.audio_codec == "aac"


def test_parse_probe_prefers_average_rate_over_timebase():
    from pathlib import Path

    payload = {
        "format": {"duration": "5", "size": "2000"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "240/1",
                "avg_frame_rate": "24/1",
            }
        ],
    }
    meta = parse_probe(payload, Path("out.mp4"))
    assert meta.fps == 24


def test_parse_silence_ranges():
    stderr = """
[silencedetect @ 0] silence_start: 4.02
[silencedetect @ 0] silence_end: 6.55 | silence_duration: 2.53
"""
    ranges = parse_silence(stderr)
    assert ranges[0]["start"] == 4.02
    assert ranges[0]["end"] == 6.55
