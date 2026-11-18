"""Download for video editors: timeline formats and the export endpoint."""

import io
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from app import create_app
from app.config import Settings
from app.models.schemas import VideoMetadata
from app.video.timeline_export import (
    Rate,
    TimelineClip,
    TimelineSource,
    build_timeline,
    edl,
    fcpxml,
    file_uri,
    pathurl,
    rate_for,
    srt,
    xmeml,
)


def _source(tmp_path, sid="v0", name="my talk & demo.mp4", fps=29.97, audio=True, segments=None):
    path = tmp_path / name
    path.write_bytes(b"x")
    return TimelineSource(
        id=sid,
        name=name,
        path=path,
        fps=fps,
        width=1920,
        height=1080,
        duration=120.0,
        has_audio=audio,
        channels=2,
        segments=segments or [],
    )


@pytest.mark.parametrize(
    "fps, expected",
    [
        (29.97, Rate(30, True)),
        (30, Rate(30, False)),
        (23.976, Rate(24, True)),
        (25, Rate(25, False)),
        (59.94, Rate(60, True)),
        (0, Rate(25, False)),
        (None, Rate(25, False)),
        (float("nan"), Rate(25, False)),
        (12, Rate(12, False)),
    ],
)
def test_rate_for_picks_standard_editing_rates(fps, expected):
    assert rate_for(fps) == expected


def test_rate_times_and_timecode():
    ntsc = Rate(30, True)
    assert ntsc.frame_duration == "1001/30000s"
    assert ntsc.rational(0) == "0s"
    assert ntsc.rational(30) == "30030/30000s"
    assert ntsc.frames(1.0) == 30
    assert Rate(25).rational(50) == "50/25s"
    assert Rate(25).timecode(25 * 3661 + 7) == "01:01:01:07"


def test_build_timeline_snaps_to_frames_and_butts_clips(tmp_path):
    src = _source(tmp_path, fps=25)
    timeline = build_timeline(
        "Talk highlight",
        [
            TimelineClip(src, 10.0, 12.5),
            TimelineClip(src, 30.004, 30.012),  # rounds to the same frame: left out
            TimelineClip(src, 40.0, 41.0),
        ],
    )
    assert timeline.rate == Rate(25)
    assert [(c.src_in, c.src_out, c.rec_in, c.rec_out) for c in timeline.clips] == [
        (250, 312, 0, 62),
        (1000, 1025, 62, 87),
    ]
    assert [c.number for c in timeline.clips] == [1, 2]
    assert timeline.duration == 87
    assert (timeline.width, timeline.height) == (1920, 1080)


def test_fcpxml_is_valid_and_points_at_the_original(tmp_path):
    src = _source(tmp_path)
    timeline = build_timeline("A & B", [TimelineClip(src, 1, 2), TimelineClip(src, 5, 6)])
    root = ET.fromstring(fcpxml(timeline).split("\n", 2)[2])
    assert root.tag == "fcpxml" and root.get("version") == "1.9"
    asset = root.find("resources/asset")
    assert asset.get("name") == "my talk & demo.mp4"
    assert asset.get("hasAudio") == "1"
    assert asset.find("media-rep").get("src") == file_uri(src.path)
    clips = root.findall(".//spine/asset-clip")
    assert [c.get("offset") for c in clips] == ["0s", "30030/30000s"]
    assert clips[0].get("start") == "30030/30000s"
    assert root.find(".//project").get("name") == "A & B"


def test_xmeml_is_valid_defines_each_file_once_and_links_audio(tmp_path):
    src = _source(tmp_path)
    timeline = build_timeline("T", [TimelineClip(src, 1, 2), TimelineClip(src, 5, 6)])
    root = ET.fromstring(xmeml(timeline).split("\n", 2)[2])
    files = root.findall(".//file")
    assert len([f for f in files if f.find("pathurl") is not None]) == 1
    assert files[0].find("pathurl").text == pathurl(src.path)
    video = root.findall("sequence/media/video/track/clipitem")
    audio = root.findall("sequence/media/audio/track/clipitem")
    assert len(video) == 2 and len(audio) == 2
    assert [v.find("start").text for v in video] == ["0", "30"]
    assert audio[0].find("sourcetrack/trackindex").text == "1"
    assert video[0].find("link/linkclipref").text == "clipitem-v1"
    assert root.find("sequence/rate/ntsc").text == "TRUE"


def test_video_without_sound_has_no_audio_track(tmp_path):
    src = _source(tmp_path, audio=False)
    timeline = build_timeline("T", [TimelineClip(src, 1, 2)])
    root = ET.fromstring(xmeml(timeline).split("\n", 2)[2])
    assert root.find("sequence/media/audio") is None
    assert root.find(".//link") is None
    fcp = ET.fromstring(fcpxml(timeline).split("\n", 2)[2])
    assert fcp.find("resources/asset").get("hasAudio") == "0"
    assert " V " in edl(timeline)


def test_two_videos_get_two_assets_and_files(tmp_path):
    a = _source(tmp_path, "v0", "a.mp4", fps=25)
    b = _source(tmp_path, "v1", "b.mov", fps=50)
    timeline = build_timeline("T", [TimelineClip(a, 0, 1), TimelineClip(b, 3, 4)])
    fcp = ET.fromstring(fcpxml(timeline).split("\n", 2)[2])
    assert len(fcp.findall("resources/asset")) == 2
    assert len(fcp.findall("resources/format")) == 2
    xml = ET.fromstring(xmeml(timeline).split("\n", 2)[2])
    assert len([f for f in xml.findall(".//file") if f.find("pathurl") is not None]) == 2


def test_edl_lines(tmp_path):
    src = _source(tmp_path, fps=25)
    timeline = build_timeline("Ünïcode\nname", [TimelineClip(src, 10, 12)])
    text = edl(timeline)
    lines = text.split("\r\n")
    assert lines[0] == "TITLE: ?n?code name"
    assert lines[1] == "FCM: NON-DROP FRAME"
    assert lines[3] == "001  AX       AA/V  C        00:00:10:00 00:00:12:00 00:00:00:00 00:00:02:00"
    assert lines[4] == "* FROM CLIP NAME: my talk & demo.mp4"
    assert lines[5].startswith("* SOURCE FILE: ")


def test_srt_moves_speech_to_its_place_in_the_edit(tmp_path):
    segments = [
        {"start": 9.0, "end": 11.0, "text": " Before and inside "},
        {"start": 11.5, "end": 13.0, "text": "Crosses the end"},
        {"start": 20.0, "end": 21.0, "text": "Not used"},
        {"start": 41.0, "end": 41.5, "text": ""},
    ]
    src = _source(tmp_path, fps=25, segments=segments)
    timeline = build_timeline("T", [TimelineClip(src, 10, 12), TimelineClip(src, 40, 42)])
    assert srt(timeline) == (
        "1\n00:00:00,000 --> 00:00:01,000\nBefore and inside\n\n"
        "2\n00:00:01,500 --> 00:00:02,000\nCrosses the end\n"
    )
    assert srt(build_timeline("T", [TimelineClip(_source(tmp_path, "v9", "x.mp4"), 0, 1)])) == ""


def test_paths_are_escaped(tmp_path):
    path = tmp_path / "a b#c.mp4"
    assert "a%20b%23c.mp4" in pathurl(path)
    assert pathurl(path).startswith("file://localhost/")
    assert file_uri(path).startswith("file:///")


# ---------------------------------------------------------------------------- endpoint


@pytest.fixture()
def client(tmp_path, monkeypatch):
    settings = Settings(
        data_dir=tmp_path / "data",
        ollama_base_url="http://127.0.0.1:9",
        enable_whisper=False,
        enable_visual_analysis=False,
    )
    settings.ensure_directories()
    import app.services.editor_export as export

    monkeypatch.setattr(
        export,
        "_default_probe",
        lambda path: VideoMetadata(duration=60, width=1280, height=720, fps=25, has_audio=True, channels=2),
    )
    app = create_app(settings, start_background=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _job(client, config=None) -> str:
    data = {
        "file": (io.BytesIO(b"video"), "talk.mp4"),
        "config": json.dumps(config or {"min_duration": 20, "max_duration": 30, "max_sequence_duration": 60}),
    }
    response = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert response.status_code == 201
    return response.get_json()["id"]


def _job_dir(job_id: str) -> Path:
    from app.services.job_manager import get_job_manager

    return get_job_manager().job_dir(job_id)


def test_export_zip_contents(client):
    job_id = _job(client)
    (_job_dir(job_id) / "transcript.json").write_text(
        json.dumps({"segments": [{"start": 10.5, "end": 11.5, "text": "Hello"}]}), encoding="utf-8"
    )
    clips = [
        {"start": 10, "end": 12, "reason": "r"},
        {"start": 20, "end": 22, "reason": "r", "kept": False},
        {"start": 30, "end": 33, "reason": "r"},
    ]
    assert client.post(f"/api/jobs/{job_id}/clips", json={"clips": clips}).status_code == 200

    response = client.get(f"/api/jobs/{job_id}/export/editor")
    assert response.status_code == 200
    assert response.mimetype == "application/zip"
    assert "talk-for-editors.zip" in response.headers["Content-Disposition"]
    archive = zipfile.ZipFile(io.BytesIO(response.data))
    assert sorted(archive.namelist()) == [
        "README.txt",
        "highlight.edl",
        "highlight.fcpxml",
        "highlight.srt",
        "highlight.xml",
    ]
    xml = ET.fromstring(archive.read("highlight.xml").decode("utf-8").split("\n", 2)[2])
    video = xml.findall("sequence/media/video/track/clipitem")
    assert [(v.find("in").text, v.find("out").text) for v in video] == [("250", "300"), ("750", "825")]
    assert "Hello" in archive.read("highlight.srt").decode("utf-8")
    readme = archive.read("README.txt").decode("utf-8")
    assert "2 clips" in readme and "25 fps" in readme


def test_export_keeps_clips_made_longer_by_hand(client):
    # The longest-clip setting is 2 s, but the clip was made 10 s long on the Review page.
    job_id = _job(client, {"min_duration": 20, "max_duration": 30, "sequence_duration_mode": "custom", "max_sequence_duration": 2})
    client.post(f"/api/jobs/{job_id}/clips", json={"clips": [{"start": 10, "end": 20, "reason": "r"}]})
    archive = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{job_id}/export/editor").data))
    xml = ET.fromstring(archive.read("highlight.xml").decode("utf-8").split("\n", 2)[2])
    clip = xml.find("sequence/media/video/track/clipitem")
    assert (clip.find("in").text, clip.find("out").text) == ("250", "500")  # all 10 s at 25 fps


def test_export_without_speech_has_no_subtitles(client):
    job_id = _job(client, {"min_duration": 20, "max_duration": 30, "use_speech": False})
    (_job_dir(job_id) / "transcript.json").write_text(
        json.dumps({"segments": [{"start": 10.5, "end": 11.5, "text": "Hello"}]}), encoding="utf-8"
    )
    client.post(f"/api/jobs/{job_id}/clips", json={"clips": [{"start": 10, "end": 12, "reason": "r"}]})
    archive = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{job_id}/export/editor").data))
    assert "highlight.srt" not in archive.namelist()


def test_export_errors(client):
    assert client.get("/api/jobs/missing/export/editor").status_code == 404
    job_id = _job(client)
    response = client.get(f"/api/jobs/{job_id}/export/editor")
    assert response.status_code == 400
    assert response.get_json()["error"] == "no clips to export"
    client.post(f"/api/jobs/{job_id}/clips", json={"clips": [{"start": 1, "end": 2, "reason": "r", "kept": False}]})
    assert client.get(f"/api/jobs/{job_id}/export/editor").status_code == 400
    client.post(f"/api/jobs/{job_id}/clips", json={"clips": [{"start": 1, "end": 2, "reason": "r"}]})
    for item in (_job_dir(job_id) / "source").iterdir():
        item.unlink()
    response = client.get(f"/api/jobs/{job_id}/export/editor")
    assert response.status_code == 404
    assert response.get_json()["error"] == "source video missing"
