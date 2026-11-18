"""Tighten clips: cut long pauses and filler words out of clips."""

import io
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from app import create_app
from app.analysis.tighten import tighten_clips, tighten_spans
from app.config import Settings
from app.models.schemas import ClipPlan, ClipPlanItem, HighlightConfig, RenderSettings, VideoMetadata
from app.video.renderer import Renderer


def _w(*items):
    return [{"word": f" {text}", "start": start, "end": end} for text, start, end in items]


SPEECH = _w(("Hi", 1.0, 1.5), ("there", 1.6, 2.0), ("Um,", 2.1, 2.5), ("so", 2.6, 3.0), ("yes", 5.0, 5.5))


def test_long_pauses_are_cut_with_a_little_room():
    assert tighten_spans(0, 10, SPEECH) == [(0.88, 3.12), (4.88, 5.62)]


def test_fillers_are_cut_completely():
    assert tighten_spans(0, 10, SPEECH, remove_fillers=True) == [(0.88, 2.1), (2.5, 3.12), (4.88, 5.62)]


def test_nothing_to_cut_keeps_the_clip():
    assert tighten_spans(0, 3, _w(("Hi", 0.1, 1.5), ("there", 1.6, 2.9))) == [(0, 3)]
    assert tighten_spans(0, 3, []) == [(0, 3)]
    # Only filler words: keep the clip rather than lose it.
    assert tighten_spans(0, 3, _w(("um", 1, 1.4)), remove_fillers=True) == [(0, 3)]


def test_short_leading_and_trailing_silence_stays():
    assert tighten_spans(0, 3, _w(("a", 0.4, 1.0), ("b", 2.0, 2.6))) == [(0, 1.12), (1.88, 3)]


def test_tiny_pieces_are_dropped_and_words_outside_are_ignored():
    words = _w(("before", -2, -1), ("x", 1.0, 1.05), ("main", 3.0, 4.0), ("after", 20, 21), ("bad", "a", 1))
    assert tighten_spans(0, 10, words) == [(2.88, 4.12)]


def test_tighten_clips_marks_joined_pieces():
    clips = [ClipPlanItem(start=0, end=10, reason="r", source_id="v0"), ClipPlanItem(start=20, end=22, reason="q")]
    out = tighten_clips(clips, lambda clip: {"words": SPEECH} if clip.source_id == "v0" else {}, remove_fillers=True)
    assert [(c.start, c.end, c.join_previous, c.reason) for c in out] == [
        (0.88, 2.1, False, "r"),
        (2.5, 3.12, True, "r"),
        (4.88, 5.62, True, "r"),
        (20, 22, False, "q"),
    ]


def test_config_turns_tighten_off_without_speech():
    assert HighlightConfig().tighten == "off"
    assert HighlightConfig(tighten="fillers", use_speech=False).tighten == "off"
    assert HighlightConfig(tighten="pauses").tighten == "pauses"


def test_fades_only_at_real_clip_edges(tmp_path: Path):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x" * 2000)
    commands = []

    class FakeRunner:
        def run(self, cmd):
            commands.append(cmd)
            Path(cmd[-1]).write_bytes(b"x" * 2000)

        def probe(self, path):
            return VideoMetadata(duration=6, width=1280, height=720, fps=30, has_audio=True, size_bytes=2000)

    clips = [
        ClipPlanItem(start=0, end=2, reason="a"),
        ClipPlanItem(start=3, end=5, reason="a", join_previous=True),
        ClipPlanItem(start=8, end=10, reason="b"),
    ]
    Renderer(FakeRunner()).render(
        source,
        ClipPlan(target_duration=6, estimated_duration=6, clips=clips),
        tmp_path / "out.mp4",
        RenderSettings(transition="fade"),
        tmp_path / "work",
    )
    cuts = [cmd for cmd in commands if Path(cmd[-1]).name.startswith("clip_")]
    video = [cmd[cmd.index("-vf") + 1] for cmd in cuts]
    audio = [cmd[cmd.index("-af") + 1] for cmd in cuts]
    assert "fade=t=in" in video[0] and "fade=t=out" not in video[0]
    assert "fade=t=in" not in video[1] and "fade=t=out" in video[1]
    assert "fade=t=in" in video[2] and "fade=t=out" in video[2]
    assert "afade=t=out" not in audio[0] and "afade=t=in" not in audio[1]


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


def test_editor_export_matches_the_tightened_video(client):
    from app.services.job_manager import get_job_manager

    config = {"min_duration": 20, "max_duration": 30, "max_sequence_duration": 60, "tighten": "pauses"}
    data = {"file": (io.BytesIO(b"video"), "talk.mp4"), "config": json.dumps(config)}
    job_id = client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()["id"]
    (get_job_manager().job_dir(job_id) / "transcript.json").write_text(json.dumps({"words": SPEECH}), encoding="utf-8")
    client.post(f"/api/jobs/{job_id}/clips", json={"clips": [{"start": 0, "end": 10, "reason": "r"}]})
    archive = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{job_id}/export/editor").data))
    xml = ET.fromstring(archive.read("highlight.xml").decode("utf-8").split("\n", 2)[2])
    items = xml.findall("sequence/media/video/track/clipitem")
    assert [(i.find("in").text, i.find("out").text) for i in items] == [("22", "78"), ("122", "140")]
