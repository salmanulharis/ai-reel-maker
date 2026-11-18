"""Burned-in subtitles in the real render path, for a job made from several videos.

Only the FFmpeg part is replaced: a stand-in renderer asks for each clip's subtitle file the way
the real one does, so this checks which transcript every clip's subtitles come from.
"""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.workers.worker as worker
from app import create_app
from app.config import Settings


@pytest.fixture()
def app_and_client(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", ollama_base_url="http://127.0.0.1:9", enable_whisper=False)
    settings.ensure_directories()
    app = create_app(settings, start_background=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield app, client


class FakeRenderer:
    """Calls clip_subtitles for every clip like Renderer.render, and records what it got."""

    calls: list = []

    def __init__(self, runner):
        pass

    def render(self, source, plan, output, settings, work_dir, metadata=None, clip_subtitles=None, **kwargs):
        canvas = SimpleNamespace(width=1280, height=720, fps=30)
        for index, clip in enumerate(plan.clips):
            path = clip_subtitles(index, clip, canvas) if clip_subtitles else None
            FakeRenderer.calls.append(
                {
                    "source_id": clip.source_id,
                    "text": path.read_text(encoding="utf-8") if path else None,
                    "kind": path.suffix if path else None,
                }
            )
        Path(output).write_bytes(b"video")
        return {"metadata": {"duration": 4, "size_bytes": 5, "width": 1280, "height": 720}, "canvas": {}, "encoder": {}}


def _two_video_job(client, config: dict) -> tuple[str, Path]:
    from app.services.job_manager import get_job_manager

    data = {
        "file": [(io.BytesIO(b"one"), "first.mp4"), (io.BytesIO(b"two"), "second.mp4")],
        "config": json.dumps({"subtitles": "burned", **config}),
    }
    job_id = client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()["id"]
    job_dir = get_job_manager().job_dir(job_id)
    # Each video has its own transcript, in its own times (both start talking at 10 s).
    for sid, words in (("v0", ["Hello", "from", "the", "first", "video."]), ("v1", ["Greetings", "from", "video", "two."])):
        folder = job_dir / "work" / sid
        folder.mkdir(parents=True)
        transcript = {
            "segments": [{"start": 10.0, "end": 12.5, "text": " ".join(words)}],
            "words": [{"word": w, "start": 10.0 + i * 0.5, "end": 10.4 + i * 0.5} for i, w in enumerate(words)],
        }
        (folder / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")
    (job_dir / "clip_plan.json").write_text(json.dumps({"target_duration": 4, "estimated_duration": 4, "clips": []}), encoding="utf-8")
    clips = [
        {"start": 10, "end": 12.5, "reason": "r", "source_id": "v0"},
        {"start": 10, "end": 12.5, "reason": "r", "source_id": "v1"},
    ]
    assert client.post(f"/api/jobs/{job_id}/clips", json={"clips": clips}).status_code == 200
    return job_id, job_dir


@pytest.mark.parametrize("style, kind", [("plain", ".srt"), ("clean", ".ass")])
def test_each_clip_gets_subtitles_from_its_own_video(app_and_client, monkeypatch, style, kind):
    app, client = app_and_client
    monkeypatch.setattr(worker, "Renderer", FakeRenderer)
    monkeypatch.setattr(worker, "preferred_hw_encoder", lambda: None)
    FakeRenderer.calls = []
    job_id, job_dir = _two_video_job(client, {"caption_style": style})

    worker._run_render(app, job_id)

    job = client.get(f"/api/jobs/{job_id}").get_json()
    assert job["status"] == "completed", job.get("error")
    first, second = FakeRenderer.calls
    assert first["source_id"] == "v0" and first["kind"] == kind
    assert "first" in first["text"] and "two" not in first["text"]
    assert second["source_id"] == "v1" and second["kind"] == kind
    assert "Greetings" in second["text"] and "first" not in second["text"]
    # Each clip's subtitles start at the clip's own start (the cut begins at 0).
    if style == "plain":
        assert "00:00:00,000 -->" in first["text"] and "00:00:00,000 -->" in second["text"]
    # The downloadable subtitle file has both, one after the other.
    srt = (job_dir / "outputs" / "highlight.srt").read_text(encoding="utf-8")
    assert "Hello from the first video." in srt and "Greetings from video two." in srt
    assert "00:00:02,500 --> 00:00:05,000" in srt
