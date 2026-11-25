"""Routes for narrated videos: settings, the brief, switching engines after a pause, the preview."""

from __future__ import annotations

import io
import json

import pytest

from app import create_app
from app.brief.schema import VideoBrief
from app.brief.service import save_brief, save_failure
from app.config import Settings


@pytest.fixture()
def client(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", ollama_base_url="http://127.0.0.1:9", enable_whisper=False)
    settings.ensure_directories()
    app = create_app(settings, start_background=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        client.settings = settings
        yield client


def _job(client, config=None):
    data = {"file": (io.BytesIO(b"video"), "trip.mp4"), "config": json.dumps(config or {})}
    return client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()["id"]


def _job_dir(client, job_id):
    return client.settings.jobs_dir / job_id


BRIEF = VideoBrief.model_validate(
    {
        "engine": "local",
        "model": "qwen3:8b",
        "title": "Trip",
        "beats": [
            {"id": "b1", "purpose": "hook", "narration": "Hello.", "max_words": 8, "clips": [{"id": "c1", "start": 0, "end": 4}]},
            {"id": "b2", "purpose": "key", "narration": "", "max_words": 8, "clips": [{"id": "c2", "start": 10, "end": 14}]},
        ],
    }
)


def test_narrated_job_has_a_brief_step(client):
    job = client.get(f"/api/jobs/{_job(client, {'video_brief': True})}").get_json()
    assert "brief" in [stage["name"] for stage in job["stages"]]
    plain = client.get(f"/api/jobs/{_job(client)}").get_json()
    assert "brief" not in [stage["name"] for stage in plain["stages"]] and plain["brief_failure"] is None


def test_brief_routes_read_and_edit(client):
    job_id = _job(client, {"video_brief": True, "narration": True})
    assert client.get(f"/api/jobs/{job_id}/brief").get_json()["brief"] is None
    assert client.put(f"/api/jobs/{job_id}/brief", json={"title": "x"}).status_code == 404
    save_brief(_job_dir(client, job_id), BRIEF)
    info = client.get(f"/api/jobs/{job_id}/brief").get_json()
    assert info["brief"]["title"] == "Trip" and info["spoken"] == [] and info["failure"] is None
    assert client.put(f"/api/jobs/{job_id}/brief", data="nope", content_type="application/json").status_code == 400
    edited = client.put(
        f"/api/jobs/{job_id}/brief",
        json={"title": "New", "beats": [{"id": "b1", "narration": " ".join(["word"] * 12)}]},
    ).get_json()
    assert edited["brief"]["title"] == "New" and edited["brief"]["edited"] and edited["warnings"]


def test_paused_claude_job_switches_to_local_and_resumes(client):
    job_id = _job(client, {"video_brief": True, "brief_engine": "claude"})
    from app.services.job_manager import get_job_manager

    manager = get_job_manager()
    manager.update(job_id, status="paused", analysis_status="pending", error="Claude's usage limit was reached.")
    save_failure(_job_dir(client, job_id), "Claude's usage limit was reached.", "limit")
    job = client.get(f"/api/jobs/{job_id}").get_json()
    assert job["brief_failure"]["reason"] == "limit"
    assert client.get(f"/api/jobs/{job_id}/brief").get_json()["failure"]["reason"] == "limit"
    assert client.post(f"/api/jobs/{job_id}/brief-engine", json={"engine": "gpt"}).status_code == 400
    job = client.post(f"/api/jobs/{job_id}/brief-engine", json={"engine": "local", "resume": True}).get_json()
    assert job["config"]["brief_engine"] == "local" and job["status"] == "queued" and job["brief_failure"] is None


def test_preview_spec_follows_the_kept_clips(client):
    job_id = _job(client, {"video_brief": True, "aspect_ratio": "9:16", "subtitles": "burned", "caption_style": "clean"})
    assert client.get(f"/api/jobs/{job_id}/preview-spec").get_json()["spec"] is None
    save_brief(_job_dir(client, job_id), BRIEF)
    from app.services.job_manager import get_job_manager

    get_job_manager().save_plan(
        job_id,
        [
            {"start": 0, "end": 4, "source_id": "v0"},
            {"start": 10, "end": 14, "source_id": "v0"},
            {"start": 20, "end": 22, "source_id": "v0", "kept": False},
        ],
    )
    spec = client.get(f"/api/jobs/{job_id}/preview-spec").get_json()["spec"]
    assert spec["width"] == 1080 and spec["height"] == 1920 and spec["duration"] == 8
    assert [c["source_start"] for c in spec["video"]["clips"]] == [0, 10]
    assert spec["video"]["clips"][0]["src"] == f"/api/media/{job_id}/source/v0"
    assert spec["texts"][0] == {"kind": "title", "text": "Trip", "start": 0.0, "end": 2.6}


def test_narration_media_only_serves_spoken_lines(client):
    job_id = _job(client)
    folder = _job_dir(client, job_id) / "narration"
    folder.mkdir(parents=True)
    (folder / ("a" * 20 + ".wav")).write_bytes(b"RIFF")
    assert client.get(f"/api/media/{job_id}/narration/{'a' * 20}.wav").status_code == 200
    assert client.get(f"/api/media/{job_id}/narration/..%2Fjob.log").status_code == 404
    assert client.get(f"/api/media/{job_id}/narration/{'b' * 20}.wav").status_code == 404
    assert client.get(f"/api/media/{job_id}/narration/source.mp4").status_code == 404


def test_narration_settings_are_validated(client):
    ok = client.post("/api/settings", json={"brief_engine": "claude", "claude_model": "sonnet", "tts_voice": "bf_emma", "tts_speed": 1.2})
    assert ok.status_code == 200
    saved = client.get("/api/settings").get_json()["settings"]
    assert saved["brief_engine"] == "claude" and saved["tts_voice"] == "bf_emma" and saved["tts_speed"] == 1.2
    assert client.post("/api/settings", json={"claude_model": "gpt-5"}).status_code == 400
    assert client.post("/api/settings", json={"render_engine": "blender"}).status_code == 400
    assert client.post("/api/settings", json={"tts_voice": "robot"}).status_code == 400
    assert client.post("/api/settings", json={"tts_speed": 3}).status_code == 400
    # The CLI's program path names a program to run: never saved from the browser.
    client.post("/api/settings", json={"claude_cli_path": "C:/evil.exe"})
    assert client.get("/api/settings").get_json()["settings"]["claude_cli_path"] == "claude"


def test_system_routes(client, monkeypatch):
    import app.brief.engines as engines
    from app.voice import kokoro

    monkeypatch.setattr(engines, "engines_status", lambda settings, refresh=False: [{"name": "local", "ready": True}])
    assert client.get("/api/system/brief-engines?refresh=1").get_json()["engines"][0]["ready"]
    tools = client.get("/api/system/video-tools").get_json()
    assert set(tools) == {"voice", "remotion"} and "af_heart" in tools["voice"]["voices"]
    assert set(tools["remotion"]) >= {"node", "installed", "bundle", "browser", "ready", "prepare"}
    started = []
    monkeypatch.setattr(kokoro, "start_download", lambda: started.append(1) or {"state": "running"})
    assert client.post("/api/system/voice-model/download").get_json()["state"] == "running" and started


def test_clear_cache_keeps_the_prepared_video_tools(tmp_path):
    from app.services.storage import clear_cache

    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_directories()
    (settings.cache_dir / "remotion-bundle" / "abc").mkdir(parents=True)
    (settings.cache_dir / "remotion-browser.json").write_text("{}")
    (settings.cache_dir / "old.json").write_text("{}")
    clear_cache(settings)
    assert (settings.cache_dir / "remotion-bundle" / "abc").is_dir() and (settings.cache_dir / "remotion-browser.json").is_file()
    assert not (settings.cache_dir / "old.json").exists()
