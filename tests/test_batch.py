"""Batch: a separate highlight for each uploaded video."""

import io
import json
import shutil
from pathlib import Path

import pytest

from app import create_app
from app.config import Settings


@pytest.fixture()
def client(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        ollama_base_url="http://127.0.0.1:9",
        enable_whisper=False,
        enable_visual_analysis=False,
    )
    settings.ensure_directories()
    app = create_app(settings, start_background=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _manager():
    from app.services.job_manager import get_job_manager

    return get_job_manager()


def _upload(client, *names) -> str:
    data = {
        "file": [(io.BytesIO(f"video {name}".encode()), name) for name in names],
        "config": json.dumps({"min_duration": 20, "max_duration": 30}),
    }
    response = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert response.status_code == 201
    return response.get_json()["id"]


def test_split_makes_one_queued_job_per_video(client):
    job_id = _upload(client, "a.mp4", "b.mov", "c.mkv")
    old_dir = _manager().job_dir(job_id)
    response = client.post(f"/api/jobs/{job_id}/split", json={"style": "exciting", "max_duration": 45})
    assert response.status_code == 200
    jobs = response.get_json()["jobs"]
    assert [j["original_filename"] for j in jobs] == ["a.mp4", "b.mov", "c.mkv"]
    for job, name in zip(jobs, ["a.mp4", "b.mov", "c.mkv"]):
        assert job["status"] == "queued"
        assert job["config"]["style"] == "exciting" and job["config"]["max_duration"] == 45
        assert len(job["sources"]) == 1 and job["sources"][0]["id"] == "v0"
        path = Path(job["sources"][0]["path"])
        assert path.is_file() and path.read_bytes() == f"video {name}".encode()
        assert path.parent.parent == _manager().job_dir(job["id"])
    # The combined job is gone, with its folder.
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert not old_dir.exists()
    listed = [j["id"] for j in client.get("/api/jobs").get_json()["jobs"]]
    assert sorted(listed) == sorted(j["id"] for j in jobs)


def test_split_needs_two_videos_and_an_unstarted_job(client):
    single = _upload(client, "a.mp4")
    response = client.post(f"/api/jobs/{single}/split", json={})
    assert response.status_code == 400 and "two videos" in response.get_json()["error"]

    started = _upload(client, "a.mp4", "b.mp4")
    client.post(f"/api/jobs/{started}/analyze", json={})
    response = client.post(f"/api/jobs/{started}/split", json={})
    assert response.status_code == 400 and "not started" in response.get_json()["error"]
    assert client.post("/api/jobs/missing/split", json={}).status_code == 404


def test_split_puts_files_back_when_a_move_fails(client, monkeypatch):
    job_id = _upload(client, "a.mp4", "b.mp4")
    sources = client.get(f"/api/jobs/{job_id}").get_json()["sources"]
    real_move = shutil.move
    calls = {"n": 0}

    def flaky_move(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk is full")
        return real_move(src, dst)

    monkeypatch.setattr(shutil, "move", flaky_move)
    response = client.post(f"/api/jobs/{job_id}/split", json={})
    assert response.status_code == 400 and response.get_json()["error"] == "disk is full"
    monkeypatch.setattr(shutil, "move", real_move)
    # The original job still has both videos, and no other job was made.
    assert all(Path(s["path"]).is_file() for s in sources)
    assert client.get(f"/api/jobs/{job_id}").get_json()["status"] == "uploaded"
    assert [j["id"] for j in client.get("/api/jobs").get_json()["jobs"]] == [job_id]
    leftovers = [p.name for p in _manager().settings.jobs_dir.iterdir() if p.is_dir()]
    assert leftovers == [job_id]
