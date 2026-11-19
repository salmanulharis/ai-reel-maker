import io
import json

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


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["ok"] is True


def test_system_info_endpoint(client):
    response = client.get("/api/system/info")
    assert response.status_code == 200
    data = response.get_json()
    assert "checks" in data
    assert "hardware" in data
    assert "locally" in data["privacy"].lower()


def test_settings_roundtrip(client):
    response = client.get("/api/settings")
    assert response.status_code == 200
    posted = client.post("/api/settings", json={"output_crf": 20})
    assert posted.status_code == 200
    assert posted.get_json()["settings"]["output_crf"] == 20


def test_sample_video_endpoint(client, tmp_path):
    import shutil
    import subprocess

    from app.config import get_settings

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is not installed")
    sample = tmp_path / "sample.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10:duration=1",
         "-pix_fmt", "yuv420p", str(sample)],
        check=True,
    )
    get_settings().sample_video = sample
    response = client.get("/api/media/sample")
    assert response.status_code == 200
    assert response.mimetype == "video/mp4"
    assert len(response.data) > 1000


def test_upload_rejects_disallowed_extension(client):
    data = {
        "file": (io.BytesIO(b"not a video"), "notes.exe"),
        "config": json.dumps({"min_duration": 20, "max_duration": 30}),
    }
    response = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert response.status_code == 400


def test_upload_multiple_videos(client):
    data = {
        "file": (io.BytesIO(b"one"), "one.mp4"),
        "files": (io.BytesIO(b"two"), "two.mp4"),
        "config": json.dumps({"min_duration": 20, "max_duration": 30}),
    }
    response = client.post("/api/jobs", data=data, content_type="multipart/form-data")
    assert response.status_code == 201
    body = response.get_json()
    sources = body.get("sources") or []
    assert len(sources) == 2
    assert sources[0]["id"] == "v0"
    assert sources[1]["id"] == "v1"
    assert "one.mp4" in sources[0]["filename"]
    assert "two.mp4" in sources[1]["filename"]


def test_client_analysis_route_is_not_404(client):
    response = client.get("/jobs/abc123/analysis")
    assert response.status_code == 200


def test_worker_can_construct_analysis_pipeline():
    from app.workers import worker
    from app.analysis.pipeline import PipelineProgress, VideoAnalyzer

    assert worker.VideoAnalyzer is VideoAnalyzer
    assert worker.PipelineProgress is PipelineProgress
