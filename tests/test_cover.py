"""Cover image: a frame of the finished highlight."""

import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from app import create_app
from app.config import Settings
from app.models.schemas import VideoMetadata
from app.services.cover import CoverError, cover_image, parse_request
from app.video.processor import VideoProcessor


def test_frame_command_scales_only_when_asked():
    plain = VideoProcessor.extract_frame_command("in.mp4", "out.jpg", 1.5)
    assert "-vf" not in plain
    small = VideoProcessor.extract_frame_command("in.mp4", "out.jpg", 1.5, width=320)
    assert small[small.index("-vf") + 1] == "scale=320:-2:flags=lanczos"
    assert small[small.index("-ss") + 1] == "1.500"


@pytest.mark.parametrize(
    "t, width, expected",
    [("2.5", None, (2.5, None)), ("", "", (0.0, None)), ("1", "320", (1.0, 320)), ("1", "9999", (1.0, 1920))],
)
def test_parse_request(t, width, expected):
    assert parse_request(t, width) == expected


@pytest.mark.parametrize("t, width", [("-1", None), ("nan", None), ("inf", None), ("x", None), ("1", "10"), ("1", "abc")])
def test_parse_request_rejects_bad_values(t, width):
    with pytest.raises(CoverError):
        parse_request(t, width)


class FakeRunner:
    def __init__(self, duration=10.0):
        self.duration = duration
        self.commands = []

    def probe(self, path):
        return VideoMetadata(duration=self.duration)

    def run(self, cmd):
        self.commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"\xff\xd8jpeg")


def _job_dir(tmp_path: Path) -> Path:
    (tmp_path / "outputs").mkdir(parents=True)
    (tmp_path / "outputs" / "highlight.mp4").write_bytes(b"video")
    return tmp_path


def test_cover_is_cached_and_clamped(tmp_path):
    job_dir = _job_dir(tmp_path)
    runner = FakeRunner(duration=10)
    first = cover_image(job_dir, 3.2, 320, runner)
    again = cover_image(job_dir, 3.2, 320, runner)
    assert first == again and len(runner.commands) == 1
    late = cover_image(job_dir, 99, None, runner)
    assert late.name.endswith("-9900-full.jpg")
    cmd = runner.commands[-1]
    assert cmd[cmd.index("-ss") + 1] == "9.900"


def test_a_new_render_removes_old_covers(tmp_path):
    job_dir = _job_dir(tmp_path)
    runner = FakeRunner()
    old = cover_image(job_dir, 1, None, runner)
    video = job_dir / "outputs" / "highlight.mp4"
    stat = video.stat()
    import os

    os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    new = cover_image(job_dir, 1, None, runner)
    assert new != old and not old.exists() and new.exists()


def test_cover_without_output(tmp_path):
    with pytest.raises(CoverError) as info:
        cover_image(tmp_path, 1, None, FakeRunner())
    assert info.value.status == 404


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


def _create_job(client) -> str:
    data = {"file": (io.BytesIO(b"video"), "my talk.mp4"), "config": json.dumps({"min_duration": 20, "max_duration": 30})}
    return client.post("/api/jobs", data=data, content_type="multipart/form-data").get_json()["id"]


def test_cover_endpoint_errors(client):
    assert client.get("/api/jobs/missing/cover?t=1").status_code == 404
    job_id = _create_job(client)
    response = client.get(f"/api/jobs/{job_id}/cover?t=1")
    assert response.status_code == 404 and response.get_json()["error"] == "no output"
    assert client.get(f"/api/jobs/{job_id}/cover?t=-4").status_code == 400


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs FFmpeg")
def test_cover_endpoint_with_a_real_video(client):
    from app.services.job_manager import get_job_manager

    job_id = _create_job(client)
    outputs = get_job_manager().job_dir(job_id) / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=2",
         str(outputs / "highlight.mp4")],
        check=True,
    )
    small = client.get(f"/api/jobs/{job_id}/cover?t=1&width=160")
    assert small.status_code == 200 and small.mimetype == "image/jpeg"
    assert small.data[:2] == b"\xff\xd8"
    full = client.get(f"/api/jobs/{job_id}/cover?t=1&download=1")
    assert full.status_code == 200
    assert "my talk-cover.jpg" in full.headers["Content-Disposition"]
    assert len(full.data) > len(small.data)
