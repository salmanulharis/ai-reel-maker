"""Processing details: where the time went, kept with every job, and captions only when asked."""

from __future__ import annotations

import json

import pytest

import app.services.job_manager as jm
from app import create_app
from app.ai.intent import interpret
from app.config import Settings, get_settings, set_settings
from app.db import get_session
from app.models.db_models import Job, JobStage


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    previous = get_settings()
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", ollama_base_url="http://127.0.0.1:9")
    settings.ensure_directories()
    set_settings(settings)
    app = create_app(settings, start_background=False)
    app.config["TESTING"] = True
    monkeypatch.setattr(jm, "_manager", None)
    job_dir = settings.jobs_dir / "j1"
    job_dir.mkdir(parents=True, exist_ok=True)
    session = get_session()
    session.add(
        Job(
            id="j1",
            status="completed",
            analysis_status="completed",
            render_status="completed",
            original_filename="phone review.mp4",
            input_path=str(tmp_path / "v.mp4"),
            config_json=json.dumps({"mode": "prompt", "prompt": "x"}),
            metadata_json=json.dumps({"duration": 1025, "width": 3840, "height": 2160, "video_codec": "av1"}),
            processing_seconds=9206,
            analysis_seconds=8645,
            render_seconds=561,
        )
    )
    for name, seconds, unit in (
        ("scenes", 2842, "time"),
        ("transcript", 655, "time"),
        ("content", 1149, "segments"),
        ("visual", 3444, "scenes"),
        ("director", 329, "requests"),
        ("planning", 0.1, ""),
        ("rendering", 561, "clips"),
    ):
        session.add(JobStage(job_id="j1", name=name, status="done", seconds=seconds, done=5, total=5, unit=unit))
    session.commit()
    session.close()
    yield app, job_dir
    set_settings(previous)


def _log(job_dir, *entries):
    with (job_dir / "job.log").open("a", encoding="utf-8") as handle:
        for level, message in entries:
            handle.write(json.dumps({"level": level, "message": message, "ts": 1795600000.0}) + "\n")


def test_insights_explain_where_the_time_went(app_env):
    app, job_dir = app_env
    _log(
        job_dir,
        ("WARNING", "Other programs are using 94% of the processor"),
        ("INFO", "Visual AI: 70 calls, 97,555 prompt tokens in 2748.6s, 5,070 reply tokens in 676.6s (7.5 tok/s), model load 7.3s"),
        ("INFO", "Transcript AI: 15 calls, 11,740 prompt tokens in 154.9s, 7,836 reply tokens in 986.7s (7.9 tok/s)"),
        ("WARNING", "The transcript looks unreliable"),
        ("WARNING", "The transcript looks unreliable"),
    )
    (job_dir / "render_timings.json").write_text(
        json.dumps({"seconds": 561, "steps": {"clips": 500, "music": 20}, "encoder": {"codec": "libx264", "preset": "medium", "crf": 14}}),
        encoding="utf-8",
    )
    data = app.test_client().get("/api/jobs/j1/insights").get_json()
    assert data["totals"]["total_seconds"] == 9206 and data["totals"]["ratio"] == pytest.approx(8.98, abs=0.01)
    names = [step["name"] for step in data["steps"]]
    assert names[0] == "scenes" and names[-1] == "rendering", "pipeline order"
    assert data["slowest"][0] == "visual"
    assert sum(step["share"] for step in data["steps"]) == pytest.approx(1.0, abs=0.01)
    # Older jobs: the AI work is read back from the log.
    visual = next(row for row in data["ai"] if row["step"] == "Visual AI")
    assert visual["calls"] == 70 and visual["prompt_tokens"] == 97555 and visual["tokens_per_second"] == pytest.approx(7.5, abs=0.1)
    bottlenecks = {item["name"]: item for item in data["bottlenecks"]}
    assert {"visual", "scenes", "computer", "ai_speed"} <= set(bottlenecks)
    assert "70 request" in bottlenecks["visual"]["why"] and bottlenecks["visual"]["tips"]
    assert "94%" in bottlenecks["computer"]["why"]
    assert [w["message"] for w in data["warnings"]].count("The transcript looks unreliable") == 1
    parts = {part["name"]: part["seconds"] for part in data["render"]["parts"]}
    assert parts["clips"] == 500 and parts["other"] == 41
    assert data["has_log"] is True and data["run"] is None


def test_saved_ai_usage_and_run_info_are_used(app_env):
    app, job_dir = app_env
    (job_dir / "ai_usage.json").write_text(
        json.dumps({"AI editor": {"calls": 3, "prompt_tokens": 9000, "prompt_sec": 100, "output_tokens": 900, "output_sec": 120}}),
        encoding="utf-8",
    )
    (job_dir / "run_info.json").write_text(
        json.dumps({
            "models": {"speech": "small", "vision": "qwen3-vl:4b-instruct", "text": "qwen3:4b", "editor": "qwen3:4b"},
            "options": {"analysis_copy": True, "editor_review": True, "vision_scope": "shortlist"},
            "found": {"scenes": 97, "scenes_ai": 40, "candidates": 120, "language": "ml"},
            "computer": {"cpu_threads": 12, "memory_gb": 15.8},
        }),
        encoding="utf-8",
    )
    data = app.test_client().get("/api/jobs/j1/insights").get_json()
    assert [row["step"] for row in data["ai"]] == ["AI editor"]
    assert data["ai"][0]["seconds_per_call"] == pytest.approx(73.3, abs=0.1)
    assert data["run"]["models"]["speech"] == "small"
    scenes = next(item for item in data["bottlenecks"] if item["name"] == "scenes")
    assert "small copy was used" in scenes["why"]


def test_full_log_downloads_as_text(app_env):
    app, job_dir = app_env
    _log(job_dir, ("INFO", "Video loaded"), ("WARNING", "Careful"))
    response = app.test_client().get("/api/jobs/j1/logs?download=1")
    assert response.status_code == 200
    assert "attachment" in response.headers["Content-Disposition"]
    text = response.get_data(as_text=True)
    assert "INFO    Video loaded" in text and "WARNING Careful" in text


def test_unknown_jobs_have_no_insights(app_env):
    app, _ = app_env
    client = app.test_client()
    assert client.get("/api/jobs/nope/insights").status_code == 404
    assert client.get("/api/jobs/nope/logs?download=1").status_code == 404


class CaptionHappyModel:
    """A model that adds captions to every request."""

    def role_options(self, role):
        return {"num_ctx": 4096}

    def chat(self, model, prompt, schema=None, options=None, **kwargs):
        return {"summary": "a reel", "subtitles": "burned", "format": "9:16"}


def test_captions_are_only_added_when_asked_for():
    plain = interpret("Make an engaging reel of the best moments", duration=600, client=CaptionHappyModel(), model="m")
    assert "subtitles" not in plain["settings"]
    asked = interpret("Make a reel with captions", duration=600, client=CaptionHappyModel(), model="m")
    assert asked["settings"]["subtitles"] == "burned"
