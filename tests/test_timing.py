"""Processing time: live attempts, totals across retries and phases, and per-stage time."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from app.config import Settings, get_settings, set_settings
from app.models.db_models import Job, get_session, init_db
from app.services import job_manager as jm
from app.services.job_manager import JobManager
from app.utils.timefmt import format_elapsed


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds: float):
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def env(tmp_path, monkeypatch):
    previous = get_settings()
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    settings.ensure_directories()
    set_settings(settings)
    init_db(settings.db_path)
    clock = Clock()
    monkeypatch.setattr(jm, "utcnow", clock)
    session = get_session()
    session.add(Job(id="j1", status="queued", original_filename="v.mp4", input_path="v.mp4"))
    session.commit()
    session.close()
    yield JobManager(settings), clock
    set_settings(previous)


def _job(manager):
    return manager.get("j1")


def test_format_elapsed():
    assert format_elapsed(12.44) == "12.4 seconds"
    assert format_elapsed(0) == "0.0 seconds"
    assert format_elapsed(135) == "2 min 15 sec"
    assert format_elapsed(120) == "2 min"
    assert format_elapsed(3900) == "1 hr 5 min"
    assert format_elapsed(7200) == "2 hr"
    assert format_elapsed(-5) == "0.0 seconds"


def test_attempt_time_is_measured_from_start_to_finish(env):
    manager, clock = env
    manager.start_attempt("j1", "analysis")
    timing = manager.timing(_job(manager))
    assert timing["attempt_started_at"] is not None and timing["total_seconds"] == 0
    clock.advance(42.5)
    assert manager.finish_attempt("j1", "completed") == pytest.approx(42.5)
    timing = manager.timing(_job(manager))
    assert timing["attempt_started_at"] is None
    assert timing["attempt_seconds"] == pytest.approx(42.5)
    assert timing["total_seconds"] == pytest.approx(42.5)
    assert timing["analysis_seconds"] == pytest.approx(42.5)
    assert timing["attempt_outcome"] == "completed"


def test_retries_and_rendering_add_up_and_failures_keep_their_time(env):
    manager, clock = env
    manager.start_attempt("j1", "analysis")
    clock.advance(30)
    manager.finish_attempt("j1", "failed")
    manager.start_attempt("j1", "analysis")
    clock.advance(20)
    manager.finish_attempt("j1", "completed")
    manager.start_attempt("j1", "render")
    clock.advance(10)
    manager.finish_attempt("j1", "cancelled")
    timing = manager.timing(_job(manager))
    assert timing["analysis_seconds"] == pytest.approx(50)
    assert timing["render_seconds"] == pytest.approx(10)
    assert timing["total_seconds"] == pytest.approx(60)
    assert timing["attempt_seconds"] == pytest.approx(10)
    assert timing["attempt_outcome"] == "cancelled"


def test_closing_an_already_finished_attempt_changes_nothing(env):
    manager, clock = env
    manager.start_attempt("j1", "analysis")
    clock.advance(5)
    manager.finish_attempt("j1", "completed")
    clock.advance(100)
    assert manager.finish_attempt("j1", "failed", only_if_running=True) == 0.0
    timing = manager.timing(_job(manager))
    assert timing["total_seconds"] == pytest.approx(5) and timing["attempt_outcome"] == "completed"


def _stage(manager, name):
    return next(s for s in manager.stages("j1") if s["name"] == name)


def test_stage_time_ignores_progress_updates_and_adds_up_runs(env):
    manager, clock = env
    manager.set_stage("j1", "visual", "running")
    clock.advance(3)
    manager.set_stage("j1", "visual", "running", "Visual batch 2/5")  # an update, not a new run
    assert _stage(manager, "visual")["started_at"] is not None
    clock.advance(4)
    manager.set_stage("j1", "visual", "done")
    assert _stage(manager, "visual")["seconds"] == pytest.approx(7)
    assert _stage(manager, "visual")["started_at"] is None
    manager.set_stage("j1", "visual", "running")  # second video
    clock.advance(5)
    manager.set_stage("j1", "visual", "done")
    assert _stage(manager, "visual")["seconds"] == pytest.approx(12)


def test_a_cached_stage_adds_no_time(env):
    manager, clock = env
    clock.advance(10)
    manager.set_stage("j1", "audio", "done", "cached")
    assert _stage(manager, "audio")["seconds"] == 0


def test_ending_an_attempt_closes_its_running_stage(env):
    manager, clock = env
    manager.start_attempt("j1", "analysis")
    manager.set_stage("j1", "transcript", "running")
    clock.advance(8)
    manager.finish_attempt("j1", "cancelled")
    stage = _stage(manager, "transcript")
    assert stage["status"] == "cancelled" and stage["seconds"] == pytest.approx(8)
    manager.start_attempt("j1", "analysis")
    manager.set_stage("j1", "transcript", "running")
    clock.advance(2)
    manager.finish_attempt("j1", "failed")
    stage = _stage(manager, "transcript")
    assert stage["status"] == "error" and stage["seconds"] == pytest.approx(10)


def test_interrupted_attempts_are_cleared_without_counting_unknown_time(env):
    manager, clock = env
    manager.start_attempt("j1", "analysis")
    manager.set_stage("j1", "scenes", "running")
    clock.advance(500)  # the app was closed somewhere in here
    assert manager.recover_interrupted() == 1
    timing = manager.timing(_job(manager))
    assert timing["attempt_started_at"] is None and timing["total_seconds"] == 0
    assert timing["attempt_outcome"] == "interrupted"
    assert _stage(manager, "scenes")["status"] == "pending"


def test_serialized_job_carries_timing_and_server_time(env):
    manager, _clock = env
    manager.start_attempt("j1", "render")
    data = manager.serialize(_job(manager))
    assert data["timing"]["attempt_phase"] == "render"
    assert data["timing"]["server_time"].startswith("2026-10-14T12:00:00")
    assert data["timing"]["attempt_started_at"].startswith("2026-10-14T12:00:00")


def test_existing_databases_gain_the_new_columns_and_keep_old_times(tmp_path):
    path = tmp_path / "old.db"
    init_db(path)
    with sqlite3.connect(path) as conn:
        for column in ("attempt_started_at", "attempt_phase", "attempt_seconds", "attempt_outcome", "analysis_seconds", "render_seconds"):
            conn.execute(f"ALTER TABLE jobs DROP COLUMN {column}")
        conn.execute("ALTER TABLE job_stages DROP COLUMN seconds")
        # A job saved by the previous version: every required column filled, 75.5 s of analysis.
        values = {"id": "old", "status": "completed", "processing_seconds": 75.5}
        for _cid, name, kind, notnull, _default, _pk in conn.execute("PRAGMA table_info(jobs)"):
            if notnull and name not in values:
                values[name] = "2026-10-01 10:00:00" if "DATETIME" in kind.upper() else (0 if kind.upper() in {"FLOAT", "INTEGER", "BOOLEAN"} else "")
        names = ", ".join(values)
        conn.execute(f"INSERT INTO jobs ({names}) VALUES ({', '.join('?' for _ in values)})", list(values.values()))
    init_db(path)
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT processing_seconds, analysis_seconds, render_seconds FROM jobs WHERE id='old'").fetchone()
        stage_columns = {r[1] for r in conn.execute("PRAGMA table_info(job_stages)")}
    assert row == (75.5, 75.5, 0)
    assert "seconds" in stage_columns


def test_worker_never_leaves_a_timer_running(env, monkeypatch):
    from app.workers import worker

    manager, clock = env
    monkeypatch.setattr(worker, "get_job_manager", lambda: manager)

    def broken(_app, job_id):
        manager.start_attempt(job_id, "analysis")
        clock.advance(3)
        raise RuntimeError("crashed before its own error handling")

    monkeypatch.setattr(worker, "_run_analysis", broken)
    with pytest.raises(RuntimeError):
        worker._tick(None)
    timing = manager.timing(_job(manager))
    assert timing["attempt_started_at"] is None
    assert timing["attempt_outcome"] == "failed"
    assert timing["total_seconds"] == pytest.approx(3)
