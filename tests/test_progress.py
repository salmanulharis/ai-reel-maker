"""Progress reporting: the job figure, counts inside steps, time left, FFmpeg progress,
downloads and the stored progress of a job."""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import threading
from datetime import timedelta
from pathlib import Path

import pytest

from app import create_app
from app.analysis.progress import PipelineProgress, ProgressModel, RenderProgress
from app.config import Settings, get_settings, set_settings
from app.models.db_models import Job, get_session, utcnow
from app.services import download_progress as dp
from app.services import job_manager as jm
from app.services.job_manager import get_job_manager
from app.video.ffmpeg_runner import FFmpegRunner, parse_progress_line, report_progress


class Clock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def _progress(clock: Clock | None = None, speeds: dict | None = None):
    events: list[dict] = []
    progress = PipelineProgress(
        lambda kind, payload: events.append(payload) if kind == "progress" else None,
        speeds=speeds,
        clock=clock or Clock(),
    )
    return progress, events


# The weighted figure


def test_model_counts_steps_by_weight_and_leaves_out_removed_ones():
    model = ProgressModel([("a", 1.0), ("b", 3.0), ("c", 6.0)])
    model.start("a", 0)
    model.finish("a", 1)
    assert model.fraction() == pytest.approx(0.1)
    model.remove("c")
    assert model.fraction() == pytest.approx(0.25), "a removed step no longer counts"
    model.set_fraction("b", 0.5, 2)
    assert model.fraction() == pytest.approx(0.625)


def test_steps_finished_earlier_are_taken_out_instead_of_holding_the_bar_back():
    model = ProgressModel([("a", 5.0), ("b", 5.0), ("c", 5.0)])
    model.start("b", 0)
    assert model.units["a"].state == "removed", "a step before the running one that never started was cached"
    model.finish("c", 1)
    assert model.units["c"].state == "removed", "done without running means it was finished earlier"


def test_a_long_step_moves_the_bar_more_than_a_short_one():
    progress, events = _progress()
    progress.source_duration(600)
    progress.stage("metadata", "running")
    progress.stage("metadata", "done")
    after_metadata = events[-1]["overall"]
    progress.stage("transcript", "running")
    progress.stage("transcript", "done")
    assert after_metadata < 1.0, "reading the video details is a tiny part of the work"
    assert events[-1]["overall"] - after_metadata > 10, "transcription is a large part"


def test_skipped_and_cached_steps_never_make_the_bar_go_back():
    progress, events = _progress()
    progress.source_duration(300)
    progress.stage("scenes", "running")
    progress.step("scenes", 150, 300, "time")
    progress.stage("scenes", "done")
    progress.stage("audio", "done", "cached")
    progress.stage("transcript", "skipped", "Speech is turned off")
    progress.stage("visual", "running")
    progress.stage("visual", "running", "again")
    figures = [event["overall"] for event in events]
    assert figures == sorted(figures), figures
    assert all(0 <= value <= 100 for value in figures)


def test_counts_inside_a_step_are_sent_with_the_update():
    clock = Clock()
    progress, events = _progress(clock)
    progress.stage("visual", "running")
    clock.now = 1.0
    progress.step("visual", 3, 12, "scenes")
    event = events[-1]
    assert (event["step_done"], event["step_total"], event["step_unit"]) == (3, 12, "scenes")
    assert event["status"] == "running" and event["stage"] == "visual"
    progress.stage("visual", "done")
    assert (events[-1]["step_done"], events[-1]["step_total"]) == (12, 12), "a finished step shows everything done"


def test_updates_inside_a_step_are_spaced_out_but_the_last_is_always_sent():
    clock = Clock()
    progress, events = _progress(clock)
    progress.stage("frames", "running")
    sent = len(events)
    for done in range(1, 50):
        clock.now += 0.01
        progress.step("frames", done, 50, "frames")
    assert len(events) - sent <= 2
    progress.step("frames", 50, 50, "frames")
    assert events[-1]["step_done"] == 50


def test_old_stage_counts_still_work():
    progress, events = _progress()
    progress.stage("visual", "running", "Visual batch 2/3", 2, 3)
    assert (events[-1]["index"], events[-1]["total"]) == (2, 3)


def test_several_videos_share_the_bar_by_length_and_it_only_moves_forward():
    clock = Clock()
    progress, events = _progress(clock)
    progress.configure_sources([100.0, 300.0], [60.0, None])
    for index in range(2):
        progress.begin_source(index)
        progress.source_duration(60.0 if index == 0 else 180.0)
        for stage in ("metadata", "scenes", "transcript", "frames", "visual"):
            clock.now += 1
            progress.stage(stage, "running")
            progress.stage(stage, "done")
        if index == 0:
            first_video = events[-1]["overall"]
    assert 10 < first_video < 50, "the first video is about a quarter of the length"
    for stage in ("repetition", "planning"):
        progress.stage(stage, "running")
        progress.stage(stage, "done")
    figures = [event["overall"] for event in events]
    assert figures == sorted(figures)
    assert figures[-1] == pytest.approx(100.0, abs=0.5)
    assert events[-1]["source_count"] == 2


def test_preparing_the_videos_keeps_its_progress_when_the_videos_are_set_up():
    progress, events = _progress()
    progress.step("prepare", 50, 100)
    before = events[-1]["overall"]
    progress.configure_sources([10.0, 10.0])
    assert progress.model.units[("prepare", None)].fraction == pytest.approx(0.5)
    progress.stage("prepare", "done")
    assert events[-1]["overall"] >= before


def test_planning_again_only_counts_the_editor_and_planning():
    progress, events = _progress()
    progress.only("director", "planning")
    progress.stage("director", "running")
    progress.stage("director", "done")
    assert events[-1]["overall"] > 90, "the AI editor is nearly all of the work"


def test_measured_speeds_change_the_expected_cost():
    slow, _ = _progress(speeds={"whisper_realtime_factor": 0.5})
    fast, _ = _progress(speeds={"whisper_realtime_factor": 20.0})
    for progress in (slow, fast):
        progress.source_duration(600)
    key = ("transcript", 0)
    assert slow.model.units[key].weight > 10 * fast.model.units[key].weight


# Time left


def test_time_left_is_hidden_until_reliable_then_follows_the_running_step():
    clock = Clock()
    progress, events = _progress(clock)
    progress.source_duration(600)
    progress.skip("metadata", "scenes", "audio", "silence", "frames", "sounds", "content", "visual")
    progress.stage("transcript", "running")
    clock.now = 2
    progress.step("transcript", 1, 600, "time")
    assert events[-1]["eta"] is None, "too early to say"
    clock.now = 60
    progress.step("transcript", 300, 600, "time")
    eta = events[-1]["eta"]
    assert eta is not None and 40 < eta < 120, eta
    clock.now = 90
    progress.step("transcript", 450, 600, "time")
    assert events[-1]["eta"] < eta, "time left goes down as the step advances"


def test_render_progress_weights_clips_by_length_and_counts_reused_clips_at_once():
    clock = Clock()
    events: list[dict] = []
    tracker = RenderProgress([[10.0, 30.0]], music=True, emit=events.append, clock=clock)
    tracker.clip(0, 0, 1.0)
    assert events[-1]["step_done"] == 1 and events[-1]["step_unit"] == "clips"
    first = events[-1]["overall"]
    assert 15 < first < 30, "a reused 10 s clip of 40 s plus music"
    clock.now = 1
    tracker.clip(0, 1, 0.5)
    tracker.clip(0, 1, 1.0)
    tracker.clip(0, 1, 0.2)
    assert tracker._clips_done == 2, "a clip encoded again after a hardware failure is not counted twice"
    tracker.music(0, 1.0)
    assert events[-1]["overall"] == pytest.approx(100.0)


def test_render_without_music_finishes_with_the_clips():
    events: list[dict] = []
    tracker = RenderProgress([[5.0], [5.0]], music=False, emit=events.append, clock=Clock())
    tracker.clip(0, 0, 1.0)
    assert events[-1]["overall"] == pytest.approx(50.0)
    tracker.clip(1, 0, 1.0)
    assert events[-1]["overall"] == pytest.approx(100.0)


# FFmpeg progress


def test_ffmpeg_progress_lines_are_read_as_a_share():
    assert parse_progress_line("out_time_us=5000000\n", 10) == pytest.approx(0.5)
    assert parse_progress_line("out_time_ms=20000000", 10) == 1.0, "out_time_ms is microseconds too, capped at 1"
    assert parse_progress_line("out_time_us=N/A", 10) is None
    assert parse_progress_line("progress=end", 10) == 1.0
    assert parse_progress_line("frame=12", 10) is None
    assert parse_progress_line("out_time_us=100", 0) is None


def test_report_progress_leaves_other_runners_alone():
    class Fake:
        def run(self, cmd):
            return cmd

    seen = []
    with report_progress(Fake(), 10, seen.append):
        assert Fake().run(["ffmpeg", "-i", "x"]) == ["ffmpeg", "-i", "x"]
    assert seen == []


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is not installed")
def test_ffmpeg_commands_report_progress_and_still_return_their_output():
    runner = FFmpegRunner()
    seen: list[float] = []
    command = ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "aevalsrc=0:d=2", "-af", "silencedetect=d=0.5", "-f", "null", "-"]
    with report_progress(runner, 2.0, seen.append):
        result = runner.run(command)
    assert seen and seen[-1] == 1.0
    assert seen == sorted(seen)
    assert "silence_start" in result.stderr, "stderr (where silencedetect writes) is still collected"
    # Outside the block nothing is reported.
    reported = len(seen)
    runner.run(command)
    assert len(seen) == reported


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is not installed")
def test_progress_is_only_reported_on_the_thread_that_asked_for_it():
    runner = FFmpegRunner()
    seen: list[float] = []
    command = ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "sine=duration=1", "-f", "null", "-"]
    with report_progress(runner, 1.0, seen.append):
        other = threading.Thread(target=lambda: runner.run(command))
        other.start()
        other.join()
    assert seen == [], "parallel frame extraction threads do not report into the step"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg is not installed")
def test_a_failing_command_with_progress_still_raises():
    from app.video.ffmpeg_runner import FFmpegError

    runner = FFmpegRunner()
    with report_progress(runner, 1.0, lambda share: None), pytest.raises(FFmpegError):
        runner.run(["ffmpeg", "-hide_banner", "-i", "missing-file.mp4", "-f", "null", "-"])


# Downloads


def test_ollama_layers_add_up_to_one_bar():
    layers = dp.LayeredProgress()
    first = layers.update("pulling a", "sha256:a", 500, 1000)
    assert (first["bytes_done"], first["bytes_total"]) == (500, 1000)
    layers.update("pulling a", "sha256:a", 1000, 1000)
    second = layers.update("pulling b", "sha256:b", 10, 20)
    assert (second["bytes_done"], second["bytes_total"]) == (1010, 1020), "a new layer does not restart the bar"
    assert layers.update("verifying sha256 digest", "", 0, 0)["phase"] == "verifying"
    assert dp.ollama_phase("pulling manifest") == "starting"
    assert dp.ollama_phase("writing manifest") == "finishing"


def test_rate_meter_waits_for_a_steady_speed():
    clock = Clock()
    meter = dp.RateMeter(clock)
    assert meter.fields(0, 1000)["eta"] is None
    clock.now = 0.5
    assert meter.fields(100, 1000)["speed"] == 0, "too early to promise a speed"
    clock.now = 2.0
    fields = meter.fields(400, 1000)
    assert fields["speed"] > 0 and fields["eta"] == pytest.approx(600 / fields["speed"], rel=0.01)
    assert dp.RateMeter(clock).fields(5, 0)["eta"] is None, "no time left without a size"


def test_hub_watcher_counts_only_new_bytes(tmp_path, monkeypatch):
    folder = tmp_path / "models--org--name"
    (folder / "blobs").mkdir(parents=True)
    (folder / "blobs" / "old").write_bytes(b"x" * 50)
    monkeypatch.setattr(dp, "hub_cache_folder", lambda repo: folder)
    updates: list[dict] = []
    watcher = dp.HubWatcher("org/name", 300, updates.append)
    (folder / "blobs" / "new.incomplete").write_bytes(b"y" * 120)
    assert watcher.written() == 120
    (folder / "blobs" / "new.incomplete").rename(folder / "blobs" / "new")
    assert watcher.written() == 120, "the finished file is the same download"
    (folder / "blobs" / "more").write_bytes(b"z" * 500)
    assert watcher.written() == 300, "never more than the expected total"
    with watcher:
        pass
    assert updates[0]["phase"] == "downloading"


def test_pull_status_is_one_bar_with_the_old_fields_kept(monkeypatch):
    from app.routes import system

    class Client:
        def pull_model(self, name, on_progress=None):
            on_progress({"status": "pulling a", "digest": "a", "completed": 100, "total": 100})
            on_progress({"status": "pulling b", "digest": "b", "completed": 5, "total": 50})

    monkeypatch.setattr(system, "get_ollama_client", lambda *a, **k: Client())
    seen = {}

    original = system._pull_state

    def remember(state, status, fields, error=None):
        result = original(state, status, fields, error)
        if state == "running":
            seen.update(result)
        return result

    monkeypatch.setattr(system, "_pull_state", remember)
    system._pull_one("m:1")
    assert (seen["bytes_done"], seen["bytes_total"]) == (105, 150)
    assert (seen["completed"], seen["total"]) == (105, 150), "older pages read completed/total"
    done = system._pull_status["m:1"]
    assert done["state"] == "done" and done["bytes_done"] == done["bytes_total"] == 150


def test_hub_model_download_reports_bytes(monkeypatch):
    from app.ai import onnx_models

    model = onnx_models.HubModel("org/name", ("a.onnx",), 10, "test model")
    monkeypatch.setattr(onnx_models, "hub_missing_bytes", lambda repo, files=(): 0)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name: None)
    assert model.begin()
    model.download()
    status = model.status(installed=False, runtime=True)
    assert status["state"] == "done" and status["files_done"] == 1
    assert {"bytes_done", "bytes_total", "speed", "eta", "phase"} <= set(status)


# Stored progress of a job


@pytest.fixture
def job_env(tmp_path, monkeypatch):
    previous = get_settings()
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_directories()
    set_settings(settings)
    app = create_app(settings, start_background=False)
    monkeypatch.setattr(jm, "_manager", None)
    session = get_session()
    session.add(Job(id="j1", status="analyzing", original_filename="v.mp4", input_path=str(tmp_path / "v.mp4")))
    session.commit()
    session.close()
    (settings.jobs_dir / "j1").mkdir(parents=True, exist_ok=True)
    yield app
    set_settings(previous)


def test_progress_is_stored_with_the_stage_counts_and_time_left(job_env):
    manager = get_job_manager()
    manager.start_attempt("j1", "analysis")
    manager.record_progress("j1", 42.5, 120.0, "visual", "running", "", 3, 12, "scenes", 1, 2)
    data = manager.serialize(manager.get("j1"))
    assert data["progress"] == 42.5 and data["source_index"] == 1 and data["source_count"] == 2
    assert 110 <= data["eta_seconds"] <= 120
    stage = next(item for item in data["stages"] if item["name"] == "visual")
    assert (stage["status"], stage["done"], stage["total"], stage["unit"]) == ("running", 3, 12, "scenes")

    # The time left counts down from when it was estimated.
    session = get_session()
    job = session.get(Job, "j1")
    job.progress_at = utcnow() - timedelta(seconds=100)
    session.commit()
    session.close()
    assert manager.serialize(manager.get("j1"))["eta_seconds"] <= 20

    manager.set_stage("j1", "visual", "done", "", 0, 12, "scenes")
    stage = next(item for item in manager.stages("j1") if item["name"] == "visual")
    assert stage["done"] == stage["total"] == 12
    manager.finish_attempt("j1", "completed")
    assert manager.serialize(manager.get("j1"))["eta_seconds"] is None, "no time left once nothing runs"


def test_preparing_is_listed_first_when_it_runs(job_env):
    manager = get_job_manager()
    manager.prepare_stages("j1", "manual")
    names = [item["name"] for item in manager.stages("j1")]
    assert "prepare" not in names, "jobs without a crop do not show the step"
    manager.record_progress("j1", 5, None, "prepare", "running", "v.mp4", 50, 100, "")
    assert manager.stages("j1")[0]["name"] == "prepare"


def test_older_databases_get_the_new_columns(tmp_path):
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE jobs (id VARCHAR(40) PRIMARY KEY, status VARCHAR(32), progress FLOAT, processing_seconds FLOAT)")
        conn.execute("CREATE TABLE job_stages (id INTEGER PRIMARY KEY, job_id VARCHAR(40), name VARCHAR(64))")
        conn.execute("INSERT INTO job_stages (job_id, name) VALUES ('old', 'scenes')")
    previous = get_settings()
    settings = Settings(data_dir=tmp_path / "data", database_path=db)
    settings.ensure_directories()
    try:
        create_app(settings, start_background=False)
    finally:
        set_settings(previous)
    with sqlite3.connect(db) as conn:
        jobs = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        stages = {row[1] for row in conn.execute("PRAGMA table_info(job_stages)")}
        kept = conn.execute("SELECT name, done, total, unit FROM job_stages").fetchall()
    assert {"eta_seconds", "progress_at", "source_index", "source_count"} <= jobs
    assert {"seconds", "done", "total", "unit"} <= stages
    assert kept == [("scenes", 0.0, 0.0, "")], "existing rows keep their data"


def test_scene_detection_reports_how_far_it_read(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is not installed")
    pytest.importorskip("scenedetect")
    from app.analysis.scenes import detect_scenes

    video = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=3:size=64x64:rate=10", str(video)],
        check=True,
    )
    seen: list[float] = []
    scenes = detect_scenes(video, runner=FFmpegRunner(), on_progress=seen.append)
    assert scenes
    assert all(0 <= share <= 1 for share in seen)
