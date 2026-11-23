"""Pause and resume, and continuing after a crash or a power cut: files are written safely, the
slow steps save their progress as they go, a render reuses finished clips, a paused job stays
paused across restarts, and a resumed job waits for Ollama."""

from __future__ import annotations

import json
import re
import sys
import threading
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from app import create_app
from app.analysis import audio as audio_module
from app.analysis import score_source
from app.analysis.pipeline import (
    SEGMENT_SCORES_PARTIAL,
    TRANSCRIPT_PARTIAL,
    VISUAL_PARTIAL,
    PipelineProgress,
    VideoAnalyzer,
)
from app.config import Settings, get_settings, set_settings
from app.models.db_models import Job, get_session
from app.models.schemas import (
    ClipPlan,
    ClipPlanItem,
    HighlightConfig,
    RenderSettings,
    Transcript,
    TranscriptSegment,
    VideoMetadata,
)
from app.services import job_manager as jm
from app.services.job_manager import get_job_manager
from app.utils.atomic import atomic_write_json, part_path, read_json
from app.video.ffmpeg_runner import CancelledError
from app.video.renderer import CLIP_MANIFEST, Renderer
from app.workers import worker


# Safe writes


def test_atomic_write_keeps_the_old_file_when_the_write_is_cut_off(tmp_path, monkeypatch):
    path = tmp_path / "clip_plan.json"
    atomic_write_json(path, {"clips": [1]})

    def power_cut(*_args):
        raise OSError("power cut")

    monkeypatch.setattr("app.utils.atomic.os.replace", power_cut)
    with pytest.raises(OSError):
        atomic_write_json(path, {"clips": [1, 2]})
    assert read_json(path) == {"clips": [1]}
    assert [p.name for p in tmp_path.iterdir()] == ["clip_plan.json"], "no temporary file is left behind"


def test_read_json_treats_a_cut_off_file_as_missing(tmp_path):
    path = tmp_path / "sources.json"
    path.write_bytes(b'[{"id": "v0", "pa')
    assert read_json(path, []) == []
    path.write_bytes(b"\xff\xfe\x00garbage")
    assert read_json(path) is None
    assert read_json(tmp_path / "missing.json", {}) == {}


def test_part_path_keeps_the_extension_last():
    assert part_path(Path("source/audio.wav")) == Path("source/.audio.part.wav")


def test_cut_off_frame_is_not_reused(tmp_path):
    from app.video.ffmpeg_runner import FFmpegError

    class Runner:
        def __init__(self, fail: bool):
            self.cancel_event = threading.Event()
            self.fail = fail

        def extract_frame(self, source, out, stamp, duration=None):
            Path(out).write_bytes(b"partial")
            if self.fail:
                raise FFmpegError("stopped mid-file")

    scenes = [{"scene_id": 0, "start": 0.0, "end": 5.0, "duration": 5.0}]
    frames = tmp_path / "frames"
    frames.mkdir()
    analyzer = VideoAnalyzer(Settings(data_dir=tmp_path / "data"), Runner(fail=True), None)
    progress = PipelineProgress(lambda *_: None)
    frame_map = analyzer._extract_frames(tmp_path / "v.mp4", scenes, frames, 5.0, "lite", progress)
    assert frame_map == {"0": []}
    assert list(frames.iterdir()) == [], "the partial image is removed, not kept under the final name"

    analyzer.runner = Runner(fail=False)
    frame_map = analyzer._extract_frames(tmp_path / "v.mp4", scenes, frames, 5.0, "lite", progress)
    assert frame_map["0"] and all(Path(entry["image_path"]).exists() for entry in frame_map["0"])
    assert not any(p.name.startswith(".") for p in frames.iterdir())


# Analysis steps that save their progress


class StopAfter:
    """A local AI that answers both prompts and stops the job after a number of calls of one kind."""

    def __init__(self, runner=None, stop_after_vision: int | None = None, stop_after_text: int | None = None):
        self.runner = runner
        self.stop_after_vision = stop_after_vision
        self.stop_after_text = stop_after_text
        self.vision_calls: list[list[int]] = []
        self.text_calls = 0

    def role_options(self, role):
        return {"num_ctx": 16384, "num_thread": 4}

    def encode_image(self, path):
        return "image"

    def chat(self, model, prompt, images=None, schema=None, options=None, schema_first=True):
        if images:
            ids = [int(match) for match in re.findall(r"scene_id=(\d+)", prompt)]
            self.vision_calls.append(ids)
            if self.stop_after_vision is not None and len(self.vision_calls) >= self.stop_after_vision:
                self.runner.cancel_event.set()
            return {"scenes": [{"scene_id": scene_id, "visual_importance": 0.9} for scene_id in ids]}
        self.text_calls += 1
        if self.stop_after_text is not None and self.text_calls >= self.stop_after_text:
            self.runner.cancel_event.set()
        count = len(re.findall(r"^\d+\. \[", prompt, re.MULTILINE))
        return {"segments": [{"i": i, "importance": 0.8, "novelty": 0.7, "information_density": 0.6} for i in range(count)]}


class Runner:
    def __init__(self):
        self.cancel_event = threading.Event()

    def extract_frame(self, source, out, stamp, duration=None):
        Path(out).write_bytes(b"jpg")


def _transcript() -> Transcript:
    return Transcript(
        segments=[
            TranscriptSegment(start=0.5, end=4.0, text="Here is the main idea behind the whole product demo."),
            TranscriptSegment(start=5.5, end=9.0, text="And this is the result you get when everything works."),
        ]
    )


def _job_dir(tmp_path: Path) -> tuple[Path, dict]:
    job_dir = tmp_path / "job"
    (job_dir / "source").mkdir(parents=True)
    (job_dir / "source" / "audio.wav").write_bytes(b"RIFF")
    cache = {
        "metadata": {"duration": 10.0, "width": 640, "height": 360, "fps": 30.0},
        "scenes": [
            {"scene_id": 0, "start": 0.0, "end": 5.0, "duration": 5.0},
            {"scene_id": 1, "start": 5.0, "end": 10.0, "duration": 5.0},
        ],
        "silences": [],
    }
    for key, name in (("metadata", "metadata.json"), ("scenes", "scenes.json"), ("silences", "silences.json")):
        (job_dir / name).write_text(json.dumps(cache[key]), encoding="utf-8")
    return job_dir, cache


def _run(tmp_path, job_dir, ollama, runner, logs=None):
    from app.workers.worker import load_cache

    settings = Settings(
        data_dir=tmp_path / "data",
        ollama_vision_batch_size=1,
        ollama_text_batch_size=1,
        vision_scope="all",
        image_matching=False,
    )
    progress = PipelineProgress(lambda kind, payload: logs.append(payload["message"]) if logs is not None and kind == "log" else None)
    analyzer = VideoAnalyzer(settings, runner, ollama, vision_enabled=True, text_enabled=True)
    return analyzer.analyze(job_dir, job_dir / "source.mp4", HighlightConfig(), progress, load_cache(job_dir), plan=False)


@pytest.fixture
def speech(monkeypatch):
    monkeypatch.setattr("app.analysis.pipeline.transcribe_detailed", lambda *a, **k: (_transcript(), {"ok": True}))


def test_visual_ai_continues_after_the_scenes_it_finished(tmp_path, speech):
    job_dir, _ = _job_dir(tmp_path)
    runner = Runner()
    first = StopAfter(runner, stop_after_vision=1)
    first.runner = runner
    with pytest.raises(CancelledError):
        _run(tmp_path, job_dir, first, runner)
    saved = read_json(job_dir / VISUAL_PARTIAL)
    assert list(saved["scenes"]) == ["0"] and saved["scenes"]["0"]["score_source"] == score_source.AI
    assert not (job_dir / "visual_analysis.json").exists()

    second = StopAfter()
    logs: list[str] = []
    result = _run(tmp_path, job_dir, second, Runner(), logs)
    assert second.vision_calls == [[1]], "only the scene that was not finished is analyzed again"
    assert set(result["visual"]) == {"0", "1"}
    assert not score_source.needs_ai(result["visual"]["0"])
    assert read_json(job_dir / "visual_analysis.json").keys() == {"0", "1"}
    assert not (job_dir / VISUAL_PARTIAL).exists()
    assert any("analyzed before the interruption" in line for line in logs)


def test_saved_visual_progress_for_another_focus_is_not_used(tmp_path, speech):
    job_dir, _ = _job_dir(tmp_path)
    atomic_write_json(
        job_dir / VISUAL_PARTIAL,
        {"focus": ["pets"], "scenes": {"0": score_source.tag({"scene_id": 0, "visual_importance": 0.1})}},
    )
    client = StopAfter()
    _run(tmp_path, job_dir, client, Runner())
    assert sorted(ids[0] for ids in client.vision_calls) == [0, 1]


def test_visual_progress_is_kept_when_every_scene_was_already_finished(tmp_path, speech):
    job_dir, _ = _job_dir(tmp_path)
    scenes = {str(i): score_source.tag({"scene_id": i, "visual_importance": 0.7}) for i in (0, 1)}
    atomic_write_json(job_dir / VISUAL_PARTIAL, {"focus": [], "scenes": scenes})
    client = StopAfter()
    _run(tmp_path, job_dir, client, Runner())
    assert client.vision_calls == []
    assert read_json(job_dir / "visual_analysis.json") == scenes
    assert not (job_dir / VISUAL_PARTIAL).exists()


def test_transcript_scoring_continues_after_the_segments_it_finished(tmp_path, speech):
    job_dir, _ = _job_dir(tmp_path)
    runner = Runner()
    first = StopAfter(runner, stop_after_text=1)
    with pytest.raises(CancelledError):
        _run(tmp_path, job_dir, first, runner)
    saved = read_json(job_dir / SEGMENT_SCORES_PARTIAL)
    assert saved["scores"][0]["score_source"] == score_source.AI and saved["scores"][1] is None

    second = StopAfter()
    result = _run(tmp_path, job_dir, second, Runner())
    assert second.text_calls == 1
    assert all(item["score_source"] == score_source.AI for item in result["segment_scores"])
    assert len(read_json(job_dir / "segment_scores.json")) == 2
    assert not (job_dir / SEGMENT_SCORES_PARTIAL).exists()


def test_saved_scores_of_another_transcript_are_not_used(tmp_path, speech):
    job_dir, _ = _job_dir(tmp_path)
    atomic_write_json(job_dir / SEGMENT_SCORES_PARTIAL, {"key": "another transcript", "scores": [{"score": 1}, None]})
    client = StopAfter()
    _run(tmp_path, job_dir, client, Runner())
    assert client.text_calls == 2


def test_transcription_saves_progress_and_continues_from_it(tmp_path, monkeypatch):
    job_dir, _ = _job_dir(tmp_path)
    calls: list[dict] = []
    state = {"until": 4.0, "segments": [{"start": 0.5, "end": 4.0, "text": "First."}], "words": [], "language": "en"}

    def interrupted(*_args, **kwargs):
        calls.append(kwargs)
        kwargs["on_progress"](state)
        raise CancelledError("Job paused")

    monkeypatch.setattr("app.analysis.pipeline.transcribe_detailed", interrupted)
    with pytest.raises(CancelledError):
        _run(tmp_path, job_dir, None, Runner())
    assert read_json(job_dir / TRANSCRIPT_PARTIAL) == state
    assert "resume" not in calls[0] and callable(calls[0]["check_cancel"])

    def finish(*_args, **kwargs):
        calls.append(kwargs)
        return _transcript(), {"ok": True}

    monkeypatch.setattr("app.analysis.pipeline.transcribe_detailed", finish)
    logs: list[str] = []
    _run(tmp_path, job_dir, None, Runner(), logs)
    assert calls[1]["resume"] == state
    assert (job_dir / "transcript.json").exists() and not (job_dir / TRANSCRIPT_PARTIAL).exists()
    assert any("Continuing transcription from 00:04" in line for line in logs)


class FakeWhisper:
    def __init__(self, segments, stop_at: int | None = None):
        self.segments = segments
        self.audio = None
        self.options = None

    def transcribe(self, audio, **options):
        self.audio = audio
        self.options = options
        info = types.SimpleNamespace(language="de")
        return iter(self.segments), info


def _segment(start, end, text):
    word = types.SimpleNamespace(word=f" {text}", start=start, end=end)
    return types.SimpleNamespace(start=start, end=end, text=f" {text}", words=[word])


def test_whisper_skips_the_audio_it_already_wrote_down(monkeypatch):
    model = FakeWhisper([_segment(0.0, 2.0, "Second"), _segment(2.5, 4.0, "Third")])
    monkeypatch.setattr(audio_module, "_get_whisper_model", lambda *a: model)
    monkeypatch.setattr(audio_module, "_audio_after", lambda path, seconds: np.zeros(16000 * 5, dtype=np.float32))
    resume = {
        "until": 10.0,
        "segments": [{"start": 1.0, "end": 10.0, "text": "First"}],
        "words": [{"word": "First", "start": 1.0, "end": 10.0}],
        "language": "en",
    }
    transcript = audio_module._transcribe_with(Path("a.wav"), "small", "cpu", "int8", 4, 1, resume=resume)
    assert model.options["language"] == "en", "the language already heard is kept"
    assert [(s.start, s.end, s.text) for s in transcript.segments] == [
        (1.0, 10.0, "First"),
        (10.0, 12.0, "Second"),
        (12.5, 14.0, "Third"),
    ]
    assert [w.start for w in transcript.words] == [1.0, 10.0, 12.5]
    assert transcript.language == "en"


def test_whisper_starts_over_when_the_audio_cannot_be_skipped(monkeypatch):
    model = FakeWhisper([_segment(0.0, 2.0, "Only")])
    monkeypatch.setattr(audio_module, "_get_whisper_model", lambda *a: model)
    monkeypatch.setattr(audio_module, "_audio_after", lambda path, seconds: None)
    resume = {"until": 10.0, "segments": [{"start": 1.0, "end": 10.0, "text": "First"}], "language": "en"}
    transcript = audio_module._transcribe_with(Path("a.wav"), "small", "cpu", "int8", 4, 1, resume=resume)
    assert model.audio == "a.wav" and "language" not in model.options
    assert [s.text for s in transcript.segments] == ["Only"]


def test_whisper_reports_progress_and_stops_between_segments(monkeypatch):
    model = FakeWhisper([_segment(0.0, 2.0, "One"), _segment(2.0, 4.0, "Two"), _segment(4.0, 6.0, "Three")])
    monkeypatch.setattr(audio_module, "_get_whisper_model", lambda *a: model)
    monkeypatch.setattr(audio_module, "PROGRESS_SAVE_SECONDS", 0.0)
    saved: list[dict] = []
    checks = {"n": 0}

    def check_cancel():
        checks["n"] += 1
        if checks["n"] == 3:
            raise CancelledError("Job paused")

    with pytest.raises(CancelledError):
        audio_module._transcribe_with(
            Path("a.wav"), "small", "cpu", "int8", 4, 1, on_progress=saved.append, check_cancel=check_cancel
        )
    assert [state["until"] for state in saved] == [2.0, 4.0]
    assert [s["text"] for s in saved[-1]["segments"]] == ["One", "Two"]


def test_whisper_does_not_run_again_when_nothing_is_left(monkeypatch):
    model = FakeWhisper([])
    monkeypatch.setattr(audio_module, "_get_whisper_model", lambda *a: model)
    monkeypatch.setattr(audio_module, "_audio_after", lambda path, seconds: np.zeros(100, dtype=np.float32))
    resume = {"until": 59.9, "segments": [{"start": 0.0, "end": 59.9, "text": "All"}], "language": "en"}
    transcript = audio_module._transcribe_with(Path("a.wav"), "small", "cpu", "int8", 4, 1, resume=resume)
    assert model.audio is None
    assert [s.text for s in transcript.segments] == ["All"]


def test_a_stop_during_transcription_is_not_treated_as_a_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=object))

    def stopped(*_args, **_kwargs):
        raise CancelledError("Job paused")

    monkeypatch.setattr(audio_module, "_transcribe_with", stopped)
    with pytest.raises(CancelledError):
        audio_module.transcribe_detailed(Path("a.wav"), device="cpu")


# Render


class RenderRunner:
    def __init__(self):
        self.cancel_event = threading.Event()
        self.commands: list[list[str]] = []

    def run(self, cmd):
        self.commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"x" * 2000)

    def probe(self, path):
        return VideoMetadata(duration=20, width=1280, height=720, fps=30.0, avg_fps=30.0, has_audio=True, size_bytes=2000)


def _render(tmp_path: Path, runner: RenderRunner, clips: list[tuple[float, float]]):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x" * 2000)
    plan = ClipPlan(
        target_duration=10,
        estimated_duration=10,
        clips=[ClipPlanItem(start=start, end=end, reason="m", score=1) for start, end in clips],
    )
    meta = runner.probe(source)
    return Renderer(runner).render(source, plan, tmp_path / "out.mp4", RenderSettings(), tmp_path / "work", metadata=meta)


def _encoded(runner: RenderRunner) -> list[str]:
    return [Path(cmd[-1]).name for cmd in runner.commands if re.search(r"clip_\d+\.mp4$", cmd[-1])]


def test_render_reuses_clips_finished_before_an_interruption(tmp_path):
    first = RenderRunner()
    _render(tmp_path, first, [(0, 2), (4, 6)])
    assert _encoded(first) == ["clip_000.mp4", "clip_001.mp4"]
    assert set(read_json(tmp_path / "work" / CLIP_MANIFEST)) == {"clip_000.mp4", "clip_001.mp4"}

    again = RenderRunner()
    _render(tmp_path, again, [(0, 2), (4, 6)])
    assert _encoded(again) == [], "finished clips are not encoded again"

    edited = RenderRunner()
    _render(tmp_path, edited, [(0, 2), (4, 7)])
    assert _encoded(edited) == ["clip_001.mp4"], "a clip changed on the Review page is encoded again"


def test_a_clip_cut_off_mid_encode_is_encoded_again(tmp_path):
    class StopsOnSecondClip(RenderRunner):
        def run(self, cmd):
            if cmd[-1].endswith("clip_001.mp4"):
                Path(cmd[-1]).write_bytes(b"partial")
                raise CancelledError("Job paused")
            super().run(cmd)

    with pytest.raises(CancelledError):
        _render(tmp_path, StopsOnSecondClip(), [(0, 2), (4, 6)])
    work = tmp_path / "work"
    assert (work / "clip_000.mp4").exists() and not (work / "clip_001.mp4").exists()
    assert not (work / "encoding" / "clip_001.mp4").exists()

    resumed = RenderRunner()
    _render(tmp_path, resumed, [(0, 2), (4, 6)])
    assert _encoded(resumed) == ["clip_001.mp4"]


# Pause, resume, and restarts


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    previous = get_settings()
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        ollama_base_url="http://127.0.0.1:9",
        enable_whisper=False,
        enable_visual_analysis=False,
    )
    settings.ensure_directories()
    set_settings(settings)
    app = create_app(settings, start_background=False)
    app.config["TESTING"] = True
    monkeypatch.setattr(jm, "_manager", None)
    session = get_session()
    session.add(Job(id="j1", status="queued", original_filename="v.mp4", input_path=str(tmp_path / "v.mp4")))
    session.commit()
    session.close()
    (settings.jobs_dir / "j1").mkdir(parents=True, exist_ok=True)
    yield app
    set_settings(previous)


def test_a_waiting_job_can_be_paused_and_resumed(app_env):
    client = app_env.test_client()
    paused = client.post("/api/jobs/j1/pause")
    assert paused.status_code == 200 and paused.get_json()["status"] == "paused"
    worker._tick(app_env)
    assert get_job_manager().get("j1").status == "paused", "the worker does not pick up a paused job"

    resumed = client.post("/api/jobs/j1/resume")
    assert resumed.status_code == 200 and resumed.get_json()["status"] == "queued"
    assert client.post("/api/jobs/j1/resume").status_code == 409, "only a paused job can be resumed"


def test_finished_jobs_cannot_be_paused(app_env):
    get_job_manager().update("j1", status="completed")
    assert app_env.test_client().post("/api/jobs/j1/pause").status_code == 409
    assert app_env.test_client().post("/api/jobs/nope/pause").status_code == 404


class _NoOllama:
    def health(self):
        return {"connected": False}


def _stub_analysis(monkeypatch, analyze_sources):
    class Analyzer:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_sources(self, job_dir, sources, config, progress, cache):
            return analyze_sources(progress)

    monkeypatch.setattr(worker, "VideoAnalyzer", Analyzer)
    monkeypatch.setattr(worker, "get_ollama_client", lambda settings: _NoOllama())
    monkeypatch.setattr(worker, "_prepare_all_sources", lambda *args: ([], args[-1]))
    monkeypatch.setattr(worker, "_run_plan", lambda *args: None)


def test_pausing_a_running_analysis_ends_it_as_paused(app_env, monkeypatch):
    manager = get_job_manager()

    def pause_mid_run(progress):
        manager.pause("j1")
        raise CancelledError("Job cancelled")

    _stub_analysis(monkeypatch, pause_mid_run)
    worker._run_analysis(app_env, "j1")
    job = manager.get("j1")
    assert (job.status, job.analysis_status, job.cancelled) == ("paused", "pending", False)
    assert job.attempt_outcome == "paused" and job.attempt_started_at is None
    assert not manager.pause_requested("j1")
    log = (manager.job_dir("j1") / "job.log").read_text(encoding="utf-8")
    assert "Analysis paused" in log


def test_a_stop_is_still_a_stop(app_env, monkeypatch):
    manager = get_job_manager()

    def stop_mid_run(progress):
        manager.cancel("j1")
        raise CancelledError("Job cancelled")

    _stub_analysis(monkeypatch, stop_mid_run)
    worker._run_analysis(app_env, "j1")
    assert manager.get("j1").status == "cancelled"


def test_a_pause_asked_for_while_the_job_is_picked_up_is_not_lost(app_env, monkeypatch):
    manager = get_job_manager()
    seen = {}

    def check(progress):
        seen["stopped"] = manager.cancel_event("j1").is_set()
        raise CancelledError("Job cancelled")

    _stub_analysis(monkeypatch, check)
    manager.pause("j1")
    worker._run_analysis(app_env, "j1")
    assert seen["stopped"] is True
    assert manager.get("j1").status == "paused"


def test_requeueing_clears_an_old_pause_request(app_env):
    manager = get_job_manager()
    manager.pause("j1")
    manager.update("j1", status="queued", render_status="queued")
    assert not manager.pause_requested("j1")


def test_paused_jobs_stay_paused_after_a_restart(app_env):
    manager = get_job_manager()
    manager.update("j1", status="paused", analysis_status="running")
    session = get_session()
    session.add(Job(id="j2", status="analyzing", analysis_status="running", original_filename="b.mp4", input_path="b.mp4"))
    session.commit()
    session.close()
    worker._recover_orphaned_jobs()
    paused, interrupted = manager.get("j1"), manager.get("j2")
    assert (paused.status, paused.analysis_status) == ("paused", "pending")
    assert (interrupted.status, interrupted.analysis_status) == ("queued", "pending")


def test_an_interrupted_attempt_counts_the_time_it_ran(app_env, monkeypatch):
    clock = {"now": datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)}
    monkeypatch.setattr(jm, "utcnow", lambda: clock["now"])
    manager = get_job_manager()
    manager.set_stage("j1", "transcript", "running")
    manager.start_attempt("j1", "analysis")
    clock["now"] += timedelta(seconds=90)
    manager.touch_attempt("j1")
    clock["now"] += timedelta(hours=5)  # the power is off until here
    assert manager.recover_interrupted() == 1
    job = manager.get("j1")
    assert job.attempt_outcome == "interrupted"
    assert job.analysis_seconds == pytest.approx(90)
    assert job.attempt_started_at is None and job.attempt_seen_at is None
    stage = next(s for s in manager.stages("j1") if s["name"] == "transcript")
    assert stage["status"] == "pending" and stage["seconds"] == pytest.approx(90)


def test_a_resumed_job_waits_for_ollama(monkeypatch):
    monkeypatch.setattr(worker, "OLLAMA_RESUME_POLL_SECONDS", 0.001)
    monkeypatch.setattr(worker, "OLLAMA_RESUME_WAIT_SECONDS", 5.0)
    answers = iter([{"connected": False}, {"connected": True, "vision_ready": True}])

    class Client:
        def health(self):
            return next(answers)

    logs: list[str] = []
    progress = PipelineProgress(lambda kind, payload: logs.append(payload["message"]) if kind == "log" else None)
    health = worker._wait_for_ollama(Client(), {"connected": False}, threading.Event(), progress)
    assert health["connected"] is True
    assert any("Waiting for Ollama" in line for line in logs) and "Ollama is ready" in logs


def test_the_wait_for_ollama_ends_on_pause_or_timeout(monkeypatch):
    monkeypatch.setattr(worker, "OLLAMA_RESUME_POLL_SECONDS", 0.001)
    monkeypatch.setattr(worker, "OLLAMA_RESUME_WAIT_SECONDS", 0.05)
    progress = PipelineProgress(lambda *_: None)
    stopped = threading.Event()
    stopped.set()
    assert worker._wait_for_ollama(_NoOllama(), {"connected": False}, stopped, progress) == {"connected": False}
    assert worker._wait_for_ollama(_NoOllama(), {"connected": False}, threading.Event(), progress) == {"connected": False}


def test_only_an_interrupted_job_waits_for_ollama(app_env, monkeypatch):
    waited: list[str] = []
    monkeypatch.setattr(worker, "_wait_for_ollama", lambda client, health, cancel, progress: waited.append("x") or health)
    _stub_analysis(monkeypatch, lambda progress: (_ for _ in ()).throw(CancelledError("stop")))
    worker._run_analysis(app_env, "j1")
    assert waited == []
    get_job_manager().update("j1", status="queued", attempt_outcome="interrupted")
    worker._run_analysis(app_env, "j1")
    assert waited == ["x"]


def test_log_lines_cut_off_by_a_power_cut_do_not_break_the_log(app_env):
    path = get_job_manager().job_dir("j1") / "job.log"
    path.write_bytes(b'{"level": "INFO", "message": "ok"}\n{"level": "INFO", "mess\xe2\x80')
    response = app_env.test_client().get("/api/jobs/j1/logs")
    assert response.status_code == 200
    assert response.get_json()["logs"][0]["message"] == "ok"


def test_resume_goes_back_to_review_when_analysis_finished_as_the_pause_arrived(app_env):
    manager = get_job_manager()
    manager.update("j1", status="paused", analysis_status="completed", render_status="pending")
    assert manager.resume("j1").status == "planning"
    manager.update("j1", status="paused", analysis_status="completed", render_status="paused")
    job = manager.resume("j1")
    assert (job.status, job.render_status) == ("queued", "queued"), "a paused render continues"
