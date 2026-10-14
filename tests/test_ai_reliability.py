"""Local AI reliability: Ollama connection fallback, error handling, readiness checks,
batched scoring, and reuse of cached fallback scores."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pytest
import requests

from app.ai import ollama as ollama_module
from app.ai.ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    OllamaPullError,
    OllamaTruncatedError,
    fallback_roots,
    predict_budget,
    version_problem,
    version_tuple,
)
from app.analysis import score_source
from app.analysis.pipeline import PipelineProgress, VideoAnalyzer, align_segment_scores
from app.analysis.visual import analyze_scenes_visual
from app.config import Settings
from app.models.schemas import HighlightConfig, Transcript, TranscriptSegment
from app.services.ai_test import explain_ai_error
from app.services.model_catalog import recommend_setup
from app.services.ollama_runtime import _models_to_warm


class FakeResponse:
    def __init__(self, status: int = 200, body=None, lines=None):
        self.status_code = status
        self.ok = status < 400
        self._body = body
        self._lines = lines or []
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(str(self.status_code))

    def iter_lines(self):
        yield from self._lines


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(ollama_module.time, "sleep", lambda _seconds: None)


def _client(**overrides) -> OllamaClient:
    values = {"ollama_base_url": "http://127.0.0.1:11434", "ollama_max_retries": 2, **overrides}
    return OllamaClient(Settings(**values))


def _patch_session(monkeypatch, handler):
    calls: list[dict] = []

    def fake_request(self, method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return handler(url, kwargs)

    monkeypatch.setattr("app.ai.ollama.requests.Session.request", fake_request)
    return calls


# Connection fallback


def test_fallback_roots_cover_both_docker_directions(monkeypatch):
    monkeypatch.setattr(ollama_module, "running_in_docker", lambda: True)
    assert fallback_roots("http://127.0.0.1:11434") == [
        "http://host.docker.internal:11434",
        "http://ollama:11434",
    ]
    assert fallback_roots("http://host.docker.internal:11434") == ["http://127.0.0.1:11434"]
    monkeypatch.setattr(ollama_module, "running_in_docker", lambda: False)
    assert fallback_roots("http://127.0.0.1:11434") == []
    assert fallback_roots("http://192.168.1.50:11434") == []


def test_docker_client_with_localhost_url_reaches_host_ollama(monkeypatch):
    monkeypatch.setattr(ollama_module, "running_in_docker", lambda: True)
    client = _client()

    def handler(url, _kwargs):
        if "127.0.0.1" in url:
            raise requests.exceptions.ConnectionError("Connection refused")
        if url.endswith("/api/tags"):
            return FakeResponse(body={"models": [{"name": "qwen3:4b"}]})
        return FakeResponse(body={"version": "0.13.0", "models": []})

    _patch_session(monkeypatch, handler)
    health = client.health()
    assert health["connected"] is True
    assert client.root == "http://host.docker.internal:11434"
    assert health["base_url"] == "http://host.docker.internal:11434"
    assert client.is_local() is False


def test_fallback_trials_use_a_short_connect_timeout(monkeypatch):
    monkeypatch.setattr(ollama_module, "running_in_docker", lambda: True)
    client = _client()

    def handler(url, _kwargs):
        if "127.0.0.1" in url:
            raise requests.exceptions.ConnectionError("refused")
        return FakeResponse(body={"message": {"content": '{"ok": 1}'}})

    calls = _patch_session(monkeypatch, handler)
    client.chat("qwen3:4b", "hi")
    fallback_call = next(call for call in calls if "host.docker.internal" in call["url"])
    assert fallback_call["timeout"][0] <= 3


# Request and chat error handling


def test_read_timeout_is_not_retried(monkeypatch):
    client = _client(ollama_timeout=30)

    def handler(_url, _kwargs):
        raise requests.exceptions.ReadTimeout("read timed out")

    calls = _patch_session(monkeypatch, handler)
    with pytest.raises(OllamaError, match="timed out") as caught:
        client.chat("qwen3:4b", "hi")
    assert not isinstance(caught.value, OllamaConnectionError)
    assert len(calls) == 1


def test_unreachable_ollama_raises_connection_error_after_retries(monkeypatch):
    client = _client()

    def handler(_url, _kwargs):
        raise requests.exceptions.ConnectionError("refused")

    calls = _patch_session(monkeypatch, handler)
    with pytest.raises(OllamaConnectionError):
        client.chat("qwen3:4b", "hi")
    assert len(calls) == 3


@pytest.mark.parametrize(
    "status,error",
    [
        (404, "model 'qwen3:4b' not found, try pulling it first"),
        (500, "model requires more system memory (9.1 GiB) than is available (4.0 GiB)"),
    ],
)
def test_chat_does_not_retry_errors_that_repeat(monkeypatch, status, error):
    client = _client()
    calls = _patch_session(monkeypatch, lambda _url, _kwargs: FakeResponse(status, {"error": error}))
    with pytest.raises(OllamaError) as caught:
        client.chat("qwen3:4b", "hi", schema={"type": "object"}, schema_first=False)
    assert len(calls) == 1
    assert error in str(caught.value)


def test_chat_retries_a_runner_crash(monkeypatch):
    client = _client()
    replies = iter(
        [
            FakeResponse(500, {"error": "llama runner process has terminated"}),
            FakeResponse(body={"message": {"content": '{"ok": true}'}}),
        ]
    )
    calls = _patch_session(monkeypatch, lambda _url, _kwargs: next(replies))
    assert client.chat("qwen3:4b", "hi") == {"ok": True}
    assert len(calls) == 2


def test_cut_off_reply_raises_truncated_error_without_retrying(monkeypatch):
    client = _client()
    body = {"message": {"content": '{"segments":[{"i":0,"importance":0.5'}, "done_reason": "length"}
    calls = _patch_session(monkeypatch, lambda _url, _kwargs: FakeResponse(body=body))
    with pytest.raises(OllamaTruncatedError):
        client.chat("qwen3:4b", "hi", schema={"type": "object"}, schema_first=False)
    assert len(calls) == 1


def test_batch_images_are_never_reduced_to_one(monkeypatch):
    client = _client(ollama_max_retries=1)
    calls = _patch_session(monkeypatch, lambda _url, _kwargs: FakeResponse(body={"message": {"content": "not json"}}))
    with pytest.raises(OllamaError):
        client.chat("qwen3-vl:4b", "scenes", images=["a", "b", "c"], schema={"type": "object"}, schema_first=False)
    assert calls
    assert all(len(call["json"]["messages"][0]["images"]) == 3 for call in calls)


def test_predict_budget_grows_with_batch_and_respects_context():
    assert predict_budget(512, 20, 40, 4096) == 864
    assert predict_budget(512, 4, 40, 4096) == 512
    assert predict_budget(256, 40, 80, 2048) == 1024


# Runner options shared by warm-up, probes, and real calls


def test_warm_probe_and_role_options_match(monkeypatch):
    client = _client(
        ollama_vision_model="qwen3-vl:4b",
        ollama_vision_num_ctx=2048,
        ollama_text_num_ctx=4096,
        ollama_num_thread=6,
    )
    body = {"message": {"content": "ready"}, "eval_count": 1, "eval_duration": 1000}
    calls = _patch_session(monkeypatch, lambda _url, _kwargs: FakeResponse(body=body))
    client.warm_model("qwen3-vl:4b")
    client.probe_chat("qwen3-vl:4b", "color?", role="vision")
    client.warm_model("qwen3:4b")
    vision_warm, vision_probe, text_warm = (call["json"]["options"] for call in calls)
    assert vision_warm["num_ctx"] == vision_probe["num_ctx"] == 2048
    assert text_warm["num_ctx"] == 4096
    assert vision_warm["num_thread"] == vision_probe["num_thread"] == 6
    assert client.role_options("vision") == {"num_ctx": 2048, "num_thread": 6}


def test_warm_up_loads_the_model_the_job_uses_first():
    settings = Settings(ollama_vision_model="qwen3-vl:4b", ollama_text_model="qwen3:4b")
    both = {"models": ["qwen3-vl:4b", "qwen3:4b"], "vision_ready": True}
    # Transcript scoring runs first (it shortlists scenes for the vision model).
    assert _models_to_warm(settings, both, warm_both=False) == [("qwen3:4b", "text")]
    assert _models_to_warm(settings, both, warm_both=True) == [("qwen3:4b", "text"), ("qwen3-vl:4b", "vision")]
    text_only = {"models": ["qwen3:4b"], "vision_ready": False}
    assert _models_to_warm(settings, text_only, warm_both=False) == [("qwen3:4b", "text")]
    settings.enable_visual_analysis = False
    assert _models_to_warm(settings, both, warm_both=False) == [("qwen3:4b", "text")]


# Model readiness


def _health_handler(version: str, capabilities):
    def handler(url, _kwargs):
        if url.endswith("/api/version"):
            return FakeResponse(body={"version": version})
        if url.endswith("/api/tags"):
            return FakeResponse(
                body={"models": [{"name": "qwen3-vl:4b", "digest": "abc"}, {"name": "qwen3:4b", "digest": "def"}]}
            )
        if url.endswith("/api/ps"):
            return FakeResponse(body={"models": []})
        if url.endswith("/api/show"):
            return FakeResponse(body={} if capabilities is None else {"capabilities": capabilities})
        raise AssertionError(url)

    return handler


def test_vision_model_without_image_support_is_not_ready(monkeypatch):
    client = _client(ollama_vision_model="qwen3:4b", ollama_text_model="qwen3:4b")
    _patch_session(monkeypatch, _health_handler("0.13.0", ["completion"]))
    health = client.health()
    assert health["vision_present"] is True
    assert health["vision_capable"] is False
    assert health["vision_ready"] is False
    assert any("cannot read images" in warning for warning in health["warnings"])


def test_old_ollama_blocks_qwen3_vl(monkeypatch):
    client = _client(ollama_vision_model="qwen3-vl:4b", ollama_text_model="qwen3:4b")
    _patch_session(monkeypatch, _health_handler("0.11.4", ["completion", "vision"]))
    health = client.health()
    assert health["vision_capable"] is True
    assert health["vision_ready"] is False
    assert any("too old" in warning for warning in health["warnings"])
    assert "newer Ollama" in explain_ai_error(health["warnings"][0])


def test_unknown_capabilities_do_not_block_vision(monkeypatch):
    client = _client(ollama_vision_model="qwen3-vl:4b", ollama_text_model="qwen3:4b")
    calls = _patch_session(monkeypatch, _health_handler("0.13.0", None))
    health = client.health()
    assert health["vision_capable"] is None
    assert health["vision_ready"] is True
    assert health["warnings"] == []
    client.health()
    assert sum(1 for call in calls if call["url"].endswith("/api/show")) == 1


def test_version_parsing():
    assert version_tuple("0.12.7") == (0, 12, 7)
    assert version_tuple("0.13.1-rc2") == (0, 13, 1)
    assert version_tuple("0.0.0") is None
    assert version_tuple(None) is None
    assert version_problem("0.12.6", "qwen3-vl:8b")
    assert version_problem("0.12.7", "qwen3-vl:8b") is None
    assert version_problem("0.1.0", "qwen3:4b") is None
    assert version_problem("dev", "qwen3-vl:8b") is None


# Model downloads


def test_pull_error_in_stream_fails_the_download(monkeypatch):
    client = _client()
    lines = [b'{"status":"pulling manifest"}', b'{"error":"pull model manifest: file does not exist"}']
    _patch_session(monkeypatch, lambda _url, _kwargs: FakeResponse(lines=lines))
    with pytest.raises(OllamaPullError, match="does not exist"):
        client.pull_model("qwen3-vl:404b")
    assert "Check the model name" in explain_ai_error("pull model manifest: file does not exist")


def test_pull_without_success_fails_even_without_progress_callback(monkeypatch):
    client = _client()
    _patch_session(monkeypatch, lambda _url, _kwargs: FakeResponse(lines=[b'{"status":"downloading"}']))
    with pytest.raises(OllamaPullError):
        client.pull_model("qwen3-vl:4b")


def test_refused_pull_does_not_retry_through_the_cli(monkeypatch):
    from app.routes import system

    class RefusingClient:
        def pull_model(self, name, on_progress=None):
            raise OllamaPullError("pull model manifest: file does not exist")

        def is_local(self):
            return True

    def fail_run(*_args, **_kwargs):
        raise AssertionError("the ollama CLI should not run")

    monkeypatch.setattr(system, "get_ollama_client", lambda: RefusingClient())
    monkeypatch.setattr(system.shutil, "which", lambda _name: "ollama")
    monkeypatch.setattr(system.subprocess, "run", fail_run)
    system._pull_one("qwen3-vl:404b")
    assert system._pull_status["qwen3-vl:404b"]["state"] == "error"


def test_disk_check_is_skipped_for_ollama_on_another_machine(monkeypatch):
    from app.routes import system

    class RemoteClient:
        def is_local(self):
            return False

    monkeypatch.setattr(system, "get_ollama_client", lambda: RemoteClient())
    assert system._disk_problem("qwen3-vl:30b") is None


# Whisper models


def test_large_gpus_default_to_multilingual_whisper():
    assert recommend_setup(16, 32, 15, 8)["whisper_model"] == "large-v3-turbo"
    for vram, free in ((16, 15), (24, 20), (12, 11)):
        assert recommend_setup(vram, 32, free, 8)["whisper_model"] != "distil-large-v3"


def test_whisper_cache_check_uses_each_models_repo(tmp_path, monkeypatch):
    from app.analysis import audio

    monkeypatch.setattr(audio, "_hf_hub_cache", lambda: tmp_path)
    assert audio.whisper_repo("distil-large-v3") == "Systran/faster-distil-whisper-large-v3"
    assert audio.whisper_repo("large-v3-turbo") == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    assert audio.whisper_repo("base") == "Systran/faster-whisper-base"

    distil = tmp_path / "models--Systran--faster-distil-whisper-large-v3" / "snapshots" / "rev1"
    distil.mkdir(parents=True)
    assert audio.whisper_model_cached("distil-large-v3") is False
    (distil / "model.bin").write_bytes(b"weights")
    assert audio.whisper_model_cached("distil-large-v3") is True
    assert audio.whisper_model_cached("base") is False


# Transcript score alignment


def test_scores_follow_the_index_not_the_reply_order():
    rows = [{"i": 1, "importance": 0.9}, {"i": 0, "importance": 0.1}]
    assert align_segment_scores(rows, 2) == [rows[1], rows[0]]


def test_missing_rows_leave_gaps_instead_of_shifting_scores():
    rows = [{"i": 0, "importance": 0.4}, {"i": 2, "importance": 0.8}]
    assert align_segment_scores(rows, 3) == [rows[0], None, rows[1]]


def test_rows_without_indexes_are_used_only_when_counts_match():
    rows = [{"importance": 0.4}, {"importance": 0.8}]
    assert align_segment_scores(rows, 2) == rows
    assert align_segment_scores(rows, 3) == [None, None, None]
    assert align_segment_scores(None, 2) == [None, None]


# Visual scoring


class FakeVisionClient:
    def __init__(self, fail_with: Exception | None = None, max_images: int = 99):
        self.fail_with = fail_with
        self.max_images = max_images
        self.calls: list[int] = []

    def role_options(self, role):
        return {"num_ctx": 16384, "num_thread": 4}  # room for a 4-image batch

    def encode_image(self, path):
        return "image"

    def chat(self, model, prompt, images=None, schema=None, options=None, schema_first=True):
        self.calls.append(len(images or []))
        if self.fail_with:
            raise self.fail_with
        if len(images or []) > self.max_images:
            raise OllamaTruncatedError("cut off")
        ids = [int(match) for match in re.findall(r"scene_id=(\d+)", prompt)]
        return {"scenes": [{"scene_id": scene_id, "visual_importance": 0.9} for scene_id in ids]}


def _visual_inputs(tmp_path: Path, count: int):
    scenes = [{"scene_id": i, "start": i * 5.0, "end": i * 5.0 + 5.0, "duration": 5.0} for i in range(count)]
    frames = {}
    for scene in scenes:
        image = tmp_path / f"scene_{scene['scene_id']}.jpg"
        image.write_bytes(b"jpg")
        frames[str(scene["scene_id"])] = [{"timestamp": scene["start"] + 1, "image_path": str(image)}]
    return scenes, frames


def test_cut_off_visual_batch_is_split_until_it_fits(tmp_path):
    scenes, frames = _visual_inputs(tmp_path, 4)
    client = FakeVisionClient(max_images=1)
    settings = Settings(ollama_vision_batch_size=4)
    results, _errors, reasons = analyze_scenes_visual(client, "qwen3-vl:4b", scenes, frames, 20.0, settings=settings)
    assert reasons == {}
    assert all(results[i].visual_importance == pytest.approx(0.9) for i in range(4))
    assert client.calls == [4, 2, 1, 1, 2, 1, 1]


def test_lost_connection_stops_visual_calls_for_the_job(tmp_path):
    scenes, frames = _visual_inputs(tmp_path, 6)
    client = FakeVisionClient(fail_with=OllamaConnectionError("refused"))
    settings = Settings(ollama_vision_batch_size=2)
    results, errors, reasons = analyze_scenes_visual(client, "qwen3-vl:4b", scenes, frames, 30.0, settings=settings)
    assert client.calls == [2]
    assert len(results) == 6
    assert set(reasons.values()) == {score_source.AI_FAILED}
    assert any("stopped responding" in message for message in errors)


def test_visual_fallback_reasons_separate_retryable_cases(tmp_path):
    scenes, frames = _visual_inputs(tmp_path, 2)
    scenes.append({"scene_id": 9, "start": 10.0, "end": 10.4, "duration": 0.4})
    _results, _errors, reasons = analyze_scenes_visual(
        None, "qwen3-vl:4b", scenes[:2], frames, 20.0, settings=Settings()
    )
    assert set(reasons.values()) == {score_source.AI_UNAVAILABLE}
    _results, _errors, reasons = analyze_scenes_visual(
        FakeVisionClient(), "qwen3-vl:4b", scenes, frames, 20.0, settings=Settings()
    )
    assert reasons == {9: score_source.SHORT_SCENE}


# Pipeline: failed transcription and cached fallback scores


class FakeRunner:
    def __init__(self):
        self.cancel_event = threading.Event()

    def extract_frame(self, source, out, stamp, duration=None):
        Path(out).write_bytes(b"jpg")


class FakeLocalAI(FakeVisionClient):
    """Answers both the visual and the transcript prompts."""

    def chat(self, model, prompt, images=None, schema=None, options=None, schema_first=True):
        if images:
            return super().chat(model, prompt, images, schema, options, schema_first)
        count = len(re.findall(r"^\d+\. \[", prompt, re.MULTILINE))
        return {"segments": [{"i": i, "importance": 0.8, "novelty": 0.7, "information_density": 0.6} for i in range(count)]}


def _job(tmp_path: Path):
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


def _transcript() -> Transcript:
    return Transcript(
        segments=[
            TranscriptSegment(start=0.5, end=4.0, text="Here is the main idea behind the whole product demo."),
            TranscriptSegment(start=5.5, end=9.0, text="And this is the result you get when everything works."),
        ]
    )


def _analyze(tmp_path, job_dir, cache, ollama=None, logs=None):
    settings = Settings(data_dir=tmp_path / "data", ollama_vision_batch_size=4)
    progress = PipelineProgress(lambda kind, payload: logs.append(payload) if logs is not None and kind == "log" else None)
    analyzer = VideoAnalyzer(settings, FakeRunner(), ollama, vision_enabled=bool(ollama), text_enabled=bool(ollama))
    return analyzer.analyze(job_dir, job_dir / "source.mp4", HighlightConfig(), progress, cache, plan=False)


def test_failed_transcription_is_not_cached_and_runs_again(tmp_path, monkeypatch):
    from app.workers.worker import load_cache

    job_dir, cache = _job(tmp_path)
    failure = (Transcript(), {"ok": False, "error": "CUDA failed with error out of memory"})
    monkeypatch.setattr("app.analysis.pipeline.transcribe_detailed", lambda *args, **kwargs: failure)
    logs: list[dict] = []
    result = _analyze(tmp_path, job_dir, cache, logs=logs)
    assert not (job_dir / "transcript.json").exists()
    assert result["transcript"]["segments"] == []
    assert any("Transcription failed" in entry["message"] and "try again" in entry["message"] for entry in logs)

    success = (_transcript(), {"ok": True})
    monkeypatch.setattr("app.analysis.pipeline.transcribe_detailed", lambda *args, **kwargs: success)
    result = _analyze(tmp_path, job_dir, load_cache(job_dir))
    assert (job_dir / "transcript.json").exists()
    assert len(result["segment_scores"]) == 2


def test_fallback_scores_are_redone_once_ollama_is_available(tmp_path, monkeypatch):
    from app.workers.worker import load_cache

    job_dir, cache = _job(tmp_path)
    monkeypatch.setattr("app.analysis.pipeline.transcribe_detailed", lambda *args, **kwargs: (_transcript(), {"ok": True}))

    offline = _analyze(tmp_path, job_dir, cache)
    assert {entry["fallback_reason"] for entry in offline["visual"].values()} == {score_source.AI_UNAVAILABLE}
    assert {item["fallback_reason"] for item in offline["segment_scores"]} == {score_source.AI_UNAVAILABLE}

    still_offline = load_cache(job_dir)
    before = (job_dir / "segment_scores.json").stat().st_mtime_ns
    _analyze(tmp_path, job_dir, still_offline)
    assert (job_dir / "segment_scores.json").stat().st_mtime_ns == before

    online = _analyze(tmp_path, job_dir, load_cache(job_dir), ollama=FakeLocalAI())
    assert {entry["score_source"] for entry in online["visual"].values()} == {score_source.AI}
    assert {item["score_source"] for item in online["segment_scores"]} == {score_source.AI}
    saved = json.loads((job_dir / "segment_scores.json").read_text(encoding="utf-8"))
    assert saved[0]["score"]["importance"] == pytest.approx(0.8)


def test_untagged_cache_from_older_versions_is_kept(tmp_path, monkeypatch):
    job_dir, cache = _job(tmp_path)
    monkeypatch.setattr("app.analysis.pipeline.transcribe_detailed", lambda *args, **kwargs: (_transcript(), {"ok": True}))
    first = _analyze(tmp_path, job_dir, dict(cache))
    legacy = dict(cache)
    legacy["transcript"] = first["transcript"]
    legacy["frames"] = first["frames"]
    legacy["visual"] = {key: {"visual_importance": 0.3} for key in first["visual"]}
    legacy["segment_scores"] = [
        {key: value for key, value in item.items() if key not in {"score_source", "fallback_reason"}}
        for item in first["segment_scores"]
    ]
    client = FakeLocalAI()
    result = _analyze(tmp_path, job_dir, legacy, ollama=client)
    assert client.calls == []
    assert result["visual"]["0"] == {"visual_importance": 0.3}
