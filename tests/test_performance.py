"""Speed work that must not change results: context-sized transcript batches, parallel frame
extraction, model usage reporting, and remote Ollama options."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
import requests

from app.ai import ollama as ollama_module
from app.ai.ollama import OllamaClient, describe_usage, predict_budget, usage_delta
from app.analysis.pipeline import (
    PipelineProgress,
    VideoAnalyzer,
    _CONTEXT_SAFETY,
    estimate_tokens,
    frame_extract_workers,
    plan_text_batches,
)
from app.config import Settings
from app.video.ffmpeg_runner import CancelledError, FFmpegRunner


class FakeResponse:
    def __init__(self, status: int = 200, body=None):
        self.status_code = status
        self.ok = status < 400
        self._body = body
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        return self._body

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(str(self.status_code))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(ollama_module.time, "sleep", lambda _seconds: None)


def _patch_session(monkeypatch, handler):
    calls: list[dict] = []

    def fake_request(self, method, url, **kwargs):
        calls.append({"method": method, "url": url, "headers": dict(self.headers), **kwargs})
        return handler(url, kwargs)

    monkeypatch.setattr("app.ai.ollama.requests.Session.request", fake_request)
    return calls


# Transcript batches sized to the context window


def _plan(texts, num_ctx, max_items=20):
    return plan_text_batches(
        texts,
        max_items=max_items,
        fixed_tokens=150,
        configured_predict=512,
        per_item_output=40,
        num_ctx=num_ctx,
    )


def test_short_segments_keep_the_configured_batch_size():
    texts = ["A short sentence about the product launch."] * 45
    batches = _plan(texts, 4096)
    assert [len(batch) for batch in batches] == [20, 20, 5]


def test_long_segments_are_split_so_prompt_and_reply_fit_a_small_context():
    long_text = "word " * 120
    texts = [long_text] * 20
    batches = _plan(texts, 2048)
    assert len(batches) > 1
    for batch in batches:
        prompt = 150 + sum(estimate_tokens(texts[i]) + 14 for i in batch)
        assert prompt + predict_budget(512, len(batch), 40, 2048) <= 2048 * _CONTEXT_SAFETY or len(batch) == 1


def test_batches_keep_every_segment_once_and_in_order():
    texts = [("x" * (i * 37 % 400)) for i in range(70)]
    batches = _plan(texts, 2048)
    assert [i for batch in batches for i in batch] == list(range(70))


def test_a_segment_too_long_for_any_batch_still_goes_alone():
    texts = ["short", "y" * 20000, "short"]
    assert _plan(texts, 2048) == [[0], [1], [2]]


def test_token_estimate_counts_non_latin_scripts_per_character():
    assert estimate_tokens("日本語のテキスト") >= 8
    assert estimate_tokens("hello world") < 6


# Parallel frame extraction


class RecordingRunner:
    def __init__(self, fail_at: set[float] | None = None, cancel_after: int | None = None):
        self.cancel_event = threading.Event()
        self.fail_at = fail_at or set()
        self.cancel_after = cancel_after
        self.calls: list[str] = []
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def extract_frame(self, source, out, stamp, duration=None):
        if self.cancel_event.is_set():
            raise CancelledError("Job cancelled")
        with self.lock:
            self.calls.append(Path(out).name)
            self.active += 1
            self.peak = max(self.peak, self.active)
            if self.cancel_after is not None and len(self.calls) >= self.cancel_after:
                self.cancel_event.set()
        threading.Event().wait(0.02)  # time.sleep is stubbed out by the autouse fixture
        with self.lock:
            self.active -= 1
        if round(stamp, 2) in self.fail_at:
            raise RuntimeError("decode error")
        Path(out).write_bytes(b"jpg")


def _scenes(count: int, length: float = 6.0):
    return [
        {"scene_id": i, "start": i * length, "end": (i + 1) * length, "duration": length} for i in range(count)
    ]


def _extract(tmp_path, runner, scenes, logs=None):
    analyzer = VideoAnalyzer(Settings(data_dir=tmp_path / "data"), runner, None)
    progress = PipelineProgress(lambda kind, payload: logs.append(payload) if logs is not None and kind == "log" else None)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(exist_ok=True)
    duration = scenes[-1]["end"]
    return analyzer._extract_frames(tmp_path / "in.mp4", scenes, frames_dir, duration, "auto", progress)


def test_parallel_frames_match_scene_and_timestamp_order(tmp_path, monkeypatch):
    monkeypatch.setattr("app.analysis.pipeline.os.cpu_count", lambda: 8)
    runner = RecordingRunner()
    frame_map = _extract(tmp_path, runner, _scenes(6))
    assert list(frame_map) == [str(i) for i in range(6)]
    for scene_id, entries in frame_map.items():
        stamps = [entry["timestamp"] for entry in entries]
        assert stamps == sorted(stamps) and stamps
        assert all(Path(entry["image_path"]).name.startswith(f"scene_{scene_id}_") for entry in entries)
    assert runner.peak > 1


def test_failed_frames_are_logged_and_left_out(tmp_path):
    scenes = _scenes(1)
    runner = RecordingRunner(fail_at={3.0})
    logs: list[dict] = []
    frame_map = _extract(tmp_path, runner, scenes, logs)
    assert all(entry["timestamp"] != 3.0 for entry in frame_map["0"])
    assert sum("Frame extract failed at 3.00s" in entry["message"] for entry in logs) == 1


def test_existing_frames_are_not_extracted_again(tmp_path):
    scenes = _scenes(2)
    first = RecordingRunner()
    _extract(tmp_path, first, scenes)
    second = RecordingRunner()
    frame_map = _extract(tmp_path, second, scenes)
    assert second.calls == []
    assert sum(len(entries) for entries in frame_map.values()) == len(first.calls)


def test_cancel_stops_frame_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr("app.analysis.pipeline.os.cpu_count", lambda: 8)
    runner = RecordingRunner(cancel_after=3)
    with pytest.raises(CancelledError):
        _extract(tmp_path, runner, _scenes(20))
    assert len(runner.calls) < 20 * 5


def test_frame_workers_stay_small_and_bounded(monkeypatch):
    monkeypatch.setattr("app.analysis.pipeline.os.cpu_count", lambda: 12)
    assert frame_extract_workers(100) == 4
    assert frame_extract_workers(2) == 2
    monkeypatch.setattr("app.analysis.pipeline.os.cpu_count", lambda: 2)
    assert frame_extract_workers(100) == 1
    monkeypatch.setattr("app.analysis.pipeline.os.cpu_count", lambda: None)
    assert frame_extract_workers(0) == 1


def test_runner_cancel_stops_every_running_command():
    runner = FFmpegRunner()

    class FakeProc:
        def __init__(self):
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    procs = [FakeProc(), FakeProc()]
    runner._procs.update(procs)
    runner.cancel()
    assert all(proc.terminated for proc in procs)
    assert runner.cancel_event.is_set()


# Stage timings and model usage


def test_stage_timings_count_only_stages_that_ran():
    progress = PipelineProgress(lambda kind, payload: None)
    progress.stage("scenes", "running")
    progress.stage("scenes", "done")
    progress.stage("audio", "done", "cached")
    progress.stage("visual", "running")
    progress.stage("visual", "running", "Visual batch 2/3")
    progress.stage("visual", "done")
    assert set(progress.timings) == {"scenes", "visual"}
    progress.timings["scenes"] = 3.25
    assert progress.timing_summary().startswith("Scene detection 3.2s")


def _chat_body(prompt_tokens=100, output_tokens=40, done_reason="stop"):
    return {
        "message": {"content": '{"ok": true}'},
        "done_reason": done_reason,
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": 2_000_000_000,
        "eval_count": output_tokens,
        "eval_duration": 4_000_000_000,
        "load_duration": 1_500_000_000,
    }


def test_chat_records_usage_per_call(monkeypatch):
    client = OllamaClient(Settings(ollama_base_url="http://127.0.0.1:11434"))
    _patch_session(monkeypatch, lambda _url, _kwargs: FakeResponse(body=_chat_body()))
    before = client.usage_snapshot()
    client.chat("qwen3:4b", "hi", options={"num_ctx": 4096})
    client.chat("qwen3:4b", "hi", options={"num_ctx": 4096})
    usage = usage_delta(before, client.usage_snapshot())
    assert usage["calls"] == 2
    assert usage["prompt_tokens"] == 200
    assert usage["output_tokens"] == 80
    assert usage["near_context_limit"] == 0
    text = describe_usage(usage)
    assert "2 calls" in text and "80 reply tokens" in text and "10.0 tok/s" in text


def test_full_context_is_flagged(monkeypatch):
    client = OllamaClient(Settings(ollama_base_url="http://127.0.0.1:11434"))
    _patch_session(monkeypatch, lambda _url, _kwargs: FakeResponse(body=_chat_body(prompt_tokens=2000, output_tokens=40)))
    client.chat("qwen3:4b", "hi", options={"num_ctx": 2048})
    assert client.usage_snapshot()["near_context_limit"] == 1


def test_usage_is_kept_per_thread(monkeypatch):
    client = OllamaClient(Settings(ollama_base_url="http://127.0.0.1:11434"))
    _patch_session(monkeypatch, lambda _url, _kwargs: FakeResponse(body=_chat_body()))
    worker = threading.Thread(target=lambda: client.chat("qwen3:4b", "hi"))
    worker.start()
    worker.join()
    assert client.usage_snapshot()["calls"] == 0


def test_describe_usage_is_empty_without_calls():
    assert describe_usage({"calls": 0}) == ""


# Remote Ollama


def test_thread_count_is_sent_only_to_ollama_on_this_computer():
    for url in ("http://127.0.0.1:11434", "http://localhost:11434", "http://host.docker.internal:11434", "http://ollama:11434"):
        client = OllamaClient(Settings(ollama_base_url=url, ollama_num_thread=5))
        assert client.role_options("text")["num_thread"] == 5, url
    remote = OllamaClient(Settings(ollama_base_url="http://192.168.1.50:11434", ollama_num_thread=5, ollama_text_num_ctx=4096))
    assert remote.role_options("text") == {"num_ctx": 4096}
    assert "num_thread" not in remote._default_options()


def test_api_key_is_sent_as_bearer_token(monkeypatch):
    client = OllamaClient(Settings(ollama_base_url="https://gpu.example.com", ollama_api_key=" secret "))
    calls = _patch_session(monkeypatch, lambda _url, _kwargs: FakeResponse(body={"version": "0.12.7"}))
    client._request("GET", "/api/version")
    assert calls[0]["headers"]["Authorization"] == "Bearer secret"
    client.settings.ollama_api_key = ""
    client.apply_auth()
    client._request("GET", "/api/version")
    assert "Authorization" not in calls[1]["headers"]


def test_unreachable_remote_falls_back_to_the_configured_local_ollama(monkeypatch):
    client = OllamaClient(
        Settings(ollama_base_url="http://192.168.1.50:11434", ollama_fallback_url="http://127.0.0.1:11434/api")
    )

    def handler(url, _kwargs):
        if "192.168.1.50" in url:
            raise requests.exceptions.ConnectionError("unreachable")
        return FakeResponse(body={"message": {"content": '{"ok": true}'}})

    _patch_session(monkeypatch, handler)
    assert client.chat("qwen3:4b", "hi") == {"ok": True}
    assert client.root == "http://127.0.0.1:11434"


def test_remote_without_fallback_still_reports_connection_error(monkeypatch):
    client = OllamaClient(Settings(ollama_base_url="http://192.168.1.50:11434", ollama_max_retries=0))

    def handler(_url, _kwargs):
        raise requests.exceptions.ConnectionError("unreachable")

    _patch_session(monkeypatch, handler)
    with pytest.raises(ollama_module.OllamaConnectionError):
        client.chat("qwen3:4b", "hi")


def test_configured_server_is_used_again_once_it_answers(monkeypatch):
    client = OllamaClient(
        Settings(ollama_base_url="http://192.168.1.50:11434", ollama_fallback_url="http://127.0.0.1:11434")
    )
    client.root = "http://127.0.0.1:11434"
    remote_up = {"value": False}

    def handler(url, _kwargs):
        if "192.168.1.50" in url and not remote_up["value"]:
            raise requests.exceptions.ConnectionError("unreachable")
        if url.endswith("/api/tags"):
            return FakeResponse(body={"models": []})
        if url.endswith("/api/ps"):
            return FakeResponse(body={"models": []})
        return FakeResponse(body={"version": "0.12.7"})

    calls = _patch_session(monkeypatch, handler)
    clock = {"now": 1000.0}
    monkeypatch.setattr(ollama_module.time, "monotonic", lambda: clock["now"])

    client.health()
    assert client.root == "http://127.0.0.1:11434"
    remote_up["value"] = True
    client.health()
    assert client.root == "http://127.0.0.1:11434", "rechecked too soon"
    probes = sum("192.168.1.50" in call["url"] for call in calls)
    assert probes == 1
    clock["now"] += 61
    client.health()
    assert client.root == "http://192.168.1.50:11434"


def test_api_key_is_never_returned_by_the_settings_api(tmp_path):
    from app import create_app
    from app.config import set_settings

    settings = Settings(data_dir=tmp_path / "data", ollama_api_key="tok-7f3a9c")
    set_settings(settings)
    try:
        app = create_app(settings, start_background=False)
        body = app.test_client().get("/api/settings").get_json()
    finally:
        set_settings(None)
    assert "ollama_api_key" not in body["settings"]
    assert "tok-7f3a9c" not in json.dumps(body)


# Thinking vision models and vision batches that fit the context


def test_bare_qwen3_vl_tags_map_to_their_instruct_versions():
    from app.ai.ollama import instruct_variant

    assert instruct_variant("qwen3-vl:2b") == "qwen3-vl:2b-instruct"
    assert instruct_variant("qwen3-vl:8b-thinking") == "qwen3-vl:8b-instruct"
    assert instruct_variant("qwen3-vl") == "qwen3-vl:8b-instruct"
    assert instruct_variant("qwen3-vl:latest") == "qwen3-vl:8b-instruct"
    assert instruct_variant("qwen3-vl:30b") == "qwen3-vl:30b-a3b-instruct"
    assert instruct_variant("qwen3-vl:32b") == "qwen3-vl:32b-instruct"
    assert instruct_variant("llava:7b") is None


def _health_with_capabilities(monkeypatch, vision_model: str, capabilities: list[str]):
    client = OllamaClient(Settings(ollama_base_url="http://127.0.0.1:11434", ollama_vision_model=vision_model))

    def handler(url, _kwargs):
        if url.endswith("/api/version"):
            return FakeResponse(body={"version": "0.12.7"})
        if url.endswith("/api/tags"):
            return FakeResponse(body={"models": [{"name": vision_model, "digest": "abc"}, {"name": "qwen3:8b"}]})
        if url.endswith("/api/ps"):
            return FakeResponse(body={"models": []})
        return FakeResponse(body={"capabilities": capabilities})

    _patch_session(monkeypatch, handler)
    return client.health()


def test_thinking_vision_model_gets_a_warning_with_the_instruct_tag(monkeypatch):
    health = _health_with_capabilities(monkeypatch, "qwen3-vl:2b", ["completion", "vision", "thinking"])
    assert health["vision_ready"] is True
    assert any("thinking model" in item and "qwen3-vl:2b-instruct" in item for item in health["warnings"])


def test_instruct_vision_model_has_no_thinking_warning(monkeypatch):
    health = _health_with_capabilities(monkeypatch, "qwen3-vl:2b-instruct", ["completion", "vision", "tools"])
    assert not any("thinking" in item for item in health["warnings"])


def test_catalog_recommends_only_instruct_vision_models():
    from app.services.model_catalog import VISION_MODELS, recommend_setup

    assert all(item["name"].endswith("-instruct") for item in VISION_MODELS)
    for vram, ram, free in ((None, 8, None), (None, 32, None), (6, 32, 5), (12, 32, 11), (24, 64, 22)):
        setup = recommend_setup(vram, ram, free, 8)
        assert setup["vision_model"].endswith("-instruct")
        assert all(name.endswith("-instruct") for name in setup["alternatives"]["vision"])


def test_vision_batches_shrink_to_fit_the_context():
    from app.analysis.visual import fitting_vision_batch

    assert fitting_vision_batch(4, 2048, 256) == 1
    assert fitting_vision_batch(2, 2048, 256) == 1
    assert fitting_vision_batch(4, 4096, 256) == 2
    assert fitting_vision_batch(4, 8192, 256) == 4
    assert fitting_vision_batch(1, 1024, 256) == 1
    assert fitting_vision_batch(0, 8192, 256) == 1


def test_visual_analysis_uses_the_fitting_batch_size(tmp_path):
    from app.analysis.visual import analyze_scenes_visual

    class SmallContextClient:
        def __init__(self):
            self.calls: list[int] = []

        def role_options(self, role):
            return {"num_ctx": 2048}

        def encode_image(self, path):
            return "image"

        def chat(self, model, prompt, images=None, schema=None, options=None, schema_first=True):
            self.calls.append(len(images or []))
            return {"scenes": []}

    scenes = [{"scene_id": i, "start": i * 5.0, "end": i * 5.0 + 5.0, "duration": 5.0} for i in range(3)]
    frames = {}
    for scene in scenes:
        image = tmp_path / f"s{scene['scene_id']}.jpg"
        image.write_bytes(b"jpg")
        frames[str(scene["scene_id"])] = [{"timestamp": scene["start"] + 1, "image_path": str(image)}]
    client = SmallContextClient()
    analyze_scenes_visual(client, "qwen3-vl:2b-instruct", scenes, frames, 15.0, settings=Settings(ollama_vision_batch_size=4))
    assert client.calls == [1, 1, 1]


def test_recommendation_change_is_applied_once_to_automatic_mode(tmp_path, monkeypatch):
    from app.config import get_settings, set_settings
    from app.models.db_models import init_db
    from app.services import ai_profile
    from app.services.model_catalog import recommend_setup
    from app.services.settings_store import write_setting

    previous = get_settings()
    settings = Settings(data_dir=tmp_path / "data", ai_config_mode="auto", ollama_vision_model="qwen3-vl:2b")
    settings.ensure_directories()
    set_settings(settings)
    init_db(settings.db_path)

    class OfflineClient:
        def health(self):
            return {"connected": False, "version": None, "models": [], "running": [], "error": "offline"}

    hardware = {
        "cpu": "Test CPU",
        "ram_gb": 15.8,
        "vram_gb": None,
        "gpu": None,
        "inference_mode": "cpu",
        "recommendation": recommend_setup(None, 15.8, None, 6),
    }
    monkeypatch.setattr("app.services.ai_profile.detect_hardware", lambda **_kwargs: hardware)
    monkeypatch.setattr("app.services.ai_profile.get_ollama_client", lambda settings=None: OfflineClient())
    try:
        # A profile saved by the previous version: same hardware, fingerprint without a version.
        old_fingerprint = ai_profile.fingerprint(hardware, None).split("|", 1)[1]
        write_setting("hw_fingerprint", old_fingerprint)
        write_setting(
            "hw_profile_json",
            json.dumps({"fingerprint": old_fingerprint, "saved_at": time.time(), "hardware": hardware}),
        )
        first = ai_profile.sync_hardware_profile(settings)
        assert first["profile"]["applied_now"] is True
        assert settings.ollama_vision_model == "qwen3-vl:2b-instruct"

        second = ai_profile.sync_hardware_profile(settings, refresh=True)
        assert second["profile"]["applied_now"] is False
        assert second["profile"]["fingerprint"].startswith(ai_profile.RECOMMENDATION_VERSION + "|")
    finally:
        set_settings(previous)
