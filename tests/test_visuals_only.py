"""Choosing moments without speech, and fast picture matching (SigLIP) as a first filter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.analysis import image_match, score_source
from app.analysis.pipeline import _apply_visual_focus, _visual_signals, _with_image_match
from app.analysis.visual import shortlist_scenes
from app.config import Settings
from app.models.schemas import HighlightConfig


# Settings and options


def test_visuals_only_turns_off_subtitles():
    config = HighlightConfig(use_speech=False, subtitles="burned")
    assert config.subtitles == "none"
    assert HighlightConfig().use_speech is True
    assert HighlightConfig.model_validate({"min_duration": 20}).use_speech is True  # older jobs


def _run(tmp_path, monkeypatch, *, use_speech=True, enable_whisper=True, ollama=None):
    from tests.test_ai_reliability import FakeLocalAI, FakeRunner, _job, _transcript
    from app.analysis.pipeline import PipelineProgress, VideoAnalyzer

    job_dir, cache = _job(tmp_path)
    calls = {"transcribe": 0}

    def transcribe(*_args, **_kwargs):
        calls["transcribe"] += 1
        return _transcript(), {"ok": True}

    monkeypatch.setattr("app.analysis.pipeline.transcribe_detailed", transcribe)
    settings = Settings(data_dir=tmp_path / "data", enable_whisper=enable_whisper, image_matching=False)
    stages: list[tuple[str, str]] = []

    def emit(kind, payload):
        if kind == "progress":
            stages.append((payload["stage"], payload["status"]))

    client = ollama if ollama is not None else FakeLocalAI()
    analyzer = VideoAnalyzer(settings, FakeRunner(), client, vision_enabled=True, text_enabled=True)
    result = analyzer.analyze(
        job_dir, job_dir / "source.mp4", HighlightConfig(use_speech=use_speech), PipelineProgress(emit), cache, plan=False
    )
    return job_dir, result, calls, stages


def test_speech_off_for_a_highlight_skips_transcription_and_speech_scoring(tmp_path, monkeypatch):
    job_dir, result, calls, stages = _run(tmp_path, monkeypatch, use_speech=False)
    assert calls["transcribe"] == 0
    assert ("transcript", "skipped") in stages and ("content", "skipped") in stages
    assert not (job_dir / "transcript.json").exists()
    assert result["candidates"]  # moments still come from pictures and sound
    assert all(c["source"] != "transcript" for c in result["candidates"])


def test_speech_off_in_settings_has_the_same_effect(tmp_path, monkeypatch):
    _job_dir, _result, calls, stages = _run(tmp_path, monkeypatch, enable_whisper=False)
    assert calls["transcribe"] == 0 and ("transcript", "skipped") in stages


def test_turning_speech_back_on_keeps_earlier_speech_scores(tmp_path, monkeypatch):
    job_dir, _result, _calls, _stages = _run(tmp_path, monkeypatch, use_speech=True)
    saved = (job_dir / "segment_scores.json").read_text(encoding="utf-8")
    from tests.test_ai_reliability import FakeLocalAI, FakeRunner
    from app.analysis.pipeline import PipelineProgress, VideoAnalyzer
    from app.workers.worker import load_cache

    analyzer = VideoAnalyzer(Settings(data_dir=tmp_path / "data", image_matching=False), FakeRunner(), FakeLocalAI(), True, True)
    analyzer.analyze(job_dir, job_dir / "source.mp4", HighlightConfig(use_speech=False), PipelineProgress(lambda *_: None), load_cache(job_dir), plan=False)
    assert (job_dir / "segment_scores.json").read_text(encoding="utf-8") == saved
    assert (job_dir / "transcript.json").exists()


# Shortlist without speech


def _scenes(n, length=10.0):
    return [{"scene_id": i, "start": i * length, "end": (i + 1) * length, "duration": length} for i in range(n)]


def test_without_speech_the_strongest_scenes_are_shortlisted():
    scenes = _scenes(20)
    signal = {i: i / 20 for i in range(20)}
    picked = shortlist_scenes(scenes, [], highlight_seconds=20, visual_signal=signal)
    assert picked is not None and 19 in picked and 0 not in picked
    assert 6 <= len(picked) < 20


def test_without_speech_or_quick_signals_every_scene_is_analyzed():
    assert shortlist_scenes(_scenes(20), [], highlight_seconds=20) is None


def test_lookalike_and_black_scenes_are_left_out():
    scenes = _scenes(20)
    signal = {i: 1.0 for i in range(20)}
    match = {5: {"interest": 1.0, "duplicate_of": 4}}
    quality = {6: {"quality": 0.03, "black": True}}
    picked = shortlist_scenes(scenes, [], 60, quality, visual_signal=signal, match=match)
    assert picked is not None and 5 not in picked and 6 not in picked


def test_picture_interest_breaks_ties_between_spoken_scenes():
    scenes = _scenes(20)
    segments = [
        {"start": i * 10, "end": i * 10 + 9, "text": "x", "score": {"importance": 0.5, "information_density": 0.5}}
        for i in range(20)
    ]
    match = {i: {"interest": 1.0 if i in (3, 4) else 0.0, "duplicate_of": None} for i in range(20)}
    picked = shortlist_scenes(scenes, segments, 10, match=match)
    assert picked is not None and {3, 4} <= picked


# Quick signals and picture matching in scoring


def test_visual_signal_combines_matching_motion_sound_and_quality():
    scenes = _scenes(2)
    signals = _visual_signals(
        scenes,
        {},
        {0: {"quality": 0.2}, 1: {"quality": 0.9}},
        [0.2] * 40 + [1.0] * 40,
        {0: {"interest": 0.1}, 1: {"interest": 0.9}},
    )
    assert signals[1] > signals[0]
    assert _visual_signals([], {}, {}, [], {}) == {}


def test_picture_matching_only_improves_quick_scores():
    visual = {
        0: {"visual_importance": 0.9, "score_source": score_source.AI},
        1: {"visual_importance": 0.4, "score_source": score_source.FALLBACK},
        2: {"visual_importance": 0.4, "score_source": score_source.FALLBACK},
    }
    blended = _with_image_match(visual, {0: {"interest": 0.0}, 1: {"interest": 1.0}})
    assert blended[0]["visual_importance"] == 0.9  # the vision model's score is kept
    assert blended[1]["visual_importance"] > 0.4
    assert blended[2] is visual[2]  # no match data
    assert _with_image_match(visual, {}) is visual


def test_visual_focus_raises_matching_moments():
    scenes = _scenes(2)
    candidates = [{"id": "a", "start": 1.0, "score": 0.5}, {"id": "b", "start": 12.0, "score": 0.55}]
    assert _apply_visual_focus(candidates, scenes, {0: {"focus": 1.0}, 1: {"focus": 0.2}}) == 1
    assert candidates[0]["id"] == "a" and candidates[0]["score"] == pytest.approx(0.6)


# Picture matching model


class FakeMatcher:
    device = "CPU"

    def __init__(self, image_vectors, text_vectors):
        self.image_vectors = image_vectors
        self.text_vectors = text_vectors

    def embed_images(self, paths):
        rows = [self.image_vectors.get(Path(p).name) for p in paths]
        vectors = np.array([r if r is not None else np.zeros(3) for r in rows], dtype=np.float32)
        return vectors, [r is not None for r in rows]

    def embed_texts(self, texts):
        return np.array([self.text_vectors.get(t, [0, 0, 1.0]) for t in texts], dtype=np.float32)


def _unit(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_score_scenes_ranks_interest_finds_lookalikes_and_matches_focus():
    positive = {text: _unit(1, 0, 0) for text in image_match.POSITIVE_PROMPTS}
    negative = {text: _unit(0, 1, 0) for text in image_match.NEGATIVE_PROMPTS}
    matcher = FakeMatcher(
        {"action.jpg": _unit(1, 0.1, 0), "black.jpg": _unit(0, 1, 0.1), "same.jpg": _unit(1, 0.1, 0), "missing.jpg": None},
        {**positive, **negative, "a red car": _unit(0, 0, 1)},
    )
    result = image_match.score_scenes(
        matcher, {0: "action.jpg", 1: "black.jpg", 2: "same.jpg", 3: "missing.jpg"}, focus="a red car"
    )
    assert set(result) == {0, 1, 2}  # an unreadable frame is skipped
    assert result[0]["interest"] > result[1]["interest"]
    assert result[2]["duplicate_of"] == 0
    assert result[0]["focus"] is not None


def test_matcher_is_unavailable_until_the_model_is_downloaded(monkeypatch):
    monkeypatch.setattr(image_match, "model_installed", lambda: False)
    assert image_match.get_matcher() is None


def test_gpu_is_preferred_and_a_failing_gpu_falls_back_to_the_cpu(monkeypatch):
    ort = pytest.importorskip("onnxruntime")

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert image_match._providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    used = []

    def fake_session(path, options, providers):
        used.append(providers)
        if "CUDAExecutionProvider" in providers:
            raise RuntimeError("CUDA driver missing")
        return "cpu-session"

    monkeypatch.setattr(ort, "InferenceSession", fake_session)
    assert image_match._session("model.onnx", 2) == "cpu-session"
    assert used[-1] == ["CPUExecutionProvider"]


def test_download_runs_in_the_background_and_reports_status(monkeypatch):
    fetched = []
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name: fetched.append(name))
    monkeypatch.setitem(image_match._download_state, "state", "idle")
    image_match._download()
    assert fetched == list(image_match.MODEL_FILES)
    assert image_match.download_status()["state"] == "done"


def test_image_model_api(tmp_path, monkeypatch):
    from app import create_app
    from app.config import get_settings, set_settings

    previous = get_settings()
    try:
        client = create_app(Settings(_env_file=None, data_dir=tmp_path / "data"), start_background=False).test_client()
        monkeypatch.setattr(image_match, "runtime_available", lambda: False)
        assert client.post("/api/system/image-model/download").status_code == 400
        status = client.get("/api/system/image-model").get_json()
        assert {"installed", "size_mb", "state"} <= set(status)
        assert client.post("/api/settings", json={"image_matching": False}).get_json()["settings"]["image_matching"] is False
    finally:
        set_settings(previous)


@pytest.mark.skipif(
    not (image_match.runtime_available() and image_match.model_installed()),
    reason="SigLIP model or onnxruntime not available",
)
def test_real_model_ranks_a_black_frame_lowest(tmp_path):
    from PIL import Image, ImageDraw

    paths = {}
    for name, color in (("black", (0, 0, 0)), ("busy", None)):
        image = Image.new("RGB", (320, 240), color or (240, 240, 240))
        if color is None:
            draw = ImageDraw.Draw(image)
            for i in range(0, 320, 20):
                draw.rectangle((i, 60, i + 10, 200), fill=(i % 255, 120, 255 - i % 255))
        path = tmp_path / f"{name}.jpg"
        image.save(path)
        paths[name] = str(path)
    matcher = image_match.get_matcher(2)
    assert matcher is not None
    result = image_match.score_scenes(matcher, {0: paths["black"], 1: paths["busy"]})
    assert result[0]["interest"] < result[1]["interest"]
