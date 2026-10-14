"""Smarter selection: picture quality, loudness, sentence-complete moments, meaning-based
matching, and the vision shortlist. All of it is optional and falls back to earlier behaviour."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from app.analysis import score_source
from app.analysis.candidates import _Loudness, _sentence_spans, generate_candidates
from app.analysis.pipeline import _candidate_limit
from app.analysis.repetition import RepetitionAnalyzer
from app.analysis.scorer import HighlightScorer
from app.analysis.semantic import Embedder, apply_focus, repetition_embedding_fn
from app.analysis.signals import POOR_QUALITY, audio_energy, quiet_point, scene_quality
from app.analysis.visual import analyze_scenes_visual, shortlist_scenes
from app.config import Settings
from app.models.schemas import HighlightConfig, Transcript

cv2 = pytest.importorskip("cv2")


# Picture quality


def _image(path: Path, kind: str) -> Path:
    rng = np.random.default_rng(1)
    base = (rng.random((360, 640)) * 255).astype(np.uint8)
    base = cv2.resize(cv2.resize(base, (80, 45)), (640, 360), interpolation=cv2.INTER_NEAREST)  # crisp blocks
    if kind == "blur":
        base = cv2.GaussianBlur(base, (0, 0), 8)
    elif kind == "dark":
        base = (base * 0.1).astype(np.uint8)
    elif kind == "black":
        base = np.zeros_like(base)
    cv2.imwrite(str(path), base)
    return path


def test_scene_quality_ranks_sharp_above_blurry_dark_and_black(tmp_path):
    q = {kind: scene_quality([_image(tmp_path / f"{kind}.jpg", kind)]) for kind in ("sharp", "blur", "dark", "black")}
    assert q["sharp"]["quality"] > POOR_QUALITY
    assert q["blur"]["quality"] < q["sharp"]["quality"]
    assert q["dark"]["quality"] < POOR_QUALITY
    assert q["black"]["black"] is True and q["black"]["quality"] < 0.1


def test_scene_quality_without_readable_frames_is_neutral(tmp_path):
    missing = scene_quality([tmp_path / "nope.jpg"])
    assert missing["quality"] is None and missing["black"] is False


# Loudness


def _wav(path: Path, levels: list[float], seconds_each: float = 1.0, rate: int = 16000) -> Path:
    samples = []
    rng = np.random.default_rng(0)
    for level in levels:
        samples.append((rng.standard_normal(int(rate * seconds_each)) * 30000 * level).clip(-32767, 32767))
    data = np.concatenate(samples).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())
    return path


def test_audio_energy_follows_loudness_and_tolerates_bad_files(tmp_path):
    energy = audio_energy(_wav(tmp_path / "a.wav", [0.05, 0.05, 0.6, 0.05]))
    assert len(energy) == 16
    assert max(energy[8:12]) > max(energy[:8])
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF")
    assert audio_energy(bad) == []


def test_audio_focus_has_no_audibility_signal_for_flat_energy():
    assert _Loudness([0.5] * 12).audibility(0, 3) is None


def test_quiet_point_moves_a_cut_into_a_pause():
    energy = [0.9] * 8 + [0.1] + [0.9] * 8
    assert quiet_point(energy, 1.9, radius=0.5) == pytest.approx(8 * 0.25 + 0.125)
    assert quiet_point([0.5] * 10, 1.0) == 1.0  # nothing quieter nearby: unchanged


# Candidates


def _segment(start, end, text, importance=0.7):
    return {
        "start": start,
        "end": end,
        "text": text,
        "score": {"importance": importance, "novelty": 0.6, "information_density": 0.6},
        "reason": "Key explanation",
    }


def _candidates(segments, scenes=None, **kwargs):
    scenes = scenes or [{"scene_id": 0, "start": 0.0, "end": 40.0, "duration": 40.0}]
    return generate_candidates(
        Transcript(),
        scenes,
        {0: {"visual_importance": 0.5}},
        segments,
        [],
        40.0,
        HighlightConfig(),
        HighlightScorer(),
        **kwargs,
    )


def test_sentence_complete_candidates_join_unfinished_segments():
    segments = [
        _segment(0, 4, "The biggest reason startups fail is"),
        _segment(4.2, 8, "that nobody wants what they build."),
        _segment(20, 24, "Next topic."),
    ]
    assert _sentence_spans(segments) == [(0, 1)]
    merged = [c for c in _candidates(segments) if c["id"].startswith("s")]
    assert len(merged) == 1
    assert merged[0]["start"] <= 0.5 and merged[0]["end"] >= 7.5
    assert merged[0]["joined_segments"] == 2


def test_segments_far_apart_are_not_joined():
    segments = [_segment(0, 4, "And then"), _segment(9, 12, "a long pause later.")]
    assert _sentence_spans(segments) == []


def test_without_new_signals_scores_are_unchanged():
    segments = [_segment(0, 4, "A complete sentence about pricing.")]
    plain = _candidates(segments)
    with_none = _candidates(segments, scene_quality=None, energy=None)
    assert [c["score"] for c in plain] == [c["score"] for c in with_none]


def test_poor_picture_lowers_scores_and_removes_the_scene_minimum():
    segments = [_segment(0, 4, "A complete sentence about pricing.")]
    good = _candidates(segments, scene_quality={0: {"quality": 0.8, "black": False}})
    poor = _candidates(segments, scene_quality={0: {"quality": 0.1, "black": False}})
    by_id = lambda items: {c["id"]: c["score"] for c in items}  # noqa: E731
    assert by_id(poor)["t0"] < by_id(good)["t0"]
    poor_scene = [c for c in poor if c["id"].startswith("v")]
    assert poor_scene and all(c["score"] < 0.42 for c in poor_scene)


def test_unusually_loud_moments_count_as_emotional():
    segments = [_segment(0, 4, "Quiet setup sentence."), _segment(20, 24, "Huge applause here.")]
    energy = [0.2] * 80 + [1.0] * 16 + [0.2] * 64  # loud from 20 s to 24 s
    items = {c["id"]: c for c in _candidates(segments, energy=energy)}
    assert items["t1"]["metrics"]["emotional_intensity"] > items["t0"]["metrics"]["emotional_intensity"]


def test_candidate_limit_grows_with_long_videos():
    assert _candidate_limit(80, 120) == 80
    assert _candidate_limit(80, 3600) == 200
    assert _candidate_limit(300, 3600) == 300


# Meaning-based matching


class FakeEmbedClient:
    def __init__(self, vectors=None, fail=False):
        self.vectors = vectors or {}
        self.fail = fail
        self.calls = 0

    def embed(self, model, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model 'nomic-embed-text' not found")
        return [self.vectors.get(text, [0.0, 0.0, 1.0]) for text in texts]


def test_focus_raises_matching_moments():
    client = FakeEmbedClient({"product demo": [1, 0, 0], "Here is the demo of the app": [0.95, 0.31, 0], "Thanks for watching": [0, 1, 0]})
    candidates = [
        {"id": "a", "text": "Thanks for watching", "score": 0.6},
        {"id": "b", "text": "Here is the demo of the app", "score": 0.55},
    ]
    assert apply_focus(candidates, "product demo", Embedder(client, "nomic-embed-text")) == 1
    assert candidates[0]["id"] == "b" and candidates[0]["score"] > 0.55


def test_missing_embedding_model_falls_back_after_one_request():
    client = FakeEmbedClient(fail=True)
    embedder = Embedder(client, "nomic-embed-text")
    candidates = [{"id": "a", "text": "hello there", "score": 0.5}]
    assert apply_focus(candidates, "demo", embedder) == 0
    assert embedder.available is False
    assert repetition_embedding_fn(embedder) is None
    apply_focus(candidates, "demo", embedder)
    assert client.calls == 1


def test_clear_paraphrases_count_as_repeats_but_loosely_related_lines_do_not():
    a, b, c = "talk to customers first", "interview buyers before building", "the second reason is cash"
    vectors = {a: [1, 0, 0], b: [0.95, 0.312, 0], c: [0.75, 0.661, 0]}  # cos(a,b)=0.95, cos(a,c)=0.75
    analyzer = RepetitionAnalyzer(repetition_embedding_fn(Embedder(FakeEmbedClient(vectors), "m")))
    sim = analyzer.similarity_matrix([a, b, c])
    plain = RepetitionAnalyzer().similarity_matrix([a, b, c])
    assert sim[0, 1] >= 0.8 > plain[0, 1]
    assert sim[0, 2] == pytest.approx(plain[0, 2])


def test_embedding_errors_inside_repetition_fall_back_to_words():
    def broken(_texts):
        raise RuntimeError("boom")

    texts = ["same words here", "same words here too"]
    assert np.allclose(RepetitionAnalyzer(broken).similarity_matrix(texts), RepetitionAnalyzer().similarity_matrix(texts))


# Vision shortlist


def _scenes(n, length=10.0):
    return [{"scene_id": i, "start": i * length, "end": (i + 1) * length, "duration": length} for i in range(n)]


def test_shortlist_keeps_visual_only_and_best_spoken_scenes():
    scenes = _scenes(20)
    segments = [_segment(i * 10, i * 10 + 9, f"line {i}", importance=0.2 + 0.03 * i) for i in range(20) if i not in (3, 7)]
    picked = shortlist_scenes(scenes, segments, highlight_seconds=20)
    assert picked is not None
    assert {3, 7} <= picked  # no speech: visual content
    assert 19 in picked and 18 in picked  # best speech
    assert 0 not in picked  # weakest speech, budget already covered
    assert len(picked) < 20


def test_shortlist_analyzes_everything_without_a_transcript_or_when_most_scenes_qualify():
    assert shortlist_scenes(_scenes(20), [], highlight_seconds=20) is None
    segments = [_segment(i * 10, i * 10 + 9, f"line {i}") for i in range(20)]
    assert shortlist_scenes(_scenes(20), segments, highlight_seconds=80) is None


def test_shortlist_skips_black_scenes():
    scenes = _scenes(20)
    segments = [_segment(i * 10, i * 10 + 9, f"line {i}") for i in range(20) if i != 5]
    picked = shortlist_scenes(scenes, segments, 20, scene_quality={5: {"quality": 0.03, "black": True}})
    assert picked is not None and 5 not in picked


class CountingVision:
    def __init__(self):
        self.calls = 0

    def role_options(self, role):
        return {"num_ctx": 16384}

    def encode_image(self, path):
        return "img"

    def chat(self, model, prompt, images=None, schema=None, options=None, schema_first=True):
        import re

        self.calls += 1
        return {"scenes": [{"scene_id": int(i), "visual_importance": 0.9} for i in re.findall(r"scene_id=(\d+)", prompt)]}


def test_scenes_outside_the_shortlist_get_quick_scoring_and_stay_retryable(tmp_path):
    scenes = _scenes(4, 5.0)
    frames = {}
    for scene in scenes:
        image = tmp_path / f"{scene['scene_id']}.jpg"
        image.write_bytes(b"jpg")
        frames[str(scene["scene_id"])] = [{"timestamp": scene["start"] + 1, "image_path": str(image)}]
    client = CountingVision()
    results, _errors, reasons = analyze_scenes_visual(
        client, "m", scenes, frames, 20.0, settings=Settings(ollama_vision_batch_size=4), shortlist={1, 2}
    )
    assert set(results) == {0, 1, 2, 3}
    assert reasons == {0: score_source.NOT_SHORTLISTED, 3: score_source.NOT_SHORTLISTED}
    assert client.calls == 1
    assert score_source.needs_ai(score_source.tag({}, score_source.NOT_SHORTLISTED))


# Pipeline order and settings


def test_speech_is_scored_before_the_vision_model_runs(tmp_path, monkeypatch):
    from tests.test_ai_reliability import FakeLocalAI, _analyze, _job, _transcript

    job_dir, cache = _job(tmp_path)
    monkeypatch.setattr("app.analysis.pipeline.transcribe_detailed", lambda *a, **k: (_transcript(), {"ok": True}))
    order: list[str] = []
    real_score = __import__("app.analysis.pipeline", fromlist=["VideoAnalyzer"]).VideoAnalyzer._score_transcript

    def tracking_score(self, *args, **kwargs):
        order.append("content")
        return real_score(self, *args, **kwargs)

    def tracking_visual(*args, **kwargs):
        order.append("visual")
        return analyze_scenes_visual(*args, **kwargs)

    monkeypatch.setattr("app.analysis.pipeline.VideoAnalyzer._score_transcript", tracking_score)
    monkeypatch.setattr("app.analysis.pipeline.analyze_scenes_visual", tracking_visual)
    result = _analyze(tmp_path, job_dir, cache, ollama=FakeLocalAI())
    assert order == ["content", "visual"]
    assert result["candidates"]
    assert (job_dir / "scene_quality.json").exists()


def test_vision_scope_setting_is_validated(tmp_path):
    from app import create_app
    from app.config import get_settings, set_settings

    previous = get_settings()
    try:
        client = create_app(Settings(_env_file=None, data_dir=tmp_path / "data"), start_background=False).test_client()
        assert client.post("/api/settings", json={"vision_scope": "everything"}).status_code == 400
        saved = client.post("/api/settings", json={"vision_scope": "all"}).get_json()
        assert saved["settings"]["vision_scope"] == "all"
    finally:
        set_settings(previous)
