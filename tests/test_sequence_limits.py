from app.analysis.sequence_limits import keep_best_span, fit_candidate, sequence_cap, sequence_min
from app.analysis.optimizer import ClipOptimizer
from app.analysis.candidates import generate_candidates
from app.analysis.scorer import HighlightScorer
from app.models.schemas import HighlightConfig, Transcript


def _c(cid, start, end, score, topic="x", text="", peak=None, metrics=None, reason="Key explanation"):
    return {
        "id": cid,
        "start": start,
        "end": end,
        "score": score,
        "topic": topic,
        "text": text or topic,
        "reason": reason,
        "peak": peak,
        "metrics": metrics or {},
    }


def test_keep_best_span_cuts_to_hard_maximum_around_peak():
    start, end = keep_best_span(10.0, 25.0, max_len=5.0, peak=18.0)
    assert end - start == 5.0
    assert 15.5 <= start <= 16.0
    assert 20.5 <= end <= 21.0


def test_keep_best_span_does_not_stretch_short_clips():
    start, end = keep_best_span(10.0, 13.0, max_len=5.0, peak=11.5)
    assert start == 10.0
    assert end == 13.0


def test_optimizer_never_exceeds_max_sequence_duration():
    optimizer = ClipOptimizer()
    candidates = [
        _c("long", 10, 40, 0.95, "a", peak=22.0),
        _c("other", 80, 100, 0.9, "b", peak=90.0),
    ]
    plan = optimizer.select(
        candidates,
        min_duration=8,
        max_duration=20,
        max_sequence_duration=5.0,
    )
    assert plan.clips
    assert all(c.end - c.start <= 5.0 + 1e-6 for c in plan.clips)


def test_disallow_repeated_content_keeps_only_strongest_duplicate():
    optimizer = ClipOptimizer()
    candidates = [
        _c("a", 10, 15, 0.95, "speed", "we deliver fast"),
        _c("b", 20, 25, 0.93, "speed", "delivery is very fast"),
        _c("c", 40, 45, 0.88, "pricing", "unique pricing example"),
    ]
    plan = optimizer.select(
        candidates,
        min_duration=8,
        max_duration=20,
        max_sequence_duration=5.0,
        allow_repeated_content=False,
        similarity_fn=lambda x, y: 0.9 if x["topic"] == y["topic"] else 0.1,
    )
    ids = [c.candidate_id for c in plan.clips]
    assert "a" in ids
    assert "b" not in ids
    assert "c" in ids


def test_allow_repeated_content_caps_total_repeat_duration():
    optimizer = ClipOptimizer()
    candidates = [
        _c("best", 10, 15, 0.96, "speed"),
        _c("rep1", 20, 25, 0.9, "speed"),
        _c("rep2", 30, 35, 0.89, "speed"),
        _c("unique", 80, 85, 0.7, "other"),
    ]
    plan = optimizer.select(
        candidates,
        min_duration=12,
        max_duration=20,
        max_sequence_duration=5.0,
        allow_repeated_content=True,
        max_repeated_duration=5.0,
        similarity_fn=lambda x, y: 0.9 if x["topic"] == y["topic"] else 0.1,
    )
    speed = [c for c in plan.clips if c.candidate_id in {"best", "rep1", "rep2"}]
    unique = [c for c in plan.clips if c.candidate_id == "unique"]
    assert unique
    extra = sum(c.end - c.start for c in speed if c.candidate_id != "best")
    assert extra <= 5.0 + 1e-6


def test_sequence_cap_is_hard_maximum_not_a_target():
    optimizer = ClipOptimizer()
    candidates = [_c("short", 10, 13, 0.95, "a")]
    plan = optimizer.select(
        candidates,
        min_duration=20,
        max_duration=30,
        max_sequence_duration=15.0,
        source_duration=40,
    )
    assert plan.clips
    assert all(c.end - c.start <= 15.0 + 1e-6 for c in plan.clips)
    assert plan.estimated_duration <= 15.0 + 1e-6


def test_sequence_cap_custom_uses_user_maximum():
    config = HighlightConfig(sequence_duration_mode="custom", max_sequence_duration=5, min_duration=20, max_duration=30)
    assert sequence_cap(config) == 5.0


def test_sequence_cap_auto_uses_overall_highlight_max_not_a_fixed_clip_cap():
    config = HighlightConfig(sequence_duration_mode="auto", max_sequence_duration=5, min_duration=20, max_duration=30)
    cap = sequence_cap(config)
    assert cap == 30.0
    assert cap > 12.0


def test_custom_fit_cuts_long_moment_to_user_max_and_explains():
    fitted = fit_candidate(_c("long", 10, 30, 0.9, peak=18.0), cap=5.0, mode="custom")
    assert abs((fitted["end"] - fitted["start"]) - 5.0) < 1e-6
    assert fitted["duration_reason"]
    assert "5" in fitted["duration_reason"]


def test_auto_fit_does_not_pad_short_moments():
    fitted = fit_candidate(
        _c("short", 10, 12, 0.95, metrics={"importance": 1.0, "information_density": 1.0}),
        cap=12.0,
        mode="auto",
    )
    assert abs((fitted["end"] - fitted["start"]) - 2.0) < 1e-6


def test_auto_fit_lets_important_moments_exceed_five_seconds():
    fitted = fit_candidate(
        _c(
            "dense",
            10,
            40,
            0.95,
            peak=22.0,
            metrics={
                "importance": 0.95,
                "information_density": 0.9,
                "visual_importance": 0.8,
                "emotional_intensity": 0.7,
            },
        ),
        cap=30.0,
        mode="auto",
    )
    duration = fitted["end"] - fitted["start"]
    assert duration >= 12.0
    assert duration <= 30.0
    assert fitted["duration_reason"]


def test_auto_fit_respects_custom_minimum_floor():
    fitted = fit_candidate(
        _c(
            "hook",
            0,
            20,
            0.9,
            peak=1.0,
            metrics={"hook_strength": 0.92, "importance": 0.55},
        ),
        cap=12.0,
        mode="auto",
        min_len=3.0,
    )
    assert fitted["end"] - fitted["start"] >= 3.0


def test_custom_fit_respects_min_and_max_sequence_duration():
    fitted = fit_candidate(_c("long", 10, 30, 0.9, peak=18.0), cap=8.0, mode="custom", min_len=3.0)
    duration = fitted["end"] - fitted["start"]
    assert 3.0 <= duration <= 8.0 + 1e-6


def test_sequence_min_uses_default_when_unset():
    config = HighlightConfig(sequence_duration_mode="auto")
    assert sequence_min(config) == 1.5


def test_sequence_min_honors_explicit_auto_minimum():
    config = HighlightConfig(sequence_duration_mode="auto", min_sequence_duration=3, min_duration=20, max_duration=30)
    assert sequence_min(config) == 3.0


def test_auto_fit_keeps_hooks_short():
    fitted = fit_candidate(
        _c(
            "hook",
            0,
            20,
            0.9,
            peak=1.0,
            metrics={"hook_strength": 0.92, "importance": 0.55},
        ),
        cap=12.0,
        mode="auto",
    )
    assert 1.5 <= fitted["end"] - fitted["start"] <= 4.0


def test_auto_optimizer_uses_varied_sequence_lengths():
    optimizer = ClipOptimizer()
    candidates = [
        _c("hook", 1, 20, 0.94, "intro", peak=2.0, metrics={"hook_strength": 0.9, "importance": 0.5}),
        _c(
            "demo",
            25,
            50,
            0.93,
            "demo",
            peak=35.0,
            metrics={"importance": 0.92, "information_density": 0.88, "visual_importance": 0.8},
        ),
        _c("beat", 55, 70, 0.8, "action", peak=60.0, metrics={"emotional_intensity": 0.8, "importance": 0.6}),
        _c("end", 80, 83, 0.78, "close", metrics={"importance": 0.7}),
    ]
    plan = optimizer.select(
        candidates,
        min_duration=10,
        max_duration=22,
        source_duration=90,
        sequence_duration_mode="auto",
        max_sequence_duration=5.0,
        allow_repeated_content=False,
    )
    durations = [round(c.end - c.start, 2) for c in plan.clips]
    assert plan.clips
    assert all(d <= 22.0 + 1e-6 for d in durations)
    assert max(durations) - min(durations) >= 1.0
    assert any(d > 5.0 for d in durations)
    assert all(c.duration_reason for c in plan.clips)


def test_auto_candidate_generation_does_not_precut_to_five_seconds():
    scenes = [{"scene_id": 0, "start": 0.0, "end": 14.0, "duration": 14.0}]
    config = HighlightConfig(sequence_duration_mode="auto", max_sequence_duration=5, min_duration=20, max_duration=30)
    candidates = generate_candidates(
        Transcript(),
        scenes,
        {},
        [],
        [],
        14.0,
        config,
        HighlightScorer(),
    )
    assert candidates
    assert any((c["end"] - c["start"]) > 5.1 for c in candidates)


def test_auto_can_keep_a_full_twelve_second_scene():
    scenes = [{"scene_id": 0, "start": 0.0, "end": 12.0, "duration": 12.0}]
    config = HighlightConfig(sequence_duration_mode="auto", max_sequence_duration=5, min_duration=20, max_duration=30)
    candidates = generate_candidates(
        Transcript(),
        scenes,
        {},
        [],
        [],
        12.0,
        config,
        HighlightScorer(),
    )
    assert any(abs((c["end"] - c["start"]) - 12.0) < 0.2 for c in candidates)
