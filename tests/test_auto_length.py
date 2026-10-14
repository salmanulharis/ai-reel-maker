"""Automatic highlight length ("No limit") and Auto longest clip."""

from __future__ import annotations

import json

from app.analysis.optimizer import ClipOptimizer
from app.analysis.sequence_limits import AUTO_LENGTH_CLIP_CAP, sequence_cap
from app.models.schemas import HighlightConfig


def _c(cid, start, score, length=10.0):
    return {
        "id": cid,
        "start": float(start),
        "end": float(start) + length,
        "score": score,
        "topic": cid,
        "text": f"distinct topic {cid}",
        "reason": "Key explanation",
    }


def _different(_a, _b):
    return 0.1


def _select(candidates, *, mode="auto", strict=False, min_duration=10, max_duration=30, source=600.0):
    return ClipOptimizer().select(
        candidates,
        min_duration=min_duration,
        max_duration=max_duration,
        source_duration=source,
        similarity_fn=_different,
        max_sequence_duration=60,
        sequence_duration_mode="custom",
        length_mode=mode,
        auto_length_strict=strict,
    )


STRONG_AND_WEAK = [
    _c("a", 0, 0.92),
    _c("b", 40, 0.88),
    _c("c", 80, 0.85),
    _c("d", 120, 0.8),
    _c("e", 160, 0.75),
    _c("f", 200, 0.62),  # above the floor (0.35) but well below the best (0.92 * 0.65 = 0.598 -> kept)
    _c("g", 240, 0.5),  # below 65% of the best -> dropped
    _c("h", 280, 0.4),
]


def test_no_limit_keeps_every_strong_moment_beyond_the_old_maximum():
    plan = _select(STRONG_AND_WEAK)
    kept = {clip.candidate_id for clip in plan.clips}
    assert kept == {"a", "b", "c", "d", "e", "f"}
    assert plan.estimated_duration == 60  # longer than max_duration=30: it is not used as a cap
    assert plan.target_duration == plan.estimated_duration


def test_range_mode_is_unchanged():
    plan = _select(STRONG_AND_WEAK, mode="range")
    assert plan.estimated_duration <= 30


def test_shorter_raises_the_quality_bar():
    strict = _select(STRONG_AND_WEAK, strict=True)
    kept = {clip.candidate_id for clip in strict.clips}
    assert kept == {"a", "b", "c", "d", "e"}  # >= 0.92 * 0.8 = 0.736; "f" (0.62) is dropped


def test_shortest_length_is_filled_with_acceptable_moments():
    # 0.45 is under 65% of the best (0.585) but above the quality floor, so it may fill the minimum.
    candidates = [_c("a", 0, 0.9), _c("b", 40, 0.45), _c("c", 80, 0.45)]
    plan = _select(candidates, min_duration=25)
    assert plan.estimated_duration >= 25


def test_poor_moments_are_not_added_just_to_reach_the_shortest_length():
    # Same rule as a fixed range: a shorter, good highlight beats padding with weak moments.
    candidates = [_c("a", 0, 0.9), _c("b", 40, 0.2), _c("c", 80, 0.2), _c("d", 120, 0.2)]
    plan = _select(candidates, min_duration=40)
    assert plan.estimated_duration < 40


def test_uniform_scores_are_capped_at_a_quarter_of_the_video():
    candidates = [_c(str(i), i * 12, 0.9) for i in range(40)]
    plan = _select(candidates, source=480.0)
    assert plan.estimated_duration <= 480 * 0.25 + 1e-6
    assert plan.estimated_duration >= 100


def test_unknown_source_length_falls_back_to_the_range_maximum():
    plan = _select(STRONG_AND_WEAK, source=None)
    assert plan.estimated_duration <= 30


def test_auto_clip_length_has_its_own_cap_without_a_highlight_maximum():
    auto = HighlightConfig(length_mode="auto", sequence_duration_mode="auto", max_duration=20)
    assert sequence_cap(auto) == AUTO_LENGTH_CLIP_CAP
    ranged = HighlightConfig(length_mode="range", sequence_duration_mode="auto", max_duration=20)
    assert sequence_cap(ranged) == 20
    limited = HighlightConfig(length_mode="auto", sequence_duration_mode="custom", max_sequence_duration=6)
    assert sequence_cap(limited) == 6


def test_auto_clip_length_with_a_shortest_value_is_valid():
    config = HighlightConfig(sequence_duration_mode="auto", min_sequence_duration=3)
    assert config.sequence_duration_mode == "auto"
    assert config.min_sequence_duration == 3


def test_social_short_keeps_its_limited_range():
    config = HighlightConfig(length_mode="auto", social_short=True, min_duration=15, max_duration=60)
    assert config.length_mode == "range"


def test_older_jobs_default_to_a_range():
    assert HighlightConfig.model_validate({"min_duration": 20, "max_duration": 30}).length_mode == "range"


def test_shorter_on_an_automatic_plan_is_stricter_not_capped(tmp_path):
    from app import create_app
    from app.config import Settings, get_settings, set_settings
    from app.services.job_manager import get_job_manager

    previous = get_settings()
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    try:
        app = create_app(settings, start_background=False)
        manager = get_job_manager()
        from app.models.db_models import Job, get_session

        session = get_session()
        config = HighlightConfig(length_mode="auto", min_duration=10, max_duration=30)
        session.add(
            Job(
                id="job1",
                status="planning",
                original_filename="v.mp4",
                input_path=str(tmp_path / "v.mp4"),
                style="auto",
                min_duration=10,
                max_duration=30,
                config_json=config.model_dump_json(),
            )
        )
        session.commit()
        session.close()
        job_dir = manager.job_dir("job1")
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "metadata.json").write_text(json.dumps({"duration": 600}), encoding="utf-8")
        (job_dir / "candidates.json").write_text(json.dumps(STRONG_AND_WEAK), encoding="utf-8")

        response = app.test_client().post("/api/jobs/job1/regenerate", json={"preset": "more_concise"})
        assert response.status_code == 200
        saved = HighlightConfig.model_validate_json(manager.get("job1").config_json)
        assert saved.length_mode == "auto"
        assert saved.auto_length_strict is True
        assert saved.max_duration == 30
    finally:
        set_settings(previous)
