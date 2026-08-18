import pytest
from pydantic import ValidationError

from app.models.schemas import CandidateScore, VisualAnalysis, ClipPlan, ClipPlanItem


def test_candidate_score_clamps_values_to_unit_interval():
    score = CandidateScore(
        importance=1.4,
        novelty=-0.2,
        visual_importance=0.5,
        information_density=0.94,
        hook_strength=0.84,
        repetition_score=0.08,
        filler_score=0.03,
        context_dependency=0.36,
    )
    assert score.importance == 1.0
    assert score.novelty == 0.0
    assert score.visual_importance == 0.5


def test_visual_analysis_rejects_non_numeric_importance_after_coercion_failure():
    with pytest.raises(ValidationError):
        VisualAnalysis(
            visual_summary="ok",
            visual_importance="high",
            action_strength=0.5,
            information_density=0.5,
            visual_change=0.5,
            repetition_likelihood=0.1,
            context_dependency=0.2,
        )


def test_clip_plan_validates_clip_order_and_duration():
    plan = ClipPlan(
        target_duration=30,
        estimated_duration=28.6,
        clips=[
            ClipPlanItem(start=72.4, end=80.1, reason="Strong key statement", score=0.94),
            ClipPlanItem(start=133.2, end=141.0, reason="Important demonstration", score=0.91),
        ],
    )
    assert len(plan.clips) == 2
    assert plan.clips[0].duration == pytest.approx(7.7, abs=0.01)
