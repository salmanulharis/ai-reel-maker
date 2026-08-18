from app.analysis.optimizer import ClipOptimizer
from app.analysis.scorer import HighlightScorer
from app.models.schemas import CandidateScore, HighlightConfig


def _score(**overrides):
    base = dict(
        importance=0.5,
        novelty=0.4,
        visual_importance=0.4,
        information_density=0.4,
        hook_strength=0.3,
        repetition_score=0.05,
        filler_score=0.05,
        context_dependency=0.2,
        emotional_intensity=0.3,
        production_quality=0.7,
    )
    base.update(overrides)
    return CandidateScore(**base)


def test_action_priority_prefers_visual_action_over_speech():
    scorer = HighlightScorer()
    action = _score(emotional_intensity=0.95, visual_importance=0.9, importance=0.3, information_density=0.2)
    speech = _score(importance=0.95, information_density=0.9, emotional_intensity=0.2, visual_importance=0.2)
    action_first = scorer.finalize(action, priorities=["action"]) > scorer.finalize(speech, priorities=["action"])
    speech_first = scorer.finalize(speech, priorities=["speech"]) > scorer.finalize(action, priorities=["speech"])
    assert action_first
    assert speech_first


def test_balanced_does_not_collapse_to_a_single_type():
    scorer = HighlightScorer()
    action = scorer.finalize(_score(emotional_intensity=0.9, visual_importance=0.85), priorities=["balanced"])
    info = scorer.finalize(_score(importance=0.9, information_density=0.9), priorities=["balanced"])
    assert abs(action - info) < 0.25


def test_highlight_config_normalizes_balanced_and_weights():
    config = HighlightConfig(content_priorities=["action", "balanced", "speech"], priority_weights={"action": 5})
    assert config.content_priorities == ["balanced"]
    config = HighlightConfig(content_priorities=["action", "speech"], priority_weights={"action": 8, "speech": 0})
    assert config.content_priorities == ["action", "speech"]
    assert 0.2 <= config.priority_weights["action"] <= 1.0
    assert config.priority_weights["speech"] >= 0.2


def test_optimizer_keeps_only_the_strongest_similar_action():
    optimizer = ClipOptimizer()
    candidates = [
        {
            "id": "weak",
            "start": 10,
            "end": 18,
            "score": 0.72,
            "topic": "jump",
            "text": "the rider jumps the gap",
            "reason": "Action",
            "metrics": {"emotional_intensity": 0.7, "visual_importance": 0.7},
        },
        {
            "id": "strong",
            "start": 40,
            "end": 50,
            "score": 0.94,
            "topic": "jump",
            "text": "another jump over the same gap",
            "reason": "Action",
            "metrics": {"emotional_intensity": 0.95, "visual_importance": 0.9},
        },
        {
            "id": "talk",
            "start": 70,
            "end": 78,
            "score": 0.8,
            "topic": "story",
            "text": "here is why this trail matters",
            "reason": "Speech",
            "metrics": {"importance": 0.85, "information_density": 0.8},
        },
    ]
    plan = optimizer.select(
        candidates,
        min_duration=10,
        max_duration=20,
        sequence_duration_mode="auto",
        allow_repeated_content=False,
        similarity_fn=lambda x, y: 0.88 if x["topic"] == y["topic"] else 0.1,
    )
    ids = [c.candidate_id for c in plan.clips]
    assert "strong" in ids
    assert "weak" not in ids
    assert "talk" in ids
