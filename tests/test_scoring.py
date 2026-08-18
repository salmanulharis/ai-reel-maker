from app.analysis.scorer import HighlightScorer, ScoreWeights
from app.models.schemas import CandidateScore


def _score(**overrides):
    base = dict(
        importance=0.9,
        novelty=0.8,
        visual_importance=0.7,
        information_density=0.85,
        hook_strength=0.6,
        repetition_score=0.1,
        filler_score=0.05,
        context_dependency=0.2,
        emotional_intensity=0.5,
        production_quality=0.8,
    )
    base.update(overrides)
    return CandidateScore(**base)


def test_composite_score_uses_configured_weights():
    scorer = HighlightScorer(ScoreWeights())
    value = scorer.composite(_score())
    assert 0.0 <= value <= 1.0
    assert value > 0.7


def test_high_repetition_and_filler_penalize_score():
    scorer = HighlightScorer(ScoreWeights())
    clean = scorer.finalize(_score(repetition_score=0.05, filler_score=0.0, silence_penalty=0.0))
    noisy = scorer.finalize(_score(repetition_score=0.9, filler_score=0.8, silence_penalty=0.4))
    assert noisy < clean


def test_style_informative_boosts_information_density():
    scorer = HighlightScorer(ScoreWeights())
    dense = _score(information_density=0.95, hook_strength=0.1, emotional_intensity=0.1)
    hooky = _score(information_density=0.2, hook_strength=0.95, emotional_intensity=0.9)
    informative = scorer.finalize(dense, style="informative")
    social = scorer.finalize(hooky, style="social")
    assert informative > scorer.finalize(hooky, style="informative")
    assert social > scorer.finalize(dense, style="social")


def test_incomplete_sentence_penalty_reduces_score():
    scorer = HighlightScorer(ScoreWeights())
    complete = scorer.finalize(_score(), incomplete_sentence=False)
    incomplete = scorer.finalize(_score(), incomplete_sentence=True)
    assert incomplete < complete
