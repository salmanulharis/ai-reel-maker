from app.analysis.optimizer import ClipOptimizer
from app.analysis.repetition import RepetitionAnalyzer
from app.analysis.scorer import HighlightScorer
from app.models.schemas import CandidateScore


def _cand(cid, start, end, text, **score):
    metrics = CandidateScore(
        importance=score.get("importance", 0.5),
        novelty=score.get("novelty", 0.5),
        information_density=score.get("information_density", 0.5),
        hook_strength=score.get("hook_strength", 0.2),
        filler_score=score.get("filler_score", 0.05),
        repetition_score=score.get("repetition_score", 0.1),
        visual_importance=score.get("visual_importance", 0.4),
    )
    final = HighlightScorer().finalize(metrics, style=score.get("style", "auto"))
    return {
        "id": cid,
        "start": start,
        "end": end,
        "text": text,
        "score": final,
        "reason": score.get("reason", "Key explanation"),
        "topic": text.split()[0],
    }


def test_interview_drops_greeting_and_repeated_conclusions():
    candidates = [
        _cand("hi", 0, 8, "Hello everyone thanks for joining today", filler_score=0.7, importance=0.2),
        _cand("ans", 70, 82, "The launch succeeded because support stayed local", importance=0.92, information_density=0.9, reason="Strong statement"),
        _cand("rep1", 120, 130, "We launched the product last year", importance=0.7),
        _cand("rep2", 190, 200, "The product was launched a year ago", importance=0.74),
        _cand("end", 210, 222, "The important conclusion is reliability", importance=0.9, reason="Conclusion"),
    ]
    kept = RepetitionAnalyzer().deduplicate(candidates, threshold=0.4)
    plan = ClipOptimizer().select(kept, min_duration=18, max_duration=28, max_sequence_duration=60)
    texts = " ".join(c.transcript or "" for c in plan.clips).lower()
    ids = [c.candidate_id for c in plan.clips]
    assert "hello everyone" not in texts
    assert "ans" in ids or "end" in ids


def test_tutorial_keeps_demonstration_over_dead_air():
    candidates = [
        _cand("pause", 20, 28, "um", filler_score=0.95, importance=0.1),
        _cand("demo", 40, 52, "Click export then choose the high quality preset", importance=0.93, visual_importance=0.88, reason="Important demonstration"),
        _cand("step", 80, 90, "The next step is naming the file", importance=0.8, information_density=0.8),
    ]
    plan = ClipOptimizer().select(candidates, min_duration=15, max_duration=25, max_sequence_duration=60)
    ids = [c.candidate_id for c in plan.clips]
    assert "demo" in ids
    assert "pause" not in ids
