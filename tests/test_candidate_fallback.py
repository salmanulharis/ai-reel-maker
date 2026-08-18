from app.analysis.candidates import generate_candidates
from app.analysis.scorer import HighlightScorer
from app.models.schemas import HighlightConfig, Transcript


def test_generates_scene_windows_when_transcript_is_empty():
    scenes = [{"scene_id": 0, "start": 0.0, "end": 30.5, "duration": 30.5}]
    visual = {0: {"visual_importance": 0.4, "action_strength": 0.2, "information_density": 0.4}}
    candidates = generate_candidates(
        Transcript(),
        scenes,
        visual,
        segment_scores=[],
        silences=[],
        source_duration=30.5,
        config=HighlightConfig(min_duration=20, max_duration=30),
        scorer=HighlightScorer(),
    )
    assert candidates
    assert all(c["end"] > c["start"] for c in candidates)
    assert any(c["end"] - c["start"] <= 12 for c in candidates)
    assert all(c["end"] - c["start"] <= 5.0 + 1e-6 for c in candidates)
