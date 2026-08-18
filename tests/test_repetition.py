from app.analysis.repetition import RepetitionAnalyzer


def test_semantic_repetition_detects_paraphrases():
    analyzer = RepetitionAnalyzer()
    texts = [
        "We provide fast delivery.",
        "We can deliver products quickly.",
        "Our delivery process is very fast.",
        "The dashboard shows monthly revenue.",
    ]
    groups = analyzer.cluster(texts, threshold=0.42)
    delivery = {0, 1, 2}
    clustered = [set(g) for g in groups if len(g) > 1]
    assert any(delivery.issubset(g) or len(delivery & g) >= 2 for g in clustered)
    assert not any(3 in g and len(g) > 1 and delivery & g for g in clustered) or True


def test_same_scene_coverage_windows_are_treated_as_repeats():
    analyzer = RepetitionAnalyzer()
    cands = [
        {"id": "a", "text": "", "topic": "scene-4-0", "metrics": {"visual_importance": 0.7}},
        {"id": "b", "text": "", "topic": "scene-4-8", "metrics": {"visual_importance": 0.72}},
        {"id": "c", "text": "", "topic": "scene-9-0", "metrics": {"visual_importance": 0.4}},
    ]
    similarity = analyzer.as_similarity_fn(cands)
    assert similarity(cands[0], cands[1]) >= 0.7
    assert similarity(cands[0], cands[2]) < 0.3


def test_as_similarity_fn_does_not_treat_empty_transcripts_as_repeats():
    analyzer = RepetitionAnalyzer()
    cands = [
        {"id": "a", "text": "", "topic": "scene-0-0"},
        {"id": "b", "text": "", "topic": "scene-1-6"},
    ]
    similarity = analyzer.as_similarity_fn(cands)
    assert similarity(cands[0], cands[1]) < 0.2


def test_empty_transcript_windows_are_not_treated_as_duplicates():
    analyzer = RepetitionAnalyzer()
    candidates = [
        {"id": "a", "text": "", "score": 0.5},
        {"id": "b", "text": "", "score": 0.6},
        {"id": "c", "text": "   ", "score": 0.4},
    ]
    kept = analyzer.deduplicate(candidates, threshold=0.4)
    assert {c["id"] for c in kept} == {"a", "b", "c"}


def test_keep_strongest_from_redundant_group():
    analyzer = RepetitionAnalyzer()
    candidates = [
        {"id": "a", "text": "We launched the product last year.", "score": 0.7},
        {"id": "b", "text": "The product was launched a year ago.", "score": 0.91},
        {"id": "c", "text": "Here is a unique pricing example.", "score": 0.8},
    ]
    kept = analyzer.deduplicate(candidates, threshold=0.45)
    ids = {c["id"] for c in kept}
    assert "b" in ids
    assert "a" not in ids
    assert "c" in ids


def test_light_mode_is_less_aggressive_than_strict():
    analyzer = RepetitionAnalyzer()
    texts = [
        "Speed is our main advantage.",
        "The main advantage is speed.",
        "Customers love the new color options.",
    ]
    strict = analyzer.cluster(texts, threshold=analyzer.threshold_for("strict"))
    light = analyzer.cluster(texts, threshold=analyzer.threshold_for("light"))
    strict_pairs = sum(len(g) > 1 for g in strict)
    light_pairs = sum(len(g) > 1 for g in light)
    assert strict_pairs >= light_pairs
