from app.analysis.optimizer import ClipOptimizer


def _c(cid, start, end, score, topic="x", text=""):
    return {
        "id": cid,
        "start": start,
        "end": end,
        "score": score,
        "topic": topic,
        "text": text or topic,
        "reason": "Key explanation",
    }


def test_selects_diverse_clips_instead_of_near_duplicates():
    optimizer = ClipOptimizer()
    candidates = [
        _c("a", 10, 20, 0.95, "speed"),
        _c("b", 22, 32, 0.93, "speed"),
        _c("c", 34, 44, 0.92, "speed"),
        _c("d", 80, 90, 0.89, "pricing"),
        _c("e", 120, 130, 0.87, "demo"),
    ]
    plan = optimizer.select(
        candidates,
        min_duration=20,
        max_duration=30,
        similarity_fn=lambda x, y: 0.9 if x["topic"] == y["topic"] else 0.1,
        max_sequence_duration=60,
    )
    ids = [c.candidate_id for c in plan.clips]
    topics = [next(x["topic"] for x in candidates if x["id"] == cid) for cid in ids]
    assert "pricing" in topics or "demo" in topics
    assert topics.count("speed") <= 1
    assert 18 <= plan.estimated_duration <= 35


def test_respects_must_include_and_must_exclude():
    optimizer = ClipOptimizer()
    candidates = [
        _c("keep", 32, 38, 0.4, "must"),
        _c("skip", 84, 110, 0.99, "exclude"),
        _c("good", 200, 215, 0.9, "unique"),
    ]
    plan = optimizer.select(
        candidates,
        min_duration=10,
        max_duration=30,
        must_include=[(32, 38)],
        must_exclude=[(84, 115)],
        max_sequence_duration=60,
    )
    ids = [c.candidate_id for c in plan.clips]
    assert "keep" in ids
    assert "skip" not in ids


def test_prefers_shorter_high_quality_over_padding_with_low_score():
    optimizer = ClipOptimizer()
    candidates = [
        _c("gold", 10, 22, 0.96, "a"),
        _c("ok", 40, 52, 0.94, "b"),
        _c("filler", 80, 110, 0.2, "c"),
    ]
    plan = optimizer.select(candidates, min_duration=20, max_duration=30, max_sequence_duration=60)
    ids = [c.candidate_id for c in plan.clips]
    assert "filler" not in ids
    assert plan.estimated_duration <= 30


def test_does_not_expand_when_source_is_already_shorter_than_target():
    optimizer = ClipOptimizer()
    candidates = [_c("whole", 0, 18, 0.8, "all")]
    plan = optimizer.select(
        candidates, min_duration=20, max_duration=30, source_duration=18, max_sequence_duration=60
    )
    assert plan.estimated_duration <= 18 + 0.05
    assert plan.notes and "shorter" in plan.notes.lower()


def test_default_order_is_chronological():
    optimizer = ClipOptimizer()
    candidates = [
        _c("late", 200, 210, 0.95, "end"),
        _c("early", 10, 20, 0.9, "start"),
    ]
    plan = optimizer.select(
        candidates, min_duration=15, max_duration=30, order_mode="chronological", max_sequence_duration=60
    )
    assert plan.clips[0].start < plan.clips[1].start
