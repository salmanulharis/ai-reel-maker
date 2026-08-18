from __future__ import annotations

from app.analysis.boundaries import context_pad, overlapping_text, snap_window
from app.analysis.scorer import HighlightScorer
from app.analysis.sequence_limits import keep_best_span, sequence_cap
from app.models.schemas import CandidateScore, HighlightConfig, Transcript


def generate_candidates(
    transcript: Transcript,
    scenes: list[dict],
    visual_by_scene: dict[int, dict],
    segment_scores: list[dict],
    silences: list[dict],
    source_duration: float,
    config: HighlightConfig,
    scorer: HighlightScorer,
    max_candidates: int = 80,
) -> list[dict]:
    pad = context_pad(config.preserve_context)
    seq_max = sequence_cap(config)
    candidates: list[dict] = []
    idx = 0

    for item in segment_scores:
        start, end, incomplete = snap_window(
            item["start"],
            item["end"],
            transcript,
            silences,
            scenes,
            source_duration,
            pad=pad,
        )
        duration = end - start
        if duration < 1.0:
            end = min(source_duration, start + 1.2)
        peak = (start + end) / 2
        hook = float((item.get("score") or {}).get("hook_strength") or 0) if isinstance(item.get("score"), dict) else 0.0
        if hook >= 0.7:
            peak = start + max(0.0, end - start) * 0.25
        start, end = keep_best_span(start, end, seq_max, peak)
        score_model = CandidateScore.model_validate(item.get("score") or item)
        text = overlapping_text(transcript, start, end) or item.get("text", "")
        visual = _visual_at(start, scenes, visual_by_scene)
        score_model.visual_importance = max(score_model.visual_importance, visual.get("visual_importance", 0))
        score_model.silence_penalty = _silence_ratio(start, end, silences)
        score_model.incomplete_sentence = incomplete
        final = scorer.finalize(
            score_model,
            style=config.style,
            incomplete_sentence=incomplete,
            priorities=config.content_priorities,
            priority_weights=config.priority_weights,
        )
        if start < 10 and (visual.get("appears_intro") or score_model.filler_score >= 0.5):
            final *= 0.68
        reason = item.get("reason") or scorer.reason_for(score_model)
        candidates.append(
            {
                "id": f"t{idx}",
                "start": start,
                "end": end,
                "score": final,
                "text": text,
                "reason": reason,
                "topic": _topic(text),
                "source": "transcript",
                "metrics": score_model.model_dump(),
                "peak": peak,
            }
        )
        idx += 1

    for scene in scenes:
        visual = visual_by_scene.get(int(scene.get("scene_id", 0)), {})
        windows = list(_coverage_windows(float(scene["start"]), float(scene["end"]), window=seq_max))
        if not windows:
            windows = [(float(scene["start"]), min(float(scene["end"]), float(scene["start"]) + seq_max))]
        for w_i, (win_start, win_end) in enumerate(windows):
            start, end, incomplete = snap_window(
                win_start,
                win_end,
                transcript,
                silences,
                scenes,
                source_duration,
                pad=pad,
            )
            start, end = keep_best_span(start, end, seq_max)
            if end - start < 0.8:
                continue
            metrics = CandidateScore(
                importance=max(0.45, visual.get("information_density", 0.4), visual.get("visual_importance", 0.4)),
                visual_importance=max(0.4, visual.get("visual_importance", 0.4)),
                information_density=max(0.4, visual.get("information_density", 0.4)),
                novelty=0.5,
                hook_strength=0.35 if start < 12 else 0.2,
                emotional_intensity=visual.get("action_strength", 0.3),
                repetition_score=visual.get("repetition_likelihood", 0.1),
                production_quality=0.7,
                silence_penalty=_silence_ratio(start, end, silences),
            )
            text = overlapping_text(transcript, start, end)
            final = scorer.finalize(
                metrics,
                style=config.style,
                incomplete_sentence=incomplete,
                priorities=config.content_priorities,
                priority_weights=config.priority_weights,
            )
            candidates.append(
                {
                    "id": f"v{scene.get('scene_id', 0)}_{w_i}",
                    "start": start,
                    "end": end,
                    "score": max(final, 0.42),
                    "text": text,
                    "reason": scorer.reason_for(metrics, "Scene coverage"),
                    "topic": _topic(text) if (text or "").strip() else f"scene-{scene.get('scene_id', 0)}-{int(start)}",
                    "source": "visual",
                    "metrics": metrics.model_dump(),
                }
            )

    if not candidates and source_duration > 0:
        for w_i, (win_start, win_end) in enumerate(_coverage_windows(0.0, source_duration, window=seq_max)):
            candidates.append(
                {
                    "id": f"fb{w_i}",
                    "start": win_start,
                    "end": win_end,
                    "score": 0.45,
                    "text": overlapping_text(transcript, win_start, win_end),
                    "reason": "Coverage window",
                    "topic": f"coverage-{w_i}",
                    "source": "fallback",
                    "metrics": {},
                }
            )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:max_candidates]


def _coverage_windows(start: float, end: float, window: float = 8.0, overlap: float = 1.0) -> list[tuple[float, float]]:
    if end - start <= window + 0.5:
        return [(start, end)] if end > start else []
    spans: list[tuple[float, float]] = []
    t = start
    while t < end - 0.8:
        nxt = min(end, t + window)
        if nxt - t >= 1.0:
            spans.append((round(t, 3), round(nxt, 3)))
        if nxt >= end:
            break
        t = nxt - overlap
    return spans


def _visual_at(t: float, scenes: list[dict], visual_by_scene: dict[int, dict]) -> dict:
    for scene in scenes:
        if scene["start"] <= t <= scene["end"]:
            return visual_by_scene.get(scene["scene_id"], {})
    return {}


def _silence_ratio(start: float, end: float, silences: list[dict]) -> float:
    dur = max(end - start, 0.01)
    covered = 0.0
    for silence in silences:
        a, b = max(start, silence["start"]), min(end, silence["end"])
        if b > a:
            covered += b - a
    return min(1.0, covered / dur)


def _topic(text: str) -> str:
    words = [w.lower() for w in text.replace(".", " ").split() if len(w) > 3]
    return " ".join(words[:6]) or "moment"
