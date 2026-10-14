from __future__ import annotations

import numpy as np

from app.analysis.boundaries import context_pad, overlapping_text, snap_window
from app.analysis.scorer import HighlightScorer
from app.analysis.sequence_limits import keep_best_span, sequence_cap
from app.analysis.signals import ENERGY_HOP, POOR_QUALITY, quiet_point
from app.models.schemas import CandidateScore, HighlightConfig, Transcript

# A moment with picture quality below POOR_QUALITY (dark, blurry, black) keeps this share of
# its score, so it is only used when nothing better covers that content.
_POOR_QUALITY_FACTOR = 0.6
# Sentence-complete candidates join at most this many following segments, across short gaps.
_MAX_JOINED_SEGMENTS = 2
_MAX_JOIN_GAP = 1.2
_SENTENCE_END = (".", "!", "?", '"', "'")


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
    scene_quality: dict[int, dict] | None = None,
    energy: list[float] | None = None,
) -> list[dict]:
    """Candidate moments from speech and from scenes, scored and snapped to natural edges.

    scene_quality (picture quality per scene) and energy (loudness curve) are optional
    signals; without them scoring is exactly as before."""
    pad = context_pad(config.preserve_context)
    seq_max = sequence_cap(config)
    candidates: list[dict] = []
    quality_of = _quality_lookup(scenes, scene_quality)
    loudness = _Loudness(energy)

    def transcript_candidate(
        cid: str,
        raw_start: float,
        raw_end: float,
        score_data,
        fallback_text: str,
        reason: str | None,
        hook: float = 0.0,
    ) -> dict:
        start, end, incomplete = snap_window(
            raw_start,
            raw_end,
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
        if hook >= 0.7:
            peak = start + max(0.0, end - start) * 0.25
        start, end = keep_best_span(start, end, seq_max, peak)
        score_model = CandidateScore.model_validate(score_data or {})
        text = overlapping_text(transcript, start, end) or fallback_text
        visual = _visual_at(start, scenes, visual_by_scene)
        score_model.visual_importance = max(score_model.visual_importance, visual.get("visual_importance", 0))
        score_model.silence_penalty = _silence_ratio(start, end, silences)
        score_model.incomplete_sentence = incomplete
        score_model.speech_presence = _speech_presence(transcript, start, end, text)
        quality = quality_of(start, end)
        if quality is not None:
            score_model.production_quality = quality
        loud = loudness.emphasis(start, end)
        audible = loudness.audibility(start, end)
        score_model.natural_audio = (audible or 0.0) * (1.0 - score_model.speech_presence)
        if loud is not None:
            score_model.emotional_intensity = max(score_model.emotional_intensity, loud)
        final = scorer.finalize(
            score_model,
            style=config.style,
            incomplete_sentence=incomplete,
            priorities=config.content_priorities,
            priority_weights=config.priority_weights,
            audio_focus=config.audio_focus,
        )
        if start < 10 and (visual.get("appears_intro") or score_model.filler_score >= 0.5):
            final *= 0.68
        if quality is not None and quality < POOR_QUALITY:
            final *= _POOR_QUALITY_FACTOR
        return {
            "id": cid,
            "start": start,
            "end": end,
            "score": final,
            "text": text,
            "reason": reason or scorer.reason_for(score_model),
            "topic": _topic(text),
            "source": "transcript",
            "metrics": score_model.model_dump(),
            "peak": peak,
        }

    for idx, item in enumerate(segment_scores):
        nested = item.get("score") if isinstance(item.get("score"), dict) else None
        hook = float(nested.get("hook_strength") or 0) if nested else 0.0
        candidates.append(
            transcript_candidate(
                f"t{idx}", item["start"], item["end"], nested or item, item.get("text", ""), item.get("reason"), hook
            )
        )

    # A Whisper segment often stops mid-sentence. Offer the full thought as well, so the
    # planner can choose a clip that does not end mid-sentence.
    for idx, (first, last) in enumerate(_sentence_spans(segment_scores)):
        members = segment_scores[first : last + 1]
        span_start, span_end = float(members[0]["start"]), float(members[-1]["end"])
        if span_end - span_start > seq_max:
            continue
        merged = transcript_candidate(
            f"s{idx}",
            span_start,
            span_end,
            _average_scores(members),
            " ".join(str(m.get("text") or "").strip() for m in members),
            members[0].get("reason"),
            hook=float(_average_scores(members).get("hook_strength") or 0),
        )
        merged["joined_segments"] = last - first + 1
        candidates.append(merged)

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
            if loudness.available and not transcript.words:
                # No speech to align to: cut between sounds rather than through them, but keep
                # cuts that already sit on a scene change.
                if abs(start - float(scene["start"])) > 0.05:
                    start = max(float(scene["start"]), quiet_point(loudness.energy, start, 0.4))
                if abs(end - float(scene["end"])) > 0.05:
                    end = min(float(scene["end"]), quiet_point(loudness.energy, end, 0.4))
            if end - start < 0.8:
                continue
            quality = quality_of(start, end)
            loud = loudness.emphasis(start, end)
            audible = loudness.audibility(start, end)
            text = overlapping_text(transcript, start, end)
            speech_presence = _speech_presence(transcript, start, end, text)
            metrics = CandidateScore(
                importance=max(0.45, visual.get("information_density", 0.4), visual.get("visual_importance", 0.4)),
                visual_importance=max(0.4, visual.get("visual_importance", 0.4)),
                information_density=max(0.4, visual.get("information_density", 0.4)),
                novelty=0.5,
                hook_strength=0.35 if start < 12 else 0.2,
                emotional_intensity=max(visual.get("action_strength", 0.3), loud or 0.0),
                natural_audio=(audible or 0.0) * (1.0 - speech_presence),
                speech_presence=speech_presence,
                repetition_score=visual.get("repetition_likelihood", 0.1),
                production_quality=0.7 if quality is None else quality,
                silence_penalty=_silence_ratio(start, end, silences),
            )
            final = scorer.finalize(
                metrics,
                style=config.style,
                incomplete_sentence=incomplete,
                priorities=config.content_priorities,
                priority_weights=config.priority_weights,
                audio_focus=config.audio_focus,
            )
            poor = quality is not None and quality < POOR_QUALITY
            candidates.append(
                {
                    "id": f"v{scene.get('scene_id', 0)}_{w_i}",
                    "start": start,
                    "end": end,
                    # Scene coverage keeps a minimum score, except when the picture is poor.
                    "score": final * _POOR_QUALITY_FACTOR if poor else max(final, 0.42),
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


def _sentence_spans(segment_scores: list[dict]) -> list[tuple[int, int]]:
    """(first, last) segment indexes for runs that start with an unfinished sentence and end
    at the next sentence end, within a few segments and short gaps."""
    spans = []
    for first, item in enumerate(segment_scores):
        if str(item.get("text") or "").rstrip().endswith(_SENTENCE_END):
            continue
        last = first
        while last + 1 < len(segment_scores) and last - first < _MAX_JOINED_SEGMENTS:
            following = segment_scores[last + 1]
            if float(following["start"]) - float(segment_scores[last]["end"]) > _MAX_JOIN_GAP:
                break
            last += 1
            if str(following.get("text") or "").rstrip().endswith(_SENTENCE_END):
                spans.append((first, last))
                break
    return spans


def _average_scores(members: list[dict]) -> dict:
    """Duration-weighted average of the members' score fields."""
    totals: dict[str, float] = {}
    weight_sum = 0.0
    for member in members:
        data = member.get("score") if isinstance(member.get("score"), dict) else member
        weight = max(0.1, float(member["end"]) - float(member["start"]))
        weight_sum += weight
        for key, value in (data or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0.0) + float(value) * weight
    return {key: value / weight_sum for key, value in totals.items()}


def _quality_lookup(scenes: list[dict], scene_quality: dict[int, dict] | None):
    """Picture quality for a span: the lowest quality among the scenes it covers, or None."""
    if not scene_quality:
        return lambda _start, _end: None
    known = {int(k): v.get("quality") for k, v in scene_quality.items() if isinstance(v, dict)}

    def lookup(start: float, end: float) -> float | None:
        values = [
            known[int(scene["scene_id"])]
            for scene in scenes
            if float(scene["start"]) < end and float(scene["end"]) > start and known.get(int(scene["scene_id"])) is not None
        ]
        return min(values) if values else None

    return lookup


class _Loudness:
    """Unusually loud stretches (applause, laughter, shouting) relative to the whole video."""

    def __init__(self, energy: list[float] | None):
        self.energy = list(energy or [])
        self.available = len(self.energy) >= 8
        self.variation = 0.0
        if self.available:
            values = np.asarray(self.energy, dtype=np.float32)
            self.mean = float(values.mean())
            self.variation = float(values.std())
            self.std = self.variation
            self.std = self.std or 1.0

    def emphasis(self, start: float, end: float) -> float | None:
        """0-1: how far above this video's usual loudness the span is; None without data."""
        if not self.available or end <= start:
            return None
        first = max(0, int(start / ENERGY_HOP))
        last = min(len(self.energy), max(first + 1, int(np.ceil(end / ENERGY_HOP))))
        window = self.energy[first:last]
        if not window:
            return None
        z = (float(np.mean(window)) - self.mean) / self.std
        return float(np.clip((z - 1.0) / 1.5, 0.0, 1.0))

    def audibility(self, start: float, end: float) -> float | None:
        """Upper-quartile normalized energy, to prefer audible non-dialogue moments."""
        if not self.available or self.variation < 0.01 or end <= start:
            return None
        first = max(0, int(start / ENERGY_HOP))
        last = min(len(self.energy), max(first + 1, int(np.ceil(end / ENERGY_HOP))))
        window = self.energy[first:last]
        return float(np.percentile(window, 75)) if window else None


def _speech_presence(transcript: Transcript, start: float, end: float, text: str | None = None) -> float:
    """Estimate how much of a candidate contains speech, using word timestamps when available."""
    duration = max(0.0, end - start)
    if duration <= 0:
        return 0.0
    if transcript.words:
        spoken = sum(
            max(0.0, min(float(word.end), end) - max(float(word.start), start))
            for word in transcript.words
        )
        return float(np.clip(spoken / (duration * 0.55), 0.0, 1.0))
    words = len((text if text is not None else overlapping_text(transcript, start, end)).split())
    return float(np.clip(words / (duration * 2.5), 0.0, 1.0))


def populate_candidate_audio_metrics(candidates: list[dict], transcript: Transcript, energy: list[float] | None) -> None:
    """Backfill audio signals for candidates cached before audio-focus scoring existed."""
    loudness = _Loudness(energy)
    for candidate in candidates:
        metrics = candidate.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        if "natural_audio" in metrics and "speech_presence" in metrics:
            continue
        start, end = float(candidate["start"]), float(candidate["end"])
        text = str(candidate.get("text") or "")
        speech_presence = _speech_presence(transcript, start, end, text)
        audible = loudness.audibility(start, end)
        candidate["metrics"] = {
            **metrics,
            "natural_audio": metrics.get("natural_audio", (audible or 0.0) * (1.0 - speech_presence)),
            "speech_presence": metrics.get("speech_presence", speech_presence),
        }


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
