from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import CandidateScore
from app.utils.timefmt import clamp


@dataclass
class ScoreWeights:
    semantic_importance: float = 0.25
    information_density: float = 0.20
    visual_importance: float = 0.15
    novelty: float = 0.10
    hook_strength: float = 0.10
    emotional: float = 0.10
    natural_audio: float = 0.0
    speech_presence: float = 0.0
    context_quality: float = 0.05
    production_quality: float = 0.05
    repetition_penalty: float = 0.35
    filler_penalty: float = 0.25
    silence_penalty: float = 0.20
    incomplete_penalty: float = 0.18
    context_dependency_penalty: float = 0.12


STYLE_EMPHASIS = {
    "auto": {},
    "informative": {
        "information_density": 0.28,
        "semantic_importance": 0.28,
        "hook_strength": 0.05,
        "emotional": 0.04,
    },
    "educational": {
        "information_density": 0.30,
        "semantic_importance": 0.26,
        "hook_strength": 0.05,
        "emotional": 0.04,
    },
    "exciting": {
        "emotional": 0.22,
        "visual_importance": 0.22,
        "hook_strength": 0.16,
        "information_density": 0.10,
    },
    "social": {
        "hook_strength": 0.24,
        "emotional": 0.16,
        "novelty": 0.16,
        "information_density": 0.12,
    },
    "custom": {},
}

PRIORITY_EMPHASIS = {
    "balanced": {},
    "action": {
        "emotional": 0.30,
        "visual_importance": 0.26,
        "hook_strength": 0.12,
        "information_density": 0.10,
        "semantic_importance": 0.12,
    },
    "speech": {
        "semantic_importance": 0.34,
        "information_density": 0.22,
        "visual_importance": 0.08,
        "emotional": 0.06,
    },
    "magic": {
        "novelty": 0.24,
        "emotional": 0.24,
        "hook_strength": 0.20,
        "visual_importance": 0.14,
    },
    "visual": {
        "visual_importance": 0.34,
        "emotional": 0.16,
        "novelty": 0.14,
        "information_density": 0.08,
    },
    "information": {
        "information_density": 0.34,
        "semantic_importance": 0.28,
        "hook_strength": 0.06,
        "emotional": 0.05,
    },
    "connection": {
        "emotional": 0.32,
        "visual_importance": 0.28,
        "novelty": 0.14,
        "semantic_importance": 0.12,
        "information_density": 0.08,
        "hook_strength": 0.06,
    },
}

READABLE_REASONS = [
    (lambda s: s.hook_strength >= 0.75, "Strong hook"),
    (lambda s: s.importance >= 0.85 and s.information_density >= 0.7, "Key explanation"),
    (lambda s: s.novelty >= 0.75, "Unique information"),
    (lambda s: s.visual_importance >= 0.75 and s.emotional_intensity >= 0.55, "Important demonstration"),
    (lambda s: s.emotional_intensity >= 0.8, "Strong reaction"),
    (lambda s: s.importance >= 0.8, "Strong statement"),
    (lambda s: s.information_density >= 0.7, "Useful information"),
]


class HighlightScorer:
    def __init__(self, weights: ScoreWeights | None = None):
        self.weights = weights or ScoreWeights()

    def _weights_for(
        self,
        style: str,
        priorities: list[str] | None = None,
        priority_weights: dict[str, float] | None = None,
        audio_focus: str = "balanced",
    ) -> ScoreWeights:
        weights = ScoreWeights(**self.weights.__dict__)
        for key, value in STYLE_EMPHASIS.get(style, {}).items():
            setattr(weights, key, value)
        selected = [item for item in (priorities or []) if item and item != "balanced"]
        if not selected:
            if audio_focus == "natural":
                weights.natural_audio = 0.25
            elif audio_focus == "dialogue":
                weights.speech_presence = 0.25
            return weights
        scale = priority_weights or {}
        blended: dict[str, float] = {}
        total = 0.0
        for name in selected:
            factor = max(0.2, float(scale.get(name, 1.0)))
            total += factor
            for key, value in PRIORITY_EMPHASIS.get(name, {}).items():
                blended[key] = blended.get(key, 0.0) + value * factor
        if total:
            for key, value in blended.items():
                setattr(weights, key, value / total)
        if audio_focus == "natural":
            weights.natural_audio = 0.25
        elif audio_focus == "dialogue":
            weights.speech_presence = 0.25
        return weights

    def composite(
        self,
        score: CandidateScore,
        style: str = "auto",
        priorities: list[str] | None = None,
        priority_weights: dict[str, float] | None = None,
        audio_focus: str = "balanced",
    ) -> float:
        w = self._weights_for(style, priorities, priority_weights, audio_focus)
        total = (
            w.semantic_importance * score.importance
            + w.information_density * score.information_density
            + w.visual_importance * score.visual_importance
            + w.novelty * score.novelty
            + w.hook_strength * score.hook_strength
            + w.emotional * score.emotional_intensity
            + w.natural_audio * score.natural_audio
            + w.speech_presence * score.speech_presence
            + w.context_quality * (1.0 - score.context_dependency)
            + w.production_quality * score.production_quality
        )
        weight_sum = (
            w.semantic_importance
            + w.information_density
            + w.visual_importance
            + w.novelty
            + w.hook_strength
            + w.emotional
            + w.natural_audio
            + w.speech_presence
            + w.context_quality
            + w.production_quality
        )
        return clamp(total / max(weight_sum, 1e-6))

    def finalize(
        self,
        score: CandidateScore,
        style: str = "auto",
        incomplete_sentence: bool | None = None,
        priorities: list[str] | None = None,
        priority_weights: dict[str, float] | None = None,
        audio_focus: str = "balanced",
    ) -> float:
        value = self.composite(
            score,
            style=style,
            priorities=priorities,
            priority_weights=priority_weights,
            audio_focus=audio_focus,
        )
        w = self.weights
        value -= w.repetition_penalty * score.repetition_score
        value -= w.filler_penalty * score.filler_score
        value -= w.silence_penalty * score.silence_penalty
        value -= w.context_dependency_penalty * max(0.0, score.context_dependency - 0.55)
        incomplete = score.incomplete_sentence if incomplete_sentence is None else incomplete_sentence
        if incomplete:
            value -= w.incomplete_penalty
        return clamp(value)

    def reason_for(self, score: CandidateScore, fallback: str = "Meaningful moment") -> str:
        for predicate, label in READABLE_REASONS:
            if predicate(score):
                return label
        if score.importance >= 0.55:
            return "Relevant moment"
        return fallback
