from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.ai.json_util import clamp_unit
from app.utils.timefmt import clamp


def _unit(value: Any) -> float:
    if isinstance(value, str):
        text = value.strip().lower().replace("%", "")
        float(text)
    return clamp_unit(value)


class CandidateScore(BaseModel):
    model_config = ConfigDict(extra="ignore")

    importance: float = 0.0
    novelty: float = 0.0
    visual_importance: float = 0.0
    information_density: float = 0.0
    hook_strength: float = 0.0
    repetition_score: float = 0.0
    filler_score: float = 0.0
    context_dependency: float = 0.0
    emotional_intensity: float = 0.0
    natural_audio: float = 0.0
    speech_presence: float = 0.0
    production_quality: float = 0.5
    silence_penalty: float = 0.0
    incomplete_sentence: bool = False

    @field_validator(
        "importance",
        "novelty",
        "visual_importance",
        "information_density",
        "hook_strength",
        "repetition_score",
        "filler_score",
        "context_dependency",
        "emotional_intensity",
        "natural_audio",
        "speech_presence",
        "production_quality",
        "silence_penalty",
        mode="before",
    )
    @classmethod
    def _clamp(cls, value: Any) -> float:
        return _unit(value)


class VisualAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    visual_summary: str = ""
    visual_importance: float = 0.0
    action_strength: float = 0.0
    information_density: float = 0.0
    visual_change: float = 0.0
    repetition_likelihood: float = 0.0
    context_dependency: float = 0.0
    has_person: bool = False
    has_text: bool = False
    appears_intro: bool = False
    appears_ending: bool = False
    surprising: bool = False
    focus_region: tuple[float, float, float, float] | None = None

    @field_validator(
        "visual_importance",
        "action_strength",
        "information_density",
        "visual_change",
        "repetition_likelihood",
        "context_dependency",
        mode="before",
    )
    @classmethod
    def _clamp(cls, value: Any) -> float:
        return _unit(value)


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class Transcript(BaseModel):
    segments: list[TranscriptSegment] = Field(default_factory=list)
    words: list[WordTimestamp] = Field(default_factory=list)
    language: str | None = None


class Scene(BaseModel):
    scene_id: int
    start: float
    end: float
    duration: float = 0.0

    @model_validator(mode="after")
    def _duration(self) -> "Scene":
        if self.duration <= 0:
            self.duration = max(0.0, self.end - self.start)
        return self


class VideoMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    # Frames divided by duration. Unlike the declared rate, it is not thrown off by small
    # timestamp gaps, such as at the joins of clips concatenated without re-encoding.
    avg_fps: float = 0.0
    bitrate: int = 0
    video_codec: str = ""
    audio_codec: str = ""
    audio_sample_rate: int = 0
    channels: int = 0
    rotation: int = 0
    pixel_format: str = ""
    has_audio: bool = False
    size_bytes: int = 0
    filename: str = ""


class ClipPlanItem(BaseModel):
    start: float
    end: float
    reason: str
    score: float = 0.0
    candidate_id: str | None = None
    transcript: str = ""
    thumbnail: str | None = None
    duration_reason: str = ""
    source_id: str = ""
    source_name: str = ""
    source_path: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @field_validator("score", mode="before")
    @classmethod
    def _score(cls, value: Any) -> float:
        return clamp(float(value), 0.0, 1.0) if value is not None else 0.0

    @model_validator(mode="after")
    def _order(self) -> "ClipPlanItem":
        if self.end < self.start:
            raise ValueError("Clip end must be after start")
        return self


class ClipPlan(BaseModel):
    target_duration: float
    estimated_duration: float
    clips: list[ClipPlanItem] = Field(default_factory=list)
    notes: str = ""
    order_mode: str = "chronological"

    @property
    def clip_count(self) -> int:
        return len(self.clips)


class CropRegion(BaseModel):
    x: float = 0.0
    y: float = 0.0
    width: float = 1.0
    height: float = 1.0

    @model_validator(mode="after")
    def _clamp(self) -> "CropRegion":
        self.x = min(max(self.x, 0.0), 1.0)
        self.y = min(max(self.y, 0.0), 1.0)
        self.width = min(max(self.width, 0.0), 1.0 - self.x)
        self.height = min(max(self.height, 0.0), 1.0 - self.y)
        return self


class HighlightConfig(BaseModel):
    min_duration: float = 20.0
    max_duration: float = 30.0
    # "range": aim for min..max. "auto": no maximum; keep every clearly strong moment, at least
    # min_duration long (max_duration is kept but not used as a cap).
    length_mode: Literal["range", "auto"] = "range"
    # Set by "Shorter" on an automatic-length plan: only keep moments closer to the very best.
    auto_length_strict: bool = False
    style: Literal["auto", "informative", "exciting", "social", "educational", "custom"] = "auto"
    custom_focus: str = ""
    preserve_context: Literal["low", "balanced", "high"] = "balanced"
    remove_repetition: bool = True
    repetition_mode: Literal["strict", "balanced", "light"] = "balanced"
    order_mode: Literal["chronological", "narrative", "ai"] = "chronological"
    social_short: bool = False
    aspect_ratio: Literal["original", "16:9", "9:16", "1:1", "4:5"] = "original"
    subtitles: Literal["none", "auto", "burned", "srt"] = "none"
    transition: Literal["cut", "fade"] = "cut"
    audio_focus: Literal["balanced", "natural", "dialogue"] = "balanced"
    # "source" keeps the original frame rate; a number converts the output to that rate.
    frame_rate: Literal["source", "24", "25", "30", "50", "60"] = "source"
    # Create in-between frames when converting would otherwise judder. Much slower.
    smooth_motion: bool = False
    output_format: str = "mp4"
    quality: Literal["high", "very_high", "source", "custom"] = "source"
    codec: str = "auto"
    resolution: str = "original"
    must_include: list[tuple[float, float]] = Field(default_factory=list)
    must_exclude: list[tuple[float, float]] = Field(default_factory=list)
    max_sequence_duration: float = 5.0
    min_sequence_duration: float | None = None
    sequence_duration_mode: Literal["auto", "custom"] = "auto"
    allow_repeated_content: bool | None = None
    max_repeated_duration: float = 15.0
    crop: CropRegion | None = None
    content_priorities: list[str] = Field(default_factory=lambda: ["balanced"])
    priority_weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _range(self) -> "HighlightConfig":
        if self.social_short and self.aspect_ratio == "original":
            self.aspect_ratio = "9:16"
        if self.social_short:
            # Social platforms limit the length, so a short always uses its range.
            self.length_mode = "range"
        if self.min_duration > self.max_duration:
            self.min_duration, self.max_duration = self.max_duration, self.min_duration
        if self.allow_repeated_content is None:
            self.allow_repeated_content = not self.remove_repetition
        else:
            self.remove_repetition = not self.allow_repeated_content
        self.max_sequence_duration = max(1.0, float(self.max_sequence_duration))
        if self.min_sequence_duration is not None:
            self.min_sequence_duration = max(0.5, float(self.min_sequence_duration))
        if self.sequence_duration_mode == "custom":
            if self.min_sequence_duration is None:
                self.min_sequence_duration = 1.5
            if self.min_sequence_duration > self.max_sequence_duration:
                self.min_sequence_duration = self.max_sequence_duration
        if self.sequence_duration_mode not in {"auto", "custom"}:
            self.sequence_duration_mode = "auto"
        self.max_repeated_duration = max(0.0, float(self.max_repeated_duration))
        allowed = {"balanced", "action", "speech", "magic", "visual", "information", "connection"}
        selected = []
        for item in self.content_priorities or []:
            key = str(item).strip().lower()
            if key in allowed and key not in selected:
                selected.append(key)
        if not selected or "balanced" in selected:
            selected = ["balanced"]
        self.content_priorities = selected
        weights = {}
        for key, raw in (self.priority_weights or {}).items():
            name = str(key).strip().lower()
            if name not in selected or name == "balanced":
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = 1.0
            if value > 1.0:
                value = value / 5.0
            weights[name] = min(1.0, max(0.2, value))
        for name in selected:
            if name != "balanced" and name not in weights:
                weights[name] = 1.0
        self.priority_weights = weights
        return self


class RenderSettings(BaseModel):
    format: str = "mp4"
    quality: str = "source"
    codec: str = "auto"
    crf: int = 18
    resolution: str = "original"
    aspect_ratio: str = "original"
    subtitles: str = "none"
    transition: str = "cut"
    frame_rate: str = "source"
    smooth_motion: bool = False
    audio_bitrate: str = "192k"
