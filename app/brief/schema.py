"""The brief's format: what the engines reply, and what is stored in ``brief.json``."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

BRIEF_FILE = "brief.json"
BRIEF_DIR = "brief"
VERSION = 1

# Speaking rate of the narration voice. Measured with scripts/voice_benchmark.py on a 6-core CPU
# at speed 1.0: af_heart 3.1, am_michael 2.8 words per second. Every beat gets a word budget from
# its length (times NARRATION_SHARE), so the voice fits the pictures it describes.
WORDS_PER_SECOND = 2.8
# Share of a beat the voice may fill: a little air before the next beat starts.
NARRATION_SHARE = 0.85
# Beats shorter than this get no narration (on-screen text only).
MIN_NARRATED_SECONDS = 1.6
MAX_BEATS = 12
MAX_CLIPS_PER_BEAT = 4

PURPOSES = ("hook", "context", "key", "build", "payoff", "highlight", "reaction", "ending")
MUSIC_MOODS = ("none", "calm", "uplifting", "energetic", "dramatic", "playful", "emotional")

_TEXT_LIMITS = {
    "title": 80,
    "hook": 160,
    "tone": 60,
    "call_to_action": 100,
    "narration": 400,
    "on_screen_text": 60,
}


def _clean(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


class BeatReply(BaseModel):
    """One beat as an engine writes it: moments by alias ("m12"), never by time."""

    model_config = ConfigDict(extra="ignore")

    clips: list[str] = Field(default_factory=list)
    purpose: str = "highlight"
    narration: str = ""
    on_screen_text: str = ""

    @field_validator("clips", mode="before")
    @classmethod
    def _aliases(cls, value: Any) -> list[str]:
        items = value if isinstance(value, list) else []
        result = []
        for item in items:
            alias = str(item or "").strip().lower()
            if alias and not alias.startswith("m"):
                alias = f"m{alias}"
            if alias:
                result.append(alias)
        return result

    @field_validator("purpose", mode="before")
    @classmethod
    def _purpose(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        return text if text in PURPOSES else "highlight"

    @field_validator("narration", mode="before")
    @classmethod
    def _narration(cls, value: Any) -> str:
        return _clean(value, _TEXT_LIMITS["narration"])

    @field_validator("on_screen_text", mode="before")
    @classmethod
    def _on_screen(cls, value: Any) -> str:
        return _clean(value, _TEXT_LIMITS["on_screen_text"])


class BriefReply(BaseModel):
    """An engine's whole reply."""

    model_config = ConfigDict(extra="ignore")

    title: str = ""
    hook: str = ""
    tone: str = ""
    music_mood: str = "none"
    call_to_action: str = ""
    beats: list[BeatReply] = Field(default_factory=list)

    @field_validator("title", "hook", "tone", "call_to_action", mode="before")
    @classmethod
    def _texts(cls, value: Any, info) -> str:
        return _clean(value, _TEXT_LIMITS[info.field_name])

    @field_validator("music_mood", mode="before")
    @classmethod
    def _mood(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        return text if text in MUSIC_MOODS else "none"

    @field_validator("beats", mode="before")
    @classmethod
    def _beats(cls, value: Any) -> list:
        items = value if isinstance(value, list) else []
        return [item for item in items if isinstance(item, (dict, BeatReply))][: MAX_BEATS * 2]


# The reply format given to both engines (Ollama's ``format`` and the CLI's ``--json-schema``).
# Written out flat (no $ref), which both accept; tests keep it in step with BriefReply.
REPLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "hook", "tone", "music_mood", "call_to_action", "beats"],
    "properties": {
        "title": {"type": "string"},
        "hook": {"type": "string"},
        "tone": {"type": "string"},
        "music_mood": {"type": "string", "enum": list(MUSIC_MOODS)},
        "call_to_action": {"type": "string"},
        "beats": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["clips", "purpose", "narration", "on_screen_text"],
                "properties": {
                    "clips": {"type": "array", "items": {"type": "string"}},
                    "purpose": {"type": "string", "enum": list(PURPOSES)},
                    "narration": {"type": "string"},
                    "on_screen_text": {"type": "string"},
                },
            },
        },
    },
}

# The local engine writes the words for beats the AI editor already chose (see engines.local).
WRITER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["beats"],
    "properties": {
        "title": {"type": "string"},
        "hook": {"type": "string"},
        "tone": {"type": "string"},
        "music_mood": {"type": "string", "enum": list(MUSIC_MOODS)},
        "call_to_action": {"type": "string"},
        "beats": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "narration", "on_screen_text"],
                "properties": {
                    "id": {"type": "string"},
                    "narration": {"type": "string"},
                    "on_screen_text": {"type": "string"},
                },
            },
        },
    },
}


class BriefClip(BaseModel):
    """A moment of a beat, resolved from its alias when the brief was checked."""

    model_config = ConfigDict(extra="ignore")

    id: str
    source_id: str = "v0"
    start: float
    end: float


class Beat(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    purpose: str = "highlight"
    narration: str = ""
    on_screen_text: str = ""
    # Most words the narration may have for the moments chosen (set when the brief is checked).
    max_words: int = 0
    clips: list[BriefClip] = Field(default_factory=list)

    @property
    def seconds(self) -> float:
        return sum(max(0.0, clip.end - clip.start) for clip in self.clips)


class VideoBrief(BaseModel):
    """A checked brief, as stored in brief.json and edited on the Review page."""

    model_config = ConfigDict(extra="ignore")

    version: int = VERSION
    key: str = ""
    engine: str = "local"
    model: str = ""
    title: str = ""
    hook: str = ""
    tone: str = ""
    music_mood: str = "none"
    call_to_action: str = ""
    beats: list[Beat] = Field(default_factory=list)
    # What the checks found after the repair request (shown on Review), and whether the user
    # edited the text since.
    checks: list[str] = Field(default_factory=list)
    edited: bool = False

    def narration_lines(self) -> list[tuple[str, str]]:
        return [(beat.id, beat.narration) for beat in self.beats if beat.narration.strip()]


def word_count(text: str) -> int:
    return len(str(text or "").split())


def word_budget(seconds: float) -> int:
    """Most narration words that fit ``seconds`` of pictures."""
    if seconds < MIN_NARRATED_SECONDS:
        return 0
    return max(3, int(seconds * WORDS_PER_SECOND * NARRATION_SHARE))


def clean_edit(value: Any, field: str) -> str:
    """User text from the Review page, limited like an engine's."""
    return _clean(value, _TEXT_LIMITS[field])
