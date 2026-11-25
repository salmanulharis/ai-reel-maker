from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

from app.brief.evidence import EvidencePack
from app.brief.schema import BriefReply


@dataclass
class EngineStatus:
    name: str
    label: str
    ready: bool
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BriefEngineError(RuntimeError):
    """An engine could not write the brief. For the Claude engine this pauses the job with the
    reason (the user decides: retry, or use the local engine); it never switches by itself."""

    def __init__(self, message: str, *, reason: str = "failed"):
        super().__init__(message)
        self.reason = reason


class BriefPaused(RuntimeError):
    """Raised by the analysis when the chosen engine (Claude) could not write the brief: the job
    is paused with this message instead of failing or switching engines."""

    def __init__(self, message: str, reason: str = "failed"):
        super().__init__(message)
        self.reason = reason


class BriefEngine(Protocol):
    name: str

    def model_name(self) -> str: ...

    def check(self, refresh: bool = False) -> EngineStatus: ...

    def write(
        self,
        pack: EvidencePack,
        *,
        repair: tuple[BriefReply, list[str]] | None = None,
        log: Callable[[str, str], None] | None = None,
        check_cancel: Callable[[], None] | None = None,
    ) -> BriefReply: ...


SYSTEM_PROMPT = """You are a video editor writing the brief for a short narrated video made from existing footage.
You only choose moments from the numbered list (m1, m2, ...); never write times and never invent moments.
Write in the language of the request (English when unsure). Keep narration natural, spoken and specific to what the moments show and say; never describe the camera or the list.

Return one JSON object:
- title: a short title (at most 8 words).
- hook: one line that makes people keep watching.
- tone: two or three words.
- music_mood: one of none, calm, uplifting, energetic, dramatic, playful, emotional.
- call_to_action: a short closing line (may be empty).
- beats: the video in order. Each beat has clips (1-4 aliases of consecutive or related moments), purpose (hook, context, key, build, payoff, highlight or ending), narration (the words spoken over the beat, following the voice-over instruction; it must fit the moments' length at the given speaking rate) and on_screen_text (at most 6 words, may be empty).
The first beat is the hook. Use each moment at most once. Respect the requested length."""
