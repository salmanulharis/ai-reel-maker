"""Where a cached visual or transcript score came from.

Scores are cached per job. Tagging fallback scores lets a later analysis run redo only the
parts that missed the model, once the model is reachable again.
"""

from __future__ import annotations

from typing import Any

AI = "ai"
FALLBACK = "fallback"

AI_UNAVAILABLE = "ai_unavailable"
AI_FAILED = "ai_failed"
SHORT_SCENE = "short_scene"
NO_FRAME = "no_frame"
# The vision model was only run on scenes that could make the highlight. Retryable, so
# choosing "every scene" later analyzes these too; the shortlist skips them again quickly.
NOT_SHORTLISTED = "not_shortlisted"

# Fallback reasons that a run with a working model can replace.
RETRYABLE = frozenset({AI_UNAVAILABLE, AI_FAILED, NOT_SHORTLISTED})


def tag(entry: dict[str, Any], reason: str | None = None) -> dict[str, Any]:
    entry["score_source"] = FALLBACK if reason else AI
    if reason:
        entry["fallback_reason"] = reason
    else:
        entry.pop("fallback_reason", None)
    return entry


def needs_ai(entry: Any) -> bool:
    """True for a cached fallback score that the model should redo. Untagged entries come from
    older versions and are kept as they are."""
    return (
        isinstance(entry, dict)
        and entry.get("score_source") == FALLBACK
        and entry.get("fallback_reason") in RETRYABLE
    )
