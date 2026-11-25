"""Checks every brief by code, whichever engine wrote it.

``check`` turns a reply into a usable brief and lists what was wrong with it. Each problem is
also fixed by a rule (unknown moments dropped, narration shortened to its word budget, required
moments added, an overlong plan trimmed), so the result is always usable. The engine gets one
chance to fix the problems itself first (see app.brief.service); the rules handle the rest.
"""

from __future__ import annotations

import re

from app.analysis.director import _clock
from app.brief.evidence import EvidencePack
from app.brief.schema import (
    MAX_BEATS,
    MAX_CLIPS_PER_BEAT,
    Beat,
    BriefClip,
    BriefReply,
    VideoBrief,
    word_budget,
    word_count,
)
from app.models.schemas import HighlightConfig

# A plan this much longer than the target is trimmed (the planner fits the exact length later).
_TOO_LONG = 1.5
MAX_SCREEN_WORDS = 6
_TRIM_TO = 1.25


def check(
    reply: BriefReply,
    pack: EvidencePack,
    config: HighlightConfig,
    *,
    engine: str,
    model: str,
    key: str,
) -> tuple[VideoBrief | None, list[str]]:
    """The checked brief (None when no beat uses a real moment) and the problems found."""
    problems: list[str] = []
    single = len({str(item.get("source_id") or "v0") for item in pack.aliases.values()}) <= 1
    excluded = list(config.must_exclude) if single else []
    used: set[str] = set()
    beats: list[Beat] = []
    for number, raw in enumerate(reply.beats, start=1):
        unknown = [alias for alias in raw.clips if alias not in pack.aliases]
        if unknown:
            problems.append(f"Beat {number} uses moments that do not exist: {', '.join(unknown[:5])}.")
        clips: list[BriefClip] = []
        for alias in raw.clips:
            item = pack.aliases.get(alias)
            if item is None or alias in used:
                continue
            start, end = float(item["start"]), float(item["end"])
            if any(_overlaps(start, end, a, b) for a, b in excluded):
                problems.append(f"Beat {number} uses {alias}, which is inside a part that must be left out.")
                continue
            used.add(alias)
            clips.append(BriefClip(id=str(item["id"]), source_id=str(item.get("source_id") or "v0"), start=start, end=end))
            if len(clips) >= MAX_CLIPS_PER_BEAT:
                break
        if not clips:
            problems.append(f"Beat {number} has no usable moment and was left out.")
            continue
        beats.append(
            Beat(
                id="",
                purpose=raw.purpose,
                narration=raw.narration,
                on_screen_text=raw.on_screen_text,
                clips=clips,
            )
        )
    if not beats:
        problems.append("No beat uses a moment from the list.")
        return None, problems
    # The story shape the purposes promise: a hook opens the video and an ending closes it.
    beats.sort(key=lambda beat: 0 if beat.purpose == "hook" else (2 if beat.purpose == "ending" else 1))
    if len(beats) > MAX_BEATS:
        problems.append(f"The plan has {len(beats)} beats; at most {MAX_BEATS} are used.")
        beats = beats[:MAX_BEATS]

    if single:
        for start, end in config.must_include:
            if any(_overlaps(c.start, c.end, start, end) for beat in beats for c in beat.clips):
                continue
            problems.append(f"The part at {_clock(start)}-{_clock(end)} must be in the video.")
            added = _beat_for_range(pack, start, end, used)
            if added is not None:
                position = sum(1 for beat in beats if beat.clips[0].start <= added.clips[0].start)
                beats.insert(position, added)

    total = sum(beat.seconds for beat in beats)
    if total > pack.target_seconds * _TOO_LONG and len(beats) > 2:
        problems.append(
            f"The beats add up to {total:.0f} seconds for a {pack.target_seconds:.0f}-second video; "
            "choose fewer or shorter moments."
        )
        beats = _trim(beats, pack, pack.target_seconds * _TRIM_TO)

    narrated = 0
    for number, beat in enumerate(beats, start=1):
        beat.id = f"b{number}"
        beat.max_words = word_budget(beat.seconds)
        if word_count(beat.on_screen_text) > MAX_SCREEN_WORDS:
            problems.append(f"Beat {number} on-screen text has more than {MAX_SCREEN_WORDS} words.")
            beat.on_screen_text = " ".join(beat.on_screen_text.split()[:MAX_SCREEN_WORDS])
        narrated += bool(beat.narration.strip())
        words = word_count(beat.narration)
        if words > beat.max_words:
            if beat.max_words == 0:
                problems.append(f"Beat {number} is too short for narration ({beat.seconds:.1f}s); leave its narration empty.")
            else:
                problems.append(
                    f"Beat {number} narration has {words} words; at most {beat.max_words} fit its {beat.seconds:.0f} seconds."
                )
            beat.narration = shorten(beat.narration, beat.max_words)
    voiced_beats = [beat for beat in beats if beat.max_words > 0]
    if pack.voice_over and voiced_beats and narrated < len(voiced_beats) / 2:
        # Rules cannot write words: this is only reported (and sent back to the engine once).
        problems.append("A voice-over was asked for, but most beats have no narration; write narration for every beat.")

    brief = VideoBrief(
        key=key,
        engine=engine,
        model=model,
        title=reply.title or pack.request[:80] or pack.video_name[:80],
        hook=reply.hook,
        tone=reply.tone,
        music_mood=reply.music_mood,
        call_to_action=reply.call_to_action,
        beats=beats,
    )
    return brief, problems


def shorten(text: str, budget: int) -> str:
    """``text`` cut to at most ``budget`` words: whole sentences when they fit, else words."""
    if budget <= 0:
        return ""
    if word_count(text) <= budget:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: list[str] = []
    for sentence in sentences:
        if word_count(" ".join([*kept, sentence])) > budget:
            break
        kept.append(sentence)
    if kept:
        return " ".join(kept)
    words = text.split()[:budget]
    return " ".join(words).rstrip(",;:-") + "."


def repair_text(problems: list[str]) -> str:
    return (
        "Your previous brief had these problems. Write the whole brief again and fix them:\n- "
        + "\n- ".join(problems)
    )


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return min(a_end, b_end) - max(a_start, b_start) > 0.05


def _beat_for_range(pack: EvidencePack, start: float, end: float, used: set[str]) -> Beat | None:
    best = None
    for alias, item in pack.aliases.items():
        if alias in used or not _overlaps(float(item["start"]), float(item["end"]), start, end):
            continue
        if best is None or float(item.get("score") or 0) > float(best[1].get("score") or 0):
            best = (alias, item)
    if best is None:
        return None
    alias, item = best
    used.add(alias)
    clip = BriefClip(
        id=str(item["id"]), source_id=str(item.get("source_id") or "v0"), start=float(item["start"]), end=float(item["end"])
    )
    return Beat(id="", purpose="key", clips=[clip])


def _trim(beats: list[Beat], pack: EvidencePack, limit: float) -> list[Beat]:
    """Drop the weakest middle beats (never the first or the last) until the plan fits."""
    scores = {str(item["id"]): float(item.get("score") or 0) for item in pack.aliases.values()}
    kept = list(beats)
    while len(kept) > 2 and sum(beat.seconds for beat in kept) > limit:
        middle = kept[1:-1]
        weakest = min(middle, key=lambda beat: sum(scores.get(c.id, 0) for c in beat.clips) / len(beat.clips))
        kept.remove(weakest)
    return kept
