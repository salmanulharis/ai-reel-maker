"""
Tighten clips: cut long pauses (and, if asked, filler words such as "um") out of a clip, using
Whisper's word timings. The clip becomes a few pieces that are joined back to back.
Without word timings a clip is left as it is.
"""

from __future__ import annotations

import re

FILLERS = {"um", "umm", "uh", "uhh", "uhm", "erm", "er", "ah", "hmm", "hm", "mm", "mhm"}
# A pause longer than this is cut; shorter pauses are natural and stay.
PAUSE = 0.6
# Room kept before and after speech at a cut, so words are not clipped.
PAD = 0.12
# Pieces shorter than this flash by; they are dropped.
MIN_PIECE = 0.3


def _norm(text: str) -> str:
    return re.sub(r"[^a-z]", "", str(text).lower())


def _words_in(words: list, start: float, end: float) -> list[tuple[float, float, str]]:
    found = []
    for item in words or []:
        if not isinstance(item, dict):
            continue
        try:
            w_start, w_end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if w_end > start and w_start < end and w_end > w_start:
            found.append((max(w_start, start), min(w_end, end), str(item.get("word") or item.get("text") or "")))
    found.sort()
    return found


def tighten_spans(
    start: float,
    end: float,
    words: list,
    remove_fillers: bool = False,
    pause: float = PAUSE,
    pad: float = PAD,
    min_piece: float = MIN_PIECE,
) -> list[tuple[float, float]]:
    """
    The parts of start..end to keep, in order. Cuts are made at pauses longer than `pause`
    (including before the first and after the last word) and, with `remove_fillers`, around
    filler words. Returns [(start, end)] when there is nothing to cut or no word timings.
    """
    timed = _words_in(words, start, end)
    if not timed:
        return [(start, end)]
    kept = [(s, e) for s, e, text in timed if not (remove_fillers and _norm(text) in FILLERS)]
    if not kept:
        return [(start, end)]
    fillers = [(s, e) for s, e, text in timed if remove_fillers and _norm(text) in FILLERS]

    pieces: list[list[float]] = []
    lead = kept[0][0] - start
    piece_start = start if lead <= pause else max(start, kept[0][0] - pad)
    for (a_start, a_end), (b_start, b_end) in zip(kept, kept[1:]):
        gap = b_start - a_end
        between = [(s, e) for s, e in fillers if a_end <= s < b_start]
        if gap > pause or between:
            room = min(pad, gap / 2)
            cut_from, cut_to = a_end + room, b_start - room
            if between:
                # Never keep any of a cut filler word.
                cut_from = min(cut_from, between[0][0])
                cut_to = max(cut_to, between[-1][1])
            pieces.append([piece_start, cut_from])
            piece_start = cut_to
    tail = end - kept[-1][1]
    pieces.append([piece_start, end if tail <= pause else min(end, kept[-1][1] + pad)])

    result = [(round(s, 3), round(e, 3)) for s, e in pieces if e - s >= min_piece]
    if not result or sum(e - s for s, e in result) < min(0.5, end - start):
        return [(start, end)]
    return result


def tighten_clips(clips: list, transcript_for, remove_fillers: bool = False) -> list:
    """
    Plan clips with pauses (and fillers) cut out. Each clip becomes one or more pieces; pieces
    after the first are marked `join_previous`, so a fade is applied only at the real clip edges.
    `transcript_for(clip)` returns the transcript of the video the clip comes from.
    """
    result = []
    for clip in clips:
        transcript = transcript_for(clip) or {}
        spans = tighten_spans(float(clip.start), float(clip.end), transcript.get("words") or [], remove_fillers)
        for index, (start, end) in enumerate(spans):
            result.append(clip.model_copy(update={"start": start, "end": end, "join_previous": index > 0}))
    return result
