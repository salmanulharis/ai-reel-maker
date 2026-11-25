from __future__ import annotations

import bisect

from app.models.schemas import Transcript, WordTimestamp


def nearest_word_boundary(
    t: float,
    words: list[WordTimestamp],
    side: str = "start",
    max_skew: float = 1.6,
) -> float:
    if not words:
        return t
    if side == "start":
        candidates = [w.start for w in words if abs(w.start - t) <= max_skew]
        return min(candidates, key=lambda x: abs(x - t)) if candidates else t
    candidates = [w.end for w in words if abs(w.end - t) <= max_skew]
    return min(candidates, key=lambda x: abs(x - t)) if candidates else t


def nearest_pause(
    t: float,
    silences: list[dict],
    side: str = "start",
    max_skew: float = 1.2,
) -> float | None:
    best = None
    best_dist = max_skew
    for silence in silences:
        point = silence["end"] if side == "start" else silence["start"]
        dist = abs(point - t)
        if dist < best_dist:
            best = point
            best_dist = dist
    return best


def snap_window(
    start: float,
    end: float,
    transcript: Transcript,
    silences: list[dict],
    scenes: list[dict],
    source_duration: float,
    pad: float = 0.35,
) -> tuple[float, float, bool]:
    start = max(0.0, start - pad)
    end = min(source_duration, end + pad)
    start = nearest_word_boundary(start, transcript.words, "start")
    end = nearest_word_boundary(end, transcript.words, "end")
    pause_s = nearest_pause(start, silences, "start")
    pause_e = nearest_pause(end, silences, "end")
    if pause_s is not None:
        start = min(start, pause_s)
    if pause_e is not None:
        end = max(end, pause_e)
    for scene in scenes:
        if scene["start"] <= start <= scene["end"] and start - scene["start"] < 0.8:
            start = scene["start"]
        if scene["start"] <= end <= scene["end"] and scene["end"] - end < 0.8:
            end = scene["end"]
    if end <= start:
        end = min(source_duration, start + 1.0)
    incomplete = False
    text = overlapping_text(transcript, start, end)
    if text and not text.rstrip().endswith((".", "!", "?", '"', "'")):
        incomplete = len(text.split()) > 4
    return round(start, 3), round(end, 3), incomplete


def overlapping_text(transcript: Transcript, start: float, end: float) -> str:
    bits = []
    for seg in transcript.segments:
        if seg.end >= start and seg.start <= end:
            bits.append(seg.text.strip())
    return " ".join(bits).strip()


def context_pad(mode: str) -> float:
    return {"low": 0.12, "balanced": 0.35, "high": 0.7}.get(mode, 0.35)


# Where a clip may start or end, with how good a place it is: 1.0 at a sentence end, a pause
# or a scene change; 0.5 at a short gap between two words.
STRONG_CUT = 1.0
WEAK_CUT = 0.5
_WORD_GAP = 0.25
_SENTENCE_MARKS = (".", "!", "?", "。", "！", "？", "।", "؟")


class CutPoints:
    """Every good cut point of a video, for looking up the ones inside a moment."""

    def __init__(self, transcript: Transcript, silences: list[dict], scenes: list[dict]):
        points: dict[float, float] = {}

        def add(time: float, strength: float) -> None:
            key = round(float(time), 3)
            points[key] = max(points.get(key, 0.0), strength)

        words = transcript.words
        for before, after in zip(words, words[1:]):
            gap = float(after.start) - float(before.end)
            if gap < 0:
                continue
            sentence = before.word.strip().endswith(_SENTENCE_MARKS)
            if sentence or gap >= _WORD_GAP:
                add(float(before.end) + gap / 2, STRONG_CUT if sentence or gap >= 0.6 else WEAK_CUT)
        for silence in silences or []:
            start, end = float(silence["start"]), float(silence["end"])
            if end > start:
                edge = min(0.15, (end - start) / 2)
                add(start + edge, STRONG_CUT)
                add(end - edge, STRONG_CUT)
        for scene in scenes or []:
            add(float(scene["start"]), STRONG_CUT)
        self.times = sorted(points)
        self.strength = [points[time] for time in self.times]

    def between(self, start: float, end: float, limit: int = 80) -> list[list[float]]:
        """[time, strength] of the cut points strictly inside start..end (at most ``limit``,
        the strongest kept)."""
        low = bisect.bisect_right(self.times, float(start) + 1e-3)
        high = bisect.bisect_left(self.times, float(end) - 1e-3)
        found = [[self.times[i], self.strength[i]] for i in range(low, high)]
        if len(found) > limit:
            found = sorted(sorted(found, key=lambda item: -item[1])[:limit])
        return found
