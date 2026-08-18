from __future__ import annotations

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
