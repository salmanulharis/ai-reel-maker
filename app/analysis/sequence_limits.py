from __future__ import annotations

AUTO_MIN = 1.5


def keep_best_span(
    start: float,
    end: float,
    max_len: float,
    peak: float | None = None,
) -> tuple[float, float]:
    """Cut a window to a hard maximum, keeping the most important moment."""
    start = float(start)
    end = float(end)
    if end <= start:
        return start, start
    max_len = max(0.4, float(max_len))
    duration = end - start
    if duration <= max_len + 1e-9:
        return start, end
    center = float(peak) if peak is not None else (start + end) / 2.0
    center = min(max(center, start), end)
    half = max_len / 2.0
    new_start = center - half
    new_end = center + half
    if new_start < start:
        new_start = start
        new_end = start + max_len
    if new_end > end:
        new_end = end
        new_start = end - max_len
    if new_start < start:
        new_start = start
        new_end = min(end, start + max_len)
    return round(new_start, 3), round(new_end, 3)


def peak_from_candidate(candidate: dict) -> float | None:
    if candidate.get("peak") is not None:
        return float(candidate["peak"])
    metrics = candidate.get("metrics") or {}
    start = float(candidate["start"])
    end = float(candidate["end"])
    hook = float(metrics.get("hook_strength") or 0)
    if hook >= 0.7:
        return start + (end - start) * 0.25
    if float(metrics.get("importance") or 0) >= 0.75:
        return (start + end) / 2.0
    return (start + end) / 2.0


def sequence_cap(config) -> float:
    mode = str(getattr(config, "sequence_duration_mode", "custom") or "custom")
    if mode == "auto":
        return max(1.0, float(getattr(config, "max_duration", 30.0)))
    return max(1.0, float(getattr(config, "max_sequence_duration", 5.0)))


def auto_cap(min_duration: float, max_duration: float) -> float:
    return max(1.0, float(max_duration))


def _priority_set(priorities) -> set[str]:
    values = {str(item) for item in (priorities or []) if item}
    if not values or "balanced" in values:
        return {"balanced"}
    return values


def desired_auto_length(candidate: dict, cap: float, priorities: list[str] | None = None) -> tuple[float, str]:
    start = float(candidate["start"])
    end = float(candidate["end"])
    natural = max(0.0, end - start)
    cap = max(AUTO_MIN, float(cap))
    if natural <= 0:
        return 0.0, "Empty span"

    metrics = candidate.get("metrics") or {}
    importance = float(metrics.get("importance") or candidate.get("score") or 0.5)
    density = float(metrics.get("information_density") or 0.45)
    hook = float(metrics.get("hook_strength") or 0.2)
    action = float(metrics.get("emotional_intensity") or 0.3)
    visual = float(metrics.get("visual_importance") or 0.4)
    repetition = float(metrics.get("repetition_score") or 0.0)
    incomplete = bool(metrics.get("incomplete_sentence"))
    silence = float(metrics.get("silence_penalty") or 0.0)
    filler = float(metrics.get("filler_score") or 0.0)
    words = len(str(candidate.get("text") or "").split())
    focus = _priority_set(priorities)

    content = (
        0.30 * importance
        + 0.22 * density
        + 0.16 * visual
        + 0.16 * action
        + 0.16 * max(0.0, hook - 0.35)
    )
    content = max(0.0, min(1.0, content - 0.40 * repetition - 0.25 * silence - 0.30 * filler))
    needed = AUTO_MIN + (16.0 * content)
    reasons: list[str] = []

    if hook >= 0.75 and importance < 0.8:
        needed = min(needed, 3.2)
        reasons.append("short hook pacing")
    if importance >= 0.8 and density >= 0.7:
        needed = max(needed, min(natural, 12.0))
        reasons.append("kept enough time for the explanation")
    elif importance >= 0.75:
        reasons.append("high importance")
    if visual >= 0.7 or action >= 0.65:
        needed = max(needed, min(natural, 8.0 if natural >= 8 else natural))
        reasons.append("continuous action/reaction")
    if words >= 24:
        needed = max(needed, min(natural, words * 0.4))
        reasons.append("speech needs time to finish")
    if incomplete:
        needed = min(cap, max(needed, min(natural, needed + 1.8)))
        reasons.append("extended toward a complete sentence")
    if repetition >= 0.55 or filler >= 0.55:
        needed = min(needed, 3.0)
        reasons.append("cut filler and repeated setup")
    if silence >= 0.45:
        needed = min(needed, max(AUTO_MIN, natural * 0.45))
        reasons.append("removed dead time")

    if "speech" in focus or "information" in focus:
        if importance >= 0.55 or density >= 0.55 or words >= 12:
            needed = max(needed, min(natural, needed * 1.2))
            reasons.append("speech/information priority")
    if "action" in focus or "visual" in focus:
        if action >= 0.5 or visual >= 0.5:
            needed = max(needed, min(natural, needed * 1.15, 12.0 if natural >= 12 else natural))
            reasons.append("action/visual priority")
    if "magic" in focus and (hook >= 0.6 or float(metrics.get("novelty") or 0) >= 0.6 or action >= 0.7):
        needed = max(needed, min(natural, 6.0))
        reasons.append("magic-moment priority")

    length = min(natural, max(AUTO_MIN, needed), cap)
    if natural <= cap and natural - length <= 0.45:
        length = natural
        lead = f"{length:.1f}s — kept the natural scene/speech length"
    elif length + 0.05 < natural:
        lead = f"{length:.1f}s — shortest span that keeps the important part"
    else:
        lead = f"{length:.1f}s — paced from the content"
    extra = "; ".join(dict.fromkeys(reasons))
    return round(length, 3), f"{lead} ({extra})" if extra else lead


def fit_candidate(
    candidate: dict,
    cap: float,
    mode: str = "custom",
    priorities: list[str] | None = None,
) -> dict:
    item = dict(candidate)
    cap = max(0.4, float(cap))
    natural = max(0.0, float(item["end"]) - float(item["start"]))
    if (mode or "custom") == "auto":
        length, reason = desired_auto_length(item, cap, priorities=priorities)
    elif natural > cap + 1e-9:
        length = cap
        reason = f"{cap:.0f}s max — kept the peak of a {natural:.1f}s moment"
    else:
        length = natural
        reason = f"{natural:.1f}s — full moment is under the {cap:.0f}s maximum"
    start, end = keep_best_span(item["start"], item["end"], length, peak_from_candidate(item))
    item["start"] = start
    item["end"] = end
    item["duration_reason"] = reason
    return item


def trim_candidate(candidate: dict, max_len: float) -> dict:
    return fit_candidate(candidate, max_len, mode="custom")


def split_span(start: float, end: float, max_len: float) -> list[tuple[float, float]]:
    start = float(start)
    end = float(end)
    max_len = max(0.4, float(max_len))
    if end <= start:
        return []
    if end - start <= max_len + 1e-9:
        return [(round(start, 3), round(end, 3))]
    spans: list[tuple[float, float]] = []
    t = start
    while t < end - 0.05:
        nxt = min(end, t + max_len)
        if nxt - t >= 0.4:
            spans.append((round(t, 3), round(nxt, 3)))
        if nxt >= end:
            break
        t = nxt
    return spans


def clamp_plan_clips(clips: list, max_len: float) -> None:
    for item in clips:
        start, end = keep_best_span(float(item.start), float(item.end), max_len)
        item.start = start
        item.end = end


def is_low_value(candidate: dict) -> bool:
    metrics = candidate.get("metrics") or {}
    start = float(candidate.get("start") or 0)
    filler = float(metrics.get("filler_score") or 0)
    silence = float(metrics.get("silence_penalty") or 0)
    intro = bool(metrics.get("appears_intro"))
    hook = float(metrics.get("hook_strength") or 0)
    importance = float(metrics.get("importance") or candidate.get("score") or 0)
    if filler >= 0.65 or silence >= 0.7:
        return True
    if intro and start < 12 and hook < 0.65 and importance < 0.75:
        return True
    return False
