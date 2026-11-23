"""Prompt mode: turn a plain description ("a 30 second TikTok of the best goals, no intro")
into highlight settings and editing requirements.

Two readers work together. Fixed rules read what has an exact form: lengths, times, formats,
captions, number of shorts. The text model reads the meaning: what to keep, what to cut, which
moments matter, what they look and sound like, and the style. Numbers from the rules win,
because small models misread them. Without Ollama the rules alone still give a usable result,
and the whole description becomes the focus that guides scoring.

The result is a settings patch (only what the description implies, plus a few quality defaults
for prompt mode), each change with its source, and the requirements. It is checked against
HighlightConfig before it is returned, so the preview never shows a value a job would reject.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.ai.prompts import INTENT_SCHEMA, PROMPT_LEAVE_OUT, PROMPT_PRIORITIES, PROMPT_STYLES, intent_prompt
from app.models.schemas import HighlightConfig, PromptRequirements

_log = logging.getLogger(__name__)

_INTENT_PREDICT = 900
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90,
}
_NUMBER = r"\b(\d+(?:\.\d+)?|" + "|".join(sorted(map(re.escape, _NUMBER_WORDS), key=len, reverse=True)) + r")"
_UNIT = r"(seconds?|secs?|s|minutes?|mins?|m)\b"
_CLOCK = r"(\d{1,2}:\d{2}(?::\d{2})?)"
_REMOVE_WORDS = r"(?:skip|cut|remove|drop|exclude|without|leave out|no)"

# Quality defaults of prompt mode, shown in the preview like every other change.
PROMPT_DEFAULTS: dict[str, Any] = {
    "transition": "auto",
    "level_audio": True,
    "order_mode": "ai",
}


def interpret(
    request: str,
    *,
    duration: float = 0.0,
    video_count: int = 1,
    client=None,
    model: str = "",
    use_speech: bool = True,
) -> dict[str, Any]:
    """Settings and requirements for a description. ``client`` is an OllamaClient or None."""
    request = " ".join(str(request or "").split())[:2000]
    rules = read_rules(request, duration)
    ai: dict[str, Any] | None = None
    warning = None
    if client is not None and model and request:
        try:
            options = client.role_options("text")
            options["num_predict"] = max(_INTENT_PREDICT, int(getattr(getattr(client, "settings", None), "ollama_text_num_predict", 0) or 0))
            ai = client.chat(model, intent_prompt(request, duration, video_count), schema=INTENT_SCHEMA, options=options)
        except Exception as exc:  # noqa: BLE001 - the rules still give a usable result
            _log.info("Prompt understanding fell back to rules: %s", exc)
            warning = "The AI model could not read the description, so only the basics (length, format, captions, times) were taken from it."
    elif request:
        warning = "The AI model is not available, so only the basics (length, format, captions, times) were taken from the description."
    patch, sources = _patch_from_ai(ai or {}, duration)
    for key, value in rules.items():
        patch[key] = value
        sources[key] = "rules"
    for key, value in PROMPT_DEFAULTS.items():
        if key not in patch:
            patch[key] = value
            sources[key] = "default"
    if patch.get("aspect_ratio") not in {None, "original", "16:9"} and "reframe" not in patch:
        patch["reframe"] = "subject"
        sources["reframe"] = "default"
    if patch.get("subtitles") == "burned" and patch.get("aspect_ratio") == "9:16" and "caption_style" not in patch:
        patch["caption_style"] = "bold"
        sources["caption_style"] = "default"
    requirements = _requirements(ai or {}, request)
    focus = "; ".join([*requirements.keep, *requirements.moments])[:500]
    if not focus and ai is None:
        # Without the model, the description itself guides scoring.
        focus = request[:500]
    if focus:
        patch["custom_focus"] = focus
        sources["custom_focus"] = "ai" if ai is not None else "rules"
    if requirements.remove:
        patch["avoid_custom"] = "; ".join(requirements.remove)[:300]
        sources["avoid_custom"] = "ai"
    if not use_speech:
        for key in ("subtitles", "tighten", "caption_style"):
            patch.pop(key, None)
            sources.pop(key, None)
    patch = _valid_patch(patch, sources, use_speech)
    return {
        "summary": requirements.summary or (str((ai or {}).get("summary") or "").strip()) or request[:200],
        "settings": patch,
        "changes": [{"field": key, "value": patch[key], "source": sources.get(key, "ai")} for key in patch],
        "requirements": requirements.model_dump(),
        "source": "ai" if ai is not None else "rules",
        "warning": warning,
    }


# Rules


def _number(text: str) -> float | None:
    text = text.lower()
    if text in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[text])
    try:
        return float(text)
    except ValueError:
        return None


def _seconds(amount: str, unit: str) -> float | None:
    value = _number(amount)
    if value is None:
        return None
    return value * 60 if unit.lower().startswith("m") else value


def _real_amount(amount: str, gap: str, unit: str) -> bool:
    """Whether "<amount><gap><unit>" is really a length: a number word needs a space and a whole
    unit word ("two minutes"), so "as long" is not read as "a" + "s"."""
    if amount.lower() in _NUMBER_WORDS:
        return bool(gap) and len(unit) > 1
    return True


def parse_clock(text: str) -> float:
    parts = [float(part) for part in text.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def read_rules(request: str, duration: float = 0.0) -> dict[str, Any]:
    """Settings with an exact form in the description."""
    text = request.lower()
    patch: dict[str, Any] = {}
    remove_times: list[tuple[float, float]] = []
    keep_times: list[tuple[float, float]] = []

    # Times in the video, read first so "skip the first 5 minutes" is not taken as a length.
    for match in re.finditer(
        rf"{_REMOVE_WORDS}\s+(?:the\s+)?(first|last|final|opening)\s+{_NUMBER}(\s*){_UNIT}", text
    ):
        span = _seconds(match.group(2), match.group(4)) if _real_amount(match.group(2), match.group(3), match.group(4)) else None
        if not span:
            continue
        if match.group(1) in {"last", "final"} and duration > 0:
            remove_times.append((max(0.0, duration - span), duration))
        elif match.group(1) in {"first", "opening"}:
            remove_times.append((0.0, span))
        text = text.replace(match.group(0), " ")
    for match in re.finditer(rf"{_CLOCK}\s*(?:to|and|until|-|–)\s*{_CLOCK}", text):
        start, end = parse_clock(match.group(1)), parse_clock(match.group(2))
        if end <= start:
            continue
        # The verb of the clause the range is in decides: "cut 0:00-0:30 and keep 5:00-5:40".
        clause = re.split(r"[,.;]|\band\b|\bthen\b", text[: match.start()])[-1]
        (remove_times if re.search(_REMOVE_WORDS + r"\b", clause) else keep_times).append((start, end))
        text = text.replace(match.group(0), " ")
    if duration > 0:
        keep_times = [(s, min(e, duration)) for s, e in keep_times if s < duration]
        remove_times = [(s, min(e, duration)) for s, e in remove_times if s < duration]
    if keep_times:
        patch["must_include"] = keep_times
    if remove_times:
        patch["must_exclude"] = remove_times

    # Number of separate shorts.
    shorts = re.search(rf"{_NUMBER}\s+(?:separate\s+|different\s+)?(?:shorts|reels|tiktoks|separate videos|short videos)\b", text)
    if shorts:
        count = _number(shorts.group(1))
        if count and 2 <= count <= 10:
            patch["shorts_count"] = int(count)
        text = text.replace(shorts.group(0), " ")

    # Length of the result.
    length = next(
        (
            m
            for m in re.finditer(rf"{_NUMBER}\s*(?:-|to|–|or)\s*{_NUMBER}(\s*){_UNIT}", text)
            if _real_amount(m.group(2), m.group(3), m.group(4))
        ),
        None,
    )
    if length:
        low, high = _seconds(length.group(1), length.group(4)), _seconds(length.group(2), length.group(4))
        if low and high:
            patch["min_duration"], patch["max_duration"] = sorted((low, high))
    else:
        single, kind = None, None
        if re.search(r"\bhalf a minute\b", text):
            single, kind = 30.0, "about"
        else:
            found = next(
                (
                    m
                    for m in re.finditer(
                        rf"(under|less than|at most|max(?:imum)?|no (?:more|longer) than|up to|at least|about|around|roughly|~)?\s*{_NUMBER}([\s-]*){_UNIT}",
                        text,
                    )
                    if _real_amount(m.group(2), m.group(3), m.group(4))
                ),
                None,
            )
            if found:
                single, kind = _seconds(found.group(2), found.group(4)), (found.group(1) or "exact")
        if single:
            if kind in {"under", "less than", "at most", "max", "maximum", "no more than", "no longer than", "up to"}:
                patch["min_duration"], patch["max_duration"] = max(5.0, round(single * 0.6)), single
            elif kind == "at least":
                patch["min_duration"], patch["max_duration"] = single, round(single * 1.5)
            elif kind in {"about", "around", "roughly", "~"}:
                patch["min_duration"], patch["max_duration"] = round(single * 0.85), round(single * 1.15)
            else:
                patch["min_duration"], patch["max_duration"] = max(5.0, round(single * 0.85)), single
    if "min_duration" in patch:
        patch["length_mode"] = "range"
    elif re.search(r"\b(as long as (?:it )?needs?|any length|however long|keep (?:all|every) (?:good|strong|best) (?:moment|part)s?)\b", text):
        patch["length_mode"] = "auto"

    # Shape.
    if re.search(r"\b(tiktok|reels?|instagram stor(?:y|ies)|youtube shorts|shorts|vertical|portrait|9:16)\b", text):
        patch["aspect_ratio"] = "9:16"
    elif re.search(r"\b(square|1:1)\b", text):
        patch["aspect_ratio"] = "1:1"
    elif re.search(r"\b4:5\b", text):
        patch["aspect_ratio"] = "4:5"

    # Captions.
    if re.search(r"\b(no|without)\s+(?:captions|subtitles)\b", text):
        patch["subtitles"] = "none"
    elif re.search(r"\bsrt\b|subtitle file", text):
        patch["subtitles"] = "srt"
    elif re.search(r"\b(captions?|subtitles?|subtitled|captioned)\b", text):
        patch["subtitles"] = "burned"

    # Cleaning up speech.
    if re.search(r"\b(ums?|uhs?|filler words?|fillers)\b", text):
        patch["tighten"] = "fillers"
    elif re.search(r"\b(remove|cut|skip|without|no)\s+(?:long\s+)?(pauses|silences?|dead air)\b", text):
        patch["tighten"] = "pauses"

    # Order and transitions.
    if re.search(r"\b(chronological|in order|original order|as it happened)\b", text):
        patch["order_mode"] = "chronological"
    elif re.search(r"\b(start|open|begin) with (?:the )?(best|strongest|most exciting)\b|\bhook\b", text):
        patch["order_mode"] = "hook"
    if re.search(r"\b(no|without) (?:transitions|fades)\b|\bhard cuts?\b", text):
        patch["transition"] = "cut"
    elif re.search(r"\bfades?\b|\bfade (?:in|out)\b", text):
        patch["transition"] = "fade"
    return patch


# The model's reply


def _patch_from_ai(raw: dict[str, Any], duration: float) -> tuple[dict[str, Any], dict[str, str]]:
    patch: dict[str, Any] = {}
    if not isinstance(raw, dict):
        return patch, {}
    low, high = _positive(raw.get("min_seconds")), _positive(raw.get("max_seconds"))
    if low or high:
        low = low or max(5.0, round((high or 0) * 0.6))
        high = high or round(low * 1.5)
        patch["min_duration"], patch["max_duration"] = sorted((low, high))
        patch["length_mode"] = "range"
    elif raw.get("length_auto") is True:
        patch["length_mode"] = "auto"
    if raw.get("format") in {"original", "16:9", "9:16", "1:1", "4:5"} and raw.get("format") != "original":
        patch["aspect_ratio"] = raw["format"]
    if raw.get("subtitles") in {"none", "burned", "srt"}:
        patch["subtitles"] = raw["subtitles"]
    if raw.get("style") in PROMPT_STYLES and raw.get("style") != "auto":
        patch["style"] = raw["style"]
    shorts = _positive(raw.get("shorts"))
    if shorts and 2 <= shorts <= 10:
        patch["shorts_count"] = int(shorts)
    priorities = [p for p in raw.get("priorities") or [] if p in PROMPT_PRIORITIES]
    if priorities:
        patch["content_priorities"] = list(dict.fromkeys(priorities))[:4]
    leave_out = [item for item in raw.get("leave_out") or [] if item in PROMPT_LEAVE_OUT]
    if leave_out:
        patch["avoid"] = list(dict.fromkeys(leave_out))
    order = {"time": "chronological", "story": "ai", "hook": "hook"}.get(str(raw.get("order") or ""))
    if order:
        patch["order_mode"] = order
    if raw.get("transition") in {"cut", "fade", "auto"}:
        patch["transition"] = raw["transition"]
    if raw.get("tighten") in {"pauses", "fillers"}:
        patch["tighten"] = raw["tighten"]
    for source_key, target_key in (("keep_times", "must_include"), ("remove_times", "must_exclude")):
        ranges = _ranges(raw.get(source_key), duration)
        if ranges:
            patch[target_key] = ranges
    return patch, {key: "ai" for key in patch}


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _ranges(items: Any, duration: float) -> list[tuple[float, float]]:
    ranges = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        start, end = _positive(item.get("start")) or 0.0, _positive(item.get("end"))
        if end is None or end <= start or (duration > 0 and start >= duration):
            continue
        ranges.append((start, min(end, duration) if duration > 0 else end))
    return ranges


def _requirements(raw: dict[str, Any], request: str) -> PromptRequirements:
    data = {key: raw.get(key) or [] for key in ("keep", "remove", "moments", "visual_cues", "audio_cues")}
    data["summary"] = str(raw.get("summary") or "") or request[:200]
    return PromptRequirements.model_validate(data)


def _valid_patch(patch: dict[str, Any], sources: dict[str, str], use_speech: bool) -> dict[str, Any]:
    """Drop the values a job would reject, one at a time, so one bad value keeps the rest."""
    base = {"mode": "prompt", "prompt": "x", "use_speech": use_speech}
    clean: dict[str, Any] = {}
    for key, value in patch.items():
        try:
            config = HighlightConfig.model_validate({**base, **clean, key: value})
        except Exception:  # noqa: BLE001 - an unusable value is left out of the preview
            sources.pop(key, None)
            continue
        clean[key] = getattr(config, key)
    # Values the validator normalised (min above max, ranges) come back in their final form.
    final = HighlightConfig.model_validate({**base, **clean})
    return {key: _plain(getattr(final, key)) for key in clean}


def _plain(value: Any) -> Any:
    if isinstance(value, list):
        return [list(item) if isinstance(item, tuple) else item for item in value]
    return value
