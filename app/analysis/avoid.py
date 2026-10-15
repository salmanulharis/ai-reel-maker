"""What to leave out: detectors for kinds of content the user does not want in the highlight.

Each candidate gets ``avoid_match``: how strongly (0-1) it shows each chosen kind of content.
The detectors use signals that are already computed: the transcript, silence, picture quality,
fast picture matching (SigLIP), the vision model's optional "avoid" score, and text embeddings.
The matches are stored on the candidate, so re-planning applies them again without analysis.

"Avoid when possible" lowers the score of a matching moment; "Never include" removes it. Must-
include ranges still win, and removing everything falls back to avoiding, so a job never fails
because too much was left out.
"""

from __future__ import annotations

import re

import numpy as np

from app.analysis.signals import POOR_QUALITY

# Every kind of content that can be left out, in the order the UI shows them.
AVOID_ITEMS = (
    "filler",
    "sponsor",
    "swearing",
    "talking_heads",
    "dark",
    "text",
    "screens",
    "empty",
    "intro_outro",
    "silence",
)
# Kinds that are detected from what is said, so they need speech.
SPEECH_ITEMS = frozenset({"filler", "sponsor", "swearing", "talking_heads"})
# Kinds the vision model is asked about (with the user's own words).
VISION_ITEMS = {
    "text": "on-screen text, titles, captions, or slides",
    "screens": "phone, computer, or TV screens",
    "empty": "empty scenes with nothing happening",
}

# A match from this strength on lowers the score ("avoid"); from NEVER_FROM on it is removed
# ("never"). In between, "never" still lowers the score.
AVOID_FROM = 0.35
NEVER_FROM = 0.6
# Share of the score a full match takes away in "avoid" mode.
AVOID_PENALTY = 0.6

# Fast picture matching: descriptions of each kind, compared with descriptions of ordinary
# frames. The margin (best kind description minus best ordinary description) is mapped to 0-1
# between the two values given. Calibrated on real footage: a title card scored about 0.03,
# ordinary frames mostly below 0.015; close-ups of people about 0.05.
PICTURE_ITEMS = {
    "text": (("a slide or title card with text", "a screen full of text"), 0.012, 0.03),
    "screens": (("a phone or computer screen", "a screen recording"), 0.0, 0.03),
    "empty": (("an empty room with nobody in it", "an empty scene"), 0.008, 0.025),
    "talking_heads": (("a close-up of a person's face talking to the camera",), 0.02, 0.05),
}
PICTURE_CONTRAST = ("a photo", "people", "a room", "outdoors", "an object", "a video frame")
# The user's own words as a picture description use the same scale.
CUSTOM_PICTURE_RANGE = (0.012, 0.04)

_SPONSOR = re.compile(
    r"\b(sponsor(ed|s|ship)?|use (my |the )?code|promo code|discount code|link (is )?in (the |my )?"
    r"(description|bio)|affiliate|brought to you by|check out .{0,30}(link|website)|"
    r"limited[- ]time offer|percent off|% off)\b",
    re.IGNORECASE,
)
_SWEARING = re.compile(
    r"\b(fuck\w*|shit\w*|bitch\w*|bastard\w*|asshole\w*|dick(head)?s?|cunt\w*|motherfuck\w*|"
    r"bullshit|damn(it)?|goddamn\w*|piss(ed)?|crap)\b",
    re.IGNORECASE,
)
_GREETING = re.compile(
    r"\b(hi|hello|hey) (guys|everyone|everybody|all|there)\b|\bwelcome (back|to)\b|"
    r"\bwhat'?s up\b|\bmy name is\b",
    re.IGNORECASE,
)
_SIGN_OFF = re.compile(
    r"\b(thanks|thank you) for watching\b|\bsee you (next time|in the next)\b|\bsubscribe\b|"
    r"\blike and (subscribe|share)\b|\bthat'?s (it|all) for (today|now)\b|\bbye\b",
    re.IGNORECASE,
)


def clean_items(items, use_speech: bool = True) -> list[str]:
    """Known kinds, in display order, without speech kinds when speech is off."""
    chosen = {str(item).strip().lower() for item in items or []}
    return [
        item for item in AVOID_ITEMS if item in chosen and (use_speech or item not in SPEECH_ITEMS)
    ]


def custom_terms(text: str) -> list[str]:
    """The user's own words, split on commas, semicolons, and new lines."""
    return [part.strip() for part in re.split(r"[,;\n]", text or "") if part.strip()][:12]


# What the text model is told to rate low, per speech kind.
SPEECH_HINTS = {
    "filler": "greetings and filler talk",
    "sponsor": "sponsor messages and ads",
    "swearing": "swearing",
}


def speech_avoid_text(items: list[str], custom: str) -> str:
    """What the transcript-scoring model should rate low, or ""."""
    parts = [SPEECH_HINTS[item] for item in items if item in SPEECH_HINTS]
    parts += custom_terms(custom)
    return "; ".join(parts)


def vision_avoid_text(items: list[str], custom: str) -> str:
    """What the vision model is asked to recognize, or "" when nothing needs it."""
    parts = [VISION_ITEMS[item] for item in items if item in VISION_ITEMS]
    parts += custom_terms(custom)
    return "; ".join(parts)


def picture_avoid_texts(items: list[str], custom: str) -> dict[str, tuple[tuple[str, ...], float, float]]:
    """The picture descriptions to check per kind (plus "custom" for the user's own words)."""
    texts = {item: PICTURE_ITEMS[item] for item in items if item in PICTURE_ITEMS}
    terms = custom_terms(custom)
    if terms:
        texts["custom"] = (tuple(terms), *CUSTOM_PICTURE_RANGE)
    return texts


def picture_avoid_scores(matcher, vectors: np.ndarray, texts: dict) -> list[dict[str, float]]:
    """Per frame vector: how strongly it shows each kind (0-1)."""
    if not texts or vectors.size == 0:
        return [{} for _ in range(len(vectors))]
    contrast = (vectors @ matcher.embed_texts(list(PICTURE_CONTRAST)).T).max(axis=1)
    rows: list[dict[str, float]] = [{} for _ in range(len(vectors))]
    for key, (descriptions, low, high) in texts.items():
        best = (vectors @ matcher.embed_texts(list(descriptions)).T).max(axis=1)
        strength = np.clip((best - contrast - low) / max(high - low, 1e-6), 0.0, 1.0)
        for row, value in enumerate(strength):
            rows[row][key] = round(float(value), 3)
    return rows


def _ramp(value: float, low: float, high: float) -> float:
    return float(np.clip((float(value) - low) / max(high - low, 1e-6), 0.0, 1.0))


def _scenes_for(start: float, end: float, scenes: list[dict]) -> list[dict]:
    """Scenes that make up a real part of a moment (at least 1 s or 40% of it)."""
    length = max(end - start, 0.01)
    chosen = []
    for scene in scenes:
        overlap = min(end, float(scene["end"])) - max(start, float(scene["start"]))
        if overlap >= 1.0 or overlap >= 0.4 * length:
            chosen.append(scene)
    return chosen


def annotate(
    candidates: list[dict],
    items: list[str],
    custom: str,
    scenes: list[dict],
    visual_by_scene: dict[int, dict],
    match: dict[int, dict],
    source_duration: float,
    embedder=None,
) -> None:
    """Store ``avoid_match`` on every candidate for the chosen kinds and the user's words."""
    terms = custom_terms(custom)
    if not items and not terms:
        return
    term_patterns = [re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in terms]
    topic = _topic_matches(candidates, terms, embedder) if terms else {}
    asks_vision = bool(vision_avoid_text(items, custom))
    edge = max(8.0, 0.04 * float(source_duration or 0))
    for index, cand in enumerate(candidates):
        start, end = float(cand["start"]), float(cand["end"])
        text = str(cand.get("text") or "")
        metrics = cand.get("metrics") or {}
        covered = _scenes_for(start, end, scenes)
        picture = [((match or {}).get(int(s["scene_id"])) or {}).get("avoid") or {} for s in covered]
        visual = [visual_by_scene.get(int(s["scene_id"])) or {} for s in covered]

        def picture_of(key: str) -> float:
            return max((float(row.get(key) or 0) for row in picture), default=0.0)

        result: dict[str, float] = {}
        for item in items:
            if item == "filler":
                greeting = 1.0 if text and _GREETING.search(text) and len(text.split()) < 14 else 0.0
                result[item] = max(_ramp(metrics.get("filler_score") or 0, 0.3, 0.8), greeting)
            elif item == "sponsor":
                result[item] = 1.0 if _SPONSOR.search(text) else 0.0
            elif item == "swearing":
                result[item] = 1.0 if _SWEARING.search(text) else 0.0
            elif item == "talking_heads":
                speech = _ramp(metrics.get("speech_presence") or 0, 0.5, 0.9)
                # Little happening in the picture (vision model's action, or measured motion).
                action = max((float(v.get("action_strength") or 0) for v in visual), default=0.3)
                still = _ramp(0.6 - action, 0.0, 0.4)
                face = picture_of("talking_heads")
                # With picture matching, a close-up face confirms it; without, speech and calm do.
                result[item] = speech * still * ((0.5 + 0.5 * face) if picture else 1.0)
            elif item == "dark":
                quality = float(metrics.get("production_quality", 0.7))
                result[item] = _ramp(POOR_QUALITY + 0.1 - quality, 0.0, 0.2)
            elif item in {"text", "screens", "empty"}:
                result[item] = picture_of(item)
            elif item == "intro_outro":
                result[item] = _intro_outro(start, end, source_duration, edge, text, visual, picture_of("text"))
            elif item == "silence":
                result[item] = _ramp(metrics.get("silence_penalty") or 0, 0.25, 0.7)
        if terms:
            said = 1.0 if any(p.search(text) for p in term_patterns) else 0.0
            result["custom"] = max(said, topic.get(index, 0.0), picture_of("custom"))
        if asks_vision:
            # One score from the vision model for everything it was asked to recognize.
            result["vision"] = max(
                (float(v.get("avoid") or 0) for v in visual if v.get("avoid") is not None), default=0.0
            )
        cand["avoid_match"] = {key: round(value, 3) for key, value in result.items()}


def _intro_outro(
    start: float, end: float, duration: float, edge: float, text: str, visual: list[dict], title: float
) -> float:
    """Openings and endings: half strength for the position alone, full with a sign of it
    (a greeting or sign-off, a title card, or the vision model calling it an intro/ending)."""
    if not duration:
        return 0.0
    if start < edge:
        sign = bool(_GREETING.search(text)) or any(v.get("appears_intro") for v in visual) or title >= 0.6
        return 1.0 if sign else 0.5
    if end > duration - edge:
        sign = bool(_SIGN_OFF.search(text)) or any(v.get("appears_ending") for v in visual) or title >= 0.6
        return 1.0 if sign else 0.5
    return 0.0


def _topic_matches(candidates: list[dict], terms: list[str], embedder) -> dict[int, float]:
    """Meaning-based match of spoken text with the user's words (needs the embedding model)."""
    if embedder is None or not getattr(embedder, "available", False):
        return {}
    spoken = [(i, str(c.get("text") or "")) for i, c in enumerate(candidates) if str(c.get("text") or "").strip()]
    if not spoken:
        return {}
    vectors = embedder(terms + [text for _, text in spoken])
    if vectors is None or len(vectors) != len(terms) + len(spoken):
        return {}
    from app.analysis.semantic import FOCUS_BASE, FOCUS_SPAN

    term_vectors, rows = vectors[: len(terms)], vectors[len(terms) :]
    best = (rows @ term_vectors.T).max(axis=1)
    return {
        index: round(float(np.clip((value - FOCUS_BASE) / FOCUS_SPAN, 0.0, 1.0)), 3)
        for (index, _), value in zip(spoken, best)
    }


def strength(candidate: dict, items: list[str], custom: str) -> float:
    """The strongest match among what the user chose to leave out."""
    matches = candidate.get("avoid_match") or {}
    keys = list(items) + (["custom"] if custom_terms(custom) else []) + ["vision"]
    return max((float(matches.get(key) or 0) for key in keys), default=0.0)


def apply(candidates: list[dict], items: list[str], custom: str, mode: str) -> tuple[list[dict], int, bool]:
    """Lower or remove matching candidates. Returns the candidates, how many were removed, and
    whether "never" had to fall back to "avoid" because it would have removed everything."""
    if not items and not custom_terms(custom):
        return candidates, 0, False
    never = mode == "never"
    kept: list[dict] = []
    removed = 0
    for cand in candidates:
        value = strength(cand, items, custom)
        if never and value >= NEVER_FROM:
            removed += 1
            continue
        kept.append(_penalized(cand, value))
    if never and not kept and candidates:
        return [_penalized(c, strength(c, items, custom)) for c in candidates], 0, True
    kept.sort(key=lambda c: float(c.get("score", 0)), reverse=True)
    return kept, removed, False


def _penalized(cand: dict, value: float) -> dict:
    if value < AVOID_FROM:
        return cand
    factor = 1.0 - AVOID_PENALTY * value
    return {**cand, "score": round(float(cand.get("score", 0)) * factor, 4), "avoid_factor": round(factor, 3)}
