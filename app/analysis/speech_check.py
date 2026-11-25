"""Is the transcript usable?

Whisper always returns text, even when it guessed the language wrong or the model is too small
for the language: the result is then a mix of scripts and invented words ("ஓPTIMAISD", "ஞஞஞஞ").
Scoring that text, cutting at its "sentences" and showing it to the AI editor makes the choice
of moments close to random, so such a transcript is marked unreliable: its word timings are kept
(they still mark where speech and pauses are), but its words are not read.

Checks, all from the text alone (no model):
- letters of several non-Latin scripts in large shares (English words in a non-Latin language
  are normal; Tamil, Devanagari and Japanese together are not),
- words that mix Latin and non-Latin letters,
- long runs of one repeated letter,
- for a language with its own script, little of the text in that script,
- a language detection Whisper itself was unsure of (reported, it decides only with another sign).
"""

from __future__ import annotations

import re
import unicodedata

from app.models.schemas import Transcript

# Scripts expected for languages that do not use Latin letters (Whisper codes).
_LANGUAGE_SCRIPTS: dict[str, set[str]] = {
    "ml": {"MALAYALAM"},
    "ta": {"TAMIL"},
    "te": {"TELUGU"},
    "kn": {"KANNADA"},
    "hi": {"DEVANAGARI"},
    "mr": {"DEVANAGARI"},
    "ne": {"DEVANAGARI"},
    "sa": {"DEVANAGARI"},
    "bn": {"BENGALI"},
    "as": {"BENGALI"},
    "gu": {"GUJARATI"},
    "pa": {"GURMUKHI"},
    "or": {"ORIYA"},
    "si": {"SINHALA"},
    "ar": {"ARABIC"},
    "ur": {"ARABIC"},
    "fa": {"ARABIC"},
    "ps": {"ARABIC"},
    "he": {"HEBREW"},
    "yi": {"HEBREW"},
    "ru": {"CYRILLIC"},
    "uk": {"CYRILLIC"},
    "bg": {"CYRILLIC"},
    "sr": {"CYRILLIC", "LATIN"},
    "be": {"CYRILLIC"},
    "kk": {"CYRILLIC"},
    "mk": {"CYRILLIC"},
    "mn": {"CYRILLIC"},
    "el": {"GREEK"},
    "th": {"THAI"},
    "lo": {"LAO"},
    "km": {"KHMER"},
    "my": {"MYANMAR"},
    "ka": {"GEORGIAN"},
    "hy": {"ARMENIAN"},
    "am": {"ETHIOPIC"},
    "zh": {"CJK"},
    "yue": {"CJK"},
    "ja": {"CJK", "HIRAGANA", "KATAKANA"},
    "ko": {"HANGUL"},
}
# Scripts that belong together (Japanese writing uses all three).
_SCRIPT_FAMILY = {"HIRAGANA": "JAPANESE", "KATAKANA": "JAPANESE", "CJK": "JAPANESE"}

_MIN_LETTERS = 40
_SECOND_SCRIPT_SHARE = 0.12
_MIXED_WORD_SHARE = 0.04
_REPEAT_WORD_SHARE = 0.02
_EXPECTED_SCRIPT_SHARE = 0.6
UNSURE_PROBABILITY = 0.5
_REPEAT_RUN = re.compile(r"(\w)\1{4,}")


def _script(char: str) -> str | None:
    if not char.isalpha():
        return None
    try:
        name = unicodedata.name(char)
    except ValueError:
        return None
    first = name.split(" ", 1)[0]
    if first == "CJK":
        return "CJK"
    return first


def check(transcript: Transcript | dict | None) -> dict:
    """{"reliable": bool, "reason": str, "language", "probability"} for a transcript."""
    if isinstance(transcript, dict):
        transcript = Transcript.model_validate(transcript)
    if transcript is None or not transcript.segments:
        return {"reliable": True, "reason": "", "language": None, "probability": None}
    language = (transcript.language or "").lower() or None
    probability = transcript.language_probability
    text = " ".join(segment.text for segment in transcript.segments)
    scripts: dict[str, int] = {}
    for char in text:
        script = _script(char)
        if script:
            scripts[script] = scripts.get(script, 0) + 1
    letters = sum(scripts.values())
    result = {"reliable": True, "reason": "", "language": language, "probability": probability}
    if letters < _MIN_LETTERS:
        return result

    words = [word for word in text.split() if any(char.isalpha() for char in word)]
    mixed = sum(1 for word in words if _mixes_latin(word))
    repeats = sum(1 for word in words if _REPEAT_RUN.search(word))
    other = {name: count for name, count in scripts.items() if name != "LATIN"}
    other_total = sum(other.values())
    families: dict[str, int] = {}
    for name, count in other.items():
        family = _SCRIPT_FAMILY.get(name, name)
        families[family] = families.get(family, 0) + count
    strong = [name for name, count in families.items() if other_total and count / other_total >= _SECOND_SCRIPT_SHARE]

    problems: list[str] = []
    if len(strong) >= 2:
        problems.append(f"it mixes {', '.join(sorted(name.title() for name in strong))} letters")
    if words and mixed / len(words) >= _MIXED_WORD_SHARE:
        problems.append("many words mix scripts")
    if words and repeats / len(words) >= _REPEAT_WORD_SHARE:
        problems.append("many words repeat one letter")
    expected = _LANGUAGE_SCRIPTS.get(language or "")
    if expected and "LATIN" not in expected and other_total >= _MIN_LETTERS:
        share = sum(count for name, count in other.items() if name in expected) / other_total
        if share < _EXPECTED_SCRIPT_SHARE:
            problems.append(f"little of it is in the script of the detected language ({language})")
    unsure = probability is not None and probability < UNSURE_PROBABILITY
    if problems:
        detail = "; ".join(problems)
        if unsure:
            detail += f"; Whisper was unsure of the language ({language}, {probability:.0%})"
        result.update(reliable=False, reason=detail)
    return result


def _mixes_latin(word: str) -> bool:
    latin = other = False
    for char in word:
        script = _script(char)
        if script == "LATIN":
            latin = True
        elif script:
            other = True
        if latin and other:
            return True
    return False


def advice(result: dict, whisper_model: str) -> str:
    """What the user can do about an unreliable transcript."""
    tips = []
    if str(whisper_model or "").lower() in {"tiny", "base", "tiny.en", "base.en"}:
        tips.append("use a larger speech model (Settings → Whisper model: small or medium)")
    tips.append("choose the spoken language on the Create page")
    return " and ".join(tips)
