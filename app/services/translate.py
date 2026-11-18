"""
Translate subtitles with the local text model. Only the spoken lines inside the kept clips are
translated; results are cached per job and language. Lines the model cannot translate keep the
original words, so subtitles never go missing.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

_log = logging.getLogger(__name__)
BATCH = 20
SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array", "items": {"type": "string"}}},
    "required": ["lines"],
}


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _prompt(language: str, lines: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
    return (
        f"Translate these subtitle lines into {language}. Keep the meaning and tone, keep each line short,\n"
        "and do not add or merge lines. Reply with JSON only: "
        f'{{"lines": [exactly {len(lines)} translated lines, in the same order]}}\n\n{numbered}'
    )


def translate_lines(lines: list[str], language: str, client, model: str, cache: dict) -> dict[str, str]:
    """Translations for `lines` (text -> translated text), using and filling `cache`."""
    todo = [line for line in dict.fromkeys(lines) if line.strip() and _key(line) not in cache]
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        try:
            options = client.role_options("text")
            options["num_predict"] = max(256, 60 * len(batch))
            reply = client.chat(model, _prompt(language, batch), schema=SCHEMA, options=options)
            got = reply.get("lines") if isinstance(reply, dict) else None
        except Exception as exc:  # noqa: BLE001 - untranslated subtitles are better than none
            _log.warning("Subtitle translation failed: %s", exc)
            got = None
        if isinstance(got, list) and len(got) == len(batch):
            for original, translated in zip(batch, got):
                text = " ".join(str(translated or "").split())
                if text:
                    cache[_key(original)] = text
    return {line: cache.get(_key(line), line) for line in lines}


def translated_transcript(
    transcript: dict, ranges: list[tuple[float, float]], language: str, client, model: str, cache_path: Path
) -> tuple[dict, int, int]:
    """
    A copy of `transcript` whose segments inside `ranges` are translated. Word timings are
    dropped (they belong to the original words), so styled captions show whole lines.
    Returns the transcript and how many lines were translated out of how many needed.
    """
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        cache = {}
    segments = [s for s in transcript.get("segments") or [] if isinstance(s, dict)]

    def used(segment: dict) -> bool:
        try:
            start, end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError):
            return False
        return any(end > a and start < b for a, b in ranges)

    needed = [" ".join(str(s.get("text") or "").split()) for s in segments if used(s)]
    needed = [line for line in needed if line]
    mapping = translate_lines(needed, language, client, model, cache) if needed else {}
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    out = []
    for segment in segments:
        text = " ".join(str(segment.get("text") or "").split())
        out.append({**segment, "text": mapping.get(text, text) if used(segment) else text})
    done = sum(1 for line in needed if mapping.get(line, line) != line)
    return {**transcript, "segments": out, "words": []}, done, len(set(needed))
