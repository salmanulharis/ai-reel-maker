"""Prompt mode: the AI editor that chooses the moments.

The analysis turns a video into candidate moments with clean cut points (sentence ends, pauses,
scene changes) and a highlight score. The editor reads them as one list, each moment with what
is said, what the picture shows (the vision model's description), and what can be heard (sound
labels), and picks the moments that fit the user's request, with a role and a reason for each.

The editor only picks moments by their number in the list; it never writes times, so it cannot
invent a cut. Its picks become a strong preference for the usual planner, which still fits the
length, the number of shorts, required and excluded times, and keeps moments from overlapping.
When the model is unavailable or its reply is unusable, the job is planned as usual.

Long videos give more candidates than fit in one request. They are read in parts (in time
order); each part proposes its best moments, and a final request chooses among the proposals.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Any, Callable

from app.ai.ollama import OllamaError, OllamaTruncatedError
from app.ai.prompts import DIRECTOR_ROLES, DIRECTOR_SCHEMA, director_prompt
from app.analysis.audio_events import sounds_between
from app.models.schemas import HighlightConfig

_log = logging.getLogger(__name__)

DIRECTOR_FILE = "director_plan.json"
CONTENT_FILE = "content_index.json"
VERSION = 1
# The most candidates the editor reads in total, best first; the rest are weak anyway.
MAX_CANDIDATES = 360
# Prompt tokens outside the list of moments (instructions, request, requirements, changes).
_PROMPT_BASE_TOKENS = 900
# Reply tokens per picked moment (id, role, a short reason) and for the rest of the reply.
_REPLY_TOKENS_PER_PICK = 40
_REPLY_BASE_TOKENS = 200
_CONTEXT_SAFETY = 0.9
_SAYS_CHARS = 220
_SEES_CHARS = 160
# Score given to the editor's picks (best first) and the share other moments keep, so picks
# come first and the planner can still fill up with the best of the rest.
_PICK_SCORE = 0.92
_PICK_STEP = 0.004
_OTHER_SHARE = 0.7


def run_director(
    client,
    model: str,
    settings,
    config: HighlightConfig,
    candidates: list[dict],
    content: dict,
    *,
    previous: list[str] | None = None,
    cached: dict | None = None,
    check_cancel: Callable[[], None] | None = None,
    log: Callable[[str, str], None] | None = None,
    on_part: Callable[[int, int], None] | None = None,
) -> dict | None:
    """The editor's plan for ``candidates``, or None when it could not make one. ``previous``
    lists the candidate ids of the plan the user asked to change. A ``cached`` plan with the
    same inputs is returned as is. ``on_part(done, total)`` receives the requests answered so
    far (long videos are read in parts, then a final pick)."""
    log = log or (lambda message, level="INFO": None)
    think = bool(getattr(settings, "director_think", False))
    pool = _pool(candidates, content)
    if not pool:
        return None
    key = plan_key(config, pool, model, think, previous)
    if isinstance(cached, dict) and cached.get("key") == key and cached.get("clips"):
        log("AI editor: using the plan made earlier for the same request", "INFO")
        return cached
    numbered = _aliases(pool)
    aliases = {alias: item["id"] for alias, item in numbered.items()}
    by_alias = {alias: _line(alias, item, content) for alias, item in numbered.items()}
    options = client.role_options("director")
    num_ctx = int(options.get("num_ctx") or 8192)
    target, picks = _target(config, pool)
    budget = _list_budget(num_ctx, picks, int(getattr(settings, "ollama_director_num_predict", 0) or 0), think)
    alias_list = list(by_alias)
    request = config.prompt or (config.requirements.summary if config.requirements else "")
    requirements = config.requirements.model_dump() if config.requirements else {}
    reverse = {cid: alias for alias, cid in aliases.items()}
    before = [reverse[cid] for cid in previous or [] if cid in reverse]

    def ask(chosen: list[str], up_to: int, final: bool) -> dict | None:
        if check_cancel:
            check_cancel()
        prompt = director_prompt(
            request,
            requirements,
            list(config.refine_notes),
            [by_alias[alias] for alias in chosen],
            target,
            up_to,
            previous=before if final else None,
            final=final,
        )
        return _call(client, model, prompt, options, settings, up_to, think, log)

    parts = _parts(alias_list, by_alias, budget)
    requests = len(parts) + 1 if len(parts) > 1 else 1
    if on_part:
        on_part(0, requests)
    if len(parts) > 1:
        log(f"AI editor: reading {len(alias_list)} moments in {len(parts)} parts", "INFO")
        proposals: list[str] = []
        per_part = max(4, math.ceil(picks * 1.5 / len(parts)))
        for number, part in enumerate(parts, start=1):
            reply = ask(part, per_part, final=False)
            chosen = _valid_picks(reply, set(part)) if reply else []
            if not chosen:
                # A part without an answer still offers its best-scored moments.
                chosen = [{"id": alias} for alias in part[:per_part]]
            proposals.extend(item["id"] for item in chosen)
            log(f"AI editor: part {number}/{len(parts)} proposed {len(chosen)} moment(s)", "INFO")
            if on_part:
                on_part(number, requests)
        ranked = sorted(dict.fromkeys(proposals), key=alias_list.index)
        final_list = _fit(ranked, by_alias, budget)
    else:
        final_list = alias_list
    reply = ask(final_list, picks, final=True)
    if on_part:
        on_part(requests, requests)
    chosen = _valid_picks(reply, set(final_list)) if reply else []
    if not chosen:
        return None
    return {
        "version": VERSION,
        "key": key,
        "model": model,
        "understanding": _text(reply.get("understanding"), 400),
        "left_out": _text(reply.get("left_out"), 400),
        "order": "story" if reply.get("order") == "story" else "time",
        "clips": [{"id": aliases[item["id"]], "role": item["role"], "why": item["why"]} for item in chosen],
        "considered": len(alias_list),
    }


def apply_director(candidates: list[dict], plan: dict | None) -> list[dict]:
    """Candidates with the editor's picks marked and ranked first. Other moments keep part of
    their score, so the planner can still fill up the length with the best of them."""
    if not isinstance(plan, dict) or not plan.get("clips"):
        return candidates
    picks = {str(item.get("id")): (rank, item) for rank, item in enumerate(plan["clips"]) if isinstance(item, dict)}
    story = plan.get("order") == "story"
    result = []
    for candidate in candidates:
        row = dict(candidate)
        found = picks.get(str(row.get("id")))
        if found is None:
            row["score"] = round(float(row.get("score") or 0) * _OTHER_SHARE, 4)
        else:
            rank, item = found
            row["director_rank"] = rank
            if story:
                row["director_order"] = rank
            row["role"] = str(item.get("role") or "")
            if item.get("why"):
                row["reason"] = str(item["why"])
            row["score"] = round(min(1.0, max(float(row.get("score") or 0), _PICK_SCORE - _PICK_STEP * rank)), 4)
        result.append(row)
    result.sort(key=lambda c: float(c.get("score") or 0), reverse=True)
    return result


def plan_key(config: HighlightConfig, pool: list[dict], model: str, think: bool, previous: list[str] | None) -> str:
    """Identifies the editor's inputs, so an unchanged request reuses its plan."""
    data = {
        "version": VERSION,
        "prompt": config.prompt,
        "requirements": config.requirements.model_dump() if config.requirements else None,
        "notes": list(config.refine_notes),
        "length": [config.min_duration, config.max_duration, config.length_mode, config.shorts_count],
        "moments": [[item["id"], round(float(item["start"]), 2), round(float(item["end"]), 2)] for item in pool],
        "model": model,
        "think": think,
        "previous": previous or [],
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


# Content


def content_entry(scenes: list[dict], visual: dict, sound_events: dict | None) -> dict:
    """What one video's moments show and sound like, for the editor (content_index.json)."""
    described = []
    for scene in scenes or []:
        entry = (visual or {}).get(str(scene.get("scene_id"))) or {}
        caption = str(entry.get("caption") or "").strip()
        if caption:
            described.append({"start": float(scene["start"]), "end": float(scene["end"]), "caption": caption})
    return {"scenes": described, "sounds": sound_events or {}}


def _sees(content: dict, source_id: str, start: float, end: float) -> str:
    scenes = ((content or {}).get(source_id) or {}).get("scenes") or []
    captions = [
        scene["caption"]
        for scene in scenes
        if float(scene["start"]) < end and float(scene["end"]) > start and scene.get("caption")
    ]
    return " / ".join(dict.fromkeys(captions))[:_SEES_CHARS]


def _hears(content: dict, source_id: str, start: float, end: float) -> tuple[str, float]:
    heard = sounds_between(((content or {}).get(source_id) or {}).get("sounds"), start, end)
    cue = float(heard.pop("cue", 0.0) or 0.0)
    labels = [name for name, share in sorted(heard.items(), key=lambda kv: -kv[1]) if share >= 0.25][:3]
    return ", ".join(labels), cue


def _relevance(item: dict, content: dict) -> float:
    _, cue = _hears(content, str(item.get("source_id") or "v0"), float(item["start"]), float(item["end"]))
    return float(item.get("score") or 0) + 0.25 * cue


def _pool(candidates: list[dict], content: dict) -> list[dict]:
    usable = [c for c in candidates if c.get("id") is not None and float(c.get("end", 0)) > float(c.get("start", 0))]
    usable.sort(key=lambda c: _relevance(c, content), reverse=True)
    pool = usable[:MAX_CANDIDATES]
    # The editor reads them in time order (per video), which makes context and repeats visible.
    pool.sort(key=lambda c: (str(c.get("source_id") or ""), float(c["start"])))
    return pool


def _aliases(pool: list[dict]) -> dict[str, dict]:
    return {f"m{index}": item for index, item in enumerate(pool, start=1)}


def _clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60:02d}:{rest % 60:02d}"


def _line(alias: str, item: dict, content: dict) -> str:
    start, end = float(item["start"]), float(item["end"])
    source_id = str(item.get("source_id") or "v0")
    video = f"{item.get('source_name')} " if len(content or {}) > 1 and item.get("source_name") else ""
    parts = [f"{alias} [{video}{_clock(start)}-{_clock(end)}, {end - start:.0f}s] {float(item.get('score') or 0):.2f}"]
    says = " ".join(str(item.get("text") or "").split())
    if says:
        parts.append(f'says: "{says[:_SAYS_CHARS]}"')
    sees = _sees(content, source_id, start, end)
    if sees:
        parts.append(f"sees: {sees}")
    hears, cue = _hears(content, source_id, start, end)
    if hears:
        parts.append(f"hears: {hears}" + (" (matches the wanted sounds)" if cue >= 0.7 else ""))
    return " | ".join(parts)


# Requests


def _tokens(text: str) -> int:
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    return int(ascii_chars / 3.5) + (len(text) - ascii_chars) + 1


def _reply_budget(picks: int, configured: int, think: bool) -> int:
    wanted = _REPLY_BASE_TOKENS + _REPLY_TOKENS_PER_PICK * picks
    if think:
        # Reasoning comes out of the same budget.
        wanted += 2500
    return max(configured, wanted)


def _list_budget(num_ctx: int, picks: int, configured: int, think: bool) -> int:
    """Prompt tokens left for the list of moments."""
    return max(800, int(num_ctx * _CONTEXT_SAFETY) - _PROMPT_BASE_TOKENS - _reply_budget(picks, configured, think))


def _parts(aliases: list[str], lines: dict[str, str], budget: int) -> list[list[str]]:
    parts: list[list[str]] = [[]]
    used = 0
    for alias in aliases:
        cost = _tokens(lines[alias])
        if parts[-1] and used + cost > budget:
            parts.append([])
            used = 0
        parts[-1].append(alias)
        used += cost
    return [part for part in parts if part]


def _fit(aliases: list[str], lines: dict[str, str], budget: int) -> list[str]:
    kept, used = [], 0
    for alias in aliases:
        cost = _tokens(lines[alias])
        if used + cost > budget:
            break
        kept.append(alias)
        used += cost
    return kept


def _target(config: HighlightConfig, pool: list[dict]) -> tuple[str, int]:
    """The length asked for, in words, and how many moments the editor may pick."""
    lengths = sorted(float(c["end"]) - float(c["start"]) for c in pool) or [5.0]
    typical = max(2.0, lengths[len(lengths) // 2])
    count = max(1, int(config.shorts_count or 0))
    if config.length_mode == "auto" and count == 1:
        text = f"as long as the strong moments need, at least {config.min_duration:.0f} seconds"
        seconds = max(config.min_duration * 2, 60.0)
    else:
        text = f"{config.min_duration:.0f}-{config.max_duration:.0f} seconds"
        seconds = config.max_duration
        if count > 1:
            text = f"{count} separate shorts of {text} each"
    picks = math.ceil(seconds * count / typical * 1.4)
    return text, max(3, min(40, picks))


def _call(client, model: str, prompt: str, options: dict, settings, picks: int, think: bool, log) -> dict | None:
    request_options = dict(options)
    request_options["num_predict"] = _reply_budget(picks, int(getattr(settings, "ollama_director_num_predict", 0) or 0), think)
    request_options["temperature"] = 0.2
    attempts = [think, False] if think else [False]
    for attempt_think in attempts:
        try:
            extra = {"think": True} if attempt_think else {}
            return client.chat(model, prompt, schema=DIRECTOR_SCHEMA, options=request_options, **extra)
        except OllamaTruncatedError:
            if attempt_think:
                log("AI editor: reasoning used the whole reply budget; asking again without it", "WARNING")
                request_options["num_predict"] = _reply_budget(picks, 0, False)
                continue
            log("AI editor: the reply was cut off", "WARNING")
            return None
        except OllamaError as exc:
            if attempt_think and "think" in str(exc).lower():
                log("AI editor: this model cannot reason first; asking without it", "INFO")
                request_options["num_predict"] = _reply_budget(picks, 0, False)
                continue
            log(f"AI editor unavailable ({str(exc)[:160]})", "WARNING")
            return None
    return None


def _valid_picks(reply: dict | None, allowed: set[str]) -> list[dict]:
    picks: list[dict] = []
    seen: set[str] = set()
    for item in (reply or {}).get("clips") or []:
        if not isinstance(item, dict):
            continue
        alias = str(item.get("id") or "").strip().lower()
        if not alias.startswith("m"):
            alias = f"m{alias}"
        if alias not in allowed or alias in seen:
            continue
        seen.add(alias)
        role = str(item.get("role") or "").strip().lower()
        picks.append(
            {
                "id": alias,
                "role": role if role in DIRECTOR_ROLES else "highlight",
                "why": _text(item.get("why"), 200),
            }
        )
    return picks


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


# Shared with the video brief (app.brief), which reads the moments the same way.


def candidate_pool(candidates: list[dict], content: dict, limit: int = MAX_CANDIDATES) -> list[dict]:
    """The best ``limit`` moments (by score and wanted sounds), in time order per video."""
    usable = [c for c in candidates if c.get("id") is not None and float(c.get("end", 0)) > float(c.get("start", 0))]
    usable.sort(key=lambda c: _relevance(c, content), reverse=True)
    pool = usable[: max(1, min(limit, MAX_CANDIDATES))]
    pool.sort(key=lambda c: (str(c.get("source_id") or ""), float(c["start"])))
    return pool


def alias_map(pool: list[dict]) -> dict[str, dict]:
    """Moments numbered m1, m2, ... in the order given."""
    return _aliases(pool)


def moment_line(alias: str, item: dict, content: dict) -> str:
    """One moment as the AI reads it: time, length, score, what is said, seen and heard."""
    return _line(alias, item, content)


def request_text(config: HighlightConfig) -> str:
    return config.prompt or (config.requirements.summary if config.requirements else "")
