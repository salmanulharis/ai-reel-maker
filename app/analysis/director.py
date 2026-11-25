"""Prompt mode: the AI editor that chooses the moments.

The analysis turns a video into candidate moments with clean cut points (sentence ends, pauses,
scene changes) and a highlight score. The editor reads them as one list, each moment with what
is said, what the picture shows (the vision model's description), and what can be heard (sound
labels), and picks the moments that fit the user's request, with a role and a reason for each.

It works like an editor, in steps:
1. Outline. The video is split into chapters (stretches of a minute or so). From a short
   summary of each chapter the editor plans the story: a hook, the context a viewer needs, the
   key moments and an ending, each beat taken from one chapter, with a length.
2. Picks. The editor chooses moments for each beat (every moment is shown with its chapter),
   in the outline's order. Long videos are read in parts, then a final pick.
3. Shape. Rules make the result a story: a beat without a moment gets the best one of its
   chapter, the picks follow the outline, one hook opens and one ending closes.
4. Review (optional, a setting). The editor reads its cut in order and may remove moments that
   repeat or confuse, or add the moment just before one that cannot be understood without it.

The editor only picks moments by their number in the list; it never writes times, so it cannot
invent a cut. Its picks become a strong preference for the usual planner, which still fits the
length, the number of shorts, required and excluded times, and keeps moments from overlapping.
When the local model is unavailable or its reply is unusable, the job is planned as usual. The
editor can also be Claude (Claude Code on the user's subscription), when the user chose it;
then any failure raises BriefEngineError and the job pauses (engines never switch by themselves).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Any, Callable

from app.ai.ollama import OllamaError, OllamaTruncatedError
from app.ai.prompts import (
    DIRECTOR_ROLES,
    DIRECTOR_SCHEMA,
    OUTLINE_SCHEMA,
    REVIEW_SCHEMA,
    director_prompt,
    outline_prompt,
    review_prompt,
)
from app.analysis.audio_events import sounds_between
from app.models.schemas import HighlightConfig

_log = logging.getLogger(__name__)

DIRECTOR_FILE = "director_plan.json"
CONTENT_FILE = "content_index.json"
VERSION = 2
# The most candidates the editor reads in total, best first; the rest are weak anyway.
MAX_CANDIDATES = 360
# Prompt tokens outside the list of moments (instructions, request, requirements, outline).
_PROMPT_BASE_TOKENS = 1100
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
# Chapters: about this long, at most this many in total, summarised in this many characters.
_CHAPTER_SECONDS = 75.0
_MAX_CHAPTERS = 12
_CHAPTER_SAYS_CHARS = 260
_CHAPTER_SEES_CHARS = 200
_OUTLINE_TOKENS_PER_BEAT = 60
# Review: moments offered as missing context are at most this far before the moment they lead to.
_CONTEXT_GAP = 20.0
_MAX_CONTEXT = 12

CLAUDE_SYSTEM = (
    "You are a video editor cutting a short video from a longer one. You only choose moments and "
    "chapters by the ids in the lists; never write times and never invent moments. Reply with JSON only."
)

Ask = Callable[[str, dict, int], "dict | None"]


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
    engine: str = "local",
    claude=None,
    claude_cwd=None,
) -> dict | None:
    """The editor's plan for ``candidates``, or None when it could not make one. ``previous``
    lists the candidate ids of the plan the user asked to change. A ``cached`` plan with the
    same inputs is returned as is. ``on_part(done, total)`` receives the requests answered so
    far. ``engine`` "claude" asks ``claude`` (a ClaudeCliEngine) instead of the local model;
    its failures raise BriefEngineError."""
    log = log or (lambda message, level="INFO": None)
    use_claude = engine == "claude" and claude is not None
    think = bool(getattr(settings, "director_think", False)) and not use_claude
    review = bool(getattr(settings, "director_review", False))
    pool = _pool(candidates, content)
    if not pool:
        return None
    engine_name = "claude" if use_claude else "local"
    key = plan_key(config, pool, model if not use_claude else f"claude:{claude.model_name()}", think, previous, engine_name, review)
    if isinstance(cached, dict) and cached.get("key") == key and cached.get("clips"):
        log("AI editor: using the plan made earlier for the same request", "INFO")
        return cached
    numbered = _aliases(pool)
    aliases = {alias: item["id"] for alias, item in numbered.items()}
    chapters, chapter_of = build_chapters(pool, content)
    alias_chapter = {alias: chapter_of.get(str(item["id"])) for alias, item in numbered.items()}
    use_outline = len(chapters) >= 2
    by_alias = {
        alias: _line(alias, item, content, alias_chapter[alias] if use_outline else None)
        for alias, item in numbered.items()
    }
    target, picks = _target(config, pool)
    if use_claude:
        ask = _claude_asker(claude, claude_cwd, check_cancel)
        budget = 10**9
    else:
        options = client.role_options("director")
        num_ctx = int(options.get("num_ctx") or 8192)
        budget = _list_budget(num_ctx, picks, int(getattr(settings, "ollama_director_num_predict", 0) or 0), think)
        ask = _local_asker(client, model, options, settings, think, log, check_cancel)
    alias_list = list(by_alias)
    request = config.prompt or (config.requirements.summary if config.requirements else "")
    requirements = config.requirements.model_dump() if config.requirements else {}
    notes = list(config.refine_notes)
    reverse = {cid: alias for alias, cid in aliases.items()}
    before = [reverse[cid] for cid in previous or [] if cid in reverse]

    parts = _parts(alias_list, by_alias, budget)
    requests = (len(parts) + 1 if len(parts) > 1 else 1) + (1 if use_outline else 0) + (1 if review else 0)
    answered = 0

    def tick() -> None:
        nonlocal answered
        answered += 1
        if on_part:
            on_part(min(answered, requests), requests)

    if on_part:
        on_part(0, requests)

    # 1. Outline.
    outline: list[dict] = []
    understanding = ""
    if use_outline:
        max_beats = max(3, min(10, picks))
        reply = ask(
            outline_prompt(request, requirements, notes, [c["line"] for c in chapters], target, max_beats, before or None),
            OUTLINE_SCHEMA,
            _REPLY_BASE_TOKENS + _OUTLINE_TOKENS_PER_BEAT * max_beats,
        )
        tick()
        outline = _valid_outline(reply, {c["id"] for c in chapters})
        if outline:
            understanding = _text((reply or {}).get("understanding"), 400)
            log(
                "AI editor: story outline — "
                + "; ".join(f"{beat['role']} ({beat['chapter']}): {beat['purpose']}" for beat in outline)[:600],
                "INFO",
            )
        else:
            log("AI editor: no usable outline; choosing moments without one", "WARNING")
    outline_lines = [
        f"{number}. {beat['role']} from {beat['chapter']}: {beat['purpose']} (about {beat['seconds']:.0f}s)"
        for number, beat in enumerate(outline, start=1)
    ] or None

    def pick(chosen: list[str], up_to: int, final: bool) -> dict | None:
        prompt = director_prompt(
            request,
            requirements,
            notes,
            [by_alias[alias] for alias in chosen],
            target,
            up_to,
            previous=before if final else None,
            final=final,
            outline=outline_lines,
        )
        return ask(prompt, DIRECTOR_SCHEMA, _REPLY_BASE_TOKENS + _REPLY_TOKENS_PER_PICK * up_to)

    # 2. Picks.
    if len(parts) > 1:
        log(f"AI editor: reading {len(alias_list)} moments in {len(parts)} parts", "INFO")
        proposals: list[str] = []
        per_part = max(4, math.ceil(picks * 1.5 / len(parts)))
        for number, part in enumerate(parts, start=1):
            reply = pick(part, per_part, final=False)
            chosen = _valid_picks(reply, set(part)) if reply else []
            if not chosen:
                # A part without an answer still offers its best-scored moments.
                chosen = [{"id": alias} for alias in part[:per_part]]
            proposals.extend(item["id"] for item in chosen)
            log(f"AI editor: part {number}/{len(parts)} proposed {len(chosen)} moment(s)", "INFO")
            tick()
        # Every beat's chapter keeps a few moments in the final list.
        for beat in outline:
            proposals.extend(
                sorted(
                    (a for a in alias_list if alias_chapter.get(a) == beat["chapter"]),
                    key=lambda a: -float(numbered[a].get("score") or 0),
                )[:3]
            )
        ranked = sorted(dict.fromkeys(proposals), key=alias_list.index)
        final_list = _fit(ranked, by_alias, budget)
    else:
        final_list = alias_list
    reply = pick(final_list, picks, final=True)
    tick()
    chosen = _valid_picks(reply, set(final_list)) if reply else []
    if not chosen:
        return None
    story = bool(outline) or (reply or {}).get("order") == "story"

    # 3. Shape.
    chosen = _shape(chosen, outline, alias_chapter, final_list, numbered, story)

    # 4. Review.
    review_notes = ""
    if review and len(chosen) >= 2:
        chosen, review_notes = _review(
            ask, chosen, alias_list, numbered, by_alias, request, requirements, notes, log
        )
        chosen = _shape(chosen, [], alias_chapter, final_list, numbered, story)
        tick()
    if on_part:
        on_part(requests, requests)
    return {
        "version": VERSION,
        "key": key,
        "model": model if not use_claude else f"claude:{claude.model_name()}",
        "engine": engine_name,
        "understanding": understanding or _text((reply or {}).get("understanding"), 400),
        "left_out": _text((reply or {}).get("left_out"), 400),
        "order": "story" if story else "time",
        "outline": outline,
        "review": review_notes,
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


def plan_key(
    config: HighlightConfig,
    pool: list[dict],
    model: str,
    think: bool,
    previous: list[str] | None,
    engine: str = "local",
    review: bool = False,
) -> str:
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
        "engine": engine,
        "review": review,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


# Content


def content_entry(
    scenes: list[dict], visual: dict, sound_events: dict | None, *, speech_reliable: bool = True
) -> dict:
    """What one video's moments show and sound like, for the editor (content_index.json).
    ``speech_reliable`` False: the transcript is garbled, so the editor is not shown its words."""
    described = []
    for scene in scenes or []:
        entry = (visual or {}).get(str(scene.get("scene_id"))) or {}
        caption = str(entry.get("caption") or "").strip()
        if caption:
            described.append({"start": float(scene["start"]), "end": float(scene["end"]), "caption": caption})
    return {"scenes": described, "sounds": sound_events or {}, "speech_reliable": bool(speech_reliable)}


def _speech_shown(content: dict, source_id: str) -> bool:
    return ((content or {}).get(source_id) or {}).get("speech_reliable", True) is not False


def _sees(content: dict, source_id: str, start: float, end: float, limit: int = _SEES_CHARS) -> str:
    scenes = ((content or {}).get(source_id) or {}).get("scenes") or []
    captions = [
        scene["caption"]
        for scene in scenes
        if float(scene["start"]) < end and float(scene["end"]) > start and scene.get("caption")
    ]
    return " / ".join(dict.fromkeys(captions))[:limit]


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


def _line(alias: str, item: dict, content: dict, chapter: str | None = None) -> str:
    start, end = float(item["start"]), float(item["end"])
    source_id = str(item.get("source_id") or "v0")
    video = f"{item.get('source_name')} " if len(content or {}) > 1 and item.get("source_name") else ""
    tag = f" ({chapter})" if chapter else ""
    parts = [f"{alias}{tag} [{video}{_clock(start)}-{_clock(end)}, {end - start:.0f}s] {float(item.get('score') or 0):.2f}"]
    says = " ".join(str(item.get("text") or "").split()) if _speech_shown(content, source_id) else ""
    if says:
        parts.append(f'says: "{says[:_SAYS_CHARS]}"')
    sees = _sees(content, source_id, start, end)
    if sees:
        parts.append(f"sees: {sees}")
    hears, cue = _hears(content, source_id, start, end)
    if hears:
        parts.append(f"hears: {hears}" + (" (matches the wanted sounds)" if cue >= 0.7 else ""))
    return " | ".join(parts)


# Chapters


def build_chapters(pool: list[dict], content: dict) -> tuple[list[dict], dict[str, str]]:
    """The video(s) split into chapters of about _CHAPTER_SECONDS, cut where the moments leave
    the largest gap near each even split. Returns the chapters ({id, source_id, start, end,
    line}) in time order and the chapter id of every moment id."""
    groups: dict[str, list[dict]] = {}
    for item in pool:
        groups.setdefault(str(item.get("source_id") or ""), []).append(item)
    spans = {
        sid: (min(float(i["start"]) for i in items), max(float(i["end"]) for i in items))
        for sid, items in groups.items()
    }
    total = sum(end - start for start, end in spans.values())
    length = max(_CHAPTER_SECONDS, total / _MAX_CHAPTERS) if total > 0 else _CHAPTER_SECONDS
    chapters: list[dict] = []
    chapter_of: dict[str, str] = {}
    multi = len(groups) > 1
    for sid, items in groups.items():
        items = sorted(items, key=lambda i: float(i["start"]))
        first, last = spans[sid]
        span = last - first
        count = max(1, min(_MAX_CHAPTERS, round(span / length)))
        bounds = [first]
        latest_end = []
        running = -math.inf
        for item in items:
            latest_end.append(running)
            running = max(running, float(item["end"]))
        for step in range(1, count):
            aim = first + span * step / count
            window = span / (2 * count)
            best, best_value = None, -math.inf
            for index, item in enumerate(items[1:], start=1):
                start = float(item["start"])
                if abs(start - aim) > window or start <= bounds[-1] + 1.0:
                    continue
                value = (start - latest_end[index]) - abs(start - aim) / max(window, 1.0)
                if value > best_value:
                    best, best_value = start, value
            bounds.append(best if best is not None else aim)
        bounds.append(last)
        for index in range(len(bounds) - 1):
            low, high = bounds[index], bounds[index + 1]
            members = [
                i
                for i in items
                if low - 1e-6 <= float(i["start"]) and (float(i["start"]) < high or index == len(bounds) - 2)
            ]
            if not members:
                continue
            chapter_id = f"c{len(chapters) + 1}"
            for member in members:
                chapter_of[str(member["id"])] = chapter_id
            chapters.append(
                {
                    "id": chapter_id,
                    "source_id": sid,
                    "start": low,
                    "end": high,
                    "line": _chapter_line(chapter_id, members, low, high, content, sid, multi),
                }
            )
    return chapters, chapter_of


def _chapter_line(chapter_id: str, members: list[dict], start: float, end: float, content: dict, sid: str, multi: bool) -> str:
    video = f"{members[0].get('source_name')} " if multi and members[0].get("source_name") else ""
    parts = [f"{chapter_id} [{video}{_clock(start)}-{_clock(end)}]"]
    if _speech_shown(content, sid or "v0"):
        best = sorted(members, key=lambda m: -float(m.get("score") or 0))[:5]
        texts = []
        for member in sorted(best, key=lambda m: float(m["start"])):
            text = " ".join(str(member.get("text") or "").split())
            if text and text not in texts:
                texts.append(text)
        says = " ... ".join(texts)[:_CHAPTER_SAYS_CHARS]
        if says:
            parts.append(f'says: "{says}"')
    sees = _sees(content, sid or "v0", start, end, _CHAPTER_SEES_CHARS)
    if sees:
        parts.append(f"sees: {sees}")
    return " | ".join(parts)


def _valid_outline(reply: dict | None, chapter_ids: set[str]) -> list[dict]:
    beats: list[dict] = []
    for item in (reply or {}).get("beats") or []:
        if not isinstance(item, dict):
            continue
        chapter = str(item.get("chapter") or "").strip().lower()
        if chapter and not chapter.startswith("c"):
            chapter = f"c{chapter}"
        if chapter not in chapter_ids:
            continue
        role = str(item.get("role") or "").strip().lower()
        try:
            seconds = max(1.0, min(600.0, float(item.get("seconds") or 0)))
        except (TypeError, ValueError):
            seconds = 10.0
        beats.append(
            {
                "chapter": chapter,
                "role": role if role in DIRECTOR_ROLES else "key",
                "purpose": _text(item.get("purpose"), 120),
                "seconds": round(seconds, 1),
            }
        )
    beats = beats[:12]
    if beats:
        beats[0]["role"] = "hook"
        if len(beats) >= 2:
            beats[-1]["role"] = "ending"
        for beat in beats[1:-1]:
            if beat["role"] in {"hook", "ending"}:
                beat["role"] = "key"
    return beats


# Story shape


def _shape(
    chosen: list[dict],
    outline: list[dict],
    alias_chapter: dict[str, str | None],
    final_list: list[str],
    numbered: dict[str, dict],
    story: bool,
) -> list[dict]:
    """The picks as a story: every outline beat has a moment (the best of its chapter when the
    editor left it empty), picks follow the outline (in time order within a beat), and in a
    story order one hook opens and one ending closes."""
    chosen = [dict(item) for item in chosen]
    taken = {item["id"] for item in chosen}
    if outline:
        covered = {alias_chapter.get(item["id"]) for item in chosen}
        for beat in outline:
            if beat["chapter"] in covered:
                continue
            options = [a for a in final_list if alias_chapter.get(a) == beat["chapter"] and a not in taken]
            if not options:
                continue
            best = max(options, key=lambda a: float(numbered[a].get("score") or 0))
            chosen.append({"id": best, "role": beat["role"], "why": beat["purpose"] or "Part of the story outline"})
            taken.add(best)
            covered.add(beat["chapter"])
        order = {}
        for index, beat in enumerate(outline):
            order.setdefault(beat["chapter"], index)

        def beat_of(alias: str) -> float:
            chapter = alias_chapter.get(alias)
            if chapter in order:
                return order[chapter]
            # A pick from a chapter outside the outline goes next to the nearest chapter.
            number = _chapter_number(chapter)
            if number is None or not order:
                return len(outline)
            nearest = min(order, key=lambda c: abs((_chapter_number(c) or 0) - number))
            return order[nearest] + 0.5

        chosen.sort(key=lambda item: (beat_of(item["id"]), float(numbered[item["id"]]["start"])))
    if story and chosen:
        _fix_roles(chosen)
    return chosen


def _chapter_number(chapter: str | None) -> int | None:
    try:
        return int(str(chapter or "")[1:])
    except ValueError:
        return None


def _fix_roles(chosen: list[dict]) -> None:
    """One hook first and one ending last (other hooks and endings become highlights)."""
    hooks = [index for index, item in enumerate(chosen) if item["role"] == "hook"]
    if hooks and hooks[0] != 0:
        chosen.insert(0, chosen.pop(hooks[0]))
    if not hooks:
        chosen[0]["role"] = "hook"
    for item in chosen[1:]:
        if item["role"] == "hook":
            item["role"] = "highlight"
    if len(chosen) < 2:
        return
    endings = [index for index, item in enumerate(chosen) if item["role"] == "ending" and index > 0]
    if endings and endings[-1] != len(chosen) - 1:
        chosen.append(chosen.pop(endings[-1]))
    if not endings:
        chosen[-1]["role"] = "ending"
    for item in chosen[1:-1]:
        if item["role"] == "ending":
            item["role"] = "highlight"


def _review(
    ask: Ask,
    chosen: list[dict],
    alias_list: list[str],
    numbered: dict[str, dict],
    by_alias: dict[str, str],
    request: str,
    requirements: dict,
    notes: list[str],
    log: Callable[[str, str], None],
) -> tuple[list[dict], str]:
    taken = {item["id"] for item in chosen}
    context: list[str] = []
    for item in chosen:
        moment = numbered[item["id"]]
        index = alias_list.index(item["id"])
        for earlier in reversed(alias_list[max(0, index - 3) : index]):
            other = numbered[earlier]
            if earlier in taken or earlier in context:
                continue
            if str(other.get("source_id") or "") != str(moment.get("source_id") or ""):
                continue
            gap = float(moment["start"]) - float(other["end"])
            if -0.5 <= gap <= _CONTEXT_GAP:
                context.append(earlier)
                break
    context = context[:_MAX_CONTEXT]
    cut = [f"{by_alias[item['id']]} | role: {item['role']}" for item in chosen]
    reply = ask(
        review_prompt(request, requirements, notes, cut, [by_alias[a] for a in context]),
        REVIEW_SCHEMA,
        _REPLY_BASE_TOKENS + 40 * (len(chosen) + len(context)),
    )
    if not reply:
        return chosen, ""
    remove = {_alias(value) for value in reply.get("remove") or []} & taken
    if len(remove) >= len(chosen):
        remove = set()
    result = [item for item in chosen if item["id"] not in remove]
    added = 0
    for item in reply.get("add") or []:
        if not isinstance(item, dict):
            continue
        alias, before = _alias(item.get("id")), _alias(item.get("before"))
        if alias not in context or alias in {r["id"] for r in result}:
            continue
        positions = [index for index, row in enumerate(result) if row["id"] == before]
        if not positions:
            continue
        result.insert(
            positions[0],
            {"id": alias, "role": "context", "why": _text(item.get("why"), 200) or "Needed to follow the next moment"},
        )
        added += 1
    review_notes = _text(reply.get("notes"), 300)
    if remove or added:
        log(f"AI editor: review removed {len(remove)} and added {added} moment(s). {review_notes}".strip(), "INFO")
    return result, review_notes


def _alias(value: Any) -> str:
    alias = str(value or "").strip().lower()
    return alias if alias.startswith("m") else f"m{alias}"


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
        if config.auto_length_strict:
            text = f"as short as the essentials allow, at least {config.min_duration:.0f} seconds"
            seconds = max(config.min_duration * 1.5, 45.0)
        else:
            text = f"as long as the strong moments need, at least {config.min_duration:.0f} seconds"
            seconds = max(config.min_duration * 2, 60.0)
    else:
        text = f"{config.min_duration:.0f}-{config.max_duration:.0f} seconds"
        seconds = config.max_duration
        if count > 1:
            text = f"{count} separate shorts of {text} each"
    picks = math.ceil(seconds * count / typical * 1.4)
    return text, max(3, min(40, picks))


def _local_asker(client, model: str, options: dict, settings, think: bool, log, check_cancel) -> Ask:
    def ask(prompt: str, schema: dict, reply_tokens: int) -> dict | None:
        if check_cancel:
            check_cancel()
        return _call(client, model, prompt, schema, options, settings, reply_tokens, think, log)

    return ask


def _claude_asker(claude, cwd, check_cancel) -> Ask:
    """Requests to Claude (Claude Code CLI). Failures raise BriefEngineError."""
    from app.brief.engines.base import BriefEngineError

    def ask(prompt: str, schema: dict, reply_tokens: int) -> dict | None:
        if check_cancel:
            check_cancel()
        result = claude._ask([{"type": "text", "text": prompt}], schema, CLAUDE_SYSTEM, cwd=cwd)
        structured = result.get("structured_output")
        if not isinstance(structured, dict):
            from app.brief.engines.claude_cli import _json_object

            structured = _json_object(str(result.get("result") or ""))
        if not structured:
            raise BriefEngineError("Claude did not answer the AI editor in the expected format.", reason="bad_reply")
        return structured

    return ask


def _call(
    client, model: str, prompt: str, schema: dict, options: dict, settings, reply_tokens: int, think: bool, log
) -> dict | None:
    request_options = dict(options)
    configured = int(getattr(settings, "ollama_director_num_predict", 0) or 0)
    request_options["num_predict"] = max(configured, reply_tokens + (2500 if think else 0))
    request_options["temperature"] = 0.2
    attempts = [think, False] if think else [False]
    for attempt_think in attempts:
        try:
            extra = {"think": True} if attempt_think else {}
            return client.chat(model, prompt, schema=schema, options=request_options, **extra)
        except OllamaTruncatedError:
            if attempt_think:
                log("AI editor: reasoning used the whole reply budget; asking again without it", "WARNING")
                request_options["num_predict"] = max(configured, reply_tokens)
                continue
            log("AI editor: the reply was cut off", "WARNING")
            return None
        except OllamaError as exc:
            if attempt_think and "think" in str(exc).lower():
                log("AI editor: this model cannot reason first; asking without it", "INFO")
                request_options["num_predict"] = max(configured, reply_tokens)
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
        alias = _alias(item.get("id"))
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
