"""Making, applying and editing a job's brief.

``make_brief`` runs one engine (the user's choice), checks its reply by code, gives the engine
one chance to fix the problems found, and stores the result in ``brief.json``, keyed by its
inputs so an unchanged request reuses it. A brief edited on the Review page is kept until the
inputs change.

The local engine never stops a job (rules write the brief when Ollama cannot). The Claude
engine raises BriefEngineError when it cannot write the brief; the pipeline then pauses the job
with the reason, and the user chooses to retry or to use the local engine.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from app.analysis.director import apply_director
from app.brief.engines import BriefEngineError
from app.brief.engines.local import LocalEngine
from app.brief.evidence import add_frame_sheets, build_pack
from app.brief.schema import (
    BRIEF_DIR,
    BRIEF_FILE,
    VERSION,
    BeatReply,
    BriefReply,
    VideoBrief,
    clean_edit,
    word_count,
)
from app.brief.validate import check
from app.models.schemas import HighlightConfig
from app.utils.atomic import atomic_write_json, read_json

FAILURE_FILE = "failure.json"


def brief_dir(job_dir: Path) -> Path:
    folder = Path(job_dir) / BRIEF_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def load_brief(job_dir: Path) -> VideoBrief | None:
    data = read_json(Path(job_dir) / BRIEF_FILE, None)
    if not isinstance(data, dict):
        return None
    try:
        return VideoBrief.model_validate(data)
    except ValueError:
        return None


def save_brief(job_dir: Path, brief: VideoBrief) -> None:
    atomic_write_json(Path(job_dir) / BRIEF_FILE, brief.model_dump())


def make_brief(
    job_dir: Path,
    config: HighlightConfig,
    candidates: list[dict],
    content: dict,
    *,
    settings,
    ollama=None,
    runner=None,
    sources: list[dict] | None = None,
    video_name: str = "",
    cached: dict | None = None,
    director_plan: dict | None = None,
    log: Callable[[str, str], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> VideoBrief:
    log = log or (lambda message, level="INFO": None)
    engine_name = config.brief_engine
    pack = build_pack(config, candidates, content, video_name=video_name)
    if engine_name == "claude":
        from app.brief.engines.claude_cli import ClaudeCliEngine

        engine = ClaudeCliEngine(settings, runner)
        if getattr(settings, "claude_send_frames", True) and runner is not None:
            sheets = add_frame_sheets(pack, Path(job_dir), sources or [], runner.run, check_cancel)
            if sheets:
                log(f"Brief: {len(sheets)} frame sheet(s) of the best moments prepared for Claude", "INFO")
        settings_key: tuple = (engine.model_name(), engine.effort(), bool(pack.sheets))
    else:
        engine = LocalEngine(settings, ollama, director_plan)
        settings_key = (engine.model_name(),)
    key = pack.key(VERSION, engine_name, *settings_key)
    previous = _valid(cached)
    if previous is not None and previous.key == key and previous.beats:
        log("Brief: using the brief made earlier for the same request" + (" (with your edits)" if previous.edited else ""), "INFO")
        clear_failure(job_dir)
        return previous

    if engine_name == "claude":
        status = engine.check(refresh=True)
        if not status.ready:
            raise BriefEngineError(status.problems[0] if status.problems else "Claude is not ready.", reason="not_ready")
    started = time.perf_counter()
    write_kwargs = {"cwd": brief_dir(job_dir)} if engine_name == "claude" else {}
    reply = engine.write(pack, log=log, check_cancel=check_cancel, **write_kwargs)
    brief, problems = check(reply, pack, config, engine=engine_name, model=engine.model_name(), key=key)
    if problems:
        log(f"Brief: {len(problems)} problem(s) found; asking the {_label(engine_name)} engine to fix them", "INFO")
        try:
            fixed = engine.write(pack, repair=(reply, problems), log=log, check_cancel=check_cancel, **write_kwargs)
            second, second_problems = check(fixed, pack, config, engine=engine_name, model=engine.model_name(), key=key)
            if second is not None and (brief is None or len(second_problems) <= len(problems)):
                brief, problems = second, second_problems
        except BriefEngineError as exc:
            if brief is None:
                raise
            log(f"Brief: the repair request failed ({exc}); the problems are fixed by rules", "WARNING")
    if brief is None:
        if engine_name == "claude":
            raise BriefEngineError("Claude's brief did not use any of the video's moments.", reason="bad_reply")
        brief, problems = check(rules_reply(pack), pack, config, engine=engine_name, model="rules", key=key)
        if brief is None:
            raise ValueError("No moments to make a brief from.")
    # What was still wrong after the repair request; the rules fixed all of it except missing
    # words, which only an engine can write.
    brief.checks = list(problems)
    for line in brief.checks:
        log(f"Brief check: {line}", "INFO")
    save_brief(job_dir, brief)
    clear_failure(job_dir)
    log(
        f"Brief ready in {time.perf_counter() - started:.1f}s: \"{brief.title}\", {len(brief.beats)} beat(s), "
        f"{sum(word_count(b.narration) for b in brief.beats)} narration words",
        "INFO",
    )
    return brief


def rules_reply(pack) -> BriefReply:
    """A brief without AI: the best moments in time order, no narration."""
    _low, high = pack.beats_range
    ranked = sorted(pack.aliases, key=lambda a: float(pack.aliases[a].get("score") or 0), reverse=True)[:high]
    order = list(pack.aliases)
    beats = [BeatReply(clips=[alias], purpose="highlight") for alias in sorted(ranked, key=order.index)]
    if beats:
        beats[0].purpose = "hook"
    return BriefReply(title=pack.request[:80], beats=beats)


def apply_brief(candidates: list[dict], brief: VideoBrief | dict | None) -> list[dict]:
    """Candidates with the brief's moments ranked first, in the brief's story order (the same
    strong preference the AI editor's picks get; the planner still fits the length)."""
    brief = _valid(brief) if isinstance(brief, dict) else brief
    if brief is None or not brief.beats:
        return candidates
    plan = {
        "order": "story",
        "clips": [
            {"id": clip.id, "role": beat.purpose, "why": beat.on_screen_text or beat.narration[:120]}
            for beat in brief.beats
            for clip in beat.clips
        ],
    }
    return apply_director(candidates, plan)


def planning_config(config: HighlightConfig) -> HighlightConfig:
    """A narrated video follows the brief's story order unless the user chose another order."""
    if config.video_brief and config.order_mode == "chronological":
        return config.model_copy(update={"order_mode": "ai"})
    return config


def update_brief(job_dir: Path, changes: dict) -> tuple[VideoBrief, list[str]]:
    """Apply text edits from the Review page. Returns the brief and warnings (a narration longer
    than its beat makes that beat's last clip run longer when the video is created)."""
    brief = load_brief(job_dir)
    if brief is None:
        raise LookupError("This job has no brief yet.")
    for name in ("title", "hook", "call_to_action"):
        if name in changes:
            setattr(brief, name, clean_edit(changes[name], name))
    by_id = {beat.id: beat for beat in brief.beats}
    for item in changes.get("beats") or []:
        if not isinstance(item, dict):
            continue
        beat = by_id.get(str(item.get("id") or ""))
        if beat is None:
            continue
        if "narration" in item:
            beat.narration = clean_edit(item["narration"], "narration")
        if "on_screen_text" in item:
            beat.on_screen_text = clean_edit(item["on_screen_text"], "on_screen_text")
    brief.edited = True
    save_brief(job_dir, brief)
    warnings = [
        f"Beat {beat.id[1:]} has {word_count(beat.narration)} words; about {beat.max_words} fit, so its last clip will run longer."
        for beat in brief.beats
        if beat.max_words and word_count(beat.narration) > beat.max_words
    ]
    return brief, warnings


# A Claude failure that paused the job, for the Analysis page.


def save_failure(job_dir: Path, message: str, reason: str) -> None:
    atomic_write_json(brief_dir(job_dir) / FAILURE_FILE, {"message": message, "reason": reason, "at": time.time()})


def read_failure(job_dir: Path) -> dict | None:
    data = read_json(Path(job_dir) / BRIEF_DIR / FAILURE_FILE, None)
    return data if isinstance(data, dict) else None


def clear_failure(job_dir: Path) -> None:
    (Path(job_dir) / BRIEF_DIR / FAILURE_FILE).unlink(missing_ok=True)


def _valid(data) -> VideoBrief | None:
    if isinstance(data, VideoBrief):
        return data
    if not isinstance(data, dict):
        return None
    try:
        brief = VideoBrief.model_validate(data)
    except ValueError:
        return None
    return brief if brief.version == VERSION else None


def _label(engine: str) -> str:
    return "Claude" if engine == "claude" else "local"
