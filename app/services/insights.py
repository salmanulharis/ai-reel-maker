"""Processing details of a job, for the page that explains where the time went.

Everything comes from what a job already keeps: its step rows (seconds per step, over every
attempt), its time totals, the log, and three small files written when a run finishes:
``ai_usage.json`` (model work per AI step), ``run_info.json`` (the models and options used and
what the analysis found) and ``render_timings.json`` (the parts of creating the video). Older
jobs have no such files: their AI usage is read back from the log lines, and the rest is left
out. Nothing here runs a model or reads the video.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.analysis.pipeline import AI_USAGE_FILE, RUN_INFO_FILE
from app.analysis.progress import STAGES
from app.utils.atomic import read_json

RENDER_TIMINGS_FILE = "render_timings.json"
LOG_FILE = "job.log"
_LABELS = {**dict(STAGES), "rendering": "Creating the video"}
_RENDER_LABELS = {
    "narration": "Speaking the narration",
    "clips": "Cutting and encoding clips",
    "compose": "Drawing titles and captions",
    "music": "Mixing the music",
}
# A step taking at least this share of the time is worth explaining.
_BOTTLENECK_SHARE = 0.2
_MAX_BOTTLENECKS = 3
_MAX_WARNINGS = 40
_USAGE_LINE = re.compile(
    r"^(?P<label>[A-Za-z][A-Za-z ]{1,30}): (?P<calls>\d+) calls?, (?P<pt>[\d,]+) prompt tokens in (?P<ps>[\d.]+)s, "
    r"(?P<ot>[\d,]+) reply tokens in (?P<os>[\d.]+)s(?: \([\d.]+ tok/s\))?(?:, model load (?P<load>[\d.]+)s)?"
)
_NUMBERS = re.compile(r"[0-9]+(?:[.,][0-9]+)*")
_BUSY_LINE = re.compile(r"Other programs are using (\d+)%")


def job_insights(manager, job) -> dict[str, Any]:
    """The processing details of ``job`` (see the module docstring)."""
    job_dir: Path = manager.job_dir(job.id)
    data = manager.serialize(job)
    meta = data.get("metadata") or {}
    timing = data.get("timing") or {}
    logs = read_log(job_dir)
    usage = _usage(job_dir, logs)
    run = read_json(job_dir / RUN_INFO_FILE, None)
    render = read_json(job_dir / RENDER_TIMINGS_FILE, None)
    steps = _steps(data.get("stages") or [])
    total_steps = sum(step["seconds"] for step in steps)
    for step in steps:
        step["share"] = round(step["seconds"] / total_steps, 4) if total_steps > 0 else 0.0
    slowest = [step["name"] for step in sorted(steps, key=lambda s: -s["seconds"]) if step["seconds"] >= 1][:3]
    video_seconds = float(meta.get("duration") or 0)
    total = float(timing.get("total_seconds") or 0) or total_steps
    warnings = _warnings(logs)
    return {
        "job": {
            "id": job.id,
            "filename": data.get("original_filename"),
            "status": data.get("status"),
            "created_at": data.get("created_at"),
            "mode": (data.get("config") or {}).get("mode", "manual"),
        },
        "video": {
            "duration": video_seconds,
            "width": int(meta.get("width") or 0),
            "height": int(meta.get("height") or 0),
            "codec": str(meta.get("video_codec") or ""),
            "fps": float(meta.get("fps") or 0),
            "size_bytes": int(meta.get("size_bytes") or 0),
            "sources": len(data.get("sources") or []) or 1,
        },
        "result": {
            "clips": sum(1 for clip in data.get("clips") or [] if clip.get("kept", True)),
            "outputs": [
                {"duration": o.get("duration"), "size_bytes": o.get("size_bytes"), "kind": o.get("kind")}
                for o in data.get("outputs") or []
            ],
        },
        "totals": {
            "total_seconds": round(total, 2),
            "analysis_seconds": float(timing.get("analysis_seconds") or 0),
            "render_seconds": float(timing.get("render_seconds") or 0),
            # How long processing took compared with the length of the video.
            "ratio": round(total / video_seconds, 2) if video_seconds > 0 and total > 0 else None,
        },
        "steps": steps,
        "slowest": slowest,
        "render": _render(render),
        "ai": usage,
        "run": run if isinstance(run, dict) else None,
        "bottlenecks": _bottlenecks(steps, usage, run if isinstance(run, dict) else {}, meta, render, logs),
        "warnings": warnings,
        "has_log": (job_dir / LOG_FILE).exists(),
    }


def read_log(job_dir: Path) -> list[dict]:
    """Every line of the job's log ({level, message, ts}); unreadable lines are kept as text."""
    path = Path(job_dir) / LOG_FILE
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            entry = {"level": "INFO", "message": line}
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def log_text(job_dir: Path) -> str:
    """The log as plain text, one line per entry, for downloading."""
    from datetime import datetime

    lines = []
    for entry in read_log(job_dir):
        stamp = ""
        try:
            stamp = datetime.fromtimestamp(float(entry.get("ts"))).strftime("%Y-%m-%d %H:%M:%S") + " "
        except (TypeError, ValueError, OSError):
            pass
        lines.append(f"{stamp}{str(entry.get('level') or 'INFO'):7} {entry.get('message') or ''}")
    return "\n".join(lines) + ("\n" if lines else "")


def _steps(stages: list[dict]) -> list[dict]:
    steps = []
    for stage in stages:
        seconds = float(stage.get("seconds") or 0)
        if seconds <= 0 and stage.get("status") not in {"done", "running", "error"}:
            continue
        name = str(stage.get("name") or "")
        steps.append(
            {
                "name": name,
                "label": _LABELS.get(name, name),
                "phase": "render" if name == "rendering" else "analysis",
                "seconds": round(seconds, 2),
                "status": stage.get("status") or "",
                "detail": stage.get("detail") or "",
                "done": stage.get("done") or 0,
                "total": stage.get("total") or 0,
                "unit": stage.get("unit") or "",
            }
        )
    return steps


def _usage(job_dir: Path, logs: list[dict]) -> list[dict]:
    saved = read_json(job_dir / AI_USAGE_FILE, None)
    rows: dict[str, dict[str, float]] = {}
    if isinstance(saved, dict) and saved:
        rows = {str(label): dict(values) for label, values in saved.items() if isinstance(values, dict)}
    else:
        # Older jobs: the per-step lines of the log (the last run of each step counts).
        for entry in logs:
            match = _USAGE_LINE.match(str(entry.get("message") or ""))
            if not match:
                continue
            rows[match["label"]] = {
                "calls": float(match["calls"]),
                "prompt_tokens": float(match["pt"].replace(",", "")),
                "prompt_sec": float(match["ps"]),
                "output_tokens": float(match["ot"].replace(",", "")),
                "output_sec": float(match["os"]),
                "load_sec": float(match["load"] or 0),
            }
    result = []
    for label, values in rows.items():
        calls = int(values.get("calls") or 0)
        if not calls:
            continue
        output_sec = float(values.get("output_sec") or 0)
        output_tokens = int(values.get("output_tokens") or 0)
        prompt_sec = float(values.get("prompt_sec") or 0)
        result.append(
            {
                "step": label,
                "calls": calls,
                "prompt_tokens": int(values.get("prompt_tokens") or 0),
                "prompt_seconds": round(prompt_sec, 1),
                "reply_tokens": output_tokens,
                "reply_seconds": round(output_sec, 1),
                "tokens_per_second": round(output_tokens / output_sec, 1) if output_sec > 0 else None,
                "seconds_per_call": round((prompt_sec + output_sec) / calls, 1),
                "load_seconds": round(float(values.get("load_sec") or 0), 1),
                "near_context_limit": int(values.get("near_context_limit") or 0),
            }
        )
    return result


def _render(render: Any) -> dict | None:
    if not isinstance(render, dict):
        return None
    steps = render.get("steps") if isinstance(render.get("steps"), dict) else {}
    total = float(render.get("seconds") or 0)
    parts = [
        {"name": name, "label": _RENDER_LABELS.get(name, name), "seconds": round(float(value or 0), 2)}
        for name, value in steps.items()
        if float(value or 0) > 0
    ]
    other = total - sum(part["seconds"] for part in parts)
    if other >= 1:
        parts.append({"name": "other", "label": "Preparing and checking", "seconds": round(other, 2)})
    return {**render, "parts": sorted(parts, key=lambda p: -p["seconds"])}


def _warnings(logs: list[dict]) -> list[dict]:
    """Warnings and errors, latest last. Messages that differ only in their numbers ("Scene 3
    skipped", "Scene 4 skipped") are one entry with a count and the first such message."""
    seen: dict[str, dict] = {}
    for entry in logs:
        level = str(entry.get("level") or "INFO").upper()
        if level not in {"WARNING", "ERROR", "CRITICAL"}:
            continue
        message = str(entry.get("message") or "").strip()
        if not message:
            continue
        key = _NUMBERS.sub("#", message)
        found = seen.pop(key, None)
        if found is None:
            found = {"level": level, "message": message, "count": 0}
        found["count"] += 1
        found["ts"] = entry.get("ts")
        seen[key] = found
    return list(seen.values())[-_MAX_WARNINGS:]


def _usage_for(usage: list[dict], prefix: str) -> dict | None:
    return next((row for row in usage if row["step"].lower().startswith(prefix.lower())), None)


def _bottlenecks(
    steps: list[dict], usage: list[dict], run: dict, meta: dict, render: Any, logs: list[dict]
) -> list[dict]:
    """The steps that took the most time, each with what made it slow and what would help."""
    models = run.get("models") or {}
    options = run.get("options") or {}
    found = run.get("found") or {}
    video_seconds = float(meta.get("duration") or 0)
    items: list[dict] = []
    for step in sorted(steps, key=lambda s: -s["seconds"]):
        if step["share"] < _BOTTLENECK_SHARE or step["seconds"] < 30 or len(items) >= _MAX_BOTTLENECKS:
            continue
        why, tips = _explain(step["name"], step, usage, models, options, found, meta, video_seconds, render)
        items.append(
            {
                "name": step["name"],
                "label": step["label"],
                "seconds": step["seconds"],
                "share": step["share"],
                "why": why,
                "tips": tips,
            }
        )
    busy = [int(match.group(1)) for entry in logs if (match := _BUSY_LINE.search(str(entry.get("message") or "")))]
    if busy and max(busy) >= 50:
        items.append(
            {
                "name": "computer",
                "label": "The computer was busy",
                "seconds": 0,
                "share": 0,
                "why": f"Other programs were using up to {max(busy)}% of the processor while this video was processed.",
                "tips": ["Close other programs (browsers, games, other video apps) while a video is processed."],
            }
        )
    slow_ai = [row for row in usage if row["tokens_per_second"] is not None and row["tokens_per_second"] < 12]
    if slow_ai and not any(item["name"] == "ai_speed" for item in items):
        speed = min(row["tokens_per_second"] for row in slow_ai)
        items.append(
            {
                "name": "ai_speed",
                "label": "The AI models write slowly on this computer",
                "seconds": 0,
                "share": 0,
                "why": f"The models wrote about {speed:.0f} tokens per second. On a processor without a graphics card, "
                "AI steps take most of the time.",
                "tips": [
                    "A graphics card (or Ollama on a faster computer) makes AI steps many times faster.",
                    "Settings → \"Measure this computer\" shows realistic times and suggests faster models.",
                ],
            }
        )
    return items


def _explain(
    name: str,
    step: dict,
    usage: list[dict],
    models: dict,
    options: dict,
    found: dict,
    meta: dict,
    video_seconds: float,
    render: Any,
) -> tuple[str, list[str]]:
    width, height = int(meta.get("width") or 0), int(meta.get("height") or 0)
    codec = str(meta.get("video_codec") or "").upper()
    if name == "scenes":
        size = f"{width}×{height} {codec}".strip() if width else "the"
        why = f"Finding scene changes reads every frame of {size} video."
        tips = []
        if options.get("analysis_copy"):
            why += " A small copy was used for this, which is already the fastest way."
        else:
            tips.append("Large (4K) and AV1/HEVC videos are read from a small copy since this version; analyze again to use it.")
        tips.append("Close other programs: this step uses the processor fully.")
        return why, tips
    if name == "visual":
        row = _usage_for(usage, "Visual")
        why = "The video model looked at the most promising scenes one by one."
        if row:
            why = (
                f"The video model ({models.get('vision') or 'vision model'}) made {row['calls']} request(s), "
                f"about {row['seconds_per_call']:.0f}s each."
            )
            if found.get("scenes_ai") is not None and found.get("scenes"):
                why += f" It looked at {found['scenes_ai']} of {found['scenes']} scenes."
        tips = [
            "Settings → AI models: a smaller video model is faster (with simpler descriptions).",
            "For one video: \"Speech and sound only\" on the Create page skips this step when the picture matters little.",
            "A graphics card makes this step many times faster.",
        ]
        if options.get("vision_scope") == "all":
            tips.insert(0, "The video model looked at every scene. Settings → \"Which scenes the video model looks at\" → \"Only scenes that could make the highlight\" is much faster.")
        return why, tips
    if name == "content":
        row = _usage_for(usage, "Transcript")
        why = "The text model scored what is said, part by part."
        if row:
            why = (
                f"The text model wrote {row['reply_tokens']:,} tokens in {row['reply_seconds']:.0f}s "
                f"({row['tokens_per_second'] or 0:.0f} per second) to score {row['calls']} batch(es) of speech."
            )
        return why, [
            "\"Visuals and sound only\" on the Create page skips this step for videos where speech does not matter.",
            "A smaller or faster text model in Settings → AI models.",
        ]
    if name == "transcript":
        model = models.get("speech") or "Whisper"
        factor = f" ({step['seconds'] / video_seconds:.2f}× the video's length)" if video_seconds > 0 else ""
        tips = ["An NVIDIA graphics card makes transcription several times faster."]
        language = found.get("language")
        if language and language != "en":
            tips.insert(0, "For languages such as Malayalam or Tamil, keep \"small\" or larger: smaller models are faster but write nonsense.")
        else:
            tips.insert(0, "A smaller Whisper model is faster, with more mistakes.")
        return f"Whisper ({model}) wrote down what is said{factor}.", tips
    if name == "director":
        row = _usage_for(usage, "AI editor")
        why = "The AI editor read every candidate moment to plan the story."
        if row:
            why = f"The AI editor made {row['calls']} request(s) and read {row['prompt_tokens']:,} tokens of moment descriptions."
        tips = ["Claude as the AI editor (Settings) answers in seconds instead of minutes on a processor."]
        if options.get("editor_review"):
            tips.append("Turn off \"Check the cut once more\" in Settings to save one request.")
        return why, tips
    if name == "rendering":
        parts = (render or {}).get("steps") if isinstance(render, dict) else None
        why = "Cutting, encoding and joining the clips."
        if isinstance(parts, dict) and parts:
            main = max(parts.items(), key=lambda kv: float(kv[1] or 0))
            why = f"Most of it was {_RENDER_LABELS.get(main[0], main[0]).lower()} ({float(main[1]):.0f}s)."
        return why, [
            "Quality \"High\" uses the graphics card's encoder when there is one; \"Source\" encodes on the processor.",
            "Following faces when the shape changes, levelling the sound and narration each add time.",
        ]
    if name == "frames":
        return "Pictures were taken from every scene for the AI and picture checks.", [
            "Large videos are read from a small copy since this version.",
        ]
    if name == "sounds":
        return "The sound model labelled the audio in 5-second windows.", ["Turn off \"Sound understanding\" in Settings when sounds do not matter."]
    return f"{step['label']} took {step['share']:.0%} of the time.", []
