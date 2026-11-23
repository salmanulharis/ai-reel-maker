"""Compare the AI editor's choice with the score-based plan for a few descriptions.

Usage:
    python scripts/prompt_eval.py JOB_ID "first description" ["another description" ...]

The job must have a finished analysis (any mode). For each description the script reads it
(as the Create page does), runs the AI editor on the job's saved candidates, and prints both
plans side by side: the clips, their reasons, and how much of the score-based plan the AI kept.
Nothing in the job is changed. Results are also saved to data/prompt_eval.json so runs with
other models or settings can be compared.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _clip_line(clip: dict) -> str:
    start, end = float(clip["start"]), float(clip["end"])
    role = f"[{clip['role']}] " if clip.get("role") else ""
    text = " ".join(str(clip.get("transcript") or "").split())[:70]
    return f"  {start:7.1f}-{end:7.1f}s  {role}{clip.get('reason', '')[:60]}  {text}"


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    from app.ai.intent import interpret
    from app.analysis.director import run_director
    from app.analysis.pipeline import director_model, plan_from_cache, planning_pool
    from app.config import get_settings
    from app.db import init_db
    from app.models.schemas import HighlightConfig
    from app.services.job_manager import get_job_manager
    from app.services.ollama_runtime import get_ollama_client
    from app.services.settings_store import effective_settings
    from app.workers.worker import load_cache

    settings = get_settings()
    init_db(settings.db_path)
    effective_settings(settings)
    job_id, prompts = argv[1], argv[2:]
    manager = get_job_manager()
    job = manager.get(job_id)
    if job is None:
        print(f"No job {job_id}")
        return 1
    cache = load_cache(manager.job_dir(job_id))
    if not cache.get("candidates_all") and not cache.get("candidates"):
        print("This job has no finished analysis.")
        return 1
    base = HighlightConfig.model_validate({**json.loads(job.config_json or "{}"), "mode": "manual", "prompt": ""})
    client = get_ollama_client(settings)
    if not client.reachable():
        print("Ollama is not reachable; the AI editor needs it.")
        return 1
    model = director_model(settings)
    duration = float((cache.get("metadata") or {}).get("duration") or 0)
    results = []
    for prompt in prompts:
        read = interpret(prompt, duration=duration, client=client, model=settings.ollama_text_model, use_speech=base.use_speech)
        values = {**base.model_dump(), **read["settings"], "mode": "prompt", "prompt": prompt, "requirements": read["requirements"]}
        config = HighlightConfig.model_validate(values)
        scored = plan_from_cache({**cache, "director_plan": None}, config.model_copy(update={"mode": "manual"}), settings)
        pool, _note, _fn = planning_pool(cache, config, settings)
        started = time.perf_counter()
        plan = run_director(
            client, model, settings, config, pool, cache.get("content_index") or {}, log=lambda m, level="INFO": print(f"  · {m}")
        )
        took = time.perf_counter() - started
        chosen = plan_from_cache({**cache, "director_plan": plan}, config, settings) if plan else None
        print(f"\n=== {prompt}\nUnderstood: {read['summary']}  (read by {read['source']})")
        print(f"Settings: {read['settings']}")
        print(f"Score-based plan ({len(scored['clips'])} clips, {scored['estimated_duration']:.0f}s):")
        for clip in scored["clips"]:
            print(_clip_line(clip))
        if chosen is None:
            print("AI editor: no usable plan")
        else:
            kept = {c.get("candidate_id") for c in scored["clips"]} & {c.get("candidate_id") for c in chosen["clips"]}
            print(
                f"AI plan in {took:.0f}s ({len(chosen['clips'])} clips, {chosen['estimated_duration']:.0f}s, "
                f"{len(kept)} shared with the score-based plan): {plan.get('understanding')}"
            )
            for clip in chosen["clips"]:
                print(_clip_line(clip))
        results.append(
            {"prompt": prompt, "read": read, "scored": scored, "ai": chosen, "director": plan, "seconds": round(took, 1)}
        )
    out = Path(settings.data_dir) / "prompt_eval.json"
    out.write_text(json.dumps({"job": job_id, "model": model, "results": results}, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
