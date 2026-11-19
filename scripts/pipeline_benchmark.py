"""Time each analysis stage and the model work behind it on real videos.

Usage:
    python scripts/pipeline_benchmark.py VIDEO [VIDEO ...] [--runs N] [--no-ai]

Every run starts from an empty job folder, so nothing is cached between runs. Results are
printed and saved to data/pipeline_benchmark.json (one entry per video and run), which makes
before/after comparisons of settings or code changes straightforward. The scores and the
clip plan of each run are saved too, so output quality can be compared alongside speed.
"""
from __future__ import annotations

import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _run_once(video: Path, use_ai: bool) -> dict:
    from app.analysis.pipeline import PipelineProgress, VideoAnalyzer
    from app.config import get_settings
    from app.models.schemas import HighlightConfig
    from app.services.ollama_runtime import get_ollama_client
    from app.services.settings_store import effective_settings
    from app.video.ffmpeg_runner import FFmpegRunner

    from app.db import init_db

    settings = get_settings()
    # Saved settings live in the app database; open it before reading them.
    init_db(settings.db_path)
    effective_settings(settings)
    client = None
    vision = text = False
    if use_ai:
        health = get_ollama_client(settings).health()
        if health.get("connected"):
            client = get_ollama_client(settings)
            vision = bool(health.get("vision_ready"))
            text = bool(health.get("text_present"))
    logs: list[str] = []

    def emit(kind: str, payload: dict) -> None:
        if kind == "log":
            logs.append(f"[{payload.get('level')}] {payload.get('message')}")

    progress = PipelineProgress(emit)
    analyzer = VideoAnalyzer(settings, FFmpegRunner(), client, vision_enabled=vision, text_enabled=text)
    job_dir = Path(tempfile.mkdtemp(prefix="bench_", dir=settings.tmp_dir))
    try:
        sources = [{"id": "v0", "path": str(video), "filename": video.name}]
        started = time.perf_counter()
        cache = analyzer.analyze_sources(job_dir, sources, HighlightConfig(), progress, {})
        total = time.perf_counter() - started
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
    return {
        "video": str(video),
        "total_sec": round(total, 2),
        "stages": progress.timings,
        "models": {
            "vision": settings.ollama_vision_model if vision else None,
            "text": settings.ollama_text_model if text else None,
            "whisper": settings.whisper_model,
            "num_thread": settings.ollama_num_thread,
        },
        "ai_logs": [line for line in logs if " AI: " in line or "Stage timings" in line],
        "segment_scores": [
            {"start": item["start"], "end": item["end"], "score": item["score"]}
            for item in cache.get("segment_scores") or []
        ],
        "visual": cache.get("visual") or {},
        "clips": [(clip["start"], clip["end"]) for clip in (cache.get("clip_plan") or {}).get("clips", [])],
    }


def main() -> None:
    args = sys.argv[1:]
    runs = 1
    if "--runs" in args:
        position = args.index("--runs")
        runs = max(1, int(args[position + 1]))
        del args[position : position + 2]
    use_ai = "--no-ai" not in args
    videos = [Path(arg) for arg in args if not arg.startswith("--")]
    missing = [str(video) for video in videos if not video.is_file()]
    if not videos or missing:
        raise SystemExit(__doc__ + (f"\nNot found: {', '.join(missing)}" if missing else ""))

    results = []
    for video in videos:
        for run in range(1, runs + 1):
            result = _run_once(video.resolve(), use_ai)
            result["run"] = run
            results.append(result)
            stages = ", ".join(f"{name} {seconds:.1f}s" for name, seconds in result["stages"].items())
            print(f"{video.name} run {run}: {result['total_sec']:.1f}s ({stages})")
            for line in result["ai_logs"]:
                print(f"    {line}")
        totals = [item["total_sec"] for item in results if item["video"] == str(video.resolve())]
        print(f"{video.name}: median {statistics.median(totals):.1f}s over {len(totals)} run(s)")

    path = ROOT / "data" / "pipeline_benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ts": time.time(), "results": results}, indent=2), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
