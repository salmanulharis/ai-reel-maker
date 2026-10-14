# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local-only app that turns a long video into a short highlight. Flask + Socket.IO backend (`app/`), React + Vite frontend (`frontend/`), FFmpeg for all media work, faster-whisper for speech, and Ollama (vision + text models) for scoring. No cloud AI: the only network calls go to the configured Ollama URL(s).

## Commands

```bash
pip install -r requirements.txt
python run.py                                  # API + built UI on APP_PORT (default 5000)
cd frontend && npm install && npm run build    # UI served from frontend/dist by Flask
cd frontend && npm run dev                     # UI on 5173, proxies /api and /socket.io to 5000
cd frontend && npx tsc --noEmit -p .           # frontend type check (there is no lint setup)

python -m pytest -q                            # all tests (pytest.ini sets testpaths and pythonpath)
python -m pytest tests/test_performance.py -q
python -m pytest tests/test_ai_reliability.py::test_predict_budget_grows_with_batch_and_respects_context

python scripts/check_system.py                 # FFmpeg / Whisper / Ollama readiness
python scripts/pipeline_benchmark.py VIDEO --runs 3   # per-stage timings + model token usage
```

Two tests fail on a clean checkout and are not regressions: `test_api.py::test_sample_video_endpoint` (needs `samples/sample.mp4`, create it with `scripts/generate_sample.py`) and `test_candidate_fallback.py::test_generates_scene_windows_when_transcript_is_empty`.

The user runs the app natively on Windows (CPU-only, ~16 GB RAM, Ollama installed on the host). Docker/Compose files exist but are not the user's current setup.

## Architecture

**Request → job → worker.** Routes (`app/routes/`) only create and inspect jobs. `JobManager` (`app/services/job_manager.py`) stores jobs in SQLite (`data/app.db`) and files under `data/jobs/<id>/`. A single daemon thread (`app/workers/worker.py`) polls for `queued` jobs and runs either analysis or render. Progress and logs go out over Socket.IO (`job_progress`, `job_log`, …) and are appended to `job.log`. Jobs left mid-run are re-queued at startup.

**Analysis pipeline** (`app/analysis/pipeline.py`, `VideoAnalyzer`): metadata → scenes (PySceneDetect) → audio → transcript (Whisper) → silence → frames (+ picture quality and loudness from `signals.py`, cached as `scene_quality.json` / `audio_energy.json`) → transcript AI → visual AI (only on `shortlist_scenes` unless `vision_scope=all`) → candidates (incl. sentence-complete joins, focus matching via `semantic.py`) → repetition removal (word-based, plus calibrated Ollama embeddings when `ollama_embed_model` is installed) → planning (`ClipOptimizer`). Transcript scoring must stay before visual analysis: it feeds the shortlist. `analyze_sources` handles one or many source videos. With several videos, each is analyzed in `job/work/<source id>/` and the candidates are merged before planning.

**Stage cache.** Each stage writes a JSON file in the job folder (`scenes.json`, `transcript.json`, `visual_analysis.json`, `segment_scores.json`, …) and `load_cache` feeds them back, so re-planning and retries skip finished stages. Changing the crop clears the cache. AI scores are tagged by `app/analysis/score_source.py`: fallback scores with a retryable reason are redone when the model is available again, and untagged entries from older versions are kept. Cached AI scores are **not** invalidated when the style, custom focus or model changes.

**Degraded modes are a feature.** Every AI step has a deterministic fallback: motion scoring for frames, heuristics for transcript segments. Ollama being down, a model missing, a reply cut off or malformed must never fail the job. Keep that property when touching `visual.py` / `pipeline.py`.

**Ollama client** (`app/ai/ollama.py`, shared instance from `services/ollama_runtime.get_ollama_client`):
- `role_options(role)` gives `num_ctx` (+ `num_thread` only when Ollama runs on this machine). Warm-up, probes and real calls must share it, because Ollama reloads a model when these options change.
- Batched JSON replies use `predict_budget`. A reply that hits the limit raises `OllamaTruncatedError`, and callers split the batch in half and retry.
- Batches are sized to fit the context window: `plan_text_batches` (transcript, token estimate) and `fitting_vision_batch` (vision; about 1,100 prompt tokens per image regardless of resolution, because Ollama upscales Qwen3-VL frames).
- Connection fallback: Docker host aliases, plus optional `OLLAMA_FALLBACK_URL`. The client returns to the configured URL when it answers again (checked at most once a minute). Optional `OLLAMA_API_KEY` is sent as a Bearer token and is filtered out of `GET /api/settings`.
- `usage_snapshot()` keeps per-thread token and time totals. The pipeline logs them per stage, and logs stage timings at the end.

**Model choice.** `app/services/model_catalog.py` recommends models from VRAM/RAM/CPU. `app/services/ai_profile.py` applies the recommendation in **auto** mode when the hardware fingerprint changes. Bump `RECOMMENDATION_VERSION` whenever recommendations change, so existing auto-mode installs pick them up once. Custom mode is never overwritten. `app/services/auto_tune.py` backs the Settings switch "Automatically optimize for my system": review of changes with reasons, undo snapshot (`auto_tune_previous`), and per-run speed history (`perf_history`) that only produces suggestions, never automatic changes. It manages exactly `ai_profile.APPLIED_FIELDS`; editing any of them turns auto mode off, and other settings are never touched.

**Settings precedence.** `.env` / environment → `Settings` (`app/config.py`). Values saved from the UI go into the SQLite `settings` table and are overlaid by `effective_settings()`; only keys in `EDITABLE` (`app/routes/settings.py`) can be saved. Editing a model-related key switches to custom mode.

**Video** (`app/video/`): `VideoProcessor` builds FFmpeg argv lists (never shell strings; paths go through `safe_path_argument`). `FFmpegRunner` executes them, can run several at once (parallel frame extraction), and `cancel()` kills all of them. `Renderer` cuts and concatenates the clip plan, with optional subtitles, 9:16 output and hardware encoders.

## Gotchas

- **Use `qwen3-vl:*-instruct` tags.** The bare `qwen3-vl:2b/4b/8b/30b/latest` tags are thinking models. They ignore `think: false`, spend the whole reply budget on reasoning, and return empty content, so every scene silently falls back to motion scoring. `health()` warns about this.
- `qwen3:4b` pretty-prints JSON (about 42–46 reply tokens per transcript segment), so `_TEXT_TOKENS_PER_SEGMENT` is 48. On CPU, reply tokens dominate model time (about 95% for transcript scoring). A compact-key / 0–10 format was tested and was **not** adopted: it was about 20% faster but slightly worse against a `qwen3:14b` reference.
- Tests stub `requests.Session.request` and use fake clients whose `chat()` takes the long-key reply format (`importance`, `visual_importance`, …). Keep reply parsing compatible with them. An autouse fixture in the Ollama tests stubs out the global `time.sleep`.
- Line endings: `core.autocrlf=true`. HEAD stores LF and working copies may be CRLF; don't mix both in one file.

See `docs/performance-and-gpu-options.md` for CPU performance measurements, external GPU / cloud API options and the planned backend integration.

## Working agreement

The user does UI/browser testing and real-video end-to-end testing themselves. Verify changes with unit tests, type checks and code review only; don't launch the app, browser sessions or full pipeline runs on videos.
