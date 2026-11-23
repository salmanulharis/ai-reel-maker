from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from pathlib import Path

from flask import Flask

from app.ai.ollama import OllamaError
from app.analysis.director import CONTENT_FILE, DIRECTOR_FILE
from app.analysis.pipeline import PARTIAL_FILES, SOUND_EVENTS_FILE, SOUND_VECTORS_FILE, PipelineProgress, VideoAnalyzer
from app.analysis.progress import RenderProgress
from app.analysis.tighten import tighten_clips
from app.config import get_settings
from app.services.ollama_runtime import get_ollama_client
from app.models.schemas import ClipPlan, ClipPlanItem, HighlightConfig, RenderSettings, VideoMetadata
from app.services.hardware import preferred_hw_encoder
from app.services.run_plan import RunPlan, build_run_plan
from app.services.job_manager import get_job_manager
from app.video.crop import normalize_crop, to_crop_pixels
from app.video.ffmpeg_runner import CancelledError, FFmpegRunner, report_progress
from app.video.processor import VideoProcessor
from app.video.captions import load_transcript, sequence_srt, write_clip_subtitles
from app.video.renderer import Renderer
from app.video.audio_mix import find_music, music_mix_command
from app.video.overlays import Extras, brand_logo
from app.utils.atomic import atomic_write_json
from app.utils.timefmt import format_elapsed
from app.video.sources import fallback_source, load_sources, save_sources

_worker_started = False
_lock = threading.Lock()
_log = logging.getLogger(__name__)

# How often a running job records that it is still running (see JobManager.touch_attempt).
HEARTBEAT_SECONDS = 15.0
# How long a job interrupted by a crash or a power cut waits for Ollama to start again.
OLLAMA_RESUME_WAIT_SECONDS = 180.0
OLLAMA_RESUME_POLL_SECONDS = 10.0


def start_worker(app: Flask) -> None:
    global _worker_started
    with _lock:
        if _worker_started:
            return
        _worker_started = True
        _recover_orphaned_jobs()
        thread = threading.Thread(target=_loop, args=(app,), daemon=True, name="highlight-worker")
        thread.start()


def _recover_orphaned_jobs() -> None:
    from sqlalchemy import and_, or_

    from app.models.db_models import Job, get_session

    session = get_session()
    try:
        # "planning" with a running analysis is the last analysis step, not the review page.
        stuck = (
            session.query(Job)
            .filter(
                or_(
                    Job.status.in_(["analyzing", "rendering"]),
                    and_(Job.status == "planning", Job.analysis_status == "running"),
                ),
                Job.cancelled.is_(False),
            )
            .all()
        )
        for job in stuck:
            if job.status in {"analyzing", "planning"}:
                job.status = "queued"
                job.analysis_status = "pending"
            elif job.status == "rendering":
                job.status = "queued"
                job.render_status = "queued"
        # Paused while it was still stopping: it stays paused until it is resumed.
        paused = session.query(Job).filter(Job.status == "paused").all()
        for job in paused:
            if job.analysis_status == "running":
                job.analysis_status = "pending"
            if job.render_status == "running":
                job.render_status = "paused"
        if stuck or paused:
            session.commit()
        if stuck:
            _log.warning("Re-queued %s job(s) left mid-run after a restart", len(stuck))
        get_job_manager().recover_interrupted()
    except Exception:
        session.rollback()
        _log.exception("Failed to recover orphaned jobs")
    finally:
        session.close()


def _loop(app: Flask) -> None:
    settings = get_settings()
    while True:
        try:
            with app.app_context():
                _tick(app)
        except Exception:
            _log.exception("Highlight worker tick failed")
        time.sleep(settings.worker_idle_sleep)


def _tick(app: Flask) -> None:
    from app.models.db_models import Job, get_session

    session = get_session()
    try:
        job = (
            session.query(Job)
            .filter(Job.status == "queued", Job.cancelled.is_(False))
            .order_by(Job.created_at.asc())
            .first()
        )
        job_id = job.id if job else None
        render_only = bool(job and job.analysis_status == "completed" and job.render_status == "queued")
    finally:
        session.close()
    if not job_id:
        return
    stop_heartbeat = threading.Event()
    threading.Thread(
        target=_heartbeat, args=(job_id, stop_heartbeat), daemon=True, name=f"heartbeat-{job_id}"
    ).start()
    try:
        if render_only:
            _run_render(app, job_id)
        else:
            _run_analysis(app, job_id)
    finally:
        stop_heartbeat.set()
        # Never leave a timer running: an attempt that ended without being closed (an error
        # before the attempt's own handling) is closed here as failed.
        get_job_manager().finish_attempt(job_id, "failed", only_if_running=True)


def _heartbeat(job_id: str, stop: threading.Event) -> None:
    """Record now and then that the job is still running, so a run cut off by a crash or a
    power cut still counts the time it ran."""
    manager = get_job_manager()
    while not stop.wait(HEARTBEAT_SECONDS):
        try:
            manager.touch_attempt(job_id)
        except Exception:  # noqa: BLE001 - timing only; never disturbs the job
            _log.debug("Could not record that job %s is running", job_id, exc_info=True)


def _start_run(manager, job_id: str) -> threading.Event:
    """The stop signal for a new run. A cancel of an earlier run must not stop this one, but a
    pause asked for while the job was being picked up does."""
    cancel = manager.cancel_event(job_id)
    cancel.clear()
    if manager.pause_requested(job_id):
        cancel.set()
    return cancel


def _wait_for_ollama(client, health: dict, cancel: threading.Event, progress: PipelineProgress) -> dict:
    """After a crash or a power cut the app can start before Ollama. A resumed job waits a
    little for it instead of quietly using quick scoring for everything. Returns the newest
    health report; a pause or stop ends the wait (the pipeline then stops at its first check)."""
    progress.log("Waiting for Ollama to start: this job was interrupted and is continuing…")
    deadline = time.monotonic() + OLLAMA_RESUME_WAIT_SECONDS
    while time.monotonic() < deadline:
        if cancel.wait(OLLAMA_RESUME_POLL_SECONDS):
            return health
        try:
            health = client.health()
        except Exception:  # noqa: BLE001 - still starting
            continue
        if health.get("connected"):
            progress.log("Ollama is ready")
            return health
    progress.log(
        f"Ollama did not start within {format_elapsed(OLLAMA_RESUME_WAIT_SECONDS)}; continuing without it. "
        "Scenes scored without AI are analyzed again on a later run.",
        "WARNING",
    )
    return health


def _emit(app: Flask, job_id: str, event: str, payload: dict) -> None:
    from app import socketio

    data = {"job_id": job_id, **payload}
    socketio.emit(event, data, namespace="/")


def _run_analysis(app: Flask, job_id: str) -> None:
    settings = get_settings()
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return
    # Cut off by a crash or a power cut last time (see JobManager.recover_interrupted).
    resumed = job.attempt_outcome == "interrupted"
    # Asked on the Review page to plan again with the AI editor (nothing is analyzed again).
    refining = job.analysis_status == "refining"
    manager.update(job_id, status="analyzing", analysis_status="running", error=None)
    manager.start_attempt(job_id, "analysis")
    cancel = _start_run(manager, job_id)
    runner = FFmpegRunner(cancel)
    job_dir = manager.job_dir(job_id)
    original = Path(job.input_path)

    def emit(kind: str, payload: dict) -> None:
        if kind == "log":
            append_log(job_dir, payload)
        if kind == "progress":
            stage = payload.get("stage") or ""
            status = payload.get("status") or ""
            manager.record_progress(
                job_id,
                payload.get("overall", 0),
                payload.get("eta"),
                stage,
                status,
                payload.get("detail", ""),
                payload.get("step_done", 0),
                payload.get("step_total", 0),
                payload.get("step_unit", ""),
                payload.get("source_index"),
                payload.get("source_count"),
            )
            if status == "running" and stage == "planning" and not cancel.is_set():
                manager.update(job_id, status="planning")
        _emit(app, job_id, f"job_{kind}", payload)

    progress = PipelineProgress(emit, speeds=_measured_speeds(settings))
    ollama = None
    vision_enabled = False
    text_enabled = False
    try:
        client = get_ollama_client(settings)
        health = client.health()
        if not health.get("connected") and resumed:
            health = _wait_for_ollama(client, health, cancel, progress)
        if health.get("connected"):
            ollama = client
            vision_enabled = bool(health.get("vision_ready", health.get("vision_present")))
            text_enabled = bool(health.get("text_present"))
            for warning in health.get("warnings") or []:
                progress.log(str(warning), "WARNING")
            if not text_enabled:
                progress.log(
                    f"Text model '{settings.ollama_text_model}' is not installed. Transcript scoring will use heuristics.",
                    "WARNING",
                )
        else:
            progress.log(
                "Ollama is unavailable. You can still run deterministic analysis using scene detection, transcript, audio analysis, and repetition detection.",
                "WARNING",
            )
            progress.log("AI visual understanding unavailable", "WARNING")
    except Exception:
        progress.log("Ollama is unavailable. Continuing in degraded mode.", "WARNING")

    analyzer = VideoAnalyzer(
        settings,
        runner,
        ollama,
        vision_enabled=vision_enabled,
        text_enabled=text_enabled,
        run_plan=_run_plan(settings, ollama, progress),
    )
    manager.register_runner(job_id, runner)
    try:
        config = HighlightConfig.model_validate(json.loads(job.config_json or "{}"))
        manager.prepare_stages(job_id, config.mode)
        cache = load_cache(job_dir)
        if refining:
            sources = load_sources(job_dir) or [fallback_source(original, job.original_filename)]
            cache = analyzer.refine(job_dir, config, progress, cache)
        else:
            sources, cache = _prepare_all_sources(manager, job, original, job_dir, config, runner, progress, cache)
            cache = analyzer.analyze_sources(job_dir, sources, config, progress, cache)
        if cancel.is_set():
            # Cancelled during the last steps, which have no cancel point of their own.
            raise CancelledError("Job cancelled")
        manager.take_pause_request(job_id)
        plan = cache.get("clip_plan") or {}
        save_sources(job_dir, cache.get("sources") or sources)
        manager.save_plan(job_id, plan.get("clips") or [])
        fields = {
            "status": "planning",
            "analysis_status": "completed",
            "progress": 100,
            "error": None,
        }
        if _wants_auto_render(job_id) and not refining:
            # Watch folder jobs, and jobs started with "create the video right away", go
            # straight on to creating the video. A change asked for on Review is reviewed first.
            fields.update(status="queued", render_status="queued")
        if cache.get("metadata") and not refining:
            fields["metadata_json"] = json.dumps(cache["metadata"])
        manager.update(job_id, **fields)
        took = manager.finish_attempt(job_id, "completed")
        progress.log(f"Analysis finished in {format_elapsed(took)}")
        try:
            from app.services.auto_tune import record_run

            if not refining:
                record_run(progress, cache, settings)
        except Exception:  # noqa: BLE001 - statistics must never fail a finished analysis
            _log.debug("Could not record performance history", exc_info=True)
        _emit(app, job_id, "job_complete", {"phase": "analysis", "plan": plan})
    except CancelledError:
        if manager.take_pause_request(job_id):
            manager.update(job_id, status="paused", analysis_status="pending")
            took = manager.finish_attempt(job_id, "paused")
            progress.log(f"Analysis paused after {format_elapsed(took)}. Finished work is kept; resume to continue.")
            _emit(app, job_id, "job_paused", {"phase": "analysis"})
        else:
            manager.update(job_id, status="cancelled", cancelled=True)
            took = manager.finish_attempt(job_id, "cancelled")
            progress.log(f"Analysis stopped after {format_elapsed(took)}", "WARNING")
            _emit(app, job_id, "job_failed", {"error": "cancelled"})
    except Exception as exc:  # noqa: BLE001
        manager.update(job_id, status="failed", error=str(exc), analysis_status="failed")
        took = manager.finish_attempt(job_id, "failed")
        progress.log(str(exc), "ERROR")
        progress.log(f"Analysis failed after {format_elapsed(took)}", "WARNING")
        _emit(app, job_id, "job_failed", {"error": str(exc)})
    finally:
        manager.unregister_runner(job_id, runner)


def _run_render(app: Flask, job_id: str) -> None:
    settings = get_settings()
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return
    manager.update(job_id, status="rendering", render_status="running")
    manager.start_attempt(job_id, "render")
    cancel = _start_run(manager, job_id)
    runner = FFmpegRunner(cancel)
    manager.register_runner(job_id, runner)
    job_dir = manager.job_dir(job_id)

    def emit_log(message: str, level: str = "INFO") -> None:
        payload = {"level": level, "message": message, "ts": time.time()}
        append_log(job_dir, payload)
        _emit(app, job_id, "job_log", payload)

    try:
        sources = load_sources(job_dir) or [fallback_source(job.input_path, job.original_filename)]
        source = Path(sources[0].get("analysis_path") or sources[0]["path"])
        source_lookup = {str(item["id"]): Path(item.get("analysis_path") or item["path"]) for item in sources}
        config = HighlightConfig.model_validate(json.loads(job.config_json or "{}"))
        plan_path = job_dir / "clip_plan.json"
        if not plan_path.exists():
            raise FileNotFoundError("No clip plan to render")
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        # The saved plan keeps clips removed on the review page (so they can be restored).
        plan_data["clips"] = [c for c in plan_data.get("clips") or [] if c.get("kept", True) is not False]
        plan = ClipPlan.model_validate(plan_data)
        kept = [c for c in (manager.serialize(job).get("clips") or []) if c.get("kept", True)]
        skipped_clips = 0
        if kept:
            # The clips exactly as kept on the Review page: the planner already kept each one
            # within the longest-clip setting, so anything longer was made longer on purpose.
            plan.clips, skipped_clips = plan_items_from_clips(kept, source_lookup, source)
        plan.estimated_duration = sum(c.duration for c in plan.clips)

        render_settings = RenderSettings(
            format=config.output_format,
            quality=config.quality,
            codec=config.codec,
            crf=settings.output_crf,
            resolution=config.resolution,
            aspect_ratio=config.aspect_ratio,
            subtitles=config.subtitles,
            transition=config.transition,
            reframe=config.reframe,
            level_audio=config.level_audio,
            frame_rate=config.frame_rate,
            smooth_motion=config.smooth_motion,
            audio_bitrate=settings.output_audio_bitrate,
        )
        outputs = job_dir / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        ext = render_settings.format if render_settings.format in {"mp4", "webm"} else "mp4"
        # Each video has its own transcript (multi-video jobs keep one per video under work/).
        multi_source = len(sources) > 1
        first_source_id = str(sources[0].get("id") or "v0")
        transcripts: dict[str, dict] = {}

        def transcript_for(clip: ClipPlanItem) -> dict:
            source_id = clip.source_id if clip.source_id in source_lookup else first_source_id
            if source_id not in transcripts:
                transcripts[source_id] = load_transcript(job_dir, source_id, multi_source)
            return transcripts[source_id]

        tighten_note = ""
        if config.tighten != "off":
            clip_count, length_before = len(plan.clips), plan.estimated_duration
            plan.clips = tighten_clips(plan.clips, transcript_for, remove_fillers=config.tighten == "fillers")
            plan.estimated_duration = sum(c.duration for c in plan.clips)
            cut = length_before - plan.estimated_duration
            what = "pauses and filler words" if config.tighten == "fillers" else "pauses"
            tighten_note = (
                f"Tightened {clip_count} clips: {cut:.1f}s of {what} cut out"
                if cut > 0.05
                else f"Tighten clips: no {what} to cut (or no word timings)"
            )

        groups = render_groups(plan, config, outputs, ext)
        extras = render_extras(config, settings)
        music = find_music(job_dir) if config.music_volume > 0 else None

        # Subtitles in another language: the spoken lines of the kept clips, translated once.
        subtitle_note = ""
        subtitle_transcripts: dict[str, dict] = {}

        def subtitle_transcript_for(clip: ClipPlanItem) -> dict:
            nonlocal subtitle_note
            original = transcript_for(clip)
            language = config.subtitle_language
            if not language or config.subtitles == "none":
                return original
            source_id = clip.source_id if clip.source_id in source_lookup else first_source_id
            if source_id not in subtitle_transcripts:
                ranges = [(c.start, c.end) for c in plan.clips if (c.source_id or first_source_id) == source_id]
                try:
                    from app.services.ollama_runtime import get_ollama_client
                    from app.services.translate import translated_transcript

                    result, done, needed = translated_transcript(
                        original,
                        ranges,
                        language,
                        get_ollama_client(settings),
                        settings.ollama_text_model,
                        job_dir / "translations" / f"{source_id}-{language.lower()}.json",
                    )
                    subtitle_note = f"Subtitles translated into {language}: {done} of {needed} lines"
                    subtitle_transcripts[source_id] = result
                except Exception as exc:  # noqa: BLE001 - original subtitles are better than none
                    subtitle_note = f"Subtitles could not be translated ({exc}); the original words are used"
                    subtitle_transcripts[source_id] = original
            return subtitle_transcripts[source_id]

        meta = VideoMetadata.model_validate(json.loads(job.metadata_json or "{}"))
        renderer = Renderer(runner)

        emit_log("Rendering highlight" if len(groups) == 1 else f"Rendering {len(groups)} shorts")
        if skipped_clips:
            emit_log(
                f"Left out {skipped_clips} clip(s) whose end is not after the start. Check their times on the Review page.",
                "WARNING",
            )
        if tighten_note:
            emit_log(tighten_note)
        encoder = None
        if render_settings.quality not in {"source", "very_high"}:
            encoder = preferred_hw_encoder()
        if encoder:
            emit_log(f"Using hardware encoder {encoder}")
        else:
            emit_log("Using high-quality CPU encoder libx264 to preserve source detail")
        if not groups:
            raise ValueError("No clips to render")

        def emit_progress(payload: dict) -> None:
            manager.record_progress(
                job_id,
                payload["overall"],
                payload.get("eta"),
                "rendering",
                "running",
                payload.get("detail", ""),
                payload.get("step_done", 0),
                payload.get("step_total", 0),
                payload.get("step_unit", ""),
            )
            _emit(app, job_id, "job_progress", payload)

        tracker = RenderProgress(
            [[max(0.0, c.end - c.start) for c in group["clips"]] for group in groups], music is not None, emit_progress
        )
        clip_total = tracker.clip_total
        manager.record_progress(job_id, 0, None, "rendering", "running", "", 0, clip_total, "clips")
        made: list[dict] = []
        for number, group in enumerate(groups, 1):
            prefix = "" if len(groups) == 1 else f"Short {group['short'] + 1}: "
            if len(groups) > 1:
                emit_log(f"Creating short {group['short'] + 1} ({number} of {len(groups)})")
                tracker.detail(f"Short {group['short'] + 1} ({number} of {len(groups)})")
            work_dir = job_dir / "tmp_render" / group["label"]
            group_plan = plan.model_copy(
                update={"clips": group["clips"], "estimated_duration": sum(c.duration for c in group["clips"])}
            )
            if config.subtitles in {"auto", "burned", "srt"}:
                group["srt"].write_text(
                    sequence_srt([(subtitle_transcript_for(c), c.start, c.end) for c in group["clips"]]),
                    encoding="utf-8",
                )
            result = renderer.render(
                source,
                group_plan,
                group["output"],
                render_settings,
                work_dir,
                metadata=meta,
                clip_subtitles=_clip_subtitle_writer(work_dir / "subtitles", subtitle_transcript_for, config)
                if config.subtitles == "burned"
                else None,
                hardware_encoder=encoder,
                source_lookup=source_lookup,
                extras=extras,
                on_clip=lambda index, share, g=number - 1: tracker.clip(g, index, share),
            )
            out_meta = result["metadata"]
            if number == 1 and subtitle_note:
                emit_log(subtitle_note)
            if result.get("leveled"):
                emit_log(f"{prefix}Sound levelled in {result['leveled']['clips']} of {result['leveled']['of']} clips")
            if music is not None:
                out_meta = add_music(
                    runner,
                    group["output"],
                    music,
                    out_meta,
                    config,
                    settings,
                    emit_log,
                    prefix,
                    on_progress=lambda share, g=number - 1: tracker.music(g, min(share, 0.99)),
                )
                tracker.music(number - 1, 1.0)
            canvas = result.get("canvas") or {}
            encoder_info = result.get("encoder") or {}
            if result.get("frame_rate") and number == 1:
                emit_log(str(result["frame_rate"]))
            if result.get("reframe"):
                followed = result["reframe"]
                emit_log(
                    f"{prefix}Framing followed faces in {followed['followed']} of {followed['clips']} clips"
                    " (the others keep the middle of the picture)"
                )
            emit_log(
                f"{prefix}Target output "
                f"{canvas.get('width') or out_meta.get('width')}×{canvas.get('height') or out_meta.get('height')} "
                f"@ {float(canvas.get('fps') or out_meta.get('fps') or 0):.3f} fps, "
                f"{encoder_info.get('codec') or out_meta.get('video_codec')} "
                f"{encoder_info.get('preset') or ''} CRF {encoder_info.get('crf') or ''}".strip()
            )
            emit_log(
                f"{prefix}Verified output "
                f"{out_meta.get('width')}×{out_meta.get('height')}, "
                f"{float(out_meta.get('avg_fps') or out_meta.get('fps') or 0):.3f} fps, "
                f"{out_meta.get('video_codec') or 'h264'}, "
                f"{int(out_meta.get('bitrate') or 0) or out_meta.get('size_bytes')} bitrate/size"
            )
            made.append(
                {
                    "path": str(group["output"]),
                    "kind": group["kind"],
                    "duration": out_meta.get("duration") or 0,
                    "size_bytes": out_meta.get("size_bytes") or group["output"].stat().st_size,
                    "codec": out_meta.get("video_codec") or "",
                    "width": out_meta.get("width") or 0,
                    "height": out_meta.get("height") or 0,
                    "version_label": config.style if group["kind"] == "highlight" else group["label"],
                }
            )
        # Only now that every video is made: replace the earlier render's outputs.
        keep = {Path(item["path"]).name for item in made}
        if config.subtitles in {"auto", "burned", "srt"}:
            keep |= {group["srt"].name for group in groups}
        remove_stale_outputs(outputs, keep)
        manager.clear_outputs(job_id)
        for item in made:
            manager.add_output(job_id, **item)
        if config.auto_render:
            try:
                from app.services.watch_folder import copy_outputs

                copied = copy_outputs(job_dir, [Path(item["path"]) for item in made])
                if copied:
                    emit_log(f"Copied to {copied[0].parent}: " + ", ".join(p.name for p in copied))
            except Exception as exc:  # noqa: BLE001 - the videos are still in the app
                emit_log(f"Could not copy the videos to the watched folder: {exc}", "WARNING")
        manager.set_stage(job_id, "rendering", "done", "", clip_total, clip_total, "clips")
        manager.take_pause_request(job_id)
        took = manager.finish_attempt(job_id, "completed")
        manager.update(job_id, status="completed", render_status="completed", progress=100)
        what = "Highlight" if len(groups) == 1 else f"{len(groups)} shorts"
        emit_log(f"{what} ready. Creating the video took {format_elapsed(took)}")
        _emit(app, job_id, "job_complete", {"phase": "render", "output": made[0]["path"]})
        if not settings.debug_analysis:
            tmp = job_dir / "tmp_render"
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
    except CancelledError:
        if manager.take_pause_request(job_id):
            manager.update(job_id, status="paused", render_status="paused")
            took = manager.finish_attempt(job_id, "paused")
            emit_log(f"Creating the video paused after {format_elapsed(took)}. Finished clips are kept; resume to continue.")
            _emit(app, job_id, "job_paused", {"phase": "render"})
        else:
            manager.update(job_id, status="cancelled", cancelled=True, render_status="cancelled")
            took = manager.finish_attempt(job_id, "cancelled")
            emit_log(f"Creating the video stopped after {format_elapsed(took)}", "WARNING")
            _emit(app, job_id, "job_failed", {"error": "cancelled"})
    except Exception as exc:  # noqa: BLE001
        manager.update(job_id, status="failed", error=str(exc), render_status="failed")
        manager.set_stage(job_id, "rendering", "error", str(exc))
        took = manager.finish_attempt(job_id, "failed")
        emit_log(str(exc), "ERROR")
        emit_log(f"Creating the video failed after {format_elapsed(took)}", "WARNING")
        _emit(app, job_id, "job_failed", {"error": str(exc)})
    finally:
        manager.unregister_runner(job_id, runner)


def _wants_auto_render(job_id: str) -> bool:
    job = get_job_manager().get(job_id)
    try:
        values = json.loads(job.config_json or "{}") if job else {}
        return bool(values.get("auto_render") or values.get("render_after_analysis"))
    except json.JSONDecodeError:
        return False


def render_extras(config: HighlightConfig, settings) -> Extras:
    """Picture extras for a render: punch-in, progress bar and logo (when one is saved)."""
    logo = brand_logo(settings.data_dir) if config.logo_position != "off" else None
    return Extras(
        punch_in=config.punch_in,
        logo=logo,
        logo_position=config.logo_position,
        logo_size=config.logo_size / 100,
        progress_bar=config.progress_bar,
    )


def add_music(
    runner,
    output: Path,
    music: Path,
    out_meta: dict,
    config,
    settings,
    emit_log,
    prefix: str = "",
    on_progress=None,
) -> dict:
    """Mix the background music into a finished video. On failure the video keeps its own sound.
    ``on_progress`` receives the share of the video mixed so far."""
    mixed = output.with_name(f"{output.stem}.music{output.suffix}")
    duration = float(out_meta.get("duration") or 0)
    try:
        with report_progress(runner, duration, on_progress):
            runner.run(
                music_mix_command(
                    output,
                    music,
                    mixed,
                    duration,
                    config.music_volume / 100,
                    config.music_duck,
                    bool(out_meta.get("has_audio", True)),
                    settings.output_audio_bitrate,
                )
            )
        mixed.replace(output)
        meta = runner.probe(output).model_dump()
        emit_log(f"{prefix}Background music added ({music.name}{', lowered under speech' if config.music_duck else ''})")
        return {**out_meta, **{k: v for k, v in meta.items() if v}}
    except CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        mixed.unlink(missing_ok=True)
        emit_log(f"{prefix}Background music could not be added: {exc}", "WARNING")
        return out_meta


def plan_items_from_clips(clips: list[dict], source_lookup: dict, source: Path) -> tuple[list[ClipPlanItem], int]:
    """
    Plan items for the kept clips, with their times exactly as saved (manual changes on the
    Review page are kept, also when a clip is longer than the longest-clip setting). Clips whose
    times are missing or whose end is not after the start are left out; returns how many.
    """
    items: list[ClipPlanItem] = []
    skipped = 0
    for c in clips:
        try:
            start, end = float(c["start"]), float(c["end"])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        if not (end > start >= 0):
            skipped += 1
            continue
        items.append(
            ClipPlanItem(
                start=start,
                end=end,
                reason=c.get("reason") or "Selected moment",
                score=c.get("score") or 0,
                transcript=c.get("transcript") or "",
                duration_reason=c.get("duration_reason") or "",
                source_id=c.get("source_id") or "",
                source_name=c.get("source_name") or "",
                source_path=c.get("source_path") or str(source_lookup.get(c.get("source_id") or "") or source),
                short_index=int(c.get("short_index") or 0),
            )
        )
    return items, skipped


def render_groups(plan: ClipPlan, config: HighlightConfig, outputs: Path, ext: str) -> list[dict]:
    """
    What to render: one highlight, or one video per short. A short keeps its number from the
    plan (so "Short 3" stays short_03 even when every clip of short 2 was removed).
    """
    clips = [c for c in plan.clips if c.end > c.start]
    if int(config.shorts_count or 0) < 2 and not any(c.short_index for c in clips):
        if not clips:
            return []
        return [
            {
                "label": "highlight",
                "kind": "highlight",
                "short": 0,
                "clips": clips,
                "output": outputs / f"highlight.{ext}",
                "srt": outputs / "highlight.srt",
            }
        ]
    groups = []
    for index in sorted({c.short_index for c in clips}):
        name = f"short_{index + 1:02d}"
        groups.append(
            {
                "label": name,
                "kind": "short",
                "short": index,
                "clips": [c for c in clips if c.short_index == index],
                "output": outputs / f"{name}.{ext}",
                "srt": outputs / f"{name}.srt",
            }
        )
    return groups


def remove_stale_outputs(outputs: Path, keep: set[str]) -> None:
    """Delete videos and subtitles of an earlier render that this render did not replace."""
    for path in list(outputs.glob("highlight.*")) + list(outputs.glob("short_*.*")):
        if path.is_file() and path.name not in keep:
            path.unlink(missing_ok=True)


def _clip_subtitle_writer(subtitle_dir: Path, transcript_for, config: HighlightConfig):
    """Writes the subtitle file burned into one clip, timed from that clip's start."""

    def write(index: int, clip: ClipPlanItem, canvas) -> Path | None:
        return write_clip_subtitles(
            subtitle_dir,
            index,
            transcript_for(clip),
            clip.start,
            clip.end,
            config.caption_style,
            canvas.width,
            canvas.height,
        )

    return write

ANALYSIS_VIDEO = "analysis.mp4"
_CACHE_FILES = (
    *PARTIAL_FILES,
    SOUND_EVENTS_FILE,
    SOUND_VECTORS_FILE,
    CONTENT_FILE,
    DIRECTOR_FILE,
    "metadata.json",
    "scenes.json",
    "transcript.json",
    "silences.json",
    "frames.json",
    "visual_analysis.json",
    "visual_connection_focus.json",
    "segment_scores.json",
    "scene_quality.json",
    "audio_energy.json",
    "image_match.json",
    "candidates_all.json",
    "candidates.json",
    "clip_plan.json",
)


def _analysis_video(original: Path) -> Path:
    cropped = original.parent / ANALYSIS_VIDEO
    if cropped.is_file() and cropped.stat().st_size > 1000:
        return cropped
    return original


def _clear_analysis_cache(job_dir: Path) -> None:
    for name in _CACHE_FILES:
        path = job_dir / name
        if path.exists():
            path.unlink()
    frames = job_dir / "frames"
    if frames.exists():
        shutil.rmtree(frames, ignore_errors=True)


def _crop_one(
    original: Path,
    cropped: Path,
    stamp: Path,
    crop_payload: dict | None,
    runner,
    progress,
    on_progress=None,
) -> tuple[Path, bool]:
    if not normalize_crop(crop_payload) and not cropped.exists() and not stamp.exists():
        return original, False
    source_meta = runner.probe(original)
    box = to_crop_pixels(int(source_meta.width or 0), int(source_meta.height or 0), crop_payload)
    if box is None:
        changed = cropped.exists() or stamp.exists()
        if cropped.exists():
            cropped.unlink()
        if stamp.exists():
            stamp.unlink()
        return original, changed
    payload = {"x": box.x, "y": box.y, "width": box.width, "height": box.height}
    if cropped.is_file() and cropped.stat().st_size > 1000 and stamp.exists():
        try:
            previous = json.loads(stamp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = None
        if previous == payload:
            progress.log(
                f"Using cropped analysis video {box.width}×{box.height} for {original.name}. Original file is unchanged."
            )
            return cropped, False
    progress.log(
        f"Cropping {original.name} to {box.width}×{box.height} at ({box.x},{box.y}) for analysis. Original file is unchanged."
    )
    settings = get_settings()
    # The stamp describes a finished crop, so it goes first: an interrupted or failed crop must
    # never leave a half-written file that looks like an earlier finished one.
    if stamp.exists():
        stamp.unlink()
    if on_progress is not None:
        on_progress(0.0)
    with report_progress(runner, float(source_meta.duration or 0), on_progress):
        runner.run(
            VideoProcessor.crop_command(
                original,
                cropped,
                width=box.width,
                height=box.height,
                x=box.x,
                y=box.y,
                crf=settings.output_crf,
            )
        )
    atomic_write_json(stamp, payload)
    cropped_meta = runner.probe(cropped)
    progress.log(f"Analysis will use the cropped {cropped_meta.width}×{cropped_meta.height} video for {original.name}.")
    return cropped, True


def _prepare_all_sources(manager, job, original: Path, job_dir: Path, config, runner, progress, cache: dict):
    sources = load_sources(job_dir)
    if not sources:
        sources = [fallback_source(original, job.original_filename)]
    crop_payload = config.crop.model_dump() if config.crop else None
    busted = False
    single = len(sources) == 1
    cropped_any = {"ran": False}
    for position, src in enumerate(sources):

        def crop_share(fraction: float, position: int = position, name: str = str(src.get("filename") or "")) -> None:
            cropped_any["ran"] = True
            progress.step("prepare", (position + fraction) / len(sources) * 100, 100, "", name)

        src_path = Path(src["path"])
        sid = str(src.get("id") or "v0")
        if single:
            dest = src_path.parent / ANALYSIS_VIDEO
            stamp = src_path.parent / "crop.json"
        else:
            dest = src_path.parent / f"{sid}_analysis.mp4"
            stamp = src_path.parent / f"{sid}_crop.json"
        analysis, changed = _crop_one(src_path, dest, stamp, crop_payload, runner, progress, crop_share)
        src["analysis_path"] = str(analysis)
        busted = busted or changed
    if cropped_any["ran"]:
        progress.stage("prepare", "done")
    save_sources(job_dir, sources)
    if busted:
        _clear_analysis_cache(job_dir)
        work = job_dir / "work"
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        cache = {}
        first = Path(sources[0].get("analysis_path") or sources[0]["path"])
        if first.is_file():
            try:
                manager.update(job.id, metadata_json=runner.probe(first).model_dump_json())
            except Exception:
                pass
    return sources, cache


def _measured_speeds(settings) -> dict[str, float]:
    """Measured model speeds for better time estimates; empty when none are known."""
    try:
        from app.services.auto_tune import read_history
        from app.services.calibration import current_speeds

        return current_speeds(settings, read_history())
    except Exception:  # noqa: BLE001 - estimates fall back to typical speeds
        _log.debug("Could not read measured speeds", exc_info=True)
        return {}


def _run_plan(settings, ollama, progress: PipelineProgress) -> RunPlan:
    """Memory decisions for this analysis, from the memory and load right now. Advisory: any
    problem working them out leaves the defaults."""
    try:
        from app.services.ai_profile import cached_hardware

        hardware = cached_hardware()
        device = str(settings.whisper_device or "auto")
        whisper_on_gpu = bool(settings.enable_whisper) and (
            device == "cuda" or (device == "auto" and bool(hardware.get("whisper_cuda")))
        )
        plan = build_run_plan(
            settings,
            hardware,
            ollama_here=ollama is not None and ollama.shares_cpu(),
            whisper_on_gpu=whisper_on_gpu,
        )
    except Exception:  # noqa: BLE001
        _log.debug("Could not work out a run plan", exc_info=True)
        return RunPlan()
    for note in plan.notes:
        progress.log(note)
    return plan


def load_cache(job_dir: Path) -> dict:
    cache = {}
    mapping = {
        "metadata": "metadata.json",
        "scenes": "scenes.json",
        "transcript": "transcript.json",
        "silences": "silences.json",
        "frames": "frames.json",
        "visual": "visual_analysis.json",
        "visual_connection_focus": "visual_connection_focus.json",
        "segment_scores": "segment_scores.json",
        "scene_quality": "scene_quality.json",
        "audio_energy": "audio_energy.json",
        "image_match": "image_match.json",
        "candidates_all": "candidates_all.json",
        "candidates": "candidates.json",
        "clip_plan": "clip_plan.json",
        "speakers": "speakers.json",
        "sound_events": SOUND_EVENTS_FILE,
        "content_index": CONTENT_FILE,
        "director_plan": DIRECTOR_FILE,
    }
    for key, name in mapping.items():
        path = job_dir / name
        if path.exists():
            try:
                cache[key] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    return cache


def load_transcripts(job_dir: Path, sources: list[dict]) -> dict[str, list[dict]]:
    """Transcript segments per source id. One video keeps its transcript in the job folder;
    with several, each video's analysis lives in ``work/<source id>/``."""
    multi = len(sources) > 1
    result: dict[str, list[dict]] = {}
    for index, src in enumerate(sources or [{}]):
        sid = str(src.get("id") or f"v{index}")
        path = (job_dir / "work" / sid if multi else job_dir) / "transcript.json"
        segments: list[dict] = []
        if path.exists():
            try:
                segments = list(json.loads(path.read_text(encoding="utf-8")).get("segments") or [])
            except (json.JSONDecodeError, AttributeError):
                segments = []
        result[sid] = segments
    return result


def append_log(job_dir: Path, payload: dict) -> None:
    path = job_dir / "job.log"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
