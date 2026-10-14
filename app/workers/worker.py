from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from pathlib import Path

from flask import Flask

from app.ai.ollama import OllamaError
from app.analysis.pipeline import PipelineProgress, VideoAnalyzer
from app.analysis.sequence_limits import clamp_plan_clips, sequence_cap
from app.config import get_settings
from app.services.ollama_runtime import get_ollama_client
from app.models.schemas import ClipPlan, ClipPlanItem, HighlightConfig, RenderSettings, VideoMetadata
from app.services.hardware import preferred_hw_encoder
from app.services.job_manager import get_job_manager
from app.video.crop import normalize_crop, to_crop_pixels
from app.video.ffmpeg_runner import CancelledError, FFmpegRunner
from app.video.processor import VideoProcessor
from app.video.renderer import Renderer, build_srt
from app.utils.timefmt import format_elapsed
from app.video.sources import fallback_source, load_sources, save_sources

_worker_started = False
_lock = threading.Lock()
_log = logging.getLogger(__name__)


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
    from app.models.db_models import Job, get_session

    session = get_session()
    try:
        stuck = (
            session.query(Job)
            .filter(Job.status.in_(["analyzing", "rendering"]), Job.cancelled.is_(False))
            .all()
        )
        for job in stuck:
            if job.status == "analyzing":
                job.status = "queued"
                job.analysis_status = "pending"
            elif job.status == "rendering":
                job.status = "queued"
                job.render_status = "queued"
        if stuck:
            session.commit()
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
    try:
        if render_only:
            _run_render(app, job_id)
        else:
            _run_analysis(app, job_id)
    finally:
        # Never leave a timer running: an attempt that ended without being closed (an error
        # before the attempt's own handling) is closed here as failed.
        get_job_manager().finish_attempt(job_id, "failed", only_if_running=True)


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
    manager.update(job_id, status="analyzing", analysis_status="running", error=None)
    manager.start_attempt(job_id, "analysis")
    cancel = manager.cancel_event(job_id)
    cancel.clear()
    runner = FFmpegRunner(cancel)
    job_dir = manager.job_dir(job_id)
    original = Path(job.input_path)
    config = HighlightConfig.model_validate(json.loads(job.config_json or "{}"))
    cache = load_cache(job_dir)

    def emit(kind: str, payload: dict) -> None:
        if kind == "log":
            append_log(job_dir, payload)
        if kind == "progress":
            manager.update(job_id, progress=payload.get("overall", 0), last_stage=payload.get("stage", ""))
            stage = payload.get("stage")
            status = payload.get("status")
            if stage and status:
                mapped = {"running": "running", "done": "done"}.get(status, status)
                manager.set_stage(job_id, stage, mapped, payload.get("detail", ""))
                if mapped == "running" and stage == "planning":
                    manager.update(job_id, status="planning")
        _emit(app, job_id, f"job_{kind}", payload)

    progress = PipelineProgress(emit)
    ollama = None
    vision_enabled = False
    text_enabled = False
    try:
        client = get_ollama_client(settings)
        health = client.health()
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

    analyzer = VideoAnalyzer(settings, runner, ollama, vision_enabled=vision_enabled, text_enabled=text_enabled)
    try:
        sources, cache = _prepare_all_sources(manager, job, original, job_dir, config, runner, progress, cache)
        cache = analyzer.analyze_sources(job_dir, sources, config, progress, cache)
        plan = cache.get("clip_plan") or {}
        save_sources(job_dir, cache.get("sources") or sources)
        manager.save_plan(job_id, plan.get("clips") or [])
        fields = {
            "status": "planning",
            "analysis_status": "completed",
            "progress": 90,
            "error": None,
        }
        if cache.get("metadata"):
            fields["metadata_json"] = json.dumps(cache["metadata"])
        manager.update(job_id, **fields)
        took = manager.finish_attempt(job_id, "completed")
        progress.log(f"Analysis finished in {format_elapsed(took)}")
        try:
            from app.services.auto_tune import record_run

            record_run(progress, cache, settings)
        except Exception:  # noqa: BLE001 - statistics must never fail a finished analysis
            _log.debug("Could not record performance history", exc_info=True)
        _emit(app, job_id, "job_complete", {"phase": "analysis", "plan": plan})
    except CancelledError:
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


def _run_render(app: Flask, job_id: str) -> None:
    settings = get_settings()
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return
    manager.update(job_id, status="rendering", render_status="running")
    manager.start_attempt(job_id, "render")
    cancel = manager.cancel_event(job_id)
    runner = FFmpegRunner(cancel)
    job_dir = manager.job_dir(job_id)
    sources = load_sources(job_dir) or [fallback_source(job.input_path, job.original_filename)]
    source = Path(sources[0].get("analysis_path") or sources[0]["path"])
    source_lookup = {str(item["id"]): Path(item.get("analysis_path") or item["path"]) for item in sources}
    config = HighlightConfig.model_validate(json.loads(job.config_json or "{}"))
    plan_path = job_dir / "clip_plan.json"
    if not plan_path.exists():
        manager.update(job_id, status="failed", error="No clip plan to render", render_status="failed")
        manager.finish_attempt(job_id, "failed")
        return
    plan = ClipPlan.model_validate(json.loads(plan_path.read_text(encoding="utf-8")))
    kept = [c for c in (manager.serialize(job).get("clips") or []) if c.get("kept", True)]
    if kept:
        plan.clips = [
            ClipPlanItem(
                start=c["start"],
                end=c["end"],
                reason=c.get("reason") or "Selected moment",
                score=c.get("score") or 0,
                transcript=c.get("transcript") or "",
                duration_reason=c.get("duration_reason") or "",
                source_id=c.get("source_id") or "",
                source_name=c.get("source_name") or "",
                source_path=c.get("source_path")
                or str(source_lookup.get(c.get("source_id") or "") or source),
            )
            for c in kept
        ]
        plan.estimated_duration = sum(c.duration for c in plan.clips)

    clamp_plan_clips(plan.clips, sequence_cap(config))
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
        frame_rate=config.frame_rate,
        smooth_motion=config.smooth_motion,
        audio_bitrate=settings.output_audio_bitrate,
    )
    outputs = job_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    ext = render_settings.format if render_settings.format in {"mp4", "webm"} else "mp4"
    output = outputs / f"highlight.{ext}"
    srt_path = None
    transcript = {}
    tpath = job_dir / "transcript.json"
    if tpath.exists():
        transcript = json.loads(tpath.read_text(encoding="utf-8"))
    if config.subtitles in {"auto", "burned", "srt"}:
        srt_path = outputs / "highlight.srt"
        build_srt(plan, transcript.get("segments") or [], srt_path)

    meta = VideoMetadata.model_validate(json.loads(job.metadata_json or "{}"))
    renderer = Renderer(runner)

    def emit_log(message: str, level: str = "INFO") -> None:
        payload = {"level": level, "message": message, "ts": time.time()}
        append_log(job_dir, payload)
        _emit(app, job_id, "job_log", payload)

    emit_log("Rendering highlight")
    encoder = None
    if render_settings.quality not in {"source", "very_high"}:
        encoder = preferred_hw_encoder()
    if encoder:
        emit_log(f"Using hardware encoder {encoder}")
    else:
        emit_log("Using high-quality CPU encoder libx264 to preserve source detail")
    manager.set_stage(job_id, "rendering", "running")
    try:
        result = renderer.render(
            source,
            plan,
            output,
            render_settings,
            job_dir / "tmp_render",
            metadata=meta,
            srt_path=srt_path if config.subtitles == "burned" else None,
            hardware_encoder=encoder,
            source_lookup=source_lookup,
        )
        out_meta = result["metadata"]
        canvas = result.get("canvas") or {}
        encoder_info = result.get("encoder") or {}
        if result.get("frame_rate"):
            emit_log(str(result["frame_rate"]))
        emit_log(
            "Target output "
            f"{canvas.get('width') or out_meta.get('width')}×{canvas.get('height') or out_meta.get('height')} "
            f"@ {float(canvas.get('fps') or out_meta.get('fps') or 0):.3f} fps, "
            f"{encoder_info.get('codec') or out_meta.get('video_codec')} "
            f"{encoder_info.get('preset') or ''} CRF {encoder_info.get('crf') or ''}".strip()
        )
        emit_log(
            "Verified output "
            f"{out_meta.get('width')}×{out_meta.get('height')}, "
            f"{float(out_meta.get('avg_fps') or out_meta.get('fps') or 0):.3f} fps, "
            f"{out_meta.get('video_codec') or 'h264'}, "
            f"{int(out_meta.get('bitrate') or 0) or out_meta.get('size_bytes')} bitrate/size"
        )
        manager.add_output(
            job_id,
            path=str(output),
            kind="highlight",
            duration=out_meta.get("duration") or 0,
            size_bytes=out_meta.get("size_bytes") or output.stat().st_size,
            codec=out_meta.get("video_codec") or "",
            width=out_meta.get("width") or 0,
            height=out_meta.get("height") or 0,
            version_label=config.style,
        )
        manager.set_stage(job_id, "rendering", "done")
        took = manager.finish_attempt(job_id, "completed")
        manager.update(job_id, status="completed", render_status="completed", progress=100)
        emit_log(f"Highlight ready. Creating the video took {format_elapsed(took)}")
        _emit(app, job_id, "job_complete", {"phase": "render", "output": str(output)})
        if not settings.debug_analysis:
            tmp = job_dir / "tmp_render"
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
    except CancelledError:
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


ANALYSIS_VIDEO = "analysis.mp4"
_CACHE_FILES = (
    "metadata.json",
    "scenes.json",
    "transcript.json",
    "silences.json",
    "frames.json",
    "visual_analysis.json",
    "segment_scores.json",
    "scene_quality.json",
    "audio_energy.json",
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
    stamp.write_text(json.dumps(payload), encoding="utf-8")
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
    for src in sources:
        src_path = Path(src["path"])
        sid = str(src.get("id") or "v0")
        if single:
            dest = src_path.parent / ANALYSIS_VIDEO
            stamp = src_path.parent / "crop.json"
        else:
            dest = src_path.parent / f"{sid}_analysis.mp4"
            stamp = src_path.parent / f"{sid}_crop.json"
        analysis, changed = _crop_one(src_path, dest, stamp, crop_payload, runner, progress)
        src["analysis_path"] = str(analysis)
        busted = busted or changed
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


def load_cache(job_dir: Path) -> dict:
    cache = {}
    mapping = {
        "metadata": "metadata.json",
        "scenes": "scenes.json",
        "transcript": "transcript.json",
        "silences": "silences.json",
        "frames": "frames.json",
        "visual": "visual_analysis.json",
        "segment_scores": "segment_scores.json",
        "scene_quality": "scene_quality.json",
        "audio_energy": "audio_energy.json",
        "candidates": "candidates.json",
        "clip_plan": "clip_plan.json",
    }
    for key, name in mapping.items():
        path = job_dir / name
        if path.exists():
            try:
                cache[key] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    return cache


def append_log(job_dir: Path, payload: dict) -> None:
    path = job_dir / "job.log"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
