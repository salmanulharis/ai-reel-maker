from __future__ import annotations

import json
import subprocess
import sys

from flask import Blueprint, abort, jsonify, request, send_file

from app.analysis.pipeline import plan_from_cache
from app.config import get_settings
from app.models.schemas import HighlightConfig
from app.services.job_manager import get_job_manager
from app.services.storage import delete_job_files
from app.utils.security import valid_job_id
from app.video.sources import load_sources
from app.workers.worker import load_cache

jobs_bp = Blueprint("jobs", __name__)


@jobs_bp.url_value_preprocessor
def _check_job_id(_endpoint, values):
    if values and "job_id" in values and not valid_job_id(values["job_id"]):
        abort(404)


REGEN_PRESETS = {
    "more_concise": {"preserve_context": "low"},
    "more_exciting": {"style": "exciting"},
    "more_informative": {"style": "informative"},
    "more_visual": {"content_priorities": ["visual"]},
    "less_repetitive": {"repetition_mode": "strict", "remove_repetition": True, "allow_repeated_content": False},
    "focus_on_speech": {"content_priorities": ["speech"]},
    "focus_on_action": {"content_priorities": ["action"]},
}


def _clean_clips(job_id: str, clips: object, strict: bool) -> list[dict]:
    """Clips sent from the review page with numeric times that are never negative.

    Strict (before rendering): times are also kept inside their video, and a kept clip that
    does not end after it starts is refused with a message naming it. Not strict (saving while
    the user is still typing): clips are stored as entered. Raises ValueError for bad input."""
    if not isinstance(clips, list):
        raise ValueError("clips must be a list")
    manager = get_job_manager()
    lengths = {
        str(src.get("id")): float(src.get("duration") or 0)
        for src in load_sources(manager.job_dir(job_id))
        if src.get("id")
    }
    job = manager.get(job_id)
    total = float(json.loads(job.metadata_json or "{}").get("duration") or 0) if job else 0.0
    cleaned = []
    for number, clip in enumerate(clips, start=1):
        if not isinstance(clip, dict):
            raise ValueError("Each clip must be an object")
        try:
            start = max(0.0, float(clip.get("start")))
            end = max(0.0, float(clip.get("end")))
        except (TypeError, ValueError):
            raise ValueError("Each clip needs a numeric start and end") from None
        if strict:
            limit = lengths.get(str(clip.get("source_id") or ""), 0.0) or (total if len(lengths) <= 1 else 0.0)
            if limit > 0:
                start = min(start, limit)
                end = min(end, limit)
            if clip.get("kept", True) and end - start < 0.1:
                raise ValueError(f"Clip {number} must end after it starts.")
        cleaned.append({**clip, "start": round(start, 3), "end": round(end, 3)})
    return cleaned


STILL_STOPPING = "This job is still stopping. Try again in a few seconds."


def _error_message(exc: ValueError) -> str:
    """A short message for a rejected request; validation errors carry one per field."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            first = errors()[0]
            return str(first.get("msg") or exc).removeprefix("Value error, ")
        except (IndexError, TypeError, AttributeError):
            pass
    return str(exc)


def _config_from_request(data: dict | None = None) -> HighlightConfig:
    data = data or {}
    payload = dict(data)
    if "min_duration" not in payload and "target_min" in payload:
        payload["min_duration"] = payload["target_min"]
    if "max_duration" not in payload and "target_max" in payload:
        payload["max_duration"] = payload["target_max"]
    return HighlightConfig.model_validate(payload)


@jobs_bp.post("/jobs")
def create_job():
    if "file" not in request.files and "files" not in request.files:
        return jsonify({"error": "file is required"}), 400
    try:
        config = _config_from_request(json.loads(request.form.get("config") or "{}"))
        files = request.files.getlist("file") + request.files.getlist("files")
        job = get_job_manager().create_from_uploads(files, config)
        job = get_job_manager().update(job.id, status="uploaded")
        return jsonify(get_job_manager().serialize(job)), 201
    except ValueError as exc:
        return jsonify({"error": _error_message(exc)}), 400


@jobs_bp.get("/jobs")
def list_jobs():
    manager = get_job_manager()
    jobs = [manager.serialize(j) for j in manager.list_jobs()]
    return jsonify({"jobs": jobs})


@jobs_bp.get("/jobs/<job_id>")
def get_job(job_id: str):
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    return jsonify(manager.serialize(job))


@jobs_bp.post("/jobs/<job_id>/analyze")
def analyze_job(job_id: str):
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    data = request.get_json(silent=True) or {}
    try:
        config = _config_from_request({**json.loads(job.config_json or "{}"), **data})
    except ValueError as exc:
        return jsonify({"error": _error_message(exc)}), 400
    job = manager.update(
        job_id,
        status="queued",
        analysis_status="pending",
        cancelled=False,
        error=None,
        style=config.style,
        min_duration=config.min_duration,
        max_duration=config.max_duration,
        config_json=config.model_dump_json(),
    )
    return jsonify(manager.serialize(job))


@jobs_bp.post("/jobs/<job_id>/cancel")
def cancel_job(job_id: str):
    job = get_job_manager().cancel(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    return jsonify(get_job_manager().serialize(job))


@jobs_bp.post("/jobs/<job_id>/retry")
def retry_job(job_id: str):
    if get_job_manager().has_runners(job_id):
        return jsonify({"error": STILL_STOPPING}), 409
    job = get_job_manager().retry(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    return jsonify(get_job_manager().serialize(job))


@jobs_bp.post("/jobs/<job_id>/render")
def render_job(job_id: str):
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    if manager.has_runners(job_id):
        return jsonify({"error": STILL_STOPPING}), 409
    data = request.get_json(silent=True) or {}
    if data.get("clips"):
        try:
            clips = _clean_clips(job_id, data["clips"], strict=True)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if not any(clip.get("kept", True) for clip in clips):
            return jsonify({"error": "Keep at least one clip to create a video."}), 400
        manager.save_plan(job_id, clips)
        plan_path = manager.job_dir(job_id) / "clip_plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["clips"] = clips
            plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    job = manager.update(job_id, status="queued", render_status="queued", cancelled=False, error=None)
    return jsonify(manager.serialize(job))


def _regenerated_config(job, data: dict) -> HighlightConfig:
    """The job's options with the request's changes and the chosen review preset applied."""
    values = {**json.loads(job.config_json or "{}"), **{k: v for k, v in data.items() if k != "preset"}}
    preset = data.get("preset")
    if preset in REGEN_PRESETS:
        values.update(REGEN_PRESETS[preset])
        if preset == "more_concise":
            current = HighlightConfig.model_validate(values)
            if current.length_mode == "auto":
                # No maximum to lower: keep only moments closer to the very best instead.
                values["auto_length_strict"] = True
            elif current.max_duration > current.min_duration:
                values["max_duration"] = min(current.max_duration, max(8.0, current.min_duration))
            else:
                # A fixed length (minimum = maximum): make the whole target about a third shorter.
                shorter = max(8.0, round(current.max_duration * 0.7))
                values["min_duration"] = min(current.min_duration, shorter)
                values["max_duration"] = min(current.max_duration, shorter)
    # Validate the result as a whole, so preset values go through the same rules as a new job.
    return HighlightConfig.model_validate(values)


@jobs_bp.post("/jobs/<job_id>/regenerate")
def regenerate_job(job_id: str):
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    if manager.has_runners(job_id):
        return jsonify({"error": STILL_STOPPING}), 409
    data = request.get_json(silent=True) or {}
    try:
        config = _regenerated_config(job, data)
    except ValueError as exc:
        return jsonify({"error": _error_message(exc)}), 400
    common = {
        "config_json": config.model_dump_json(),
        "style": config.style,
        "min_duration": config.min_duration,
        "max_duration": config.max_duration,
    }
    cache = load_cache(manager.job_dir(job_id))
    if cache.get("candidates"):
        plan = plan_from_cache(cache, config, get_settings())
        plan_path = manager.job_dir(job_id) / "clip_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        manager.save_plan(job_id, plan.get("clips") or [])
        manager.update(job_id, status="planning", **common)
        return jsonify({"job": manager.serialize(manager.get(job_id)), "plan": plan})
    job = manager.update(
        job_id,
        status="queued",
        analysis_status="pending",
        cancelled=False,
        error=None,
        **common,
    )
    return jsonify({"job": manager.serialize(job), "plan": None})


@jobs_bp.get("/jobs/<job_id>/analysis")
def job_analysis(job_id: str):
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    return jsonify(load_cache(manager.job_dir(job_id)))


@jobs_bp.get("/jobs/<job_id>/timeline")
def job_timeline(job_id: str):
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    cache = load_cache(manager.job_dir(job_id))
    return jsonify(
        {
            "metadata": cache.get("metadata"),
            "scenes": cache.get("scenes") or [],
            "clips": manager.serialize(job).get("clips") or [],
            "plan": cache.get("clip_plan"),
        }
    )


@jobs_bp.post("/jobs/<job_id>/clips")
def update_clips(job_id: str):
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    data = request.get_json(silent=True) or {}
    try:
        clips = _clean_clips(job_id, data.get("clips") or [], strict=False)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    manager.save_plan(job_id, clips)
    plan_path = manager.job_dir(job_id) / "clip_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {"clips": []}
    plan["clips"] = clips
    plan["estimated_duration"] = sum(max(0, c["end"] - c["start"]) for c in clips)
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return jsonify({"clips": clips, "estimated_duration": plan["estimated_duration"]})


@jobs_bp.get("/jobs/<job_id>/logs")
def job_logs(job_id: str):
    manager = get_job_manager()
    path = manager.job_dir(job_id) / "job.log"
    if not path.exists():
        return jsonify({"logs": []})
    logs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            logs.append(json.loads(line))
        except json.JSONDecodeError:
            logs.append({"level": "INFO", "message": line})
    return jsonify({"logs": logs[-400:]})


@jobs_bp.delete("/jobs/<job_id>")
def delete_job(job_id: str):
    from app.models.db_models import (
        CandidateRow,
        Job,
        JobStage,
        OutputRow,
        SceneRow,
        SelectedClip,
        TranscriptSegmentRow,
        get_session,
    )

    session = get_session()
    try:
        job = session.get(Job, job_id)
        if not job:
            return jsonify({"error": "not found"}), 404
        running = job.status in {"analyzing", "rendering"} or (
            job.status == "planning" and job.analysis_status == "running"
        )
        if (running and not job.cancelled) or get_job_manager().has_runners(job_id):
            # The worker is still writing into the job folder; deleting now would leave files
            # behind (Windows keeps open files) and break the running step.
            return jsonify({"error": "This job is still running. Stop it first, then delete it."}), 409
        for model in (JobStage, SceneRow, TranscriptSegmentRow, CandidateRow, SelectedClip, OutputRow):
            session.query(model).filter(model.job_id == job_id).delete()
        session.delete(job)
        session.commit()
    finally:
        session.close()
    delete_job_files(job_id)
    return jsonify({"ok": True})


@jobs_bp.get("/jobs/<job_id>/download")
def download_job(job_id: str):
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    outputs = manager.job_dir(job_id) / "outputs"
    files = sorted(outputs.glob("highlight.*"))
    video = next((f for f in files if f.suffix.lower() in {".mp4", ".webm"}), None)
    if not video or not video.is_file():
        return jsonify({"error": "no output"}), 404
    video = video.resolve()
    return send_file(video, as_attachment=True, download_name=video.name)


@jobs_bp.post("/jobs/<job_id>/open-folder")
def open_folder(job_id: str):
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    folder = manager.job_dir(job_id) / "outputs"
    folder.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        subprocess.Popen(["explorer", str(folder)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)])
    return jsonify({"ok": True, "path": str(folder)})
