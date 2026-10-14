from __future__ import annotations

import json
import subprocess
import sys

from flask import Blueprint, jsonify, request, send_file

from app.analysis.pipeline import plan_from_cache
from app.config import get_settings
from app.models.schemas import HighlightConfig
from app.services.job_manager import get_job_manager
from app.services.storage import delete_job_files
from app.workers.worker import load_cache

jobs_bp = Blueprint("jobs", __name__)

REGEN_PRESETS = {
    "more_concise": {"preserve_context": "low"},
    "more_exciting": {"style": "exciting"},
    "more_informative": {"style": "informative"},
    "more_visual": {"content_priorities": ["visual"]},
    "less_repetitive": {"repetition_mode": "strict", "remove_repetition": True, "allow_repeated_content": False},
    "focus_on_speech": {"content_priorities": ["speech"]},
    "focus_on_action": {"content_priorities": ["action"]},
}


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
        return jsonify({"error": str(exc)}), 400


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
    config = _config_from_request({**json.loads(job.config_json or "{}"), **data})
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
    data = request.get_json(silent=True) or {}
    if data.get("clips"):
        manager.save_plan(job_id, data["clips"])
        plan_path = manager.job_dir(job_id) / "clip_plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["clips"] = data["clips"]
            plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    job = manager.update(job_id, status="queued", render_status="queued", cancelled=False, error=None)
    return jsonify(manager.serialize(job))


@jobs_bp.post("/jobs/<job_id>/regenerate")
def regenerate_job(job_id: str):
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    data = request.get_json(silent=True) or {}
    config = HighlightConfig.model_validate({**json.loads(job.config_json or "{}"), **data})
    preset = data.get("preset")
    if preset in REGEN_PRESETS:
        for key, value in REGEN_PRESETS[preset].items():
            if hasattr(config, key):
                setattr(config, key, value)
        if preset == "more_concise":
            if config.length_mode == "auto":
                # No maximum to lower: keep only moments closer to the very best instead.
                config.auto_length_strict = True
            else:
                config.max_duration = min(config.max_duration, max(8.0, config.min_duration))
    cache = load_cache(manager.job_dir(job_id))
    if cache.get("candidates"):
        plan = plan_from_cache(cache, config, get_settings())
        plan_path = manager.job_dir(job_id) / "clip_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        manager.save_plan(job_id, plan.get("clips") or [])
        manager.update(job_id, config_json=config.model_dump_json(), style=config.style, status="planning")
        return jsonify({"job": manager.serialize(manager.get(job_id)), "plan": plan})
    job = manager.update(
        job_id,
        status="queued",
        analysis_status="pending",
        config_json=config.model_dump_json(),
        style=config.style,
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
    clips = data.get("clips") or []
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
