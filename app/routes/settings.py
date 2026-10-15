from __future__ import annotations

from flask import Blueprint, jsonify, request

from app.config import get_settings
from app.models.db_models import get_session
from app.services.job_manager import get_job_manager
from app.services.settings_store import apply_values, effective_settings, validate_values
from app.services.storage import clear_cache, clear_tmp, delete_job_files, storage_report

settings_bp = Blueprint("settings", __name__)

EDITABLE = {
    "ollama_base_url",
    "ollama_vision_model",
    "ollama_text_model",
    "ollama_timeout",
    "ollama_max_retries",
    "ollama_vision_batch_size",
    "ollama_vision_num_ctx",
    "ollama_text_num_ctx",
    "ollama_num_thread",
    "whisper_model",
    "whisper_device",
    "whisper_cpu_threads",
    "ai_config_mode",
    "scene_threshold",
    "maximum_candidates",
    "silence_threshold_db",
    "output_video_codec",
    "output_crf",
    "output_audio_bitrate",
    "output_format",
    "cache_retention_days",
    "debug_analysis",
    "enable_whisper",
    "enable_visual_analysis",
    "enable_repetition_analysis",
    "vision_scope",
    "ollama_embed_model",
    "image_matching",
}

CUSTOM_ON_EDIT = {
    "ollama_vision_model",
    "ollama_text_model",
    "whisper_model",
    "whisper_device",
    "ollama_vision_batch_size",
    "ollama_vision_num_ctx",
    "ollama_text_num_ctx",
    "ollama_num_thread",
    "whisper_cpu_threads",
}


def _current() -> dict:
    return effective_settings(get_settings())


@settings_bp.get("/settings")
def get_settings_route():
    return jsonify({"settings": _current(), "storage": storage_report()})


@settings_bp.post("/settings")
def save_settings():
    payload = dict(request.get_json(silent=True) or {})
    editing_models = any(key in payload for key in CUSTOM_ON_EDIT)
    if editing_models and payload.get("ai_config_mode") != "auto":
        payload["ai_config_mode"] = "custom"
    values = {key: value for key, value in payload.items() if key in EDITABLE}
    if "vision_scope" in values and values["vision_scope"] not in {"shortlist", "all"}:
        return jsonify({"error": "vision_scope must be shortlist or all"}), 400
    try:
        values = validate_values(values, get_settings())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    apply_values(values, get_settings())
    return jsonify({"settings": _current()})


@settings_bp.post("/settings/storage/clear-tmp")
def api_clear_tmp():
    clear_tmp()
    return jsonify(storage_report())


@settings_bp.post("/settings/storage/clear-cache")
def api_clear_cache():
    clear_cache()
    return jsonify(storage_report())


@settings_bp.post("/settings/storage/delete-completed")
def delete_completed():
    manager = get_job_manager()
    for job in manager.list_jobs(200):
        if job.status == "completed":
            request_delete(job.id)
    return jsonify(storage_report())


def request_delete(job_id: str) -> None:
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
            return
        for model in (JobStage, SceneRow, TranscriptSegmentRow, CandidateRow, SelectedClip, OutputRow):
            session.query(model).filter(model.job_id == job_id).delete()
        session.delete(job)
        session.commit()
    finally:
        session.close()
    delete_job_files(job_id)
