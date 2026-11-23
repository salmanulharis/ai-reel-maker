from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from app.config import get_settings
from app.models.db_models import get_session
from app.services.ai_profile import APPLIED_FIELDS, pinned_fields, save_pins
from app.services.job_manager import get_job_manager
from app.services.presets import PresetError, delete_preset, list_presets, save_preset
from app.services.watch_folder import read_config as read_watch_config
from app.services.watch_folder import save_config as save_watch_config
from app.video.overlays import LOGO_SUFFIXES, brand_dir, brand_logo
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



def _current() -> dict:
    return effective_settings(get_settings())


@settings_bp.get("/settings")
def get_settings_route():
    return jsonify({"settings": _current(), "storage": storage_report()})


@settings_bp.post("/settings")
def save_settings():
    payload = dict(request.get_json(silent=True) or {})
    settings = get_settings()
    values = {key: value for key, value in payload.items() if key in EDITABLE}
    if "vision_scope" in values and values["vision_scope"] not in {"shortlist", "all"}:
        return jsonify({"error": "vision_scope must be shortlist or all"}), 400
    try:
        values = validate_values(values, settings)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    # A setting automatic optimization manages, changed by hand while it is on, becomes the
    # user's own choice: it is kept, and the other settings stay automatic.
    chosen = {key for key in APPLIED_FIELDS if key in values and str(values[key]) != str(getattr(settings, key))}
    automatic = values.get("ai_config_mode", settings.ai_config_mode) == "auto"
    apply_values(values, settings)
    if chosen and automatic:
        save_pins(settings, pinned_fields(settings) | chosen)
    return jsonify({"settings": _current()})


@settings_bp.get("/presets")
def get_presets():
    return jsonify({"presets": list_presets()})


@settings_bp.post("/presets")
def post_preset():
    """Save the Create-page choices under a name (an existing preset with that name is replaced)."""
    data = request.get_json(silent=True) or {}
    try:
        presets = save_preset(data.get("name"), data.get("form"))
    except PresetError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"presets": presets})


@settings_bp.delete("/presets")
def remove_preset():
    try:
        presets = delete_preset(request.args.get("name"))
    except PresetError as exc:
        status = 404 if str(exc) == "preset not found" else 400
        return jsonify({"error": str(exc)}), status
    return jsonify({"presets": presets})


LOGO_MAX_BYTES = 5 * 1024 * 1024


@settings_bp.get("/brand/logo")
def get_logo():
    logo = brand_logo(get_settings().data_dir)
    if not logo:
        return jsonify({"error": "no logo"}), 404
    return send_file(logo, max_age=0)


@settings_bp.post("/brand/logo")
def upload_logo():
    """The logo shown on videos when a logo position is chosen (replaces an earlier one)."""
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "file is required"}), 400
    suffix = Path(file.filename).suffix.lower()
    if suffix not in LOGO_SUFFIXES:
        return jsonify({"error": "unsupported image format"}), 400
    folder = brand_dir(get_settings().data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("logo.*"):
        old.unlink(missing_ok=True)
    dest = folder / f"logo{suffix}"
    file.save(dest)
    if dest.stat().st_size > LOGO_MAX_BYTES:
        dest.unlink()
        return jsonify({"error": "image is too large"}), 400
    return jsonify({"logo": True})


@settings_bp.delete("/brand/logo")
def delete_logo():
    logo = brand_logo(get_settings().data_dir)
    if logo:
        logo.unlink(missing_ok=True)
    return jsonify({"logo": False})


@settings_bp.get("/watch-folder")
def get_watch_folder():
    return jsonify(read_watch_config())


@settings_bp.post("/watch-folder")
def post_watch_folder():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(save_watch_config(data.get("enabled"), data.get("folder"), data.get("preset")))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


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
