from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request

from app.config import get_settings
from app.models.db_models import SettingRow, get_session
from app.services.job_manager import get_job_manager
from app.services.storage import clear_cache, clear_tmp, delete_job_files, storage_report

settings_bp = Blueprint("settings", __name__)

EDITABLE = {
    "ollama_base_url",
    "ollama_vision_model",
    "ollama_text_model",
    "ollama_timeout",
    "ollama_max_retries",
    "whisper_model",
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
}


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _current() -> dict:
    settings = get_settings()
    data = settings.model_dump()
    data["data_dir"] = str(settings.data_dir)
    data["temp_dir"] = str(settings.tmp_dir)
    session = get_session()
    try:
        rows = session.query(SettingRow).all()
        for row in rows:
            if row.key in data:
                current = data[row.key]
                if isinstance(current, bool):
                    data[row.key] = row.value.lower() in {"1", "true", "yes"}
                elif isinstance(current, int):
                    data[row.key] = int(row.value)
                elif isinstance(current, float):
                    data[row.key] = float(row.value)
                else:
                    data[row.key] = row.value
                setattr(settings, row.key, data[row.key])
    finally:
        session.close()
    return _json_safe({k: data[k] for k in data if not str(k).startswith("_")})


@settings_bp.get("/settings")
def get_settings_route():
    return jsonify({"settings": _current(), "storage": storage_report()})


@settings_bp.post("/settings")
def save_settings():
    payload = request.get_json(silent=True) or {}
    session = get_session()
    settings = get_settings()
    try:
        for key, value in payload.items():
            if key not in EDITABLE:
                continue
            row = session.get(SettingRow, key) or SettingRow(key=key)
            row.value = str(value)
            session.merge(row)
            if hasattr(settings, key):
                target = getattr(settings, key)
                if isinstance(target, bool):
                    setattr(settings, key, str(value).lower() in {"1", "true", "yes"})
                elif isinstance(target, int):
                    setattr(settings, key, int(value))
                elif isinstance(target, float):
                    setattr(settings, key, float(value))
                else:
                    setattr(settings, key, value)
        session.commit()
    finally:
        session.close()
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
