from __future__ import annotations

from pathlib import Path

from flask import Blueprint, abort, send_file

from app.config import get_settings
from app.services.job_manager import get_job_manager
from app.utils.security import is_relative_to, valid_job_id
from app.video.sources import load_sources, lookup_path

media_bp = Blueprint("media", __name__)


@media_bp.url_value_preprocessor
def _check_job_id(_endpoint, values):
    if values and "job_id" in values and not valid_job_id(values["job_id"]):
        abort(404)


@media_bp.get("/media/sample")
def sample_media():
    path = get_settings().sample_video_path
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype="video/mp4", download_name="sample.mp4")


def _send_job_video(job_id: str, path: Path | None):
    manager = get_job_manager()
    job_dir = manager.job_dir(job_id)
    if not path or not path.is_file() or not is_relative_to(path, job_dir):
        abort(404)
    return send_file(path)


def _source_file(job_id: str, source_id: str | None = None) -> Path | None:
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return None
    job_dir = manager.job_dir(job_id)
    sources = load_sources(job_dir)
    path = lookup_path(sources, source_id, Path(job.input_path) if job.input_path else None)
    if path and path.is_file():
        return path
    source = job_dir / "source"
    analysis = source / "analysis.mp4"
    if analysis.is_file() and analysis.stat().st_size > 1000 and not source_id:
        return analysis
    files = [item for item in source.glob("*") if item.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}]
    files = [item for item in files if not item.name.endswith("_analysis.mp4") and item.name != "analysis.mp4"]
    files.sort()
    if source_id:
        match = next((item for item in files if item.name.startswith(f"{source_id}_")), None)
        return match
    return files[0] if files else None


@media_bp.get("/media/<job_id>/source")
def source_media(job_id: str):
    return _send_job_video(job_id, _source_file(job_id))


@media_bp.get("/media/<job_id>/source/<source_id>")
def source_media_by_id(job_id: str, source_id: str):
    if not source_id.startswith("v") or not source_id[1:].isdigit():
        abort(404)
    return _send_job_video(job_id, _source_file(job_id, source_id))


@media_bp.get("/media/<job_id>/output")
def output_media(job_id: str):
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        abort(404)
    outputs = manager.job_dir(job_id) / "outputs"
    files = sorted(outputs.glob("highlight.*"))
    video = next((f for f in files if f.suffix.lower() in {".mp4", ".webm"}), None)
    if not video or not is_relative_to(video, manager.job_dir(job_id)):
        abort(404)
    return send_file(video)


@media_bp.get("/media/<job_id>/srt")
def srt_media(job_id: str):
    manager = get_job_manager()
    path = manager.job_dir(job_id) / "outputs" / "highlight.srt"
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=True, download_name="highlight.srt")
