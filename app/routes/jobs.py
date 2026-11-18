from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from app.analysis.pipeline import alternative_moments, plan_from_cache
from app.config import get_settings
from app.models.schemas import HighlightConfig
from app.services.cover import CoverError, cover_image
from app.services.cover import parse_request as parse_cover_request
from app.services.editor_export import ExportError, build_editor_package
from app.services.job_manager import get_job_manager
from app.analysis.tighten import tighten_spans
from app.video.audio_mix import MUSIC_SUFFIXES, find_music
from app.video.captions import load_transcript
from app.video.sources import load_sources
from app.services.outputs import ShortParamError, find_output, parse_short
from app.services.post_text import PostTextError
from app.services.post_text import saved as saved_post_text
from app.services.post_text import suggest as suggest_post_text
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


@jobs_bp.post("/jobs/<job_id>/split")
def split_job(job_id: str):
    """Make a separate highlight for each video of a job that has not started: one queued job
    per video with these settings (the body, as for /analyze). The original job is removed."""
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    data = request.get_json(silent=True) or {}
    try:
        config = _config_from_request({**json.loads(job.config_json or "{}"), **data})
        jobs = manager.split_into_jobs(job_id, config)
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400
    from app.routes.settings import request_delete

    request_delete(job_id)
    return jsonify({"jobs": [manager.serialize(item) for item in jobs]})


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


MOMENTS_DEFAULT = 12
MOMENTS_MAX = 50


@jobs_bp.get("/jobs/<job_id>/moments")
def job_moments(job_id: str):
    """Strong moments the current clips leave out, best first (for "More moments" on Review)."""
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    cache = load_cache(manager.job_dir(job_id))
    if not cache.get("candidates"):
        return jsonify({"moments": [], "available": False})
    limit = request.args.get("limit", default=MOMENTS_DEFAULT, type=int) or MOMENTS_DEFAULT
    limit = max(1, min(MOMENTS_MAX, limit))
    config = HighlightConfig.model_validate(json.loads(job.config_json or "{}"))
    serialized = manager.serialize(job)
    moments = alternative_moments(
        cache,
        config,
        serialized.get("clips") or [],
        single_source=len(serialized.get("sources") or []) <= 1,
        limit=limit,
    )
    return jsonify({"moments": moments, "available": True})


@jobs_bp.get("/jobs/<job_id>/export/editor")
def export_for_editors(job_id: str):
    """The kept clips as timelines for Premiere Pro, DaVinci Resolve and Final Cut Pro (zip)."""
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    try:
        data, filename = build_editor_package(
            job, manager.job_dir(job_id), manager.serialize(job).get("clips") or []
        )
    except ExportError as exc:
        return jsonify({"error": str(exc)}), exc.status
    return send_file(io.BytesIO(data), mimetype="application/zip", as_attachment=True, download_name=filename)


@jobs_bp.get("/jobs/<job_id>/cover")
def job_cover(job_id: str):
    """A frame of the finished highlight: ?t=seconds, &width= for a small preview, &download=1 to save."""
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    try:
        short = parse_short(request.args.get("short"))
        seconds, width = parse_cover_request(request.args.get("t"), request.args.get("width"))
        path = cover_image(manager.job_dir(job_id), seconds, width, short=short)
    except ShortParamError as exc:
        return jsonify({"error": str(exc)}), 400
    except CoverError as exc:
        return jsonify({"error": str(exc)}), exc.status
    download = request.args.get("download") == "1"
    stem = Path(job.original_filename or "highlight").stem or "highlight"
    part = "" if short is None else f"-short-{short}"
    return send_file(
        path,
        mimetype="image/jpeg",
        as_attachment=download,
        download_name=f"{stem}{part}-cover.jpg",
        max_age=0 if download else 3600,
    )


@jobs_bp.get("/jobs/<job_id>/post-text")
def get_post_text(job_id: str):
    """The last suggested titles, description and hashtags, if any were made."""
    manager = get_job_manager()
    if not manager.get(job_id):
        return jsonify({"error": "not found"}), 404
    try:
        short = parse_short(request.args.get("short"))
    except ShortParamError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"suggestion": saved_post_text(manager.job_dir(job_id), short)})


@jobs_bp.post("/jobs/<job_id>/post-text")
def make_post_text(job_id: str):
    """Suggest titles, a description and hashtags for posting the highlight (local AI, with a
    simple fallback when the model is not available)."""
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    try:
        short = parse_short(request.args.get("short"))
        result = suggest_post_text(
            job, manager.job_dir(job_id), manager.serialize(job).get("clips") or [], get_settings(), short=short
        )
    except ShortParamError as exc:
        return jsonify({"error": str(exc)}), 400
    except PostTextError as exc:
        return jsonify({"error": str(exc)}), exc.status
    return jsonify({"suggestion": result})


MUSIC_MAX_BYTES = 200 * 1024 * 1024
RENDER_OPTIONS = {"music_volume", "music_duck"}


@jobs_bp.get("/jobs/<job_id>/music")
def get_music(job_id: str):
    manager = get_job_manager()
    if not manager.get(job_id):
        return jsonify({"error": "not found"}), 404
    music = find_music(manager.job_dir(job_id))
    return jsonify({"name": music.name.split("_", 1)[-1] if music else None})


@jobs_bp.post("/jobs/<job_id>/music")
def upload_music(job_id: str):
    """Background music for this job's video (replaces an earlier file)."""
    manager = get_job_manager()
    if not manager.get(job_id):
        return jsonify({"error": "not found"}), 404
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "file is required"}), 400
    suffix = Path(file.filename).suffix.lower()
    if suffix not in MUSIC_SUFFIXES:
        return jsonify({"error": "unsupported music format"}), 400
    folder = manager.job_dir(job_id) / "music"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.iterdir():
        if old.is_file():
            old.unlink()
    # The video-name helper would force a video extension; keep the music file's own.
    stem = re.sub(r"[^\w\- .]", "_", Path(file.filename).stem).strip(" .")[:80] or "music"
    name = f"{stem}{suffix}"
    dest = folder / f"music_{name}"
    file.save(dest)
    if dest.stat().st_size > MUSIC_MAX_BYTES:
        dest.unlink()
        return jsonify({"error": "music file is too large"}), 400
    return jsonify({"name": name})


@jobs_bp.delete("/jobs/<job_id>/music")
def delete_music(job_id: str):
    manager = get_job_manager()
    if not manager.get(job_id):
        return jsonify({"error": "not found"}), 404
    music = find_music(manager.job_dir(job_id))
    if music:
        music.unlink(missing_ok=True)
    return jsonify({"name": None})


@jobs_bp.post("/jobs/<job_id>/render-options")
def render_options(job_id: str):
    """Change choices that only affect creating the video (no new analysis or clip choice)."""
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    data = request.get_json(silent=True) or {}
    changes = {key: value for key, value in data.items() if key in RENDER_OPTIONS}
    config = HighlightConfig.model_validate({**json.loads(job.config_json or "{}"), **changes})
    manager.update(job_id, config_json=config.model_dump_json())
    return jsonify({"config": {key: getattr(config, key) for key in RENDER_OPTIONS}})


@jobs_bp.post("/jobs/<job_id>/tighten-spans")
def tighten_spans_route(job_id: str):
    """The parts of each clip the video will keep after pauses (and fillers) are cut, so the
    Review preview can skip the same pauses."""
    manager = get_job_manager()
    job = manager.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    config = HighlightConfig.model_validate(json.loads(job.config_json or "{}"))
    clips = (request.get_json(silent=True) or {}).get("clips") or []
    job_dir = manager.job_dir(job_id)
    sources = load_sources(job_dir)
    multi = len(sources) > 1
    known = {str(s.get("id") or "v0") for s in sources}
    first = str(sources[0].get("id") or "v0") if sources else "v0"
    transcripts: dict[str, dict] = {}
    spans = []
    for clip in clips[:500]:
        try:
            start, end = float(clip["start"]), float(clip["end"])
        except (KeyError, TypeError, ValueError):
            spans.append([])
            continue
        if config.tighten == "off" or end <= start:
            spans.append([[start, end]] if end > start else [])
            continue
        source_id = str(clip.get("source_id") or "")
        source_id = source_id if source_id in known else first
        if source_id not in transcripts:
            transcripts[source_id] = load_transcript(job_dir, source_id, multi)
        pieces = tighten_spans(start, end, transcripts[source_id].get("words") or [], config.tighten == "fillers")
        spans.append([[a, b] for a, b in pieces])
    return jsonify({"mode": config.tighten, "spans": spans})


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
    try:
        short = parse_short(request.args.get("short"))
    except ShortParamError as exc:
        return jsonify({"error": str(exc)}), 400
    video = find_output(manager.job_dir(job_id), short)
    if not video:
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
