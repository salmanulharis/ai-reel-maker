from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import current_app
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from app.config import Settings, get_settings
from app.models.db_models import (
    CandidateRow,
    Job,
    JobStage,
    OutputRow,
    SceneRow,
    SelectedClip,
    TranscriptSegmentRow,
    get_session,
    utcnow,
)
from app.models.schemas import HighlightConfig, VideoMetadata
from app.utils.security import ALLOWED_EXTENSIONS, is_allowed_extension, sanitize_filename
from app.video.ffmpeg_runner import FFmpegRunner
from app.video.sources import load_sources, save_sources, source_id_for

ANALYSIS_STAGES = [
    "metadata",
    "scenes",
    "audio",
    "transcript",
    "silence",
    "frames",
    "visual",
    "content",
    "repetition",
    "planning",
]


class JobManager:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def cancel_event(self, job_id: str) -> threading.Event:
        with self._lock:
            event = self._cancel_events.get(job_id)
            if event is None:
                event = threading.Event()
                self._cancel_events[job_id] = event
            return event

    def create_from_upload(self, file: FileStorage, config: HighlightConfig) -> Job:
        return self.create_from_uploads([file], config)

    def create_from_uploads(self, files: list[FileStorage], config: HighlightConfig) -> Job:
        uploads = [item for item in files if item and item.filename]
        if not uploads:
            raise ValueError("No file uploaded")
        if len(uploads) > 8:
            raise ValueError("You can upload at most 8 videos in one job")
        for item in uploads:
            if not is_allowed_extension(item.filename):
                raise ValueError(f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.settings.jobs_dir / job_id
        source_dir = job_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        runner = FFmpegRunner()
        sources = []
        used_names: set[str] = set()
        total_duration = 0.0
        first_meta = None
        for index, item in enumerate(uploads):
            filename = sanitize_filename(item.filename)
            if filename in used_names:
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                filename = f"{stem}-{index}{suffix}"
            used_names.add(filename)
            dest = source_dir / f"{source_id_for(index)}_{filename}"
            item.save(dest)
            try:
                meta = runner.probe(dest)
            except Exception:
                meta = VideoMetadata(filename=filename, size_bytes=dest.stat().st_size)
            if first_meta is None:
                first_meta = meta
            total_duration += float(meta.duration or 0)
            sources.append(
                {
                    "id": source_id_for(index),
                    "index": index,
                    "filename": filename,
                    "path": str(dest),
                    "analysis_path": str(dest),
                    "duration": meta.duration,
                    "width": meta.width,
                    "height": meta.height,
                    "size_bytes": meta.size_bytes,
                }
            )
        save_sources(job_dir, sources)
        display = ", ".join(s["filename"] for s in sources)
        combined = VideoMetadata(
            filename=display,
            duration=total_duration,
            width=first_meta.width if first_meta else 0,
            height=first_meta.height if first_meta else 0,
            fps=first_meta.fps if first_meta else 0,
            video_codec=first_meta.video_codec if first_meta else "",
            audio_codec=first_meta.audio_codec if first_meta else "",
            size_bytes=sum(int(s.get("size_bytes") or 0) for s in sources),
            has_audio=bool(first_meta and first_meta.has_audio),
        )
        session = get_session()
        try:
            job = Job(
                id=job_id,
                status="uploaded",
                original_filename=display[:255],
                input_path=sources[0]["path"],
                style=config.style,
                min_duration=config.min_duration,
                max_duration=config.max_duration,
                config_json=config.model_dump_json(),
                metadata_json=combined.model_dump_json(),
                last_stage="uploaded",
            )
            session.add(job)
            for name in ANALYSIS_STAGES + ["rendering"]:
                session.add(JobStage(job_id=job_id, name=name, status="pending"))
            session.commit()
            session.refresh(job)
            return job
        finally:
            session.close()

    def get(self, job_id: str) -> Job | None:
        session = get_session()
        try:
            return session.get(Job, job_id)
        finally:
            session.close()

    def list_jobs(self, limit: int = 50) -> list[Job]:
        session = get_session()
        try:
            return session.query(Job).order_by(Job.created_at.desc()).limit(limit).all()
        finally:
            session.close()

    def update(self, job_id: str, **fields) -> Job | None:
        session = get_session()
        try:
            job = session.get(Job, job_id)
            if not job:
                return None
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = utcnow()
            session.commit()
            session.refresh(job)
            return job
        finally:
            session.close()

    def set_stage(self, job_id: str, name: str, status: str, detail: str = "") -> None:
        session = get_session()
        try:
            stage = (
                session.query(JobStage)
                .filter(JobStage.job_id == job_id, JobStage.name == name)
                .one_or_none()
            )
            if stage is None:
                stage = JobStage(job_id=job_id, name=name)
                session.add(stage)
            stage.status = status
            stage.detail = detail
            if status == "running" and stage.started_at is None:
                stage.started_at = utcnow()
            if status in {"done", "error", "skipped"}:
                stage.finished_at = utcnow()
            job = session.get(Job, job_id)
            if job:
                job.last_stage = name
                job.updated_at = utcnow()
            session.commit()
        finally:
            session.close()

    def stages(self, job_id: str) -> list[dict]:
        session = get_session()
        try:
            rows = session.query(JobStage).filter(JobStage.job_id == job_id).all()
            return [
                {
                    "name": r.name,
                    "status": r.status,
                    "detail": r.detail,
                }
                for r in rows
            ]
        finally:
            session.close()

    def save_plan(self, job_id: str, clips: list[dict]) -> None:
        session = get_session()
        try:
            session.query(SelectedClip).filter(SelectedClip.job_id == job_id).delete()
            for i, clip in enumerate(clips):
                session.add(
                    SelectedClip(
                        job_id=job_id,
                        start=clip["start"],
                        end=clip["end"],
                        score=clip.get("score") or 0,
                        reason=clip.get("reason") or "",
                        transcript=clip.get("transcript") or clip.get("text") or "",
                        duration_reason=clip.get("duration_reason") or "",
                        source_id=clip.get("source_id") or "",
                        source_name=clip.get("source_name") or "",
                        sort_order=i,
                        kept=clip.get("kept", True),
                    )
                )
            session.commit()
        finally:
            session.close()

    def add_output(self, job_id: str, **fields) -> None:
        session = get_session()
        try:
            session.add(OutputRow(job_id=job_id, **fields))
            session.commit()
        finally:
            session.close()

    def cancel(self, job_id: str) -> Job | None:
        self.cancel_event(job_id).set()
        return self.update(job_id, status="cancelled", cancelled=True)

    def retry(self, job_id: str) -> Job | None:
        self.cancel_event(job_id).clear()
        job = self.get(job_id)
        if not job:
            return None
        if job.analysis_status == "completed":
            return self.update(
                job_id,
                status="queued",
                error=None,
                cancelled=False,
                render_status="queued",
            )
        return self.update(
            job_id,
            status="queued",
            error=None,
            cancelled=False,
            analysis_status="pending",
        )

    def job_dir(self, job_id: str) -> Path:
        return self.settings.jobs_dir / job_id

    def serialize(self, job: Job) -> dict:
        meta = json.loads(job.metadata_json or "{}")
        config = json.loads(job.config_json or "{}")
        session = get_session()
        try:
            outputs = session.query(OutputRow).filter(OutputRow.job_id == job.id).all()
            clips = (
                session.query(SelectedClip)
                .filter(SelectedClip.job_id == job.id)
                .order_by(SelectedClip.sort_order)
                .all()
            )
        finally:
            session.close()
        return {
            "id": job.id,
            "status": job.status,
            "created_at": _iso(job.created_at),
            "updated_at": _iso(job.updated_at),
            "original_filename": job.original_filename,
            "analysis_status": job.analysis_status,
            "render_status": job.render_status,
            "error": job.error,
            "style": job.style,
            "min_duration": job.min_duration,
            "max_duration": job.max_duration,
            "progress": job.progress,
            "last_stage": job.last_stage,
            "processing_seconds": job.processing_seconds,
            "metadata": meta,
            "config": config,
            "sources": load_sources(self.job_dir(job.id)),
            "stages": self.stages(job.id),
            "clips": [
                {
                    "id": c.id,
                    "start": c.start,
                    "end": c.end,
                    "score": c.score,
                    "reason": c.reason,
                    "duration_reason": getattr(c, "duration_reason", "") or "",
                    "source_id": getattr(c, "source_id", "") or "",
                    "source_name": getattr(c, "source_name", "") or "",
                    "transcript": c.transcript,
                    "kept": c.kept,
                }
                for c in clips
            ],
            "outputs": [
                {
                    "id": o.id,
                    "path": o.path,
                    "kind": o.kind,
                    "duration": o.duration,
                    "size_bytes": o.size_bytes,
                    "codec": o.codec,
                    "width": o.width,
                    "height": o.height,
                    "version_label": o.version_label,
                }
                for o in outputs
            ],
        }


def _iso(value) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager
