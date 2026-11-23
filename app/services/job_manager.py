from __future__ import annotations

import json
import shutil
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
    as_utc,
    get_session,
    utcnow,
)
from app.models.schemas import HighlightConfig, VideoMetadata
from app.utils.security import ALLOWED_EXTENSIONS, is_allowed_extension, sanitize_filename
from app.video.ffmpeg_runner import FFmpegRunner
from app.video.quality import source_frame_rate
from app.video.sources import load_sources, save_sources, source_id_for

def _short_index(value) -> int:
    try:
        return max(0, min(9, int(value or 0)))
    except (TypeError, ValueError):
        return 0


# Stage states that end a run of the stage.
_STAGE_FINISHED = {"done", "error", "skipped", "cancelled"}
# How a stage left running ends when its attempt ends.
_STAGE_ON_OUTCOME = {
    "completed": "done",
    "failed": "error",
    "cancelled": "cancelled",
    "interrupted": "pending",
    "paused": "pending",
}

# Steps that run only in prompt mode.
PROMPT_STAGES = {"sounds", "director"}
ANALYSIS_STAGES = [
    "metadata",
    "scenes",
    "audio",
    "transcript",
    "silence",
    "frames",
    "sounds",
    "content",
    "visual",
    "repetition",
    "director",
    "planning",
]


class JobManager:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._cancel_events: dict[str, threading.Event] = {}
        # FFmpeg runners working on a job, so a cancel can stop the command that is running.
        self._runners: dict[str, list[FFmpegRunner]] = {}
        # Jobs asked to pause: their run stops like a cancel but ends as "paused".
        self._pause_requests: set[str] = set()
        self._lock = threading.Lock()

    def register_runner(self, job_id: str, runner: FFmpegRunner) -> None:
        with self._lock:
            self._runners.setdefault(job_id, []).append(runner)

    def has_runners(self, job_id: str) -> bool:
        """Whether a run of this job is still working (or still stopping after a cancel)."""
        with self._lock:
            return bool(self._runners.get(job_id))

    def unregister_runner(self, job_id: str, runner: FFmpegRunner) -> None:
        with self._lock:
            runners = self._runners.get(job_id) or []
            if runner in runners:
                runners.remove(runner)
            if not runners:
                self._runners.pop(job_id, None)

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
            dest = dest.resolve()
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
                    "fps": source_frame_rate(meta),
                    "size_bytes": meta.size_bytes,
                }
            )
        return self._create_record(job_id, job_dir, sources, first_meta, total_duration, config)

    def _create_record(
        self,
        job_id: str,
        job_dir: Path,
        sources: list[dict],
        first_meta: VideoMetadata | None,
        total_duration: float,
        config: HighlightConfig,
        status: str = "uploaded",
    ) -> Job:
        """Save the sources list and add the job (with its pending stages) to the database."""
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
                status=status,
                analysis_status="pending",
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
            for name in _stage_names(config.mode):
                session.add(JobStage(job_id=job_id, name=name, status="pending"))
            session.commit()
            session.refresh(job)
            return job
        finally:
            session.close()

    def split_into_jobs(self, job_id: str, config: HighlightConfig) -> list[Job]:
        """
        Turn a job with several videos that has not started yet into one queued job per video,
        all with `config`. The uploaded files are moved, not copied. The original job is left
        without videos; the caller deletes it. Nothing changes if a file cannot be moved.
        """
        job = self.get(job_id)
        if not job:
            raise LookupError("not found")
        if job.status != "uploaded":
            raise ValueError("only a job that has not started can be split")
        sources = load_sources(self.job_dir(job_id))
        if len(sources) < 2:
            raise ValueError("splitting needs at least two videos")

        moved: list[tuple[Path, Path, str, dict]] = []
        new_ids: list[str] = []
        try:
            for source in sources:
                original = Path(str(source.get("path") or ""))
                if not original.is_file():
                    raise ValueError(f"video file is missing: {source.get('filename')}")
                new_id = uuid.uuid4().hex[:12]
                new_ids.append(new_id)
                new_dir = self.settings.jobs_dir / new_id / "source"
                new_dir.mkdir(parents=True, exist_ok=True)
                dest = new_dir / f"{source_id_for(0)}_{source.get('filename') or original.name}"
                shutil.move(str(original), str(dest))
                moved.append((original, dest.resolve(), new_id, source))
        except Exception:
            # Put every file back and remove the new folders, so the original job still works.
            for original, dest, _, _ in moved:
                if dest.is_file():
                    shutil.move(str(dest), str(original))
            for new_id in new_ids:
                shutil.rmtree(self.settings.jobs_dir / new_id, ignore_errors=True)
            raise

        runner = FFmpegRunner()
        jobs = []
        for _, dest, new_id, source in moved:
            try:
                meta = runner.probe(dest)
            except Exception:  # noqa: BLE001 - the upload's own record is enough to start
                meta = VideoMetadata(
                    filename=str(source.get("filename") or dest.name),
                    duration=float(source.get("duration") or 0),
                    width=int(source.get("width") or 0),
                    height=int(source.get("height") or 0),
                    fps=float(source.get("fps") or 0),
                    size_bytes=int(source.get("size_bytes") or 0),
                )
            entry = {
                "id": source_id_for(0),
                "index": 0,
                "filename": source.get("filename") or dest.name,
                "path": str(dest),
                "analysis_path": str(dest),
                "duration": meta.duration,
                "width": meta.width,
                "height": meta.height,
                "fps": source_frame_rate(meta),
                "size_bytes": meta.size_bytes or dest.stat().st_size,
            }
            jobs.append(
                self._create_record(
                    new_id, self.settings.jobs_dir / new_id, [entry], meta, float(meta.duration or 0), config, "queued"
                )
            )
        save_sources(self.job_dir(job_id), [])
        return jobs

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
        if fields.get("status") == "queued":
            # Queued again (resume, retry, a new render): an earlier pause request is over.
            with self._lock:
                self._pause_requests.discard(job_id)
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

    def set_stage(
        self,
        job_id: str,
        name: str,
        status: str,
        detail: str = "",
        done: float = 0.0,
        total: float = 0.0,
        unit: str = "",
    ) -> None:
        session = get_session()
        try:
            _apply_stage(session, job_id, name, status, detail, done, total, unit)
            job = session.get(Job, job_id)
            if job:
                job.last_stage = name
                job.updated_at = utcnow()
            session.commit()
        finally:
            session.close()

    def record_progress(
        self,
        job_id: str,
        overall: float,
        eta: float | None = None,
        stage: str = "",
        status: str = "",
        detail: str = "",
        done: float = 0.0,
        total: float = 0.0,
        unit: str = "",
        source_index: int | None = None,
        source_count: int | None = None,
    ) -> None:
        """One progress update: the job's figure and time left, and its stage, in one write."""
        session = get_session()
        try:
            job = session.get(Job, job_id)
            if job is None:
                return
            job.progress = max(0.0, min(100.0, float(overall or 0)))
            job.eta_seconds = None if eta is None else max(0.0, float(eta))
            job.progress_at = utcnow()
            if source_index is not None:
                job.source_index = int(source_index)
            if source_count is not None:
                job.source_count = int(source_count)
            if stage:
                job.last_stage = stage
                if status:
                    _apply_stage(session, job_id, stage, status, detail, done, total, unit)
            session.commit()
        finally:
            session.close()

    def stages(self, job_id: str) -> list[dict]:
        session = get_session()
        try:
            rows = session.query(JobStage).filter(JobStage.job_id == job_id).all()
            order = {name: index for index, name in enumerate(["prepare", *ANALYSIS_STAGES, "rendering"])}
            rows.sort(key=lambda row: order.get(row.name, len(order)))
            return [
                {
                    "name": r.name,
                    "status": r.status,
                    "detail": r.detail,
                    # Finished time in this stage; a running stage adds now - started_at.
                    "seconds": round(float(r.seconds or 0), 3),
                    "started_at": _iso(r.started_at) if r.status == "running" else None,
                    "done": round(float(r.done or 0), 2),
                    "total": round(float(r.total or 0), 2),
                    "unit": r.unit or "",
                }
                for r in rows
            ]
        finally:
            session.close()

    def prepare_stages(self, job_id: str, mode: str) -> None:
        """The step list for the job's mode: prompt mode adds understanding sounds and the AI
        editor; manual mode does not show them (rows that never ran are removed)."""
        wanted = _stage_names(mode)
        session = get_session()
        try:
            rows = {row.name: row for row in session.query(JobStage).filter(JobStage.job_id == job_id)}
            for name in wanted:
                if name not in rows:
                    session.add(JobStage(job_id=job_id, name=name, status="pending"))
            for name, row in rows.items():
                if name in PROMPT_STAGES and name not in wanted and row.status == "pending":
                    session.delete(row)
            session.commit()
        finally:
            session.close()

    def start_attempt(self, job_id: str, phase: str) -> None:
        """Mark the real start of an analysis or render attempt."""
        session = get_session()
        try:
            job = session.get(Job, job_id)
            if job is None:
                return
            job.attempt_started_at = utcnow()
            job.attempt_seen_at = job.attempt_started_at
            job.attempt_phase = phase
            job.attempt_outcome = "running"
            session.commit()
        finally:
            session.close()

    def touch_attempt(self, job_id: str) -> None:
        """Record that the attempt in progress is still running (see recover_interrupted)."""
        session = get_session()
        try:
            job = session.get(Job, job_id)
            if job is None or job.attempt_started_at is None:
                return
            job.attempt_seen_at = utcnow()
            session.commit()
        finally:
            session.close()

    def finish_attempt(self, job_id: str, outcome: str, *, only_if_running: bool = False) -> float:
        """End the running attempt: add its real length to the totals, keep it as the last
        attempt's length, and close any stage it left running. Returns the attempt's seconds.
        With only_if_running, an attempt that was already finished is left alone."""
        session = get_session()
        try:
            job = session.get(Job, job_id)
            if job is None or (only_if_running and job.attempt_started_at is None):
                return 0.0
            seconds = _close_attempt(session, job, utcnow(), outcome)
            session.commit()
            return seconds
        finally:
            session.close()

    def recover_interrupted(self) -> int:
        """After a restart, close attempts that were cut off (a crash or a power cut). They
        count up to their last sign of life; their stages go back to pending so they run again."""
        session = get_session()
        try:
            jobs = session.query(Job).filter(Job.attempt_started_at.isnot(None)).all()
            for job in jobs:
                # Older rows have no sign of life: their length is unknown and not counted.
                seen = as_utc(job.attempt_seen_at) or as_utc(job.attempt_started_at)
                _close_attempt(session, job, seen, "interrupted")
            session.commit()
            return len(jobs)
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
                        short_index=_short_index(clip.get("short_index")),
                        role=str(clip.get("role") or "")[:32],
                        sort_order=i,
                        kept=clip.get("kept", True),
                    )
                )
            session.commit()
        finally:
            session.close()

    def clear_outputs(self, job_id: str) -> None:
        """Forget the outputs of an earlier render (a new render replaces them)."""
        session = get_session()
        try:
            session.query(OutputRow).filter(OutputRow.job_id == job_id).delete()
            session.commit()
        finally:
            session.close()

    def add_output(self, job_id: str, **fields) -> None:
        session = get_session()
        try:
            # A new render replaces the file at the same path, so it replaces its record too.
            if fields.get("path"):
                session.query(OutputRow).filter(
                    OutputRow.job_id == job_id, OutputRow.path == fields["path"]
                ).delete()
            session.add(OutputRow(job_id=job_id, **fields))
            session.commit()
        finally:
            session.close()

    def _stop_run(self, job_id: str) -> None:
        """Signal a running job to stop and end the commands it is running."""
        self.cancel_event(job_id).set()
        with self._lock:
            runners = list(self._runners.get(job_id) or [])
        for runner in runners:
            # Stopping a process can take a few seconds; the request does not wait for it.
            threading.Thread(target=runner.cancel, daemon=True, name=f"cancel-{job_id}").start()

    def cancel(self, job_id: str) -> Job | None:
        with self._lock:
            self._pause_requests.discard(job_id)
        self._stop_run(job_id)
        return self.update(job_id, status="cancelled", cancelled=True)

    @staticmethod
    def can_pause(job: Job) -> bool:
        """Waiting or working on analysis or rendering (not on the Review page or finished)."""
        if job.status in {"queued", "analyzing", "rendering"}:
            return True
        return job.status == "planning" and job.analysis_status == "running"

    def pause(self, job_id: str) -> Job | None:
        """Stop the job at the next safe point and keep everything finished so far. A paused
        job stays paused across restarts until it is resumed."""
        with self._lock:
            self._pause_requests.add(job_id)
        self._stop_run(job_id)
        return self.update(job_id, status="paused")

    def pause_requested(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._pause_requests

    def take_pause_request(self, job_id: str) -> bool:
        """Whether the job was asked to pause; the request is used up."""
        with self._lock:
            if job_id in self._pause_requests:
                self._pause_requests.discard(job_id)
                return True
            return False

    def resume(self, job_id: str) -> Job | None:
        """Queue a paused job again; it continues after its finished steps."""
        with self._lock:
            self._pause_requests.discard(job_id)
        job = self.get(job_id)
        if job and job.analysis_status == "completed" and job.render_status not in {"queued", "running", "paused"}:
            # Analysis finished just as the pause arrived and no video was asked for: back to
            # the Review page instead of starting a render nobody asked for.
            return self.update(job_id, status="planning", error=None, cancelled=False)
        return self.retry(job_id)

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

    @staticmethod
    def timing(job: Job) -> dict:
        """Elapsed-time facts for the UI. Totals cover finished attempts only; while an attempt
        runs, the UI adds (server_time - attempt_started_at) plus the time since it fetched."""
        return {
            "total_seconds": round(float(job.processing_seconds or 0), 3),
            "analysis_seconds": round(float(getattr(job, "analysis_seconds", 0) or 0), 3),
            "render_seconds": round(float(getattr(job, "render_seconds", 0) or 0), 3),
            "attempt_seconds": round(float(getattr(job, "attempt_seconds", 0) or 0), 3),
            "attempt_started_at": _iso(getattr(job, "attempt_started_at", None)),
            "attempt_phase": getattr(job, "attempt_phase", "") or "",
            "attempt_outcome": getattr(job, "attempt_outcome", "") or "",
            "server_time": _iso(utcnow()),
        }

    def job_dir(self, job_id: str) -> Path:
        return self.settings.jobs_dir / job_id

    def serialize(self, job: Job) -> dict:
        meta = json.loads(job.metadata_json or "{}")
        config = json.loads(job.config_json or "{}")
        session = get_session()
        try:
            outputs = (
                session.query(OutputRow)
                .filter(OutputRow.job_id == job.id)
                .order_by(OutputRow.id.desc())
                .all()
            )
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
            "eta_seconds": _time_left(job),
            "source_index": int(job.source_index or 0),
            "source_count": int(job.source_count or 0),
            "last_stage": job.last_stage,
            "processing_seconds": job.processing_seconds,
            "timing": self.timing(job),
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
                    "short_index": int(getattr(c, "short_index", 0) or 0),
                    "role": getattr(c, "role", "") or "",
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


def _stage_names(mode: str) -> list[str]:
    names = [name for name in ANALYSIS_STAGES if mode == "prompt" or name not in PROMPT_STAGES]
    return names + ["rendering"]


def _close_attempt(session, job: Job, end, outcome: str) -> float:
    """End ``job``'s attempt at ``end``: add its length to the totals, keep it as the last
    attempt's length, and close the stages it left running. Returns the attempt's seconds."""
    started = as_utc(job.attempt_started_at)
    seconds = max(0.0, (end - started).total_seconds()) if started else 0.0
    if started:
        job.attempt_seconds = round(seconds, 3)
        job.processing_seconds = round(float(job.processing_seconds or 0) + seconds, 3)
        if job.attempt_phase == "render":
            job.render_seconds = round(float(job.render_seconds or 0) + seconds, 3)
        else:
            job.analysis_seconds = round(float(job.analysis_seconds or 0) + seconds, 3)
    job.attempt_started_at = None
    job.attempt_seen_at = None
    job.attempt_outcome = outcome
    job.eta_seconds = None
    final_status = _STAGE_ON_OUTCOME.get(outcome, "error")
    running = session.query(JobStage).filter(JobStage.job_id == job.id, JobStage.status == "running").all()
    for stage in running:
        _close_stage_run(stage, end)
        stage.finished_at = end
        stage.status = final_status
        if final_status == "pending":
            stage.started_at = None
    return seconds


def _apply_stage(
    session, job_id: str, name: str, status: str, detail: str, done: float, total: float, unit: str
) -> None:
    stage = session.query(JobStage).filter(JobStage.job_id == job_id, JobStage.name == name).one_or_none()
    if stage is None:
        stage = JobStage(job_id=job_id, name=name)
        session.add(stage)
    now = utcnow()
    if status == "running" and stage.status != "running":
        # A new run of this stage (first time, another video, or a retry).
        stage.started_at = now
        stage.finished_at = None
    elif status in _STAGE_FINISHED:
        _close_stage_run(stage, now)
        stage.finished_at = now
    stage.status = status
    stage.detail = detail
    if status == "done" and total:
        done = total
    stage.done = max(0.0, float(done or 0))
    stage.total = max(0.0, float(total or 0))
    stage.unit = (unit or "")[:16]


def _time_left(job: Job) -> float | None:
    """Seconds left for the running attempt: the last estimate less the time since it was made."""
    if job.attempt_started_at is None or job.eta_seconds is None:
        return None
    made = as_utc(job.progress_at)
    passed = (utcnow() - made).total_seconds() if made else 0.0
    return round(max(0.0, float(job.eta_seconds) - max(0.0, passed)), 1)


def _close_stage_run(stage: JobStage, now) -> None:
    """Add the time since a running stage started to its total."""
    started = as_utc(stage.started_at)
    if stage.status == "running" and started is not None:
        stage.seconds = round(float(stage.seconds or 0) + max(0.0, (now - started).total_seconds()), 3)


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
