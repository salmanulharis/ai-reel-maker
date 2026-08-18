from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, scoped_session


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    original_filename: Mapped[str] = mapped_column(String(255), default="")
    input_path: Mapped[str] = mapped_column(Text, default="")
    analysis_status: Mapped[str] = mapped_column(String(32), default="pending")
    render_status: Mapped[str] = mapped_column(String(32), default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    style: Mapped[str] = mapped_column(String(32), default="auto")
    min_duration: Mapped[float] = mapped_column(Float, default=20)
    max_duration: Mapped[float] = mapped_column(Float, default=30)
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    last_stage: Mapped[str] = mapped_column(String(64), default="")
    progress: Mapped[float] = mapped_column(Float, default=0)
    processing_seconds: Mapped[float] = mapped_column(Float, default=0)
    cancelled: Mapped[bool] = mapped_column(Boolean, default=False)


class JobStage(Base):
    __tablename__ = "job_stages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(40), index=True)
    name: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="pending")
    detail: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SceneRow(Base):
    __tablename__ = "scenes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(40), index=True)
    scene_id: Mapped[int] = mapped_column(Integer)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    duration: Mapped[float] = mapped_column(Float)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")


class TranscriptSegmentRow(Base):
    __tablename__ = "transcript_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(40), index=True)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    speaker: Mapped[str | None] = mapped_column(String(64), nullable=True)


class CandidateRow(Base):
    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(40), index=True)
    candidate_id: Mapped[str] = mapped_column(String(64))
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float, default=0)
    reason: Mapped[str] = mapped_column(Text, default="")
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    selected: Mapped[bool] = mapped_column(Boolean, default=False)


class SelectedClip(Base):
    __tablename__ = "selected_clips"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(40), index=True)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float, default=0)
    reason: Mapped[str] = mapped_column(Text, default="")
    transcript: Mapped[str] = mapped_column(Text, default="")
    duration_reason: Mapped[str] = mapped_column(Text, default="")
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    kept: Mapped[bool] = mapped_column(Boolean, default=True)
    source_id: Mapped[str] = mapped_column(String(32), default="")
    source_name: Mapped[str] = mapped_column(String(255), default="")


class OutputRow(Base):
    __tablename__ = "outputs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(40), index=True)
    path: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(32), default="highlight")
    duration: Mapped[float] = mapped_column(Float, default=0)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    codec: Mapped[str] = mapped_column(String(32), default="")
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    version_label: Mapped[str] = mapped_column(String(64), default="default")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SettingRow(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")


engine = None
SessionLocal = None


def init_db(path) -> None:
    global engine, SessionLocal
    url = f"sqlite:///{path}"
    engine = create_engine(url, echo=False, future=True, connect_args={"check_same_thread": False})
    SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
    Base.metadata.create_all(engine)
    _ensure_columns()


def _ensure_columns() -> None:
    if engine is None:
        return
    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(selected_clips)")).fetchall()
        names = {row[1] for row in rows}
        if "duration_reason" not in names:
            conn.execute(text("ALTER TABLE selected_clips ADD COLUMN duration_reason TEXT DEFAULT ''"))
        if "source_id" not in names:
            conn.execute(text("ALTER TABLE selected_clips ADD COLUMN source_id VARCHAR(32) DEFAULT ''"))
        if "source_name" not in names:
            conn.execute(text("ALTER TABLE selected_clips ADD COLUMN source_name VARCHAR(255) DEFAULT ''"))


def get_session():
    if SessionLocal is None:
        raise RuntimeError("Database not initialized")
    return SessionLocal()
