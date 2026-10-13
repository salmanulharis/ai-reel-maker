from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (_project_root() / candidate).resolve()


def running_in_docker() -> bool:
    docker_markers = (
        Path("/.dockerenv"),
        Path("\\.dockerenv"),
        _project_root() / ".dockerenv",
    )
    return any(marker.exists() for marker in docker_markers)


def default_ollama_base_url() -> str:
    if running_in_docker():
        return "http://host.docker.internal:11434"
    return "http://127.0.0.1:11434"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_host: str = "0.0.0.0"
    app_port: int = 5000
    flask_env: str = "production"
    secret_key: str = "change-me-local-only"

    ollama_base_url: str = Field(default_factory=default_ollama_base_url)
    ollama_vision_model: str = "qwen3-vl:8b"
    ollama_text_model: str = "qwen3:8b"
    ollama_timeout: int = 900
    ollama_max_retries: int = 2
    ollama_num_ctx: int = 8192
    ollama_vision_max_images: int = 1
    ollama_vision_max_pixels: int = 512
    ollama_vision_jpeg_quality: int = 75
    ollama_vision_num_ctx: int = 4096
    ollama_vision_num_predict: int = 256
    ollama_vision_batch_size: int = 4
    ollama_vision_min_scene_duration: float = 1.0
    ollama_text_batch_size: int = 20
    ollama_text_num_ctx: int = 4096
    ollama_text_num_predict: int = 512
    ollama_keep_alive: str = "15m"
    ollama_num_thread: int = 6
    ollama_warm_on_start: bool = True

    whisper_cpu_threads: int = 6
    whisper_num_workers: int = 1
    worker_idle_sleep: float = 1.5

    data_dir: Path = Field(default_factory=lambda: _project_root() / "data")
    temp_dir: Path | None = None
    database_path: Path | None = None
    frontend_dist: Path = Field(default_factory=lambda: _project_root() / "frontend" / "dist")
    sample_video: Path | None = None

    max_upload_size_gb: float = 10.0
    target_min_duration: float = 20.0
    target_max_duration: float = 30.0
    default_min_duration: float | None = None
    default_max_duration: float | None = None

    enable_whisper: bool = True
    whisper_model: str = "base"
    whisper_device: str = "auto"
    ai_config_mode: str = "auto"
    enable_scene_detection: bool = True
    enable_visual_analysis: bool = True
    enable_repetition_analysis: bool = True

    output_format: str = "mp4"
    output_video_codec: str = "auto"
    output_crf: int = 18
    output_audio_bitrate: str = "192k"

    keep_cache: bool = True
    cache_retention_days: int = 7
    keep_analysis_cache_days: int | None = None
    debug_analysis: bool = False

    scene_threshold: float = 27.0
    frame_sampling_density: str = "auto"
    maximum_candidates: int = 80
    silence_threshold_db: float = -35.0
    silence_min_duration: float = 0.6

    @field_validator("data_dir", "frontend_dist", mode="before")
    @classmethod
    def _as_path(cls, value: str | Path) -> Path:
        return Path(value)

    @field_validator("sample_video", mode="before")
    @classmethod
    def _optional_path(cls, value: str | Path | None) -> Path | None:
        if value is None or value == "":
            return None
        return Path(value)

    @model_validator(mode="after")
    def _normalize_paths(self) -> Settings:
        self.data_dir = resolve_project_path(self.data_dir)
        self.frontend_dist = resolve_project_path(self.frontend_dist)
        if self.temp_dir:
            self.temp_dir = resolve_project_path(self.temp_dir)
        if self.database_path:
            self.database_path = resolve_project_path(self.database_path)
        if self.sample_video:
            self.sample_video = resolve_project_path(self.sample_video)
        return self

    @property
    def sample_video_path(self) -> Path:
        if self.sample_video:
            path = Path(self.sample_video)
            if path.is_file():
                return path
        folder = _project_root() / "samples"
        preferred = folder / "sample.mp4"
        if preferred.is_file():
            return preferred
        videos: list[Path] = []
        for ext in (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"):
            videos.extend(folder.glob(f"*{ext}"))
        if videos:
            return max(videos, key=lambda item: item.stat().st_mtime)
        return preferred

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_size_gb * 1024 * 1024 * 1024)

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def tmp_dir(self) -> Path:
        if self.temp_dir:
            return Path(self.temp_dir)
        return self.data_dir / "tmp"

    @property
    def db_path(self) -> Path:
        if self.database_path:
            return Path(self.database_path)
        return self.data_dir / "app.db"

    @property
    def min_duration(self) -> float:
        return float(self.default_min_duration or self.target_min_duration)

    @property
    def max_duration(self) -> float:
        return float(self.default_max_duration or self.target_max_duration)

    @property
    def cache_days(self) -> int:
        return int(self.keep_analysis_cache_days or self.cache_retention_days)

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.jobs_dir, self.cache_dir, self.tmp_dir, self.db_path.parent):
            path.mkdir(parents=True, exist_ok=True)
        for name in ("jobs", "cache", "tmp"):
            folder = self.data_dir / name
            folder.mkdir(parents=True, exist_ok=True)
            (folder / ".gitkeep").touch()


_override: Settings | None = None


def set_settings(settings: Settings | None) -> None:
    global _override
    _override = settings
    get_settings.cache_clear()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    if _override is not None:
        _override.ensure_directories()
        return _override
    settings = Settings()
    settings.ensure_directories()
    return settings
