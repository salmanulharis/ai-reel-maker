from __future__ import annotations

import logging
import math
from pathlib import Path

from app.config import Settings, get_settings
from app.models.db_models import SettingRow, get_session


# Credentials stay in the environment and are never returned to the browser.
_NOT_SHOWN = {"ollama_api_key", "secret_key"}

_log = logging.getLogger(__name__)

# Allowed ranges for numbers saved from the Settings page: (lowest, highest or None).
_RANGES = {
    "ollama_timeout": (10, None),
    "ollama_max_retries": (0, 10),
    "ollama_vision_batch_size": (1, 32),
    "ollama_vision_num_ctx": (1024, None),
    "ollama_text_num_ctx": (1024, None),
    "ollama_num_thread": (0, 256),
    "whisper_cpu_threads": (0, 256),
    "maximum_candidates": (1, 1000),
    "output_crf": (0, 51),
    "cache_retention_days": (0, None),
    "scene_threshold": (1, 100),
    "silence_threshold_db": (-90, 0),
}


def validate_values(values: dict, settings: Settings | None = None) -> dict:
    """Saved values converted to the type of their setting and checked against its range.
    Raises ValueError with a message for the user, so nothing is stored that would later fail
    to load (a decimal in a whole-number field used to break every later read)."""
    settings = settings or get_settings()
    clean = {}
    for key, value in values.items():
        if not hasattr(settings, key):
            continue
        current = getattr(settings, key)
        label = key.replace("_", " ")
        if isinstance(current, bool):
            clean[key] = value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}
            continue
        if isinstance(current, (int, float)):
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{label} must be a number") from None
            if not math.isfinite(number):
                raise ValueError(f"{label} must be a number")
            if isinstance(current, int):
                if number != int(number):
                    raise ValueError(f"{label} must be a whole number")
                number = int(number)
            low, high = _RANGES.get(key, (None, None))
            if low is not None and number < low:
                raise ValueError(f"{label} must be at least {low}")
            if high is not None and number > high:
                raise ValueError(f"{label} must be at most {high}")
            clean[key] = number
            continue
        clean[key] = "" if value is None else str(value).strip()
    return clean


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def read_setting(key: str) -> str | None:
    session = get_session()
    try:
        row = session.get(SettingRow, key)
        return row.value if row else None
    finally:
        session.close()


def write_setting(key: str, value: str) -> None:
    session = get_session()
    try:
        row = session.get(SettingRow, key) or SettingRow(key=key)
        row.value = value
        session.merge(row)
        session.commit()
    finally:
        session.close()


def apply_values(payload: dict, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    session = get_session()
    try:
        for key, value in payload.items():
            if key == "ollama_base_url" and not str(value or "").strip():
                # An empty address means "use the default" again (from .env or the platform).
                row = session.get(SettingRow, key)
                if row is not None:
                    session.delete(row)
                setattr(settings, key, Settings().ollama_base_url)
                continue
            row = session.get(SettingRow, key) or SettingRow(key=key)
            row.value = str(value)
            session.merge(row)
            _assign(settings, key, value)
        session.commit()
    finally:
        session.close()


def effective_settings(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    data = settings.model_dump()
    data["data_dir"] = str(settings.data_dir)
    data["temp_dir"] = str(settings.tmp_dir)
    session = get_session()
    try:
        rows = session.query(SettingRow).all()
        for row in rows:
            if row.key not in data:
                continue
            try:
                value = _coerce(data[row.key], row.value)
            except (TypeError, ValueError):
                # One unreadable saved value must not block every other setting.
                _log.warning("Ignoring saved setting %s=%r: not a valid value", row.key, row.value)
                continue
            data[row.key] = value
            _assign(settings, row.key, value)
    finally:
        session.close()
    return json_safe(
        {key: value for key, value in data.items() if not str(key).startswith("_") and key not in _NOT_SHOWN}
    )


def _assign(settings: Settings, key: str, value) -> None:
    if not hasattr(settings, key):
        return
    target = getattr(settings, key)
    if isinstance(target, bool):
        setattr(settings, key, str(value).lower() in {"1", "true", "yes"})
    elif isinstance(target, int) and not isinstance(target, bool):
        setattr(settings, key, int(value))
    elif isinstance(target, float):
        setattr(settings, key, float(value))
    else:
        setattr(settings, key, value)


def _coerce(current, raw: str):
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes"}
    if isinstance(current, int) and not isinstance(current, bool):
        number = float(raw)
        if number != int(number):
            raise ValueError(f"not a whole number: {raw}")
        return int(number)
    if isinstance(current, float):
        return float(raw)
    return raw
