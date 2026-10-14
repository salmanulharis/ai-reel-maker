from __future__ import annotations

from pathlib import Path

from app.config import Settings, get_settings
from app.models.db_models import SettingRow, get_session


# Credentials stay in the environment and are never returned to the browser.
_NOT_SHOWN = {"ollama_api_key"}


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
            data[row.key] = _coerce(data[row.key], row.value)
            _assign(settings, row.key, data[row.key])
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
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw
