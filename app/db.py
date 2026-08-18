from __future__ import annotations

from pathlib import Path

from app.models.db_models import init_db as _init_db, get_session, SessionLocal


def init_db(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _init_db(path)


__all__ = ["init_db", "get_session", "SessionLocal"]
