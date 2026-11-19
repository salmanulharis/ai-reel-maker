"""Shared test setup."""

import pytest

from app.config import Settings


@pytest.fixture(autouse=True)
def _ignore_developer_env_file(monkeypatch):
    """Tests use the code defaults, not the models, threads, or addresses in a developer's .env."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
