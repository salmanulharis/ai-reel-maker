"""The engines that write a brief, and their readiness checks."""

from __future__ import annotations

from app.brief.engines.base import BriefEngineError, EngineStatus

ENGINES = ("local", "claude")


def get_engine(name: str, settings, ollama=None, runner=None):
    """The engine called ``name`` ("local" or "claude")."""
    if name == "claude":
        from app.brief.engines.claude_cli import ClaudeCliEngine

        return ClaudeCliEngine(settings, runner)
    from app.brief.engines.local import LocalEngine

    return LocalEngine(settings, ollama)


def engines_status(settings, ollama=None, refresh: bool = False) -> list[dict]:
    """Readiness of every engine, for the Create and Settings pages. Nothing is chosen here."""
    result = []
    for name in ENGINES:
        try:
            status = get_engine(name, settings, ollama).check(refresh=refresh)
        except Exception as exc:  # noqa: BLE001 - a broken check shows as "not ready"
            status = EngineStatus(name=name, label=name, ready=False, problems=[f"Could not check: {exc}"[:200]])
        result.append(status.as_dict())
    return result


__all__ = ["BriefEngineError", "EngineStatus", "ENGINES", "engines_status", "get_engine"]
