"""The Claude engine: the Claude Code CLI on the user's own Claude subscription.

One request per brief (plus at most one repair). The CLI runs in print mode with no tools, no
MCP servers and no user/project settings (``--restricted``), from the job's brief folder (so no
project instructions are read), and returns JSON checked against the reply schema
(``structured_output``). The evidence goes in through ``--input-format stream-json``: the moment
list as text, plus labelled frame sheets as images when the user allows sending frames.

``--bare`` is never used: it only accepts an API key and would ignore the subscription login.
For the same reason ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN are removed from the CLI's
environment, so the subscription is what is used.

Any failure (not installed, logged out, usage limit, timeout, no usable reply) raises
BriefEngineError: the job then pauses and the user chooses to retry or to use the local engine.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.brief.engines.base import SYSTEM_PROMPT, BriefEngineError, EngineStatus
from app.brief.evidence import EvidencePack
from app.brief.schema import REPLY_SCHEMA, BriefReply
from app.video.ffmpeg_runner import CancelledError, CommandError, FFmpegRunner

MODELS = ("opus", "sonnet", "haiku")
EFFORTS = ("low", "medium", "high")
_STATUS_SECONDS = 60.0
_status_cache: dict[str, tuple[float, EngineStatus]] = {}
# Environment variables that would make the CLI use something other than the subscription, or
# think it runs inside another Claude Code session.
_DROP_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SSE_PORT")

TEST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ok"],
    "properties": {"ok": {"type": "boolean"}},
}


def cli_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in _DROP_ENV}


class ClaudeCliEngine:
    name = "claude"

    def __init__(self, settings, runner: FFmpegRunner | None = None):
        self.settings = settings
        self.runner = runner or FFmpegRunner()

    def model_name(self) -> str:
        model = str(getattr(self.settings, "claude_model", "opus") or "opus").strip().lower()
        return model if model in MODELS else "opus"

    def effort(self) -> str:
        effort = str(getattr(self.settings, "claude_effort", "medium") or "medium").strip().lower()
        return effort if effort in EFFORTS else "medium"

    def executable(self) -> str | None:
        configured = str(getattr(self.settings, "claude_cli_path", "claude") or "claude").strip()
        if Path(configured).is_file():
            return configured
        return shutil.which(configured)

    # Readiness

    def check(self, refresh: bool = False) -> EngineStatus:
        exe = self.executable() or ""
        cached = _status_cache.get(exe)
        if cached and not refresh and time.monotonic() - cached[0] < _STATUS_SECONDS:
            return cached[1]
        status = EngineStatus(
            name="claude",
            label="Claude (Claude Code)",
            ready=False,
            details={"model": self.model_name(), "effort": self.effort()},
        )
        if not exe:
            status.problems.append(
                "Claude Code is not installed on this computer (the `claude` command was not found)."
            )
            _status_cache[exe] = (time.monotonic(), status)
            return status
        try:
            code, lines, errors = self.runner.run_streaming([exe, "--version"], timeout=30, env=cli_environment())
            status.details["version"] = (lines[0] if lines else errors).strip()[:80]
            code, lines, errors = self.runner.run_streaming([exe, "auth", "status"], timeout=30, env=cli_environment())
            auth = _json_object("\n".join(lines))
        except (CommandError, OSError) as exc:
            status.problems.append(f"Claude Code could not be started: {exc}"[:200])
            _status_cache[exe] = (time.monotonic(), status)
            return status
        if not auth.get("loggedIn"):
            status.problems.append(
                "Claude Code is not logged in. Open a terminal, run `claude` and log in with your Claude account."
            )
        else:
            method = str(auth.get("authMethod") or "")
            status.details["auth"] = method
            expires = _parse_time(auth.get("loginExpiresAt"))
            if expires is not None:
                status.details["login_expires"] = expires.isoformat()
            if expires is not None and expires < datetime.now(timezone.utc):
                status.problems.append("Claude Code's login has expired. Run `claude` in a terminal and log in again.")
            if method == "claude.ai":
                status.notes.append("Uses your Claude subscription; its usage limits apply.")
            elif method:
                status.notes.append(f"Claude Code is signed in with {method}, not a Claude subscription.")
        frames = " and small frame sheets" if getattr(self.settings, "claude_send_frames", True) else ""
        status.notes.append(f"The list of moments (what is said, seen and heard){frames} is sent to Anthropic.")
        status.ready = not status.problems
        _status_cache[exe] = (time.monotonic(), status)
        return status

    def test(self) -> dict[str, Any]:
        """One tiny request with the chosen model, to show that it really works."""
        started = time.monotonic()
        try:
            result = self._ask(
                [{"type": "text", "text": "This is a connection test. Reply with ok set to true."}],
                TEST_SCHEMA,
                "Reply with JSON only.",
                effort="low",
                timeout=120,
            )
            ok = bool((result.get("structured_output") or {}).get("ok"))
            return {"ok": ok, "seconds": round(time.monotonic() - started, 1), "model": _used_model(result), "error": None}
        except BriefEngineError as exc:
            return {"ok": False, "seconds": round(time.monotonic() - started, 1), "model": None, "error": str(exc)}

    # Writing

    def write(
        self,
        pack: EvidencePack,
        *,
        repair: tuple[BriefReply, list[str]] | None = None,
        log: Callable[[str, str], None] | None = None,
        check_cancel: Callable[[], None] | None = None,
        cwd: Path | None = None,
    ) -> BriefReply:
        log = log or (lambda message, level="INFO": None)
        if check_cancel:
            check_cancel()
        content: list[dict[str, Any]] = [{"type": "text", "text": pack.text()}]
        for number, sheet in enumerate(pack.sheets, start=1):
            try:
                data = base64.standard_b64encode(sheet.read_bytes()).decode("ascii")
            except OSError:
                continue
            content.append({"type": "text", "text": f"Frame sheet {number} (each picture is labelled with its moment's alias):"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
        if repair is not None:
            previous, problems = repair
            content.append({"type": "text", "text": "Your previous brief:\n" + previous.model_dump_json()})
            from app.brief.validate import repair_text

            content.append({"type": "text", "text": repair_text(problems)})
        else:
            content.append({"type": "text", "text": "Write the brief."})
        result = self._ask(content, REPLY_SCHEMA, SYSTEM_PROMPT, cwd=cwd)
        structured = result.get("structured_output")
        if not isinstance(structured, dict):
            structured = _json_object(str(result.get("result") or ""))
        if not structured:
            raise BriefEngineError("Claude did not return a brief in the expected format.", reason="bad_reply")
        seconds = float(result.get("duration_ms") or 0) / 1000
        log(f"Brief: Claude ({_used_model(result) or self.model_name()}) replied in {seconds:.1f}s", "INFO")
        return BriefReply.model_validate(structured)

    def _ask(
        self,
        content: list[dict[str, Any]],
        schema: dict,
        system: str,
        *,
        effort: str | None = None,
        timeout: float | None = None,
        cwd: Path | None = None,
    ) -> dict[str, Any]:
        exe = self.executable()
        if not exe:
            raise BriefEngineError("Claude Code is not installed on this computer.", reason="not_installed")
        model = self.model_name()
        command = [
            exe,
            "-p",
            "--model",
            model,
            "--effort",
            effort or self.effort(),
            "--fallback-model",
            "sonnet" if model != "sonnet" else "haiku",
            "--restricted",
            "--tools",
            "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--max-budget-usd",
            f"{max(0.05, float(getattr(self.settings, 'claude_max_budget_usd', 1.0) or 1.0)):.2f}",
            "--json-schema",
            json.dumps(schema, separators=(",", ":")),
            "--system-prompt",
            system,
        ]
        message = {"type": "user", "message": {"role": "user", "content": content}}
        try:
            code, lines, errors = self.runner.run_streaming(
                command,
                stdin_text=json.dumps(message) + "\n",
                timeout=timeout or float(getattr(self.settings, "claude_timeout", 300) or 300),
                cwd=cwd,
                env=cli_environment(),
            )
        except CancelledError:
            raise
        except CommandError as exc:
            raise BriefEngineError(f"Claude did not answer: {exc}", reason="timeout") from exc
        result = parse_result(lines)
        if result is None:
            raise _failure(errors or "\n".join(lines[-5:]) or f"exit code {code}")
        if result.get("is_error") or result.get("subtype") != "success":
            raise _failure(
                " ".join(str(result.get(k) or "") for k in ("subtype", "result", "api_error_status")).strip()
            )
        return result


def parse_result(lines: list[str]) -> dict[str, Any] | None:
    """The final ``result`` event of the CLI's stream-json output."""
    for line in reversed(lines):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            return event
    return None


def _failure(text: str) -> BriefEngineError:
    lowered = text.lower()
    if "max_budget" in lowered or "budget" in lowered:
        return BriefEngineError(
            "Claude stopped at the safety budget for one brief. Raise it in Settings or use the local engine.",
            reason="budget",
        )
    if any(word in lowered for word in ("usage limit", "rate limit", "429", "overloaded", "limit reached")):
        return BriefEngineError(
            "Claude's usage limit was reached. Try again later, or use the local engine.", reason="limit"
        )
    if any(word in lowered for word in ("not logged in", "login", "log in", "401", "authentication", "oauth")):
        return BriefEngineError(
            "Claude Code is not logged in. Run `claude` in a terminal and log in, then retry.", reason="login"
        )
    return BriefEngineError(f"Claude could not write the brief: {text[:300] or 'no reply'}", reason="failed")


def _json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _used_model(result: dict[str, Any]) -> str | None:
    usage = result.get("modelUsage")
    if isinstance(usage, dict) and usage:
        return max(usage, key=lambda name: float((usage[name] or {}).get("outputTokens") or 0))
    return None
