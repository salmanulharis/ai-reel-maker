"""The local engine: Ollama on this computer, in small steps a 4B model handles well.

1. The AI editor (app.analysis.director) picks the moments and their roles, as in prompt mode;
   each pick becomes a beat (neighbours are merged when there are too many).
2. A writer request per few beats writes the narration and on-screen text, each beat with its
   own word budget; the first request also writes the title, hook and closing line.

Without Ollama, or when a request fails, the brief is made by rules (best moments in time
order, no narration), like every other AI step's fallback. It never stops the job.
"""

from __future__ import annotations

import time
from typing import Callable

from app.ai.ollama import OllamaError, OllamaTruncatedError
from app.analysis.director import run_director
from app.brief.engines.base import EngineStatus
from app.brief.evidence import EvidencePack, voice_over_text
from app.brief.schema import WRITER_SCHEMA, BeatReply, BriefReply, word_budget

# Beats written per request, and reply tokens per beat and for the rest of a reply.
_BEATS_PER_REQUEST = 4
_TOKENS_PER_BEAT = 200
_TOKENS_BASE = 220
_STATUS_SECONDS = 30.0
_HEADING = ("title", "hook", "tone", "music_mood", "call_to_action")
_status_cache: dict[str, tuple[float, EngineStatus]] = {}


def text_model(settings) -> str:
    return str(getattr(settings, "ollama_director_model", "") or "").strip() or settings.ollama_text_model


class LocalEngine:
    name = "local"

    def __init__(self, settings, ollama=None, director_plan: dict | None = None):
        self.settings = settings
        self.ollama = ollama
        # A plan the AI editor already made for this job (prompt mode), reused when it fits.
        self.director_plan = director_plan

    def model_name(self) -> str:
        return text_model(self.settings)

    def _client(self):
        if self.ollama is None:
            from app.services.ollama_runtime import get_ollama_client

            self.ollama = get_ollama_client(self.settings)
        return self.ollama

    def check(self, refresh: bool = False) -> EngineStatus:
        model = self.model_name()
        cached = _status_cache.get(model)
        if cached and not refresh and time.monotonic() - cached[0] < _STATUS_SECONDS:
            return cached[1]
        status = EngineStatus(name="local", label="Local (Ollama)", ready=False, details={"model": model})
        try:
            health = self._client().health()
        except Exception as exc:  # noqa: BLE001
            health = {"connected": False, "error": str(exc)}
        if not health.get("connected"):
            status.problems.append("Ollama is not running. Start Ollama, then check again.")
            status.notes.append("Without Ollama the brief is made by simple rules (no narration text).")
        else:
            from app.ai.ollama import _model_present

            present = _model_present(model, [str(name) for name in health.get("models") or []])
            if not present:
                status.problems.append(f"The text model '{model}' is not installed in Ollama.")
            for warning in health.get("warnings") or []:
                status.notes.append(str(warning))
            status.ready = present
        status.notes.append("Everything stays on this computer.")
        _status_cache[model] = (time.monotonic(), status)
        return status

    def write(
        self,
        pack: EvidencePack,
        *,
        repair: tuple[BriefReply, list[str]] | None = None,
        log: Callable[[str, str], None] | None = None,
        check_cancel: Callable[[], None] | None = None,
    ) -> BriefReply:
        log = log or (lambda message, level="INFO": None)
        client = None
        try:
            client = self._client()
            if not client.reachable():
                client = None
        except Exception:  # noqa: BLE001
            client = None
        if repair is not None:
            previous, problems = repair
            if client is None:
                return previous
            # Only what needs it is written again: beats without narration (when a voice-over is
            # wanted) or with too many words, and the opening lines when they are missing.
            redo = [
                index
                for index, beat in enumerate(previous.beats)
                if (pack.voice_over and not beat.narration.strip() and word_budget(_seconds(pack, beat)) > 0)
                or len(beat.narration.split()) > word_budget(_seconds(pack, beat))
            ]
            heading = not (previous.title and previous.hook)
            return self._words(pack, previous, client, log, check_cancel, redo, heading, problems)
        beats = self._beats(pack, client, log, check_cancel)
        draft = BriefReply(beats=beats)
        if client is None:
            log("Brief: Ollama is not available; the brief is made by rules, without narration text.", "WARNING")
            draft.title = pack.request[:80]
            return draft
        return self._words(pack, draft, client, log, check_cancel, list(range(len(beats))), True, [])

    # Step 1: which moments, in which order

    def _beats(self, pack: EvidencePack, client, log, check_cancel) -> list[BeatReply]:
        _low, high = pack.beats_range
        by_id = {str(item.get("id")): alias for alias, item in pack.aliases.items()}
        plan = None
        if self.director_plan and all(str(c.get("id")) in by_id for c in self.director_plan.get("clips") or []):
            plan = self.director_plan
            log("Brief: using the moments the AI editor chose", "INFO")
        elif client is not None:
            from app.models.schemas import HighlightConfig

            config = HighlightConfig(
                mode="prompt",
                prompt=pack.request,
                min_duration=max(1.0, pack.target_seconds * 0.7) if pack.shorts < 2 else 20,
                max_duration=pack.target_seconds if pack.shorts < 2 else pack.target_seconds / max(1, pack.shorts),
                shorts_count=pack.shorts,
                refine_notes=pack.notes,
            )
            plan = run_director(
                client,
                self.model_name(),
                self.settings,
                config,
                list(pack.aliases.values()),
                pack.content,
                check_cancel=check_cancel,
                log=log,
            )
        if plan and plan.get("clips"):
            picks = [(by_id[str(c["id"])], str(c.get("role") or "highlight")) for c in plan["clips"] if str(c.get("id")) in by_id]
            if plan.get("order") != "story":
                order = list(pack.aliases)
                picks.sort(key=lambda pick: order.index(pick[0]))
        else:
            ranked = sorted(pack.aliases, key=lambda a: float(pack.aliases[a].get("score") or 0), reverse=True)[:high]
            order = list(pack.aliases)
            picks = [(alias, "highlight") for alias in sorted(ranked, key=order.index)]
        beats = [BeatReply(clips=[alias], purpose=role) for alias, role in picks]
        order = list(pack.aliases)
        while len(beats) > high:
            # Too many beats for the length: join two neighbouring beats whose moments follow each
            # other in the video; when none do, leave out the weakest middle beat.
            joinable = [
                i
                for i in range(len(beats) - 1)
                if abs(order.index(beats[i].clips[-1]) - order.index(beats[i + 1].clips[0])) == 1
            ]
            if joinable:
                index = min(joinable, key=lambda i: _seconds(pack, beats[i]) + _seconds(pack, beats[i + 1]))
                first, second = beats[index], beats[index + 1]
                beats[index : index + 2] = [BeatReply(clips=first.clips + second.clips, purpose=first.purpose)]
                continue
            middle = range(1, len(beats) - 1) if len(beats) > 2 else range(len(beats))
            weakest = min(middle, key=lambda i: _score(pack, beats[i]))
            del beats[weakest]
        if beats:
            beats[0].purpose = "hook"
            if len(beats) > 1 and beats[-1].purpose == "highlight":
                beats[-1].purpose = "ending"
        return beats

    # Step 2: the words

    def _words(
        self,
        pack: EvidencePack,
        draft: BriefReply,
        client,
        log,
        check_cancel,
        indexes: list[int],
        heading: bool,
        problems: list[str],
    ) -> BriefReply:
        """``draft`` with the words of the beats at ``indexes`` (and the opening lines when
        ``heading``) written by the text model. A reply cut off at its limit is asked again in
        halves; a beat that still gets no answer keeps what it had."""
        reply = draft.model_copy(deep=True)
        options = dict(client.role_options("text"))
        options["temperature"] = 0.5
        queue = [indexes[i : i + _BEATS_PER_REQUEST] for i in range(0, len(indexes), _BEATS_PER_REQUEST)]
        if heading and not queue:
            queue = [[]]
        first = heading
        while queue:
            if check_cancel:
                check_cancel()
            part = queue.pop(0)
            prompt = _writer_prompt(pack, reply.beats, part, first, problems)
            options["num_predict"] = _TOKENS_BASE * (2 if first else 1) + _TOKENS_PER_BEAT * max(1, len(part))
            try:
                answer = client.chat(self.model_name(), prompt, schema=WRITER_SCHEMA, options=options)
            except OllamaTruncatedError:
                if len(part) > 1:
                    half = len(part) // 2
                    queue[:0] = [part[:half], part[half:]]
                    continue
                log(f"Brief: the text model's reply for beat {part[0] + 1 if part else 0} was cut off", "WARNING")
                first = False
                continue
            except OllamaError as exc:
                log(f"Brief: the text model could not write the words ({str(exc)[:120]})", "WARNING")
                first = False
                continue
            if first:
                values = BriefReply.model_validate({k: v for k, v in answer.items() if k in _HEADING and v})
                for name in _HEADING:
                    if answer.get(name):
                        setattr(reply, name, getattr(values, name))
                first = False
            written = {str(item.get("id") or ""): item for item in answer.get("beats") or [] if isinstance(item, dict)}
            for index in part:
                item = written.get(f"b{index + 1}")
                if item is None:
                    continue
                beat = reply.beats[index]
                reply.beats[index] = BeatReply.model_validate(
                    {
                        "clips": beat.clips,
                        "purpose": beat.purpose,
                        "narration": item.get("narration"),
                        "on_screen_text": item.get("on_screen_text") or beat.on_screen_text,
                    }
                )
        return reply


def _score(pack: EvidencePack, beat: BeatReply) -> float:
    scores = [float(pack.aliases[a].get("score") or 0) for a in beat.clips if a in pack.aliases]
    return sum(scores) / len(scores) if scores else 0.0


def _seconds(pack: EvidencePack, beat: BeatReply) -> float:
    return sum(float(pack.aliases[a]["end"]) - float(pack.aliases[a]["start"]) for a in beat.clips if a in pack.aliases)


def _writer_prompt(pack: EvidencePack, beats: list[BeatReply], part: list[int], first: bool, problems: list[str]) -> str:
    lines = [
        "You write the voice-over for a short video made from existing footage.",
        f"Video: {pack.video_name or 'untitled'}. Request: {pack.request}",
    ]
    if pack.notes:
        lines.append("Changes asked for: " + " / ".join(pack.notes))
    lines.append(
        "For each beat below write: narration (spoken words over that beat, natural and specific to what it shows "
        "and says, never more words than its budget) and on_screen_text (at most 6 words, may be empty). "
        "Reply with the beat ids given."
    )
    lines.append(voice_over_text(pack.voice_over))
    if first:
        lines.append(
            "Also write: title (at most 8 words), hook (one line that makes people keep watching), tone (2-3 words), "
            "music_mood (none, calm, uplifting, energetic, dramatic, playful or emotional) and call_to_action (short, may be empty)."
        )
    if problems:
        lines.append("Fix these problems of the previous version: " + " ".join(problems))
    total = len(beats)
    for index in part:
        beat = beats[index]
        seconds = _seconds(pack, beat)
        budget = word_budget(seconds)
        lines.append(f"\nBeat b{index + 1} of {total} ({beat.purpose}, {seconds:.0f}s, at most {budget} words):")
        lines.extend(pack.lines[alias] for alias in beat.clips if alias in pack.lines)
    return "\n".join(lines)
