"""Video brief: evidence, checks and repair, the local and Claude engines, applying and editing."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.ai.prompts import DIRECTOR_SCHEMA
from app.brief import service
from app.brief.engines import BriefEngineError, engines_status
from app.brief.engines.base import BriefPaused
from app.brief.engines.claude_cli import ClaudeCliEngine, parse_result
from app.brief.engines.local import LocalEngine
from app.brief.evidence import build_pack
from app.brief.schema import (
    REPLY_SCHEMA,
    WORDS_PER_SECOND,
    WRITER_SCHEMA,
    BeatReply,
    BriefReply,
    VideoBrief,
    word_budget,
)
from app.brief.validate import check, shorten
from app.config import Settings
from app.models.schemas import HighlightConfig
from app.video.ffmpeg_runner import CommandError


def candidates(count: int = 8, length: float = 6.0) -> list[dict]:
    return [
        {
            "id": f"c{n}",
            "start": n * 10.0,
            "end": n * 10.0 + length,
            "score": 0.5 + n / 100,
            "text": f"Sentence number {n} about the trip.",
            "source_id": "v0",
        }
        for n in range(count)
    ]


def config(**values) -> HighlightConfig:
    base = {"video_brief": True, "min_duration": 20, "max_duration": 30, "prompt": "a fun recap", "mode": "prompt"}
    return HighlightConfig.model_validate({**base, **values})


def settings(tmp_path: Path, **values) -> Settings:
    return Settings(data_dir=tmp_path / "data", ollama_base_url="http://127.0.0.1:9", **values)


# Schema


def test_reply_schema_matches_the_models():
    assert set(REPLY_SCHEMA["properties"]) == set(BriefReply.model_fields)
    assert set(REPLY_SCHEMA["required"]) == set(BriefReply.model_fields)
    beat = REPLY_SCHEMA["properties"]["beats"]["items"]
    assert set(beat["properties"]) == set(BeatReply.model_fields) == set(beat["required"])
    assert beat["additionalProperties"] is False and REPLY_SCHEMA["additionalProperties"] is False
    # Flat: both Ollama's format and the CLI's --json-schema read it without $ref support.
    assert "$ref" not in json.dumps(REPLY_SCHEMA) and "$ref" not in json.dumps(WRITER_SCHEMA)


def test_reply_cleans_engine_output():
    reply = BriefReply.model_validate(
        {
            "title": "  A   day  ",
            "music_mood": "loud",
            "beats": [{"clips": ["3", "M4", ""], "purpose": "nonsense", "narration": "x" * 900}, "not a beat"],
        }
    )
    assert reply.title == "A day" and reply.music_mood == "none"
    assert reply.beats[0].clips == ["m3", "m4"] and reply.beats[0].purpose == "highlight"
    assert len(reply.beats[0].narration) == 400 and len(reply.beats) == 1


def test_word_budget():
    assert word_budget(1.0) == 0
    assert word_budget(2.0) == 4
    assert word_budget(10.0) == int(10 * WORDS_PER_SECOND * 0.85)


# Evidence


def test_pack_numbers_moments_in_time_order_and_keys_inputs():
    pack = build_pack(config(), candidates(), {}, video_name="trip.mp4")
    assert list(pack.aliases) == [f"m{n}" for n in range(1, 9)]
    assert [pack.aliases[a]["id"] for a in pack.aliases] == [f"c{n}" for n in range(8)]
    assert pack.lines["m1"].startswith("m1 [00:00-00:06, 6s]")
    assert "Request: a fun recap" in pack.text() and "m8" in pack.text()
    assert pack.key("local") == build_pack(config(), candidates(), {}, video_name="trip.mp4").key("local")
    assert pack.key("local") != build_pack(config(prompt="serious"), candidates(), {}).key("local")
    assert pack.key("local") != pack.key("claude")
    low, high = pack.beats_range
    assert 2 <= low < high


def test_pack_keeps_the_best_moments_when_limited():
    many = candidates(20)
    pack = build_pack(config(), many, {}, limit=5)
    assert sorted(item["id"] for item in pack.aliases.values()) == sorted(f"c{n}" for n in range(15, 20))
    starts = [pack.aliases[a]["start"] for a in pack.aliases]
    assert starts == sorted(starts)


# Checks


def test_check_resolves_aliases_and_reports_problems():
    pack = build_pack(config(), candidates(), {})
    reply = BriefReply(
        title="Trip",
        beats=[
            BeatReply(clips=["m1", "m99"], purpose="hook", narration="Short line."),
            BeatReply(clips=["m1"], purpose="key", narration="Uses a moment twice."),
            BeatReply(clips=["m3"], purpose="ending", narration=" ".join(["word"] * 40)),
        ],
    )
    brief, problems = check(reply, pack, config(), engine="local", model="m", key="k")
    assert [beat.id for beat in brief.beats] == ["b1", "b2"]
    assert brief.beats[0].clips[0].id == "c0" and brief.beats[0].clips[0].end == 6.0
    assert any("m99" in p for p in problems)
    assert any("no usable moment" in p for p in problems)
    assert any("narration has 40 words" in p for p in problems)
    assert len(brief.beats[1].narration.split()) <= brief.beats[1].max_words == word_budget(6.0)


def test_check_respects_must_include_and_exclude():
    pack = build_pack(config(), candidates(), {})
    cfg = config(must_include=[(41, 44)], must_exclude=[(20, 26)])
    reply = BriefReply(beats=[BeatReply(clips=["m1"]), BeatReply(clips=["m3"]), BeatReply(clips=["m8"])])
    brief, problems = check(reply, pack, cfg, engine="local", model="m", key="k")
    used = [clip.id for beat in brief.beats for clip in beat.clips]
    assert "c2" not in used and "c4" in used
    assert used == sorted(used, key=lambda cid: int(cid[1:]))
    assert any("must be left out" in p for p in problems) and any("must be in the video" in p for p in problems)


def test_check_trims_an_overlong_plan_but_keeps_first_and_last():
    pack = build_pack(config(max_duration=12, min_duration=8), candidates(10, length=8), {})
    reply = BriefReply(beats=[BeatReply(clips=[f"m{n}"]) for n in range(1, 11)])
    brief, problems = check(reply, pack, config(max_duration=12, min_duration=8), engine="local", model="m", key="k")
    assert brief.beats[0].clips[0].id == "c0" and brief.beats[-1].clips[0].id == "c9"
    assert sum(beat.seconds for beat in brief.beats) <= 12 * 1.25 + 8
    assert any("add up to" in p for p in problems)


def test_check_without_any_real_moment_gives_none():
    pack = build_pack(config(), candidates(), {})
    brief, problems = check(BriefReply(beats=[BeatReply(clips=["m77"])]), pack, config(), engine="x", model="m", key="k")
    assert brief is None and problems


def test_shorten_keeps_whole_sentences():
    assert shorten("One two. Three four five. Six.", 5) == "One two. Three four five."
    assert shorten("One two three four five six", 3) == "One two three."
    assert shorten("Anything", 0) == ""


# Local engine


class FakeOllama:
    def __init__(self, writer=None, reachable=True):
        self.calls: list[tuple[dict, str]] = []
        self.writer = writer
        self._reachable = reachable

    def reachable(self):
        return self._reachable

    def role_options(self, role):
        return {"num_ctx": 8192}

    def health(self):
        return {"connected": self._reachable, "models": ["qwen3:8b"], "text_present": True, "warnings": []}

    def chat(self, model, prompt, schema=None, options=None, **kwargs):
        self.calls.append((schema, prompt))
        if schema is DIRECTOR_SCHEMA:
            return {
                "understanding": "a recap",
                "left_out": "",
                "order": "story",
                "clips": [{"id": "m5", "role": "hook", "why": ""}, {"id": "m1", "role": "context", "why": ""}, {"id": "m7", "role": "ending", "why": ""}],
            }
        if schema is WRITER_SCHEMA:
            ids = re.findall(r"Beat (b\d+) of", prompt)
            if self.writer:
                return self.writer(ids, prompt)
            return {
                "title": "Our trip",
                "hook": "You will not believe the view.",
                "music_mood": "uplifting",
                "beats": [{"id": i, "narration": f"Narration for {i}.", "on_screen_text": i.upper()} for i in ids],
            }
        raise AssertionError("unexpected schema")


def test_local_engine_uses_the_editor_then_writes_the_words(tmp_path):
    client = FakeOllama()
    pack = build_pack(config(), candidates(), {})
    reply = LocalEngine(settings(tmp_path), client).write(pack)
    assert [beat.clips for beat in reply.beats] == [["m5"], ["m1"], ["m7"]]
    assert reply.beats[0].purpose == "hook" and reply.beats[-1].purpose == "ending"
    assert reply.title == "Our trip" and reply.music_mood == "uplifting"
    assert reply.beats[1].narration == "Narration for b2." and reply.beats[1].on_screen_text == "B2"


def test_local_engine_reuses_the_editor_plan_of_prompt_mode(tmp_path):
    client = FakeOllama()
    plan = {"order": "time", "clips": [{"id": "c6", "role": "key"}, {"id": "c2", "role": "highlight"}]}
    pack = build_pack(config(), candidates(), {})
    reply = LocalEngine(settings(tmp_path), client, director_plan=plan).write(pack)
    assert [beat.clips for beat in reply.beats] == [["m3"], ["m7"]]
    assert all(schema is not DIRECTOR_SCHEMA for schema, _ in client.calls)


def test_local_engine_without_ollama_writes_a_rules_brief(tmp_path):
    pack = build_pack(config(), candidates(), {})
    reply = LocalEngine(settings(tmp_path), FakeOllama(reachable=False)).write(pack)
    assert reply.beats and all(not beat.narration for beat in reply.beats)
    brief, _ = check(reply, pack, config(), engine="local", model="rules", key="k")
    assert brief is not None


def test_make_brief_repairs_once_then_caches(tmp_path):
    def long_then_short(ids, prompt):
        words = 3 if "Fix these problems" in prompt else 60
        return {"title": "T", "beats": [{"id": i, "narration": " ".join(["word"] * words) + ".", "on_screen_text": ""} for i in ids]}

    client = FakeOllama(writer=long_then_short)
    job = tmp_path / "job"
    job.mkdir()
    cfg = config()
    brief = service.make_brief(job, cfg, candidates(), {}, settings=settings(tmp_path), ollama=client)
    assert all(len(beat.narration.split()) == 3 for beat in brief.beats)
    assert brief.checks == []
    assert (job / "brief.json").is_file()
    calls = len(client.calls)
    again = service.make_brief(job, cfg, candidates(), {}, settings=settings(tmp_path), ollama=client, cached=brief.model_dump())
    assert len(client.calls) == calls and again.model_dump() == brief.model_dump()
    changed = service.make_brief(
        job, config(prompt="something else"), candidates(), {}, settings=settings(tmp_path), ollama=client, cached=brief.model_dump()
    )
    assert len(client.calls) > calls and changed.key != brief.key


def test_rules_fix_what_the_repair_did_not(tmp_path):
    client = FakeOllama(writer=lambda ids, prompt: {"beats": [{"id": i, "narration": " ".join(["long"] * 80), "on_screen_text": ""} for i in ids]})
    job = tmp_path / "job"
    job.mkdir()
    brief = service.make_brief(job, config(), candidates(), {}, settings=settings(tmp_path), ollama=client)
    assert brief.checks and all("narration has" in c for c in brief.checks)
    assert all(len(beat.narration.split()) <= beat.max_words for beat in brief.beats)


# Claude engine


class FakeCli:
    """Stands in for FFmpegRunner.run_streaming, recording the CLI calls."""

    def __init__(self, result: dict | None = None, auth: dict | None = None, code: int = 0, stderr: str = ""):
        self.calls: list[dict] = []
        self.result = result
        self.auth = auth if auth is not None else {"loggedIn": True, "authMethod": "claude.ai", "loginExpiresAt": "2099-01-01T00:00:00Z"}
        self.code = code
        self.stderr = stderr

    def run(self, command):
        raise AssertionError("no FFmpeg expected")

    def run_streaming(self, command, **kwargs):
        self.calls.append({"command": command, **kwargs})
        if command[1:] == ["--version"]:
            return 0, ["2.1.331 (Claude Code)"], ""
        if command[1:] == ["auth", "status"]:
            return 0, [json.dumps(self.auth)], ""
        lines = ['{"type":"system","subtype":"init"}']
        if self.result is not None:
            lines.append(json.dumps(self.result))
        return self.code, lines, self.stderr


@pytest.fixture()
def claude_exe(tmp_path):
    exe = tmp_path / "claude.exe"
    exe.write_text("")
    return str(exe)


def test_claude_command_uses_the_subscription_safely(tmp_path, claude_exe, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    structured = {
        "title": "From Claude",
        "hook": "",
        "tone": "warm",
        "music_mood": "calm",
        "call_to_action": "",
        "beats": [{"clips": ["m2"], "purpose": "hook", "narration": "Hi.", "on_screen_text": ""}],
    }
    cli = FakeCli({"type": "result", "subtype": "success", "is_error": False, "structured_output": structured, "duration_ms": 1200})
    engine = ClaudeCliEngine(settings(tmp_path, claude_cli_path=claude_exe, claude_model="sonnet"), cli)
    pack = build_pack(config(), candidates(), {})
    sheet = tmp_path / "sheet_01.jpg"
    sheet.write_bytes(b"\xff\xd8jpeg")
    pack.sheets = [sheet]
    reply = engine.write(pack, cwd=tmp_path)
    assert reply.title == "From Claude" and reply.beats[0].clips == ["m2"]
    call = cli.calls[-1]
    command = call["command"]
    assert "--bare" not in command
    for flag in ("-p", "--restricted", "--strict-mcp-config", "--no-session-persistence", "--json-schema", "--system-prompt"):
        assert flag in command
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--model") + 1] == "sonnet" and command[command.index("--fallback-model") + 1] == "haiku"
    assert json.loads(command[command.index("--json-schema") + 1]) == REPLY_SCHEMA
    assert "ANTHROPIC_API_KEY" not in call["env"] and call["cwd"] == tmp_path
    message = json.loads(call["stdin_text"])
    kinds = [block["type"] for block in message["message"]["content"]]
    assert kinds.count("image") == 1 and kinds[0] == "text"
    assert "m8" in message["message"]["content"][0]["text"]


@pytest.mark.parametrize(
    "result,reason",
    [
        ({"type": "result", "subtype": "error_during_execution", "is_error": True, "result": "Claude usage limit reached"}, "limit"),
        ({"type": "result", "subtype": "error_max_budget_usd", "is_error": True}, "budget"),
        (None, "login"),
    ],
)
def test_claude_failures_are_explained(tmp_path, claude_exe, result, reason):
    cli = FakeCli(result, code=1, stderr="Not logged in. Please run /login")
    engine = ClaudeCliEngine(settings(tmp_path, claude_cli_path=claude_exe), cli)
    with pytest.raises(BriefEngineError) as error:
        engine.write(build_pack(config(), candidates(), {}))
    assert error.value.reason == reason


def test_claude_timeout_is_a_brief_error(tmp_path, claude_exe):
    class Slow(FakeCli):
        def run_streaming(self, command, **kwargs):
            raise CommandError("claude did not finish within 300 seconds")

    engine = ClaudeCliEngine(settings(tmp_path, claude_cli_path=claude_exe), Slow())
    with pytest.raises(BriefEngineError) as error:
        engine.write(build_pack(config(), candidates(), {}))
    assert error.value.reason == "timeout"


def test_claude_check_reports_login_and_missing_cli(tmp_path, claude_exe):
    out = ClaudeCliEngine(settings(tmp_path, claude_cli_path=claude_exe), FakeCli(auth={"loggedIn": False})).check(refresh=True)
    assert not out.ready and "not logged in" in out.problems[0]
    ok = ClaudeCliEngine(settings(tmp_path, claude_cli_path=claude_exe), FakeCli()).check(refresh=True)
    assert ok.ready and any("subscription" in note for note in ok.notes) and ok.details["version"].startswith("2.1")
    expired = FakeCli(auth={"loggedIn": True, "authMethod": "claude.ai", "loginExpiresAt": "2001-01-01T00:00:00Z"})
    assert not ClaudeCliEngine(settings(tmp_path, claude_cli_path=claude_exe), expired).check(refresh=True).ready
    missing = ClaudeCliEngine(settings(tmp_path, claude_cli_path=str(tmp_path / "nope" / "claude-missing")), FakeCli()).check(refresh=True)
    assert not missing.ready and "not installed" in missing.problems[0]


def test_parse_result_takes_the_last_result_event():
    lines = ["noise", '{"type":"assistant"}', '{"type":"result","subtype":"success","n":1}', "{broken"]
    assert parse_result(lines)["n"] == 1
    assert parse_result(["{}", "text"]) is None


def test_make_brief_with_claude_not_ready_raises(tmp_path, claude_exe):
    job = tmp_path / "job"
    job.mkdir()
    engine_settings = settings(tmp_path, claude_cli_path=claude_exe, claude_send_frames=False)
    import app.brief.engines.claude_cli as claude_cli

    original = claude_cli.ClaudeCliEngine.__init__

    def init(self, settings, runner=None):
        original(self, settings, FakeCli(auth={"loggedIn": False}))

    claude_cli.ClaudeCliEngine.__init__ = init
    try:
        with pytest.raises(BriefEngineError) as error:
            service.make_brief(job, config(brief_engine="claude"), candidates(), {}, settings=engine_settings)
    finally:
        claude_cli.ClaudeCliEngine.__init__ = original
    assert error.value.reason == "not_ready"
    assert not (job / "brief.json").exists()


def test_pipeline_pauses_when_claude_fails(tmp_path, monkeypatch):
    from app.analysis import pipeline
    from app.video.ffmpeg_runner import FFmpegRunner

    def fail(*args, **kwargs):
        raise BriefEngineError("Claude's usage limit was reached.", reason="limit")

    monkeypatch.setattr(pipeline, "make_brief", fail)
    analyzer = pipeline.VideoAnalyzer(settings(tmp_path), FFmpegRunner(), None)
    progress = pipeline.PipelineProgress(lambda kind, payload: None)
    job = tmp_path / "job"
    job.mkdir()
    with pytest.raises(BriefPaused) as error:
        analyzer._brief(job, config(brief_engine="claude"), candidates(), {}, {}, progress, [])
    assert error.value.reason == "limit"
    assert service.read_failure(job)["message"].startswith("Claude's usage limit")


def test_engines_status_never_chooses(tmp_path, monkeypatch):
    import app.brief.engines.local as local

    monkeypatch.setattr(local.LocalEngine, "check", lambda self, refresh=False: local.EngineStatus("local", "Local", True))
    statuses = engines_status(settings(tmp_path, claude_cli_path=str(tmp_path / "missing")), refresh=True)
    assert [s["name"] for s in statuses] == ["local", "claude"]
    assert statuses[0]["ready"] and not statuses[1]["ready"]


# Applying and editing


def test_apply_brief_ranks_its_moments_first_in_story_order():
    brief = VideoBrief.model_validate(
        {
            "beats": [
                {"id": "b1", "purpose": "hook", "clips": [{"id": "c5", "start": 50, "end": 56}]},
                {"id": "b2", "purpose": "key", "clips": [{"id": "c1", "start": 10, "end": 16}]},
            ]
        }
    )
    ranked = service.apply_brief(candidates(), brief)
    assert [c["id"] for c in ranked[:2]] == ["c5", "c1"]
    assert ranked[0]["director_order"] == 0 and ranked[1]["director_order"] == 1 and ranked[0]["role"] == "hook"
    assert service.apply_brief(candidates(), None) == candidates()
    assert service.planning_config(config()).order_mode == "ai"
    assert service.planning_config(config(order_mode="hook")).order_mode == "hook"
    assert service.planning_config(config(video_brief=False)).order_mode == "chronological"


def test_narration_turns_on_the_brief():
    assert HighlightConfig(narration=True).video_brief is True
    assert HighlightConfig().video_brief is False and HighlightConfig().brief_engine == "local"


def test_update_brief_edits_text_and_warns(tmp_path):
    brief = VideoBrief.model_validate(
        {
            "title": "Old",
            "beats": [{"id": "b1", "max_words": 5, "narration": "a b", "clips": [{"id": "c1", "start": 0, "end": 3}]}],
        }
    )
    service.save_brief(tmp_path, brief)
    updated, warnings = service.update_brief(
        tmp_path, {"title": "  New   title ", "beats": [{"id": "b1", "narration": "one two three four five six seven"}, {"id": "zz"}]}
    )
    assert updated.title == "New title" and updated.edited
    assert warnings and "7 words" in warnings[0]
    assert service.load_brief(tmp_path).beats[0].narration.startswith("one two")
    with pytest.raises(LookupError):
        service.update_brief(tmp_path / "missing", {})


def test_voice_over_requests_narration_and_screen_text_stays_short():
    voiced = build_pack(config(narration=True), candidates(), {})
    assert voiced.voice_over and "A voice-over is wanted" in voiced.text()
    silent = build_pack(config(), candidates(), {})
    assert not silent.voice_over and "No voice-over" in silent.text() and silent.key("x") != voiced.key("x")
    reply = BriefReply(
        beats=[
            BeatReply(clips=["m1"], on_screen_text="one two three four five six seven eight"),
            BeatReply(clips=["m2"]),
            BeatReply(clips=["m3"], narration="Only one line."),
        ]
    )
    brief, problems = check(reply, voiced, config(narration=True), engine="claude", model="m", key="k")
    assert brief.beats[0].on_screen_text == "one two three four five six"
    assert any("most beats have no narration" in p for p in problems)
    _, quiet = check(reply, silent, config(), engine="claude", model="m", key="k")
    assert not any("no narration" in p for p in quiet)


def test_too_many_picks_join_only_time_neighbours(tmp_path):
    class ManyPicks(FakeOllama):
        def chat(self, model, prompt, schema=None, options=None, **kwargs):
            if schema is DIRECTOR_SCHEMA:
                ids = ["m5", "m1", "m2", "m8", "m3", "m6", "m7", "m4"]
                return {"order": "story", "clips": [{"id": i, "role": "highlight"} for i in ids]}
            return super().chat(model, prompt, schema=schema, options=options, **kwargs)

    pack = build_pack(config(max_duration=24, min_duration=20), candidates(), {})
    _low, high = pack.beats_range
    reply = LocalEngine(settings(tmp_path), ManyPicks()).write(pack)
    assert len(reply.beats) <= high
    order = list(pack.aliases)
    for beat in reply.beats:
        positions = [order.index(a) for a in beat.clips]
        assert positions == list(range(positions[0], positions[0] + len(positions))), beat.clips


def test_a_cut_off_writer_reply_is_asked_again_in_halves(tmp_path):
    from app.ai.ollama import OllamaTruncatedError

    class Truncating(FakeOllama):
        def chat(self, model, prompt, schema=None, options=None, **kwargs):
            if schema is WRITER_SCHEMA and len(re.findall(r"Beat (b\d+) of", prompt)) > 1:
                self.calls.append((schema, prompt))
                raise OllamaTruncatedError("cut off")
            return super().chat(model, prompt, schema=schema, options=options, **kwargs)

    pack = build_pack(config(narration=True), candidates(), {})
    reply = LocalEngine(settings(tmp_path), Truncating()).write(pack)
    assert all(beat.narration.startswith("Narration for") for beat in reply.beats)
    assert reply.title == "Our trip"


def test_local_repair_rewrites_only_the_beats_that_need_it(tmp_path):
    client = FakeOllama()
    pack = build_pack(config(narration=True), candidates(), {})
    previous = BriefReply(
        title="Kept",
        hook="Kept hook",
        beats=[BeatReply(clips=["m1"], narration="Fine words."), BeatReply(clips=["m2"], narration=""), BeatReply(clips=["m3"], narration=" ".join(["w"] * 40))],
    )
    fixed = LocalEngine(settings(tmp_path), client).write(pack, repair=(previous, ["problem"]))
    asked = re.findall(r"Beat (b\d+) of", client.calls[-1][1])
    assert asked == ["b2", "b3"] and fixed.title == "Kept"
    assert fixed.beats[0].narration == "Fine words." and fixed.beats[1].narration == "Narration for b2."


def test_hook_opens_and_ending_closes():
    pack = build_pack(config(), candidates(), {})
    reply = BriefReply(
        beats=[BeatReply(clips=["m1"], purpose="key"), BeatReply(clips=["m2"], purpose="ending"), BeatReply(clips=["m3"], purpose="hook"), BeatReply(clips=["m4"], purpose="highlight")]
    )
    brief, _ = check(reply, pack, config(), engine="local", model="m", key="k")
    assert [b.purpose for b in brief.beats] == ["hook", "key", "highlight", "ending"]
