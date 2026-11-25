"""Story editing and analysis speed: lengths that describe the source, unreliable transcripts,
cuts at pauses, the AI editor's outline and story shape, the planner's handling of its picks,
joined neighbours, candidate coverage, the analysis copy and the vision shortlist size."""

from __future__ import annotations

import pytest

from app.ai.intent import read_rules, _patch_from_ai
from app.analysis import director
from app.analysis.boundaries import CutPoints
from app.analysis.candidates import keep_with_coverage
from app.analysis.director import _fix_roles, _shape, _valid_outline, build_chapters, run_director
from app.analysis.optimizer import ClipOptimizer, _join_neighbours
from app.analysis.pipeline import _highlight_seconds, _needs_proxy, director_engine
from app.analysis.sequence_limits import fit_candidate, keep_best_span
from app.analysis.speech_check import check
from app.brief.engines.base import BriefEngineError
from app.config import Settings
from app.models.schemas import HighlightConfig, Transcript, TranscriptSegment, WordTimestamp
from app.video.processor import VideoProcessor


# Lengths in the description


@pytest.mark.parametrize(
    "text",
    [
        "Make a short video from the original 17-minute video.",
        "Cut my 45 minute podcast down to the best parts",
        "Summarize this 20-minute review for someone in a hurry",
        "Highlights from the whole 2 hour stream",
    ],
)
def test_a_length_that_describes_the_source_is_not_the_wanted_length(text):
    patch = read_rules(text, duration=1025)
    assert "max_duration" not in patch and "min_duration" not in patch


@pytest.mark.parametrize(
    "text, high",
    [
        ("Make a 30 second video of the best moments", 30),
        ("Under 60 seconds please", 60),
        ("A 2 minute highlight of the original 17-minute video", 120),
        ("Summarize it in 45 seconds", 45),
    ],
)
def test_a_wanted_length_is_still_read(text, high):
    patch = read_rules(text, duration=1025)
    assert patch["max_duration"] == high and patch["length_mode"] == "range"


def test_as_short_as_possible_means_strict_automatic_length():
    patch = read_rules("Make the final video as short as possible without losing the essentials", duration=1025)
    assert patch["length_mode"] == "auto" and patch["auto_length_strict"] is True
    assert HighlightConfig(**{k: v for k, v in patch.items()}).auto_length_strict


def test_a_length_about_as_long_as_the_video_is_ignored():
    assert "max_duration" not in read_rules("make it 17 minutes", duration=1025)
    patch, _sources = _patch_from_ai({"min_seconds": 900, "max_seconds": 1020}, 1025)
    assert "max_duration" not in patch
    patch, _sources = _patch_from_ai({"min_seconds": 30, "max_seconds": 60}, 1025)
    assert patch["max_duration"] == 60


# Transcript reliability


def _transcript(texts, language="ml", probability=None):
    return Transcript(
        segments=[TranscriptSegment(start=i * 3.0, end=i * 3.0 + 2.5, text=t) for i, t in enumerate(texts)],
        language=language,
        language_probability=probability,
    )


def test_a_garbled_transcript_is_unreliable():
    garbled = [
        "Ladies and gentlemen, Vivo X500 Pro Max கiddillum phone",
        "ஓட்டு வேண்டும் VS1COP PROCESSA ஓPTIMAISD for this particular device",
        "எந்து ஒன்று கiddillum phoneுடன் என்றை 12GB RAMを足りない because",
        "v پڑ into their scope of life இந்தwoobarch is absolutely nothing",
        "தரம்ஞஞஞஞஞஞஞஞஞஞ मकी गान mr",
    ] * 3
    result = check(_transcript(garbled))
    assert result["reliable"] is False and result["reason"]


def test_clean_transcripts_are_reliable():
    english = ["This phone has a large camera sensor.", "The battery lasts two full days.", "Gaming is smooth at 120 frames."] * 5
    assert check(_transcript(english, "en", 0.98))["reliable"]
    malayalam = ["ഈ ഫോണിന്റെ ക്യാമറ വളരെ നല്ലതാണ്", "ബാറ്ററി രണ്ട് ദിവസം നിൽക്കും", "ഗെയിമിംഗ് വളരെ സുഗമമാണ് 120 Hz display"] * 5
    assert check(_transcript(malayalam, "ml", 0.9))["reliable"]
    assert check(Transcript())["reliable"]


def test_the_spoken_language_is_cleaned():
    assert HighlightConfig(speech_language=" ML ").speech_language == "ml"
    assert HighlightConfig(speech_language="malayalam!").speech_language == ""


# Cuts at pauses and sentence ends


def _words(*spans):
    return [WordTimestamp(word=w, start=s, end=e) for w, s, e in spans]


def test_cut_points_come_from_pauses_sentence_ends_and_scenes():
    transcript = Transcript(
        words=_words(("Hello", 0.0, 0.4), ("there.", 0.5, 0.9), ("Next", 1.0, 1.3), ("bit", 2.0, 2.3)),
    )
    cuts = CutPoints(transcript, [{"start": 5.0, "end": 6.0}], [{"start": 8.0}])
    times = dict((round(t, 2), w) for t, w in cuts.between(0, 10))
    assert times[0.95] == 1.0, "sentence end"
    assert times[1.65] == 1.0, "long pause between words"
    assert 5.15 in times and 5.85 in times and 8.0 in times
    assert all(0 < t < 10 for t in times)


def test_a_shortened_clip_starts_and_ends_at_cut_points():
    cuts = [[3.2, 1.0], [6.1, 0.5], [9.4, 1.0], [12.7, 1.0]]
    start, end = keep_best_span(0.0, 20.0, 8.0, peak=7.0, cuts=cuts, target=7.0)
    assert (start, end) in {(3.2, 9.4)}, (start, end)
    # Without cuts the old time-based cut is kept.
    assert keep_best_span(0.0, 20.0, 8.0, peak=7.0) == (3.0, 11.0)


def test_fit_candidate_uses_the_moments_cut_points():
    candidate = {"start": 0.0, "end": 20.0, "peak": 7.0, "cuts": [[3.2, 1.0], [9.4, 1.0], [12.7, 1.0]], "text": "x"}
    fitted = fit_candidate(candidate, cap=7.0, mode="custom")
    assert (fitted["start"], fitted["end"]) == (3.2, 9.4)


def test_spoken_moments_are_not_cut_below_a_few_seconds():
    candidate = {
        "start": 0.0,
        "end": 6.0,
        "text": "we have optimized the chipset for this device",
        "metrics": {"speech_presence": 0.9, "hook_strength": 0.9, "importance": 0.3},
    }
    fitted = fit_candidate(candidate, cap=30.0, mode="auto")
    assert fitted["end"] - fitted["start"] >= 2.5


# Candidates


def test_candidate_cap_keeps_the_best_moment_of_every_stretch():
    strong = [{"id": f"s{i}", "start": 10.0 + i, "end": 14.0 + i, "score": 0.9} for i in range(20)]
    weak = [{"id": f"w{i}", "start": 200.0 + 70 * i, "end": 204.0 + 70 * i, "score": 0.3} for i in range(5)]
    kept = keep_with_coverage(strong + weak, 20, 600)
    assert len(kept) == 20
    assert {c["id"] for c in kept} >= {"w0", "w1", "w2", "w3", "w4"}
    assert kept == sorted(kept, key=lambda c: c["score"], reverse=True)
    assert len(keep_with_coverage(strong, 50, 600)) == 20


# The planner and the AI editor's picks


def _moment(cid, start, end, score=0.6, **extra):
    return {"id": cid, "start": start, "end": end, "score": score, "text": cid, "source_id": "v0", **extra}


def test_other_moments_only_fill_up_to_the_minimum_when_the_editor_chose():
    picks = [
        _moment("a", 10, 18, 0.92, director_rank=0, director_order=0, role="hook"),
        _moment("b", 100, 110, 0.91, director_rank=1, director_order=1, role="ending"),
    ]
    others = [_moment(f"o{i}", 200 + 20 * i, 208 + 20 * i, 0.6) for i in range(10)]
    plan = ClipOptimizer().select(picks + others, 10, 120, order_mode="ai", max_sequence_duration=30, sequence_duration_mode="custom")
    assert [c.candidate_id for c in plan.clips] == ["a", "b"], "the minimum is met by the picks alone"


def test_moments_added_to_reach_the_minimum_go_inside_the_story():
    picks = [
        _moment("hook", 300, 306, 0.92, director_rank=0, director_order=0, role="hook"),
        _moment("key", 100, 106, 0.91, director_rank=1, director_order=1, role="key"),
        _moment("end", 500, 506, 0.9, director_rank=2, director_order=2, role="ending"),
    ]
    filler = [_moment("after_key", 120, 128, 0.6), _moment("after_end", 520, 528, 0.59), _moment("early", 5, 13, 0.58)]
    plan = ClipOptimizer().select(picks + filler, 40, 60, order_mode="ai", max_sequence_duration=10, sequence_duration_mode="custom")
    order = [c.candidate_id for c in plan.clips]
    assert order[0] == "hook" and order[-1] == "end", order
    assert order.index("after_key") == order.index("key") + 1
    assert order.index("after_end") < order.index("end")


def test_neighbouring_clips_are_joined():
    clips = [_moment("a", 10, 15), _moment("b", 15.8, 20), _moment("c", 40, 45), _moment("d", 45.5, 50, source_id="v1")]
    joined = _join_neighbours(clips, max_duration=60)
    assert [(c["start"], c["end"]) for c in joined] == [(10, 20), (40, 45), (45.5, 50)]
    assert "joined" in joined[0]["duration_reason"]
    # Not when the footage in between would pass the maximum length.
    assert len(_join_neighbours(clips[:2], max_duration=9.3)) == 2
    # Not when the plan goes back in time.
    assert len(_join_neighbours([_moment("b", 15.8, 20), _moment("a", 10, 15)], 60)) == 2


# The AI editor


def _pool(count=40, spacing=10.0):
    return [
        {"id": f"v0:t{i}", "start": i * spacing, "end": i * spacing + 6, "score": 0.5 + (i % 5) / 10, "text": f"line {i}", "source_id": "v0"}
        for i in range(count)
    ]


def test_chapters_split_the_video_and_cover_every_moment():
    pool = _pool(40, 10.0)
    chapters, chapter_of = build_chapters(pool, {})
    assert 3 <= len(chapters) <= 12
    assert set(chapter_of) == {c["id"] for c in pool}
    assert [c["start"] for c in chapters] == sorted(c["start"] for c in chapters)
    assert all(c["line"].startswith(c["id"]) for c in chapters)


def test_chapters_hide_words_of_an_unreliable_transcript():
    chapters, _ = build_chapters(_pool(40), {"v0": {"scenes": [], "sounds": {}, "speech_reliable": False}})
    assert all("says:" not in c["line"] for c in chapters)
    line = director._line("m1", _pool(1)[0], {"v0": {"speech_reliable": False}})
    assert "says:" not in line


def test_an_outline_opens_with_a_hook_and_closes_with_an_ending():
    beats = _valid_outline(
        {"beats": [
            {"chapter": "c2", "role": "context", "purpose": "setup", "seconds": 8},
            {"chapter": "3", "role": "hook", "purpose": "x", "seconds": 5},
            {"chapter": "c9", "role": "key", "purpose": "unknown chapter", "seconds": 5},
            {"chapter": "c4", "role": "key", "purpose": "result", "seconds": 6},
        ]},
        {"c1", "c2", "c3", "c4"},
    )
    assert [(b["chapter"], b["role"]) for b in beats] == [("c2", "hook"), ("c3", "key"), ("c4", "ending")]


def test_shape_fills_empty_beats_and_follows_the_outline():
    numbered = {f"m{i}": {"start": float(i * 10), "score": 0.5 + i / 100} for i in range(1, 7)}
    chapter = {"m1": "c1", "m2": "c1", "m3": "c2", "m4": "c2", "m5": "c3", "m6": "c3"}
    outline = [
        {"chapter": "c3", "role": "hook", "purpose": "the result", "seconds": 5},
        {"chapter": "c1", "role": "context", "purpose": "setup", "seconds": 5},
        {"chapter": "c2", "role": "ending", "purpose": "wrap up", "seconds": 5},
    ]
    chosen = [{"id": "m1", "role": "context", "why": ""}, {"id": "m5", "role": "key", "why": ""}]
    shaped = _shape(chosen, outline, chapter, list(numbered), numbered, story=True)
    assert [c["id"] for c in shaped] == ["m5", "m1", "m4"], "c2 got its best moment; outline order"
    assert shaped[0]["role"] == "hook" and shaped[-1]["role"] == "ending"


def test_fix_roles_keeps_one_hook_and_one_ending():
    chosen = [
        {"id": "a", "role": "key"},
        {"id": "b", "role": "ending"},
        {"id": "c", "role": "hook"},
        {"id": "d", "role": "hook"},
        {"id": "e", "role": "key"},
    ]
    _fix_roles(chosen)
    assert [c["id"] for c in chosen] == ["c", "a", "d", "e", "b"]
    assert [c["role"] for c in chosen] == ["hook", "key", "highlight", "key", "ending"]


class StoryClient:
    """A local model that answers the outline, the picks and the review."""

    def __init__(self, review=None):
        self.calls: list[str] = []
        self.review = review or {"remove": [], "add": [], "notes": "fine"}

    def role_options(self, role):
        return {"num_ctx": 16384}

    def chat(self, model, prompt, images=None, schema=None, options=None, schema_first=True, think=False):
        import re

        self.calls.append(prompt)
        if "planning the story" in prompt:
            chapters = re.findall(r"^(c\d+) \[", prompt, re.MULTILINE)
            return {
                "understanding": "a story",
                "beats": [
                    {"chapter": chapters[0], "role": "hook", "purpose": "open", "seconds": 6},
                    {"chapter": chapters[1], "role": "key", "purpose": "middle", "seconds": 6},
                    {"chapter": chapters[-1], "role": "ending", "purpose": "close", "seconds": 6},
                ],
            }
        if "checking a cut" in prompt:
            return self.review
        ids = re.findall(r"^(m\d+)(?: \(c\d+\))? \[", prompt, re.MULTILINE)
        return {"understanding": "picked", "order": "story", "clips": [{"id": ids[0], "role": "key", "why": "first"}]}


def _story_config():
    return HighlightConfig(mode="prompt", prompt="tell the story", min_duration=20, max_duration=40)


def test_the_editor_plans_an_outline_then_fills_every_beat():
    client = StoryClient()
    plan = run_director(client, "m", Settings(_env_file=None), _story_config(), _pool(40), {})
    assert "planning the story" in client.calls[0] and "The story outline" in client.calls[1]
    assert plan["order"] == "story" and len(plan["outline"]) == 3
    assert len(plan["clips"]) >= 3, "beats without a pick get their chapter's best moment"
    assert plan["clips"][0]["role"] == "hook" and plan["clips"][-1]["role"] == "ending"
    assert plan["engine"] == "local" and plan["version"] == director.VERSION


def test_the_review_can_add_missing_context():
    pool = _pool(40)
    client = StoryClient()
    first = run_director(client, "m", Settings(_env_file=None), _story_config(), pool, {})
    first_ids = [c["id"] for c in first["clips"]]
    # The moment right before the second pick (m-numbers follow the time order of the pool).
    numbered = {c["id"]: f"m{i}" for i, c in enumerate(pool, start=1)}
    target = numbered[first_ids[1]]
    before = f"m{int(target[1:]) - 1}"
    reviewing = StoryClient(review={"remove": [], "add": [{"id": before, "before": target, "why": "setup"}], "notes": "ok"})
    plan = run_director(reviewing, "m", Settings(_env_file=None, director_review=True), _story_config(), pool, {})
    assert any("checking a cut" in call for call in reviewing.calls)
    ids = [c["id"] for c in plan["clips"]]
    added = pool[int(before[1:]) - 1]["id"]
    if added not in first_ids:
        assert ids.index(added) == ids.index(first_ids[1]) - 1
    assert plan["review"] == "ok"


class FailingClaude:
    def model_name(self):
        return "opus"

    def _ask(self, content, schema, system, cwd=None):
        raise BriefEngineError("Claude's usage limit was reached.", reason="limit")


def test_claude_as_the_editor_raises_instead_of_switching(tmp_path):
    with pytest.raises(BriefEngineError):
        run_director(None, "m", Settings(_env_file=None), _story_config(), _pool(40), {}, engine="claude", claude=FailingClaude(), claude_cwd=tmp_path)


class AnsweringClaude(FailingClaude):
    def __init__(self):
        self.client = StoryClient()

    def _ask(self, content, schema, system, cwd=None):
        return {"structured_output": self.client.chat("claude", content[0]["text"])}


def test_claude_as_the_editor_reads_everything_at_once(tmp_path):
    claude = AnsweringClaude()
    plan = run_director(None, "m", Settings(_env_file=None), _story_config(), _pool(200), {}, engine="claude", claude=claude, claude_cwd=tmp_path)
    assert plan["engine"] == "claude" and plan["model"] == "claude:opus"
    assert not any("one part of a longer video" in call for call in claude.client.calls)


def test_the_editor_engine_comes_from_the_job_or_settings():
    settings = Settings(_env_file=None, director_engine="claude")
    assert director_engine(HighlightConfig(), settings) == "claude"
    assert director_engine(HighlightConfig(director_engine="local"), settings) == "local"
    assert director_engine(HighlightConfig(), Settings(_env_file=None)) == "local"


# Speed: the analysis copy and the vision shortlist


def test_large_or_slow_videos_get_an_analysis_copy():
    assert _needs_proxy({"width": 3840, "height": 2160, "duration": 60, "video_codec": "h264"})
    assert _needs_proxy({"width": 1280, "height": 720, "duration": 60, "video_codec": "av1"})
    assert not _needs_proxy({"width": 1280, "height": 720, "duration": 60, "video_codec": "h264"})
    assert not _needs_proxy({"width": 640, "height": 360, "duration": 60, "video_codec": "av1"})
    assert not _needs_proxy({"width": 0, "height": 0, "duration": 60})


def test_the_analysis_copy_keeps_every_frame_time_and_drops_sound(tmp_path):
    cmd = VideoProcessor.analysis_proxy_command(tmp_path / "in.mp4", tmp_path / "out.mp4", 540)
    assert cmd[cmd.index("-fps_mode") + 1] == "passthrough"
    assert "-an" in cmd and cmd[cmd.index("-map") + 1] == "0:v:0"
    assert "540" in cmd[cmd.index("-vf") + 1]


def test_the_shortlist_is_sized_for_a_realistic_highlight():
    misread = HighlightConfig(min_duration=30, max_duration=1020)
    assert _highlight_seconds(misread, 1025) == pytest.approx(1025 * 0.4)
    assert _highlight_seconds(HighlightConfig(min_duration=30, max_duration=60), 1025) == 45
    assert _highlight_seconds(HighlightConfig(min_duration=30, length_mode="auto"), 1025) == pytest.approx(1025 * 0.25)
    assert _highlight_seconds(HighlightConfig(min_duration=30, max_duration=60), 50) == 30
