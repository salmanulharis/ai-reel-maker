"""Subtitles: per-clip timing for burning in, styled word-by-word captions, and the highlight SRT."""

import json

import pytest
from pathlib import Path

from app.models.schemas import ClipPlan, ClipPlanItem, HighlightConfig, RenderSettings, VideoMetadata
from app.video.captions import (
    ass_text,
    clip_cues,
    load_transcript,
    sequence_srt,
    srt_text,
    write_clip_subtitles,
)
from app.video.renderer import Renderer


def _words(*items):
    return [{"word": f" {text}", "start": start, "end": end} for text, start, end in items]


TRANSCRIPT = {
    "segments": [
        {"start": 10.0, "end": 12.0, "text": " Hello there everyone. "},
        {"start": 30.0, "end": 31.5, "text": "Second clip words"},
    ],
    "words": _words(
        ("Hello", 10.0, 10.4),
        ("there", 10.5, 10.9),
        ("everyone.", 11.0, 11.6),
        ("Next", 11.7, 11.9),
        ("Second", 30.0, 30.4),
        ("clip", 30.5, 30.8),
        ("words", 30.9, 31.5),
    ),
}


def test_plain_cues_are_whole_segments_timed_from_the_clip_start():
    cues = clip_cues(TRANSCRIPT, 30.0, 32.0, "plain")
    assert [(c.start, c.end, c.text) for c in cues] == [(0.0, 1.5, "Second clip words")]
    # A segment that runs past the clip is cut at the clip's end.
    cues = clip_cues(TRANSCRIPT, 9.0, 11.0, "plain")
    assert [(c.start, c.end) for c in cues] == [(1.0, 2.0)]


def test_word_cues_break_after_a_sentence_and_when_full():
    cues = clip_cues(TRANSCRIPT, 10.0, 12.0, "clean")
    assert [c.text for c in cues] == ["Hello there everyone.", "Next"]
    assert cues[0].start == 0.0
    # Lingers a moment after the last word, but not into the next line.
    assert cues[0].end == pytest.approx(1.7)
    portrait = clip_cues(
        {"words": _words(*[(f"w{i}", 1 + i * 0.3, 1.25 + i * 0.3) for i in range(9)])}, 0, 10, "bold", portrait=True
    )
    assert [len(c.words) for c in portrait] == [4, 4, 1]


def test_styled_without_word_times_falls_back_to_segments():
    cues = clip_cues({"segments": TRANSCRIPT["segments"]}, 10.0, 12.0, "boxed")
    assert [c.text for c in cues] == ["Hello there everyone."]
    assert cues[0].words == []


def test_bad_transcript_entries_are_skipped():
    transcript = {
        "segments": [{"start": "x", "end": 2, "text": "a"}, "junk", {"start": 0, "end": 1, "text": "  "}],
        "words": [{"word": "a"}, {"word": "ok", "start": 0.1, "end": 0.4}, None],
    }
    assert [c.text for c in clip_cues(transcript, 0, 5, "clean")] == ["ok"]
    assert clip_cues(transcript, 0, 5, "plain") == []


def test_ass_highlights_the_spoken_word_and_escapes_text():
    cues = clip_cues(
        {"words": _words(("{odd}", 0.0, 0.4), ("back\\slash", 0.5, 0.9))}, 0, 2, "bold"
    )
    text = ass_text(cues, "bold", 1080, 1920)
    assert "PlayResX: 1080" in text and "PlayResY: 1920" in text
    events = [line for line in text.splitlines() if line.startswith("Dialogue:")]
    assert len(events) == 2
    assert events[0].startswith("Dialogue: 0,0:00:00.00,0:00:00.50,")
    assert r"{\c&H0000D7FF&\fscx112\fscy112}(ODD){\r}" in events[0]
    assert "BACK∖SLASH" in events[1]
    assert "{odd}" not in text
    # Portrait captions sit higher, clear of on-screen buttons.
    style = next(line for line in text.splitlines() if line.startswith("Style: "))
    assert style.endswith(f",2,65,65,{round(1920 * 0.2)},1")


def test_boxed_uses_an_opaque_box():
    text = ass_text(clip_cues(TRANSCRIPT, 30, 32, "boxed"), "boxed", 1920, 1080)
    style = next(line for line in text.splitlines() if line.startswith("Style: ")).split(",")
    assert style[15] == "3"  # BorderStyle


def test_write_clip_subtitles(tmp_path: Path):
    plain = write_clip_subtitles(tmp_path, 0, TRANSCRIPT, 30.0, 32.0, "plain", 1920, 1080)
    assert plain.name == "clip_000.srt"
    assert plain.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,500\nSecond clip words\n"
    styled = write_clip_subtitles(tmp_path, 1, TRANSCRIPT, 30.0, 32.0, "clean", 1920, 1080)
    assert styled.name == "clip_001.ass"
    assert write_clip_subtitles(tmp_path, 2, TRANSCRIPT, 50.0, 52.0, "clean", 1920, 1080) is None
    assert write_clip_subtitles(tmp_path, 3, TRANSCRIPT, 30.0, 32.0, "unknown", 1920, 1080).suffix == ".srt"


def test_sequence_srt_places_each_clip_after_the_previous():
    other = {"segments": [{"start": 5.0, "end": 6.0, "text": "From video two"}]}
    text = sequence_srt([(TRANSCRIPT, 10.0, 12.0), (other, 4.0, 7.0), (TRANSCRIPT, 30.0, 32.0)])
    assert text == (
        "1\n00:00:00,000 --> 00:00:02,000\nHello there everyone.\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nFrom video two\n\n"
        "3\n00:00:05,000 --> 00:00:06,500\nSecond clip words\n"
    )
    assert sequence_srt([({}, 0, 5)]) == ""
    assert srt_text([]) == ""


def test_load_transcript_per_video(tmp_path: Path):
    (tmp_path / "transcript.json").write_text(json.dumps({"segments": [{"text": "main"}]}), encoding="utf-8")
    (tmp_path / "work" / "v1").mkdir(parents=True)
    (tmp_path / "work" / "v1" / "transcript.json").write_text(json.dumps({"segments": [{"text": "second"}]}), encoding="utf-8")
    assert load_transcript(tmp_path)["segments"][0]["text"] == "main"
    assert load_transcript(tmp_path, "v1", multi=True)["segments"][0]["text"] == "second"
    assert load_transcript(tmp_path, "v2", multi=True) == {}
    (tmp_path / "transcript.json").write_text("{broken", encoding="utf-8")
    assert load_transcript(tmp_path) == {}


def test_caption_style_defaults_to_the_original_look():
    assert HighlightConfig().caption_style == "plain"
    assert HighlightConfig.model_validate({"subtitles": "burned"}).caption_style == "plain"


def test_renderer_burns_each_clips_own_subtitles(tmp_path: Path):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x" * 2000)
    commands: list[list[str]] = []

    class FakeRunner:
        def run(self, cmd):
            commands.append(cmd)
            Path(cmd[-1]).write_bytes(b"x" * 2000)

        def probe(self, path):
            return VideoMetadata(duration=4, width=1280, height=720, fps=30, has_audio=True, size_bytes=2000)

    asked = []

    def subtitles(index, clip, canvas):
        asked.append((index, clip.start, canvas.width, canvas.height))
        if index == 1:
            return None
        path = tmp_path / f"sub_{index}.ass"
        path.write_text("x", encoding="utf-8")
        return path

    Renderer(FakeRunner()).render(
        source,
        ClipPlan(
            target_duration=4,
            estimated_duration=4,
            clips=[
                ClipPlanItem(start=0, end=2, reason="a"),
                ClipPlanItem(start=5, end=7, reason="b"),
                ClipPlanItem(start=9, end=11, reason="c"),
            ],
        ),
        tmp_path / "out.mp4",
        RenderSettings(subtitles="burned"),
        tmp_path / "work",
        clip_subtitles=subtitles,
    )
    assert asked == [(0, 0, 1280, 720), (1, 5, 1280, 720), (2, 9, 1280, 720)]
    cuts = [cmd for cmd in commands if any(str(part).endswith(".mp4") and "clip_" in str(part) for part in cmd[-1:])]
    filters = [cmd[cmd.index("-vf") + 1] for cmd in cuts]
    assert "sub_0.ass" in filters[0] and "subtitles=" in filters[0]
    assert "subtitles=" not in filters[1]
    assert "sub_2.ass" in filters[2]
