"""Narrated videos: beats on the timeline, fitting the voice, the shared spec, the FFmpeg commands,
and composing a finished video (without Remotion or a real FFmpeg)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.analysis.progress import RenderProgress
from app.brief.schema import VideoBrief
from app.compose import compose
from app.compose.render_spec import (
    LEAD_IN,
    SCHEMA_PATH,
    SpecVideo,
    assign_beats,
    build_spec,
    fit_narration,
    json_schema,
    narration_srt,
    preview_video,
    spec_ass,
)
from app.config import Settings
from app.models.schemas import ClipPlanItem, HighlightConfig, VideoMetadata
from app.video.audio_mix import narration_mix_command

FIXTURES = Path(__file__).parent / "fixtures"


def clip(start, end, source="v0"):
    return ClipPlanItem(start=start, end=end, reason="", source_id=source)


BRIEF = VideoBrief.model_validate(
    {
        "title": "Our trip",
        "call_to_action": "Follow for more",
        "beats": [
            {"id": "b1", "purpose": "hook", "on_screen_text": "Day one", "clips": [{"id": "c1", "start": 10, "end": 14}]},
            {"id": "b2", "purpose": "key", "clips": [{"id": "c2", "start": 30, "end": 36}, {"id": "c3", "start": 40, "end": 44}]},
        ],
    }
)
LINES = {
    "b1": {"file": "aaaa.wav", "duration": 2.0, "words": [{"text": "Hello", "start": 0.0, "end": 0.5}, {"text": "trip.", "start": 0.6, "end": 1.0}]},
    "b2": {"file": "bbbb.wav", "duration": 3.0, "words": [{"text": "Then", "start": 0.0, "end": 0.3}, {"text": "this.", "start": 0.4, "end": 0.9}]},
}


def test_clips_join_the_beat_they_overlap_most():
    clips = [clip(5, 9), clip(10, 13), clip(13.5, 14), clip(31, 35), clip(80, 85), clip(41, 43)]
    assert assign_beats(clips, BRIEF) == [None, "b1", "b1", "b2", "b2", "b2"]
    assert assign_beats([clip(10, 12, "v1")], BRIEF) == [None]
    assert assign_beats(clips, None) == [None] * len(clips)


def test_a_long_line_lengthens_its_beats_last_clip():
    clips = [clip(10, 11.5), clip(31, 33)]
    fitted, notes = fit_narration(clips, BRIEF, LINES, {"v0": 100.0})
    assert fitted[0].end == pytest.approx(10 + LEAD_IN + 2.0 + 0.25)
    assert fitted[1].end == pytest.approx(31 + LEAD_IN + 3.0 + 0.25)
    assert len(notes) == 2 and clips[0].end == 11.5
    near_end, notes = fit_narration([clip(10, 11.5)], BRIEF, LINES, {"v0": 12.0})
    assert near_end[0].end == 12.0 and any("ends before" in n for n in notes)
    roomy, notes = fit_narration([clip(10, 20)], BRIEF, LINES, {"v0": 100.0})
    assert roomy[0].end == 20 and notes == []


def spec_for(caption_style="bold"):
    clips = [clip(10, 14), clip(30, 36), clip(40, 44)]
    return build_spec(
        clips,
        BRIEF,
        LINES,
        width=1080,
        height=1920,
        fps=30,
        caption_style=caption_style,
        video=SpecVideo(kind="base", src="base.mp4"),
        narration_src=lambda line: line["file"],
    )


def test_spec_places_texts_voice_and_captions():
    spec = spec_for()
    assert spec.duration == 14 and spec.caption_style == "bold"
    kinds = [(t.kind, t.text) for t in spec.texts]
    assert kinds[0] == ("title", "Our trip") and ("beat", "Day one") in kinds and kinds[-1] == ("cta", "Follow for more")
    assert [(n.beat, n.start) for n in spec.narration] == [("b1", LEAD_IN), ("b2", 4 + LEAD_IN)]
    assert spec.narration[1].src == "bbbb.wav"
    first_words = [w for c in spec.captions for w in c.words]
    assert first_words[0].start == pytest.approx(LEAD_IN) and first_words[2].start == pytest.approx(4 + LEAD_IN)
    plain = spec_for("none")
    assert plain.caption_style == "none" and plain.captions  # worked out for the SRT, not drawn
    assert "Hello trip." in narration_srt(plain)


def test_spec_schema_file_is_current_and_fixtures_are_written():
    assert json.loads(SCHEMA_PATH.read_text(encoding="utf-8")) == json_schema(), (
        "Run: python -c \"from app.compose.render_spec import write_json_schema; write_json_schema()\""
    )
    FIXTURES.mkdir(exist_ok=True)
    preview = build_spec(
        [clip(10, 14), clip(30, 36)],
        BRIEF,
        {},
        width=1920,
        height=1080,
        fps=25,
        caption_style="clean",
        video=preview_video([clip(10, 14), clip(30, 36)], lambda c: f"/api/media/job/source/{c.source_id}"),
        narration_src=lambda line: "",
    )
    assert preview.video.clips[1].start == 4 and preview.video.clips[1].source_start == 30
    for name, spec in (("render_spec_base.json", spec_for()), ("render_spec_preview.json", preview)):
        (FIXTURES / name).write_text(json.dumps(spec.model_dump(), indent=2) + "\n", encoding="utf-8")


def test_ass_drawing_has_titles_and_captions():
    text = spec_ass(spec_for())
    assert "Style: Caption," in text and "Style: Title," in text and "Style: Beat," in text
    assert text.index("Style: Title,") < text.index("[Events]")
    assert ",Title,,0,0,0,,{\\fad(250,250)}Our trip" in text and "Day one" in text
    assert text.count("Dialogue:") >= 4
    no_captions = spec_ass(spec_for("none"))
    assert "Hello" not in no_captions and "Our trip" in no_captions


def test_narration_mix_commands():
    voices = [(Path("a.wav"), 0.2), (Path("b.wav"), 4.2)]
    audio = ["-c:a", "aac", "-b:a", "192k"]
    drawn = narration_mix_command(Path("base.mp4"), voices, Path("out.mp4"), 14.0, True, audio_args=audio, picture=Path("pic.mp4"))
    graph = drawn[drawn.index("-filter_complex") + 1]
    assert drawn[drawn.index("-map") + 1] == "1:v:0" and "copy" in drawn
    assert "adelay=200|200" in graph and "adelay=4200|4200" in graph and "sidechaincompress" in graph
    assert "amix=inputs=2" in graph and drawn[-1] == "out.mp4" and "+faststart" in drawn
    burned = narration_mix_command(
        Path("base.mp4"), voices[:1], Path("out.webm"), 5.0, False, audio_args=audio, subtitles=Path("C:/x/o.ass"),
        video_args=["-c:v", "libvpx-vp9"],
    )
    graph = burned[burned.index("-filter_complex") + 1]
    assert "subtitles='C\\:/x/o.ass'" in graph and "[0:a]" not in graph and "libvpx-vp9" in burned
    assert "+faststart" not in burned
    with pytest.raises(ValueError):
        narration_mix_command(Path("b.mp4"), [], Path("o.mp4"), 1, True, audio_args=audio)


class FakeRunner:
    def __init__(self, meta: VideoMetadata):
        self.meta = meta
        self.commands: list[list[str]] = []

    def probe(self, path):
        return self.meta

    def run(self, command):
        self.commands.append(command)
        Path(command[-1]).write_bytes(b"video")


def test_compose_draws_with_ffmpeg_when_remotion_is_not_ready(tmp_path, monkeypatch):
    job = tmp_path / "job"
    (job / "narration").mkdir(parents=True)
    for line in LINES.values():
        (job / "narration" / line["file"]).write_bytes(b"wav")
    output = job / "outputs" / "highlight.mp4"
    output.parent.mkdir()
    output.write_bytes(b"base")
    monkeypatch.setattr(compose.remotion, "ready", lambda settings: False)
    runner = FakeRunner(VideoMetadata(duration=14.0, fps=30, width=1080, height=1920, has_audio=True))
    logs: list[str] = []
    config = HighlightConfig(video_brief=True, narration=True, subtitles="burned", caption_style="bold")
    spec = compose.compose_video(
        runner,
        output,
        [clip(10, 14), clip(30, 36), clip(40, 44)],
        BRIEF,
        LINES,
        config,
        Settings(data_dir=tmp_path / "data"),
        job / "tmp_render" / "highlight",
        job,
        log=lambda message, level="INFO": logs.append(message),
    )
    assert spec is not None and len(spec.narration) == 2
    command = runner.commands[-1]
    graph = command[command.index("-filter_complex") + 1]
    assert "subtitles=" in graph and "adelay" in graph
    assert output.read_bytes() == b"video" and not list(output.parent.glob(".*part*"))
    assert (job / "render_spec.json").is_file() and (job / "tmp_render" / "highlight" / "compose" / "overlay.ass").is_file()
    assert any("not prepared" in line for line in logs) and any("drawn with FFmpeg" in line for line in logs)
    srt = job / "outputs" / "highlight.srt"
    assert compose.write_srt(spec, srt) and "Hello trip." in srt.read_text(encoding="utf-8")


def test_compose_without_texts_or_voice_leaves_the_video(tmp_path):
    output = tmp_path / "highlight.mp4"
    output.write_bytes(b"base")
    runner = FakeRunner(VideoMetadata(duration=4.0, fps=30, width=640, height=360, has_audio=True))
    bare = VideoBrief.model_validate({"beats": [{"id": "b1", "clips": [{"id": "c1", "start": 0, "end": 4}]}]})
    result = compose.compose_video(
        runner, output, [clip(0, 4)], bare, {}, HighlightConfig(video_brief=True), Settings(data_dir=tmp_path / "d"),
        tmp_path / "work", tmp_path, log=lambda *a, **k: None,
    )
    assert result is None and runner.commands == [] and output.read_bytes() == b"base"


def test_prepare_narration_explains_what_is_missing(tmp_path, monkeypatch):
    logs: list[str] = []
    log = lambda message, level="INFO": logs.append(message)  # noqa: E731
    settings = Settings(data_dir=tmp_path / "data")
    assert compose.prepare_narration(tmp_path, HighlightConfig(narration=True), settings, log=log) == (None, {})
    from app.brief.service import save_brief

    brief = VideoBrief.model_validate({"beats": [{"id": "b1", "narration": "これは日本語です", "clips": [{"id": "c", "start": 0, "end": 5}]}]})
    save_brief(tmp_path, brief)
    loaded, lines = compose.prepare_narration(tmp_path, HighlightConfig(narration=True), settings, log=log)
    assert loaded is not None and lines == {} and any("not in English" in line for line in logs)
    save_brief(tmp_path, BRIEF.model_copy(update={"beats": [b.model_copy(update={"narration": "Hi there."}) for b in BRIEF.beats]}))
    monkeypatch.setattr(compose.kokoro, "installed", lambda: False)
    assert compose.prepare_narration(tmp_path, HighlightConfig(narration=True), settings, log=log)[1] == {}
    assert any("not downloaded" in line for line in logs)
    assert compose.prepare_narration(tmp_path, HighlightConfig(video_brief=True), settings, log=log)[1] == {}


def test_render_progress_counts_the_composition():
    sent: list[dict] = []
    progress = RenderProgress([[5.0, 5.0]], False, sent.append, compose=True)
    progress.clip(0, 0, 1.0)
    progress.clip(0, 1, 1.0)
    before = sent[-1]["overall"]
    assert before < 100
    progress.compose(0, 1.0)
    assert sent[-1]["overall"] == 100
