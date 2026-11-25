"""The timeline of a narrated video: one file read by the Review preview, the Remotion render and
the FFmpeg fallback, so the three never disagree.

The clips come from the plan as kept on the Review page. Each clip belongs to the brief beat
whose moments it overlaps most (a clip added by hand joins the beat before it). A beat's
narration starts shortly after the beat starts; when a line is longer than its beat, the beat's
last clip is lengthened (within its video) before the video is made, so the voice never runs
into the next beat. Times in the spec are seconds on the finished video's timeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from app.brief.schema import VideoBrief
from app.models.schemas import ClipPlanItem
from app.video.captions import Word, word_cues

SPEC_VERSION = 1
# The voice starts this long after its beat starts, and the beat keeps this much after it.
LEAD_IN = 0.2
TAIL = 0.25
TITLE_SECONDS = 2.6
BEAT_TEXT_SECONDS = 3.5
CTA_SECONDS = 2.8
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "remotion" / "src" / "spec.schema.json"


class SpecWord(BaseModel):
    text: str
    start: float
    end: float


class SpecCaption(BaseModel):
    start: float
    end: float
    text: str
    words: list[SpecWord] = Field(default_factory=list)


class SpecText(BaseModel):
    kind: Literal["title", "beat", "cta"]
    text: str
    start: float
    end: float


class SpecNarration(BaseModel):
    beat: str
    src: str
    start: float
    duration: float


class SpecClip(BaseModel):
    """A piece of a source video (the preview plays these instead of a finished video)."""

    src: str
    start: float
    end: float
    source_start: float


class SpecVideo(BaseModel):
    kind: Literal["base", "clips"]
    src: str = ""
    clips: list[SpecClip] = Field(default_factory=list)


class RenderSpec(BaseModel):
    version: int = SPEC_VERSION
    width: int
    height: int
    fps: float
    duration: float
    video: SpecVideo
    texts: list[SpecText] = Field(default_factory=list)
    captions: list[SpecCaption] = Field(default_factory=list)
    caption_style: Literal["none", "plain", "clean", "bold", "boxed"] = "none"
    narration: list[SpecNarration] = Field(default_factory=list)
    accent: str = "#FFD700"


def json_schema() -> dict:
    """The spec's JSON schema, kept in remotion/src/spec.schema.json (a test checks they match)."""
    return RenderSpec.model_json_schema()


def write_json_schema(path: Path = SCHEMA_PATH) -> None:
    path.write_text(json.dumps(json_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


# Beats


def assign_beats(clips: list[ClipPlanItem], brief: VideoBrief | None) -> list[str | None]:
    """The beat id of every clip (None before the first matched clip)."""
    result: list[str | None] = []
    current: str | None = None
    for clip in clips:
        best, overlap = None, 0.0
        for beat in brief.beats if brief else []:
            for moment in beat.clips:
                if (moment.source_id or "v0") != (clip.source_id or "v0") and clip.source_id:
                    continue
                shared = min(clip.end, moment.end) - max(clip.start, moment.start)
                if shared > overlap + 1e-6:
                    best, overlap = beat.id, shared
        if best is not None and overlap > 0.05:
            current = best
        result.append(current)
    return result


def _segments(clips: list[ClipPlanItem], beats: list[str | None]) -> list[tuple[str | None, int, int]]:
    """Runs of consecutive clips with the same beat: (beat, first index, last index)."""
    runs: list[tuple[str | None, int, int]] = []
    for index, beat in enumerate(beats):
        if runs and runs[-1][0] == beat:
            runs[-1] = (beat, runs[-1][1], index)
        else:
            runs.append((beat, index, index))
    return runs


def fit_narration(
    clips: list[ClipPlanItem],
    brief: VideoBrief | None,
    narration: dict[str, dict],
    source_durations: dict[str, float],
) -> tuple[list[ClipPlanItem], list[str]]:
    """Clips with each narrated beat's last clip lengthened where its voice needs more time.
    Returns the clips and notes for the log."""
    beats = assign_beats(clips, brief)
    result = list(clips)
    notes: list[str] = []
    voiced: set[str] = set()
    for beat, first, last in _segments(clips, beats):
        line = narration.get(beat or "")
        if not beat or not line or beat in voiced:
            continue
        voiced.add(beat)
        have = sum(result[i].duration for i in range(first, last + 1))
        need = LEAD_IN + float(line.get("duration") or 0) + TAIL
        if need <= have + 0.01:
            continue
        clip = result[last]
        limit = float(source_durations.get(clip.source_id or "v0") or source_durations.get("") or 0) or clip.end + need
        new_end = min(limit, clip.end + (need - have))
        if new_end > clip.end + 0.01:
            result[last] = clip.model_copy(update={"end": round(new_end, 3)})
            notes.append(f"Beat {beat[1:]}: clip lengthened by {new_end - clip.end:.1f}s to fit its narration")
        if new_end < clip.end + (need - have) - 0.05:
            notes.append(f"Beat {beat[1:]}: the video ends before its narration does")
    return result, notes


# Spec


def build_spec(
    clips: list[ClipPlanItem],
    brief: VideoBrief | None,
    narration: dict[str, dict],
    *,
    width: int,
    height: int,
    fps: float,
    caption_style: str,
    video: SpecVideo,
    narration_src: Callable[[dict], str],
    duration: float | None = None,
) -> RenderSpec:
    """The timeline for ``clips`` (in video order). ``narration_src`` gives the address of a
    spoken line (a file name for rendering, a URL for the preview)."""
    beats = assign_beats(clips, brief)
    starts: list[float] = []
    cursor = 0.0
    for clip in clips:
        starts.append(cursor)
        cursor += clip.duration
    total = float(duration) if duration else cursor
    by_id = {beat.id: beat for beat in brief.beats} if brief else {}
    texts: list[SpecText] = []
    voices: list[SpecNarration] = []
    words: list[Word] = []
    voiced: set[str] = set()
    for beat_id, first, last in _segments(clips, beats):
        beat = by_id.get(beat_id or "")
        if beat is None:
            continue
        begin = starts[first]
        end = starts[last] + clips[last].duration
        if beat.on_screen_text and beat_id not in voiced:
            texts.append(
                SpecText(kind="beat", text=beat.on_screen_text, start=round(begin + 0.3, 3), end=round(min(end, begin + BEAT_TEXT_SECONDS), 3))
            )
        line = narration.get(beat_id or "")
        if line and beat_id not in voiced:
            start = begin + LEAD_IN
            voices.append(
                SpecNarration(beat=beat_id, src=narration_src(line), start=round(start, 3), duration=float(line["duration"]))
            )
            for word in line.get("words") or []:
                words.append(Word(start + float(word["start"]), start + float(word["end"]), str(word["text"])))
        voiced.add(beat_id or "")
    if brief and brief.title:
        texts.insert(0, SpecText(kind="title", text=brief.title, start=0.0, end=round(min(total, TITLE_SECONDS), 3)))
    if brief and brief.call_to_action and total > CTA_SECONDS + TITLE_SECONDS:
        texts.append(SpecText(kind="cta", text=brief.call_to_action, start=round(total - CTA_SECONDS, 3), end=round(total, 3)))
    # Always worked out (the SRT download uses them); ``caption_style`` decides if they are drawn.
    captions: list[SpecCaption] = []
    style = caption_style if caption_style in {"plain", "clean", "bold", "boxed"} else "none"
    if words:
        for cue in word_cues(words, portrait=height > width):
            captions.append(
                SpecCaption(
                    start=round(cue.start, 3),
                    end=round(min(total, cue.end), 3),
                    text=cue.text,
                    words=[SpecWord(text=w.text, start=round(w.start, 3), end=round(w.end, 3)) for w in cue.words],
                )
            )
    return RenderSpec(
        width=int(width),
        height=int(height),
        fps=float(fps),
        duration=round(total, 3),
        video=video,
        texts=texts,
        captions=captions,
        caption_style=style,
        narration=voices,
    )


def preview_video(clips: list[ClipPlanItem], source_src: Callable[[ClipPlanItem], str]) -> SpecVideo:
    """The clips as pieces of their source videos (for the browser preview)."""
    items = []
    cursor = 0.0
    for clip in clips:
        items.append(
            SpecClip(src=source_src(clip), start=round(cursor, 3), end=round(cursor + clip.duration, 3), source_start=round(clip.start, 3))
        )
        cursor += clip.duration
    return SpecVideo(kind="clips", clips=items)


def narration_srt(spec: RenderSpec) -> str:
    """The narration as an SRT file (for subtitle downloads)."""
    from app.video.captions import Cue, srt_text

    return srt_text([Cue(c.start, c.end, c.text) for c in spec.captions if c.text])


def spec_ass(spec: RenderSpec) -> str:
    """Captions, title, beat texts and closing line as ASS subtitles (the FFmpeg drawing)."""
    from app.video.captions import Cue, ass_text, _ass_escape, _ass_time

    # One "clip" covering the whole video: cue and word times are already on its timeline.
    cues = [Cue(c.start, c.end, c.text, [Word(w.start, w.end, w.text) for w in c.words]) for c in spec.captions]
    style = spec.caption_style if spec.caption_style != "none" else "clean"
    text = ass_text(cues if spec.caption_style != "none" else [], style, spec.width, spec.height)
    short_side = min(spec.width, spec.height)
    big = max(18, round(short_side * 0.085))
    small = max(14, round(short_side * 0.05))
    margin = round(spec.height * 0.1)
    extra_styles = [
        f"Style: Title,Arial,{big},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{round(big * 0.08, 1)},{round(big * 0.05, 1)},5,40,40,0,1",
        f"Style: Beat,Arial,{small},&H00FFFFFF,&H00FFFFFF,&H00000000,&HB0000000,-1,0,0,0,100,100,0,0,3,{round(small * 0.25, 1)},0,8,40,40,{margin},1",
    ]
    header, _, events = text.partition("\n[Events]\n")
    lines = [header.rstrip("\n"), *extra_styles, "", "[Events]", events.rstrip("\n")]
    for item in spec.texts:
        name = "Title" if item.kind in {"title", "cta"} else "Beat"
        fade = r"{\fad(250,250)}"
        lines.append(
            f"Dialogue: 1,{_ass_time(item.start)},{_ass_time(item.end)},{name},,0,0,0,,{fade}{_ass_escape(item.text)}"
        )
    return "\n".join(line for line in lines if line is not None) + "\n"
