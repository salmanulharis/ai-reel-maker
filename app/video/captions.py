"""
Subtitles for the highlight: a subtitle file per clip for burning into the video (timed from the
clip's own start, because each clip is cut on its own), and one SRT for the whole highlight.

"plain" keeps the original look (one line of text per spoken segment, FFmpeg's default style).
The other styles are ASS subtitles built from Whisper's word timings, with the word being spoken
highlighted. Without word timings they fall back to whole segments, still in the chosen style.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

CAPTION_STYLES = ("plain", "clean", "bold", "boxed")


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Cue:
    """One caption line, with times relative to the start of its clip."""

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def load_transcript(job_dir: Path, source_id: str = "", multi: bool = False) -> dict:
    """The transcript of one source video: multi-video jobs keep one per video."""
    path = job_dir / "work" / source_id / "transcript.json" if multi and source_id else job_dir / "transcript.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _segment_cues(segments: list, clip_start: float, clip_end: float) -> list[Cue]:
    cues = []
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        start, end = _number(segment.get("start")), _number(segment.get("end"))
        text = " ".join(str(segment.get("text") or "").split())
        if start is None or end is None or not text or end <= clip_start or start >= clip_end:
            continue
        cue_start = max(start, clip_start) - clip_start
        cue_end = min(end, clip_end) - clip_start
        if cue_end - cue_start >= 0.05:
            cues.append(Cue(cue_start, cue_end, text))
    return cues


def _clip_words(words: list, clip_start: float, clip_end: float) -> list[Word]:
    found = []
    for item in words or []:
        if not isinstance(item, dict):
            continue
        start, end = _number(item.get("start")), _number(item.get("end"))
        text = str(item.get("word") or item.get("text") or "").strip()
        if start is None or end is None or not text or end <= clip_start or start >= clip_end:
            continue
        found.append(Word(max(start, clip_start) - clip_start, min(end, clip_end) - clip_start, text))
    found.sort(key=lambda w: w.start)
    return found


def _word_cues(words: list[Word], max_words: int, max_chars: int, max_seconds: float = 3.0) -> list[Cue]:
    """Short caption lines: a new line after a sentence ends, after a pause, or when full."""
    groups: list[list[Word]] = []
    for word in words:
        current = groups[-1] if groups else None
        if current:
            chars = sum(len(w.text) + 1 for w in current) + len(word.text)
            ends_sentence = bool(re.search(r"[.!?…]$", current[-1].text))
            pause = word.start - current[-1].end > 0.7
            too_long = word.end - current[0].start > max_seconds
            if len(current) >= max_words or chars > max_chars or ends_sentence or pause or too_long:
                current = None
        if current is None:
            groups.append([word])
        else:
            current.append(word)
    cues = []
    for index, group in enumerate(groups):
        start, end = group[0].start, group[-1].end
        # Keep the line up a moment after the last word, but never into the next line.
        following = groups[index + 1][0].start if index + 1 < len(groups) else end + 0.4
        cues.append(Cue(start, max(end, min(end + 0.25, following)), " ".join(w.text for w in group), list(group)))
    return cues


def word_cues(words: list[Word], portrait: bool = False) -> list[Cue]:
    """Caption lines for timed words (narration), sized like the word-by-word styles."""
    return _word_cues(words, max_words=4 if portrait else 6, max_chars=22 if portrait else 34)


def clip_cues(transcript: dict, clip_start: float, clip_end: float, style: str = "plain", portrait: bool = False) -> list[Cue]:
    """Caption lines for one clip, timed from the clip's start."""
    if style != "plain":
        words = _clip_words(transcript.get("words") or [], clip_start, clip_end)
        if words:
            return _word_cues(words, max_words=4 if portrait else 6, max_chars=22 if portrait else 34)
    return _segment_cues(transcript.get("segments") or [], clip_start, clip_end)


# ---------------------------------------------------------------------------------------- SRT


def _srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt_text(cues: list[Cue], offset: float = 0.0, first_number: int = 1) -> str:
    blocks = [
        f"{first_number + i}\n{_srt_time(cue.start + offset)} --> {_srt_time(cue.end + offset)}\n{cue.text}\n"
        for i, cue in enumerate(cues)
    ]
    return "\n".join(blocks)


def sequence_srt(clips: list[tuple[dict, float, float]]) -> str:
    """One SRT for the whole highlight. `clips` holds (transcript, start, end) in highlight order."""
    blocks: list[str] = []
    cursor = 0.0
    count = 1
    for transcript, start, end in clips:
        cues = clip_cues(transcript, start, end, "plain")
        if cues:
            blocks.append(srt_text(cues, offset=cursor, first_number=count))
            count += len(cues)
        cursor += max(0.0, end - start)
    return "\n".join(blocks)


# ---------------------------------------------------------------------------------------- ASS

# Colours are &HAABBGGRR (alpha 00 = solid).
_WHITE = "&H00FFFFFF"
_BLACK = "&H00000000"
_HIGHLIGHT = "&H0000D7FF"  # warm yellow
_SHADOW = "&H80000000"
_BOX = "&H00181818"  # opaque: a see-through box shows darker overlaps where a line is split


@dataclass(frozen=True)
class _Look:
    size: float  # share of the shorter side of the video
    bold: bool
    border_style: int  # 1 outline, 3 box
    outline: float  # share of the font size
    shadow: float
    back: str
    upper: bool
    pop: bool


_LOOKS = {
    "clean": _Look(size=0.058, bold=True, border_style=1, outline=0.07, shadow=0.03, back=_SHADOW, upper=False, pop=False),
    "bold": _Look(size=0.074, bold=True, border_style=1, outline=0.11, shadow=0.0, back=_SHADOW, upper=True, pop=True),
    "boxed": _Look(size=0.054, bold=False, border_style=3, outline=0.18, shadow=0.0, back=_BOX, upper=False, pop=False),
}


def _ass_time(seconds: float) -> str:
    cs = max(0, int(round(seconds * 100)))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    """Braces start override blocks and a backslash starts a command; show them as plain text."""
    return text.replace("\\", "∖").replace("{", "(").replace("}", ")").replace("\n", " ")


def _word_text(word: str, upper: bool) -> str:
    return _ass_escape(word.upper() if upper else word)


def ass_text(cues: list[Cue], style: str, width: int, height: int) -> str:
    """ASS subtitles for one clip, sized for a width × height video. The word being spoken is
    highlighted (and, for "bold", enlarged); lines without word timings are shown whole."""
    look = _LOOKS.get(style, _LOOKS["clean"])
    width, height = max(2, int(width)), max(2, int(height))
    short_side = min(width, height)
    size = max(12, round(short_side * look.size))
    portrait = height > width
    margin_v = round(height * (0.2 if portrait else 0.08))
    margin_h = round(width * 0.06)
    header = "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Caption,Arial,{size},{_WHITE},{_HIGHLIGHT},{_BLACK if look.border_style == 1 else look.back},"
            f"{look.back},{-1 if look.bold else 0},0,0,0,100,100,0,0,{look.border_style},"
            f"{round(size * look.outline, 1)},{round(size * look.shadow, 1)},2,{margin_h},{margin_h},{margin_v},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )
    lines = [header]
    for cue in cues:
        if not cue.words:
            text = _ass_escape(cue.text.upper() if look.upper else cue.text)
            lines.append(f"Dialogue: 0,{_ass_time(cue.start)},{_ass_time(cue.end)},Caption,,0,0,0,,{text}")
            continue
        for index, word in enumerate(cue.words):
            start = cue.start if index == 0 else word.start
            end = cue.words[index + 1].start if index + 1 < len(cue.words) else cue.end
            if end - start < 0.01:
                continue
            parts = []
            for other_index, other in enumerate(cue.words):
                text = _word_text(other.text, look.upper)
                if other_index == index:
                    pop = r"\fscx112\fscy112" if look.pop else ""
                    text = f"{{\\c{_HIGHLIGHT}&{pop}}}{text}{{\\r}}"
                parts.append(text)
            lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,{' '.join(parts)}")
    return "\n".join(lines) + "\n"


def write_clip_subtitles(
    dest_dir: Path,
    index: int,
    transcript: dict,
    clip_start: float,
    clip_end: float,
    style: str,
    width: int,
    height: int,
) -> Path | None:
    """Write the subtitle file to burn into one clip, or return None when the clip has no speech."""
    style = style if style in CAPTION_STYLES else "plain"
    cues = clip_cues(transcript, clip_start, clip_end, style, portrait=height > width)
    if not cues:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    if style == "plain":
        path = dest_dir / f"clip_{index:03d}.srt"
        path.write_text(srt_text(cues), encoding="utf-8")
    else:
        path = dest_dir / f"clip_{index:03d}.ass"
        path.write_text(ass_text(cues, style, width, height), encoding="utf-8")
    return path
