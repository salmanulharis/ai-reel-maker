"""
The clip plan as timelines for video editors: FCPXML (Final Cut Pro, DaVinci Resolve), Final Cut
Pro 7 XML (Premiere Pro, DaVinci Resolve) and a CMX 3600 EDL, plus subtitles timed to the edit.

Everything here is pure text building: no files are read or written. Clips are snapped to whole
frames at a standard rate, placed back to back from 00:00:00:00, and point at the original video
files, so the editor works from full-quality media.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape, quoteattr

# (timebase, NTSC): NTSC rates run at timebase × 1000/1001 (23.976, 29.97, 59.94).
_STANDARD_RATES = [(24, True), (24, False), (25, False), (30, True), (30, False), (48, False), (50, False), (60, True), (60, False)]
_DEFAULT_RATE = (25, False)


@dataclass(frozen=True)
class Rate:
    timebase: int
    ntsc: bool = False

    @property
    def fps(self) -> float:
        return self.timebase * 1000 / 1001 if self.ntsc else float(self.timebase)

    @property
    def frame_duration(self) -> str:
        """One frame as an FCPXML rational time."""
        return f"1001/{self.timebase * 1000}s" if self.ntsc else f"1/{self.timebase}s"

    def frames(self, seconds: float) -> int:
        return max(0, int(round(float(seconds) * self.fps)))

    def seconds(self, frames: int) -> float:
        return frames / self.fps

    def rational(self, frames: int) -> str:
        """A frame count as an FCPXML rational time, such as 3003/30000s."""
        if frames <= 0:
            return "0s"
        if self.ntsc:
            return f"{frames * 1001}/{self.timebase * 1000}s"
        return f"{frames}/{self.timebase}s"

    def timecode(self, frames: int) -> str:
        """Non-drop-frame timecode HH:MM:SS:FF, counting at the timebase."""
        frames = max(0, int(frames))
        ff = frames % self.timebase
        total = frames // self.timebase
        return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}:{ff:02d}"


def rate_for(fps: float | None) -> Rate:
    """The standard editing rate closest to a source rate (25 when it is unknown)."""
    try:
        value = float(fps or 0)
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value) or value <= 0:
        return Rate(*_DEFAULT_RATE)
    best = min(_STANDARD_RATES, key=lambda item: abs(Rate(*item).fps - value))
    if abs(Rate(*best).fps - value) <= 0.5:
        return Rate(*best)
    # An unusual rate (a 12 fps screen recording): keep it as a whole number.
    return Rate(max(1, min(120, int(round(value)))))


@dataclass
class TimelineSource:
    """A source video as the editor will see it."""

    id: str
    name: str
    path: Path
    fps: float = 0.0
    width: int = 0
    height: int = 0
    duration: float = 0.0
    has_audio: bool = True
    channels: int = 2
    # Speech in this video: [{"start", "end", "text"}], used for the subtitles.
    segments: list[dict] = field(default_factory=list)

    @property
    def rate(self) -> Rate:
        return rate_for(self.fps)


@dataclass
class TimelineClip:
    source: TimelineSource
    start: float
    end: float


@dataclass
class PlacedClip:
    """A clip snapped to frames at the sequence rate and given its place on the timeline."""

    number: int
    source: TimelineSource
    src_in: int
    src_out: int
    rec_in: int
    rec_out: int

    @property
    def length(self) -> int:
        return self.src_out - self.src_in


@dataclass
class Timeline:
    name: str
    rate: Rate
    width: int
    height: int
    clips: list[PlacedClip]

    @property
    def duration(self) -> int:
        return self.clips[-1].rec_out if self.clips else 0

    @property
    def sources(self) -> list[TimelineSource]:
        seen: dict[str, TimelineSource] = {}
        for clip in self.clips:
            seen.setdefault(clip.source.id, clip.source)
        return list(seen.values())


def build_timeline(name: str, clips: list[TimelineClip]) -> Timeline:
    """Snap clips to whole frames at the first video's rate and place them back to back.
    Clips shorter than one frame are left out."""
    first = clips[0].source if clips else None
    rate = first.rate if first else Rate(*_DEFAULT_RATE)
    placed: list[PlacedClip] = []
    cursor = 0
    for clip in clips:
        src_in = rate.frames(clip.start)
        src_out = rate.frames(clip.end)
        if src_out <= src_in:
            continue
        placed.append(
            PlacedClip(
                number=len(placed) + 1,
                source=clip.source,
                src_in=src_in,
                src_out=src_out,
                rec_in=cursor,
                rec_out=cursor + (src_out - src_in),
            )
        )
        cursor += src_out - src_in
    width = int(first.width) if first and first.width else 1920
    height = int(first.height) if first and first.height else 1080
    return Timeline(name=name or "Highlight", rate=rate, width=width, height=height, clips=placed)


def file_uri(path: Path) -> str:
    """file:///C:/Videos/my%20talk.mp4 (FCPXML)."""
    return Path(path).absolute().as_uri()


def pathurl(path: Path) -> str:
    """file://localhost/C:/Videos/my%20talk.mp4 (Final Cut Pro 7 XML, as Premiere writes it)."""
    text = Path(path).absolute().as_posix()
    if not text.startswith("/"):
        text = "/" + text
    return "file://localhost" + quote(text, safe="/:")


# ---------------------------------------------------------------------------------- FCPXML


def fcpxml(timeline: Timeline) -> str:
    seq = timeline.rate
    formats: dict[tuple, str] = {}
    lines: list[str] = []

    def format_id(rate: Rate, width: int, height: int) -> str:
        key = (rate, width, height)
        if key not in formats:
            formats[key] = f"r{len(formats) + 1}"
            lines.append(
                f'    <format id="{formats[key]}" frameDuration="{rate.frame_duration}" '
                f'width="{width}" height="{height}"/>'
            )
        return formats[key]

    seq_format = format_id(seq, timeline.width, timeline.height)
    assets: dict[str, str] = {}
    for source in timeline.sources:
        rate = source.rate
        fmt = format_id(rate, int(source.width or timeline.width), int(source.height or timeline.height))
        asset = f"a{len(assets) + 1}"
        assets[source.id] = asset
        frames = rate.frames(source.duration) or max(
            (rate.frames(seq.seconds(c.src_out)) for c in timeline.clips if c.source.id == source.id), default=0
        )
        audio = (
            f' hasAudio="1" audioSources="1" audioChannels="{max(1, int(source.channels or 2))}" audioRate="48000"'
            if source.has_audio
            else ' hasAudio="0"'
        )
        lines.append(
            f'    <asset id="{asset}" name={quoteattr(source.name)} start="0s" '
            f'duration="{rate.rational(frames)}" hasVideo="1" format="{fmt}"{audio}>'
        )
        lines.append(f'      <media-rep kind="original-media" src={quoteattr(file_uri(source.path))}/>')
        lines.append("    </asset>")

    spine: list[str] = []
    for clip in timeline.clips:
        rate = clip.source.rate
        # The in-point on the source's own frame grid; offset and length on the sequence grid.
        start = rate.rational(rate.frames(seq.seconds(clip.src_in)))
        spine.append(
            f'          <asset-clip ref="{assets[clip.source.id]}" name={quoteattr(clip.source.name)} '
            f'offset="{seq.rational(clip.rec_in)}" start="{start}" duration="{seq.rational(clip.length)}" '
            f'tcFormat="NDF"/>'
        )

    name = quoteattr(timeline.name)
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<!DOCTYPE fcpxml>",
            '<fcpxml version="1.9">',
            "  <resources>",
            *lines,
            "  </resources>",
            "  <library>",
            f"    <event name={name}>",
            f"      <project name={name}>",
            f'        <sequence format="{seq_format}" duration="{seq.rational(timeline.duration)}" '
            'tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">',
            "          <spine>",
            *spine,
            "          </spine>",
            "        </sequence>",
            "      </project>",
            "    </event>",
            "  </library>",
            "</fcpxml>",
            "",
        ]
    )


# ------------------------------------------------------------------------ Final Cut Pro 7 XML


def _rate_xml(rate: Rate) -> str:
    return f"<rate><timebase>{rate.timebase}</timebase><ntsc>{'TRUE' if rate.ntsc else 'FALSE'}</ntsc></rate>"


def _timecode_xml(rate: Rate) -> str:
    return (
        f"<timecode>{_rate_xml(rate)}<string>00:00:00:00</string><frame>0</frame>"
        "<displayformat>NDF</displayformat></timecode>"
    )


def xmeml(timeline: Timeline) -> str:
    """Final Cut Pro 7 XML (xmeml version 4). All frame numbers use the sequence rate."""
    seq = timeline.rate
    rate = _rate_xml(seq)
    file_ids: dict[str, str] = {}
    defined: set[str] = set()

    def file_xml(source: TimelineSource) -> str:
        file_id = file_ids.setdefault(source.id, f"file-{len(file_ids) + 1}")
        if file_id in defined:
            return f'<file id="{file_id}"/>'
        defined.add(file_id)
        frames = seq.frames(source.duration) or max(
            (c.src_out for c in timeline.clips if c.source.id == source.id), default=0
        )
        audio = (
            "<audio><samplecharacteristics><depth>16</depth><samplerate>48000</samplerate>"
            f"</samplecharacteristics><channelcount>{max(1, int(source.channels or 2))}</channelcount></audio>"
            if source.has_audio
            else ""
        )
        width = int(source.width or timeline.width)
        height = int(source.height or timeline.height)
        return (
            f'<file id="{file_id}"><name>{escape(source.name)}</name>'
            f"<pathurl>{escape(pathurl(source.path))}</pathurl>{rate}<duration>{frames}</duration>"
            f"{_timecode_xml(seq)}<media><video><samplecharacteristics>{rate}<width>{width}</width>"
            f"<height>{height}</height><pixelaspectratio>square</pixelaspectratio></samplecharacteristics>"
            f"</video>{audio}</media></file>"
        )

    def links(clip: PlacedClip) -> str:
        if not clip.source.has_audio:
            return ""
        return (
            f"<link><linkclipref>clipitem-v{clip.number}</linkclipref><mediatype>video</mediatype>"
            f"<trackindex>1</trackindex><clipindex>{clip.number}</clipindex></link>"
            f"<link><linkclipref>clipitem-a{clip.number}</linkclipref><mediatype>audio</mediatype>"
            f"<trackindex>1</trackindex><clipindex>{clip.number}</clipindex><groupindex>1</groupindex></link>"
        )

    def item(clip: PlacedClip, kind: str) -> str:
        source_frames = seq.frames(clip.source.duration) or clip.src_out
        extra = (
            "<sourcetrack><mediatype>audio</mediatype><trackindex>1</trackindex></sourcetrack>"
            if kind == "a"
            else ""
        )
        return (
            f'<clipitem id="clipitem-{kind}{clip.number}"><name>{escape(clip.source.name)}</name>'
            f"<enabled>TRUE</enabled><duration>{source_frames}</duration>{rate}"
            f"<start>{clip.rec_in}</start><end>{clip.rec_out}</end><in>{clip.src_in}</in><out>{clip.src_out}</out>"
            f"{file_xml(clip.source)}{extra}{links(clip)}</clipitem>"
        )

    # Video first, so each file is fully described the first time it appears.
    video_items = "".join(item(c, "v") for c in timeline.clips)
    audio_items = "".join(item(c, "a") for c in timeline.clips if c.source.has_audio)
    audio = (
        "<audio><numOutputChannels>2</numOutputChannels><format><samplecharacteristics><depth>16</depth>"
        "<samplerate>48000</samplerate></samplecharacteristics></format>"
        f"<track>{audio_items}<enabled>TRUE</enabled><locked>FALSE</locked></track></audio>"
        if audio_items
        else ""
    )
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<!DOCTYPE xmeml>",
            '<xmeml version="4">',
            f'<sequence id="sequence-1"><name>{escape(timeline.name)}</name>'
            f"<duration>{timeline.duration}</duration>{rate}{_timecode_xml(seq)}<media>"
            f"<video><format><samplecharacteristics>{rate}<width>{timeline.width}</width>"
            f"<height>{timeline.height}</height><pixelaspectratio>square</pixelaspectratio>"
            "<fielddominance>none</fielddominance></samplecharacteristics></format>"
            f"<track>{video_items}<enabled>TRUE</enabled><locked>FALSE</locked></track></video>"
            f"{audio}</media></sequence>",
            "</xmeml>",
            "",
        ]
    )


# ------------------------------------------------------------------------------- CMX 3600 EDL


def _edl_text(text: str, limit: int = 70) -> str:
    """EDL lines are plain ASCII on one line."""
    cleaned = re.sub(r"[\r\n\t]+", " ", text).encode("ascii", "replace").decode("ascii")
    return cleaned[:limit]


def edl(timeline: Timeline) -> str:
    seq = timeline.rate
    lines = [f"TITLE: {_edl_text(timeline.name)}", "FCM: NON-DROP FRAME", ""]
    for clip in timeline.clips[:999]:
        track = "AA/V" if clip.source.has_audio else "V"
        lines.append(
            f"{clip.number:03d}  AX       {track:<5} C        "
            f"{seq.timecode(clip.src_in)} {seq.timecode(clip.src_out)} "
            f"{seq.timecode(clip.rec_in)} {seq.timecode(clip.rec_out)}"
        )
        lines.append(f"* FROM CLIP NAME: {_edl_text(clip.source.name, 200)}")
        lines.append(f"* SOURCE FILE: {_edl_text(str(clip.source.path), 400)}")
        lines.append("")
    return "\r\n".join(lines)


# --------------------------------------------------------------------------------- Subtitles


def _srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt(timeline: Timeline) -> str:
    """Subtitles for the edited sequence: each clip's speech, moved to where the clip sits.
    Empty when no clip has speech."""
    seq = timeline.rate
    blocks: list[str] = []
    for clip in timeline.clips:
        clip_start = seq.seconds(clip.src_in)
        clip_end = seq.seconds(clip.src_out)
        offset = seq.seconds(clip.rec_in) - clip_start
        for segment in clip.source.segments:
            try:
                seg_start, seg_end = float(segment["start"]), float(segment["end"])
            except (KeyError, TypeError, ValueError):
                continue
            text = str(segment.get("text") or "").strip()
            if not text or seg_end <= clip_start or seg_start >= clip_end:
                continue
            start = max(clip_start, seg_start) + offset
            end = min(clip_end, seg_end) + offset
            if end - start < 0.05:
                continue
            blocks.append(f"{len(blocks) + 1}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n")
    return "\n".join(blocks)
