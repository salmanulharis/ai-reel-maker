"""What an engine reads to write a brief: the same for every engine.

The moments come from the finished analysis exactly as the AI editor reads them (one line per
moment: alias, time, length, score, what is said, seen and heard). For Claude, small labelled
frame sheets can be added: a few thumbnails per image, each marked with its moment's alias, so a
few images cover dozens of moments.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.analysis.director import _target, alias_map, candidate_pool, moment_line, request_text
from app.brief.schema import BRIEF_DIR, MAX_BEATS, WORDS_PER_SECOND
from app.models.schemas import HighlightConfig
from app.utils.security import safe_path_argument

_log = logging.getLogger(__name__)

# Moments an engine reads at most (best first, then in time order).
MAX_MOMENTS = 150
# Frame sheets: this many moments in total, in sheets of COLS x ROWS thumbnails.
SHEET_MOMENTS = 36
SHEET_COLS = 3
SHEET_ROWS = 2
THUMB_WIDTH = 320


@dataclass
class EvidencePack:
    request: str
    requirements: dict
    notes: list[str]
    video_name: str
    target_text: str
    target_seconds: float
    shorts: int
    aliases: dict[str, dict]
    lines: dict[str, str]
    sheets: list[Path] = field(default_factory=list)
    # What the moments show and sound like (content_index.json), for the AI editor.
    content: dict = field(default_factory=dict)
    # A voice-over was asked for: narration is written for (nearly) every beat.
    voice_over: bool = True

    @property
    def beats_range(self) -> tuple[int, int]:
        low = max(2, round(self.target_seconds / 12))
        high = min(MAX_BEATS, max(low + 1, round(self.target_seconds / 5)))
        return low, high

    def text(self) -> str:
        """The evidence as one block of text (the prompt adds the instructions)."""
        low, high = self.beats_range
        parts = [
            f"Video: {self.video_name or 'untitled'}",
            f"Request: {self.request}",
        ]
        wanted = {k: v for k, v in (self.requirements or {}).items() if v and k != "summary"}
        if wanted:
            parts.append("Requirements: " + json.dumps(wanted, ensure_ascii=False))
        if self.notes:
            parts.append("Changes asked for since the first version: " + " / ".join(self.notes))
        parts.append(
            f"Length: {self.target_text}. Plan {low}-{high} beats. Narration is read at about "
            f"{WORDS_PER_SECOND:.1f} words per second, so a beat of 6 seconds takes at most about "
            f"{int(6 * WORDS_PER_SECOND * 0.85)} words."
        )
        parts.append(voice_over_text(self.voice_over))
        parts.append("Moments (alias [time, length] score | says | sees | hears):")
        parts.extend(self.lines[alias] for alias in self.lines)
        return "\n".join(parts)

    def key(self, *extra: object) -> str:
        """Identifies these inputs (plus engine settings in ``extra``), so a brief is reused."""
        data = {
            "request": self.request,
            "requirements": self.requirements,
            "notes": self.notes,
            "target": [self.target_text, self.target_seconds, self.shorts],
            "moments": [
                [alias, item.get("id"), round(float(item["start"]), 2), round(float(item["end"]), 2)]
                for alias, item in self.aliases.items()
            ],
            "sheets": [path.name for path in self.sheets],
            "voice_over": self.voice_over,
            "extra": [str(item) for item in extra],
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def voice_over_text(voice_over: bool) -> str:
    if voice_over:
        return (
            "A voice-over is wanted: write narration for every beat (it plays over the video while the video's own "
            "sound is lowered). Leave a beat's narration empty only when its own spoken words must be heard."
        )
    return "No voice-over: leave every narration empty and tell the story with on_screen_text."


def target_seconds(config: HighlightConfig) -> float:
    count = max(1, int(config.shorts_count or 0))
    if config.length_mode == "auto" and count == 1:
        return max(float(config.min_duration) * 2, 60.0)
    return float(config.max_duration) * count


def build_pack(
    config: HighlightConfig,
    candidates: list[dict],
    content: dict,
    *,
    video_name: str = "",
    limit: int = MAX_MOMENTS,
) -> EvidencePack:
    pool = candidate_pool(candidates, content, limit)
    numbered = alias_map(pool)
    target_text, _picks = _target(config, pool or [{"start": 0, "end": 5}])
    request = request_text(config) or config.custom_focus.strip() or "the most engaging moments of this video"
    return EvidencePack(
        request=request,
        requirements=config.requirements.model_dump() if config.requirements else {},
        notes=list(config.refine_notes),
        video_name=video_name,
        target_text=target_text,
        target_seconds=target_seconds(config),
        shorts=int(config.shorts_count or 0),
        aliases=numbered,
        lines={alias: moment_line(alias, item, content) for alias, item in numbered.items()},
        content=content or {},
        voice_over=bool(config.narration),
    )


def add_frame_sheets(
    pack: EvidencePack,
    job_dir: Path,
    sources: list[dict],
    run: Callable[[list[str]], object],
    check_cancel: Callable[[], None] | None = None,
) -> list[Path]:
    """Make labelled frame sheets of the best moments (for Claude). Frames are kept between
    runs. ``run`` runs an FFmpeg command. Moments whose frame cannot be read are left out."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return []
    folder = job_dir / BRIEF_DIR / "frames"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {str(s.get("id") or "v0"): Path(s.get("analysis_path") or s.get("path") or "") for s in sources or []}
    first = next(iter(paths.values()), None)
    ranked = sorted(pack.aliases, key=lambda a: float(pack.aliases[a].get("score") or 0), reverse=True)[:SHEET_MOMENTS]
    chosen = [a for a in pack.aliases if a in set(ranked)]
    thumbs: list[tuple[str, Path]] = []
    for alias in chosen:
        if check_cancel:
            check_cancel()
        item = pack.aliases[alias]
        middle = (float(item["start"]) + float(item["end"])) / 2
        source = paths.get(str(item.get("source_id") or "v0")) or first
        if not source or not source.is_file():
            continue
        name = hashlib.sha1(f"{item.get('id')}|{source}|{middle:.2f}".encode()).hexdigest()[:16]
        frame = folder / f"{name}.jpg"
        if not frame.is_file():
            try:
                run(_frame_command(source, middle, frame))
            except Exception as exc:  # noqa: BLE001 - one unreadable frame only leaves it out
                from app.video.ffmpeg_runner import CancelledError

                if isinstance(exc, CancelledError):
                    raise
                _log.debug("No frame for %s: %s", alias, exc)
                continue
        if frame.is_file():
            thumbs.append((alias, frame))
    sheets: list[Path] = []
    per_sheet = SHEET_COLS * SHEET_ROWS
    sheet_dir = job_dir / BRIEF_DIR / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for old in sheet_dir.glob("sheet_*.jpg"):
        old.unlink(missing_ok=True)
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:  # Pillow before 10.1
        font = ImageFont.load_default()
    for number, start in enumerate(range(0, len(thumbs), per_sheet), start=1):
        group = thumbs[start : start + per_sheet]
        images = []
        for alias, path in group:
            try:
                with Image.open(path) as image:
                    images.append((alias, image.convert("RGB").copy()))
            except OSError:
                continue
        if not images:
            continue
        cell_w = THUMB_WIDTH
        cell_h = max(image.height for _, image in images)
        sheet = Image.new("RGB", (cell_w * SHEET_COLS, cell_h * SHEET_ROWS), (16, 16, 16))
        draw = ImageDraw.Draw(sheet)
        for index, (alias, image) in enumerate(images):
            x, y = (index % SHEET_COLS) * cell_w, (index // SHEET_COLS) * cell_h
            sheet.paste(image, (x, y))
            box = draw.textbbox((0, 0), alias, font=font)
            draw.rectangle((x, y, x + box[2] - box[0] + 16, y + box[3] - box[1] + 14), fill=(0, 0, 0))
            draw.text((x + 8, y + 5 - box[1]), alias, fill=(255, 215, 0), font=font)
        path = sheet_dir / f"sheet_{number:02d}.jpg"
        sheet.save(path, "JPEG", quality=80)
        sheets.append(path)
    pack.sheets = sheets
    return sheets


def _frame_command(source: Path, at: float, dest: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, at):.3f}",
        "-i",
        safe_path_argument(source),
        "-frames:v",
        "1",
        "-vf",
        f"scale={THUMB_WIDTH}:-2",
        "-q:v",
        "5",
        safe_path_argument(dest),
    ]
