"""
Picture extras added to every clip: a closer "punch-in" framing on every other moment, a logo in
a corner, and a progress bar along the bottom. They are built into the clip's -vf filter graph,
so no extra encoding pass is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.video.reframe import ASPECT_RATIOS

PUNCH_IN = 1.1
LOGO_POSITIONS = {"top-left", "top-right", "bottom-left", "bottom-right"}
LOGO_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _even(value: float) -> int:
    return max(2, int(value) // 2 * 2)


def output_size(canvas_w: int, canvas_h: int, aspect_ratio: str) -> tuple[int, int]:
    """The size of a clip after the shape crop (FFmpeg rounds crops down to even sizes)."""
    ratio = ASPECT_RATIOS.get(aspect_ratio)
    if ratio is None:
        return _even(canvas_w), _even(canvas_h)
    if canvas_w / max(1, canvas_h) >= ratio:
        return _even(canvas_h * ratio), _even(canvas_h)
    return _even(canvas_w), _even(canvas_w / ratio)


def filter_path(path: Path) -> str:
    """A file path inside an FFmpeg filter argument (as for the subtitles filter)."""
    return Path(path).as_posix().replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


@dataclass
class Extras:
    punch_in: bool = False
    logo: Path | None = None
    logo_position: str = "bottom-right"
    logo_size: float = 0.12  # share of the video width
    logo_opacity: float = 0.85
    progress_bar: bool = False

    @property
    def active(self) -> bool:
        return self.punch_in or self.progress_bar or (self.logo is not None and self.logo_position in LOGO_POSITIONS)


def punch_in_filters(width: int, height: int) -> list[str]:
    """Frame the picture 10% closer, keeping the output size."""
    return [
        f"crop='trunc(iw/{PUNCH_IN}/2)*2':'trunc(ih/{PUNCH_IN}/2)*2'",
        f"scale={width}:{height}:flags=lanczos",
        "setsar=1",
    ]


def build_graph(
    chain: str,
    extras: Extras,
    width: int,
    height: int,
    fps: float,
    offset: float,
    total: float,
) -> str:
    """
    The clip's filter chain with a logo and/or progress bar laid over it. `offset` is where the
    clip starts in the finished video and `total` its length, so the bar grows smoothly across
    clips. Returns `chain` unchanged when there is nothing to overlay.
    """
    graph = chain
    label = 0
    margin = max(8, round(min(width, height) * 0.03))
    if extras.logo is not None and extras.logo_position in LOGO_POSITIONS:
        logo_w = _even(width * min(0.4, max(0.04, extras.logo_size)))
        opacity = min(1.0, max(0.1, extras.logo_opacity))
        x = f"{margin}" if "left" in extras.logo_position else f"main_w-overlay_w-{margin}"
        y = f"{margin}" if "top" in extras.logo_position else f"main_h-overlay_h-{margin}"
        graph = (
            f"{graph}[b{label}];movie='{filter_path(extras.logo)}',format=rgba,"
            f"colorchannelmixer=aa={opacity:.2f},scale={logo_w}:-1[logo];[b{label}][logo]overlay={x}:{y}"
        )
        label += 1
    if extras.progress_bar and total > 0:
        bar_h = max(4, _even(height * 0.008))
        graph = (
            f"{graph}[b{label}];color=c=white@0.9:s={width}x{bar_h}:r={max(1, round(fps))}[bar];"
            f"[b{label}][bar]overlay=x='-{width}+{width}*({offset:.3f}+t)/{total:.3f}':y={height - bar_h}:shortest=1"
        )
    return graph


def brand_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "brand"


def brand_logo(data_dir: Path) -> Path | None:
    """The saved logo image, if one was added."""
    folder = brand_dir(data_dir)
    if not folder.is_dir():
        return None
    return next((p for p in sorted(folder.glob("logo.*")) if p.suffix.lower() in LOGO_SUFFIXES and p.is_file()), None)
