"""Several scene frames tiled into one labelled picture for the vision model.

Ollama scales every Qwen3-VL picture up to about a megapixel before reading it, so one 512 px
frame costs as many prompt tokens as a 2x2 grid of them (measured: 1,044 tokens each). Tiling
keeps each frame at the size it is sent at today and lets the model read several scenes for
the price of one. Every tile has a label with its scene id above it (never over the picture),
so replies are matched to scenes by id, not by position.
"""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Height of the label strip above each tile, the gap between tiles, and their colours.
LABEL_HEIGHT = 34
GAP = 8
_BACKGROUND = (24, 24, 24)
_LABEL_BACKGROUND = (0, 0, 0)
_LABEL_TEXT = (255, 255, 255)
_FONT_SIZE = 24
# Longest grid shape preferred (log of width / height, about 2.2:1).
_MAX_SHAPE = 0.8


def can_tile(path: Path | str) -> bool:
    """True when the frame is a picture that can be opened (only its header is read)."""
    try:
        with Image.open(path) as image:
            width, height = image.size
        return width > 0 and height > 0
    except Exception:  # noqa: BLE001 - such a frame is sent on its own, as before
        return False


def tile_label(scene_id: int) -> str:
    return f"Scene {scene_id}"


def layout(count: int, tile_width: int, tile_height: int) -> tuple[int, int]:
    """(columns, rows) for ``count`` tiles: among grids no longer than about 2.2:1, the one with
    the fewest empty cells, then the most square one (the most square overall if none is)."""
    count = max(1, int(count))
    options: list[tuple[bool, int, float, int, int]] = []
    for columns in range(1, count + 1):
        rows = math.ceil(count / columns)
        if (columns - 1) * rows >= count:
            continue  # a whole column would be empty
        width = columns * tile_width + (columns - 1) * GAP
        height = rows * (tile_height + LABEL_HEIGHT) + (rows - 1) * GAP
        shape = abs(math.log(width / height))
        options.append((shape > _MAX_SHAPE, columns * rows - count, round(shape, 3), columns, rows))
    best = min(options)
    return best[3], best[4]


def compose(frames: list[tuple[int, Path | str]], longest_side: int) -> Image.Image:
    """One picture with every (scene id, frame) as a tile, in order, left to right and top to
    bottom. Each frame is scaled to ``longest_side`` like a frame sent on its own."""
    if not frames:
        raise ValueError("no frames to tile")
    tiles: list[tuple[int, Image.Image]] = []
    for scene_id, path in frames:
        with Image.open(path) as source:
            tiles.append((scene_id, _fit(source.convert("RGB"), longest_side)))
    cell_width = max(tile.width for _, tile in tiles)
    cell_height = max(tile.height for _, tile in tiles)
    columns, rows = layout(len(tiles), cell_width, cell_height)
    width = columns * cell_width + (columns - 1) * GAP
    height = rows * (cell_height + LABEL_HEIGHT) + (rows - 1) * GAP
    grid = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(grid)
    font = _font()
    for index, (scene_id, tile) in enumerate(tiles):
        column, row = index % columns, index // columns
        left = column * (cell_width + GAP)
        top = row * (cell_height + LABEL_HEIGHT + GAP)
        draw.rectangle((left, top, left + cell_width - 1, top + LABEL_HEIGHT - 1), fill=_LABEL_BACKGROUND)
        draw.text((left + 8, top + (LABEL_HEIGHT - _FONT_SIZE) // 2 - 1), tile_label(scene_id), fill=_LABEL_TEXT, font=font)
        # Centred in its cell when frames differ in shape.
        grid.paste(
            tile,
            (left + (cell_width - tile.width) // 2, top + LABEL_HEIGHT + (cell_height - tile.height) // 2),
        )
    return grid


def encode(image: Image.Image, quality: int = 75) -> str:
    """Base64 JPEG, as the Ollama client sends single frames."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _fit(image: Image.Image, longest_side: int) -> Image.Image:
    """Scaled down so its longest side is at most ``longest_side`` (the same scaling the Ollama
    client applies to a frame sent on its own)."""
    width, height = image.size
    longest = max(width, height)
    if longest_side > 0 and longest > longest_side:
        scale = longest_side / longest
        return image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.BILINEAR,
        )
    return image


def _font():
    try:
        return ImageFont.load_default(size=_FONT_SIZE)
    except Exception:  # noqa: BLE001 - Pillow without FreeType: the small built-in font
        return ImageFont.load_default()
