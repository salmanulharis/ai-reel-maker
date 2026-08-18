from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CropPixels:
    x: int
    y: int
    width: int
    height: int

    def ffmpeg_filter(self) -> str:
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"


def _even(value: int) -> int:
    return value if value % 2 == 0 else value - 1


def normalize_crop(crop: dict | None) -> dict | None:
    if not crop:
        return None
    try:
        x = float(crop.get("x", 0))
        y = float(crop.get("y", 0))
        width = float(crop.get("width", 1))
        height = float(crop.get("height", 1))
    except (TypeError, ValueError):
        return None
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0 - x)
    height = min(max(height, 0.0), 1.0 - y)
    if width >= 0.985 and height >= 0.985 and x <= 0.01 and y <= 0.01:
        return None
    if width < 0.05 or height < 0.05:
        return None
    return {"x": x, "y": y, "width": width, "height": height}


def to_crop_pixels(source_width: int, source_height: int, crop: dict | None) -> CropPixels | None:
    region = normalize_crop(crop)
    if not region:
        return None
    width = max(2, _even(int(round(region["width"] * source_width))))
    height = max(2, _even(int(round(region["height"] * source_height))))
    x = max(0, _even(int(round(region["x"] * source_width))))
    y = max(0, _even(int(round(region["y"] * source_height))))
    if x + width > source_width:
        width = _even(source_width - x)
    if y + height > source_height:
        height = _even(source_height - y)
    if width < 2 or height < 2:
        return None
    if width >= source_width - 2 and height >= source_height - 2:
        return None
    return CropPixels(x=x, y=y, width=width, height=height)
