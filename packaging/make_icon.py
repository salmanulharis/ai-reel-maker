"""Draw the app icon: packaging/app.ico (installer, shortcuts) and frontend/public/favicon.png
(the app window). Run again only to change the design: python packaging/make_icon.py

A soft white rounded tile (an iOS-style squircle) with a violet-to-pink play button and a
slanted cut through it: a clip cut from a video. Drawn at 4x size and scaled down, so the edges
are smooth at every size.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
S = 4096  # drawing size; the master image is 1024


def squircle(inset: float = 0.06, n: float = 5.0) -> Image.Image:
    y, x = np.mgrid[0:S, 0:S].astype(np.float32)
    c, r = (S - 1) / 2, S * (0.5 - inset)
    inside = np.abs((x - c) / r) ** n + np.abs((y - c) / r) ** n <= 1.0
    return Image.fromarray((inside * 255).astype(np.uint8), "L")


def gradient(first, last, diagonal: bool = True) -> Image.Image:
    y, x = np.mgrid[0:S, 0:S].astype(np.float32) / (S - 1)
    t = (x + y) / 2 if diagonal else y
    a, b = np.array(first, np.float32), np.array(last, np.float32)
    rgb = a + (b - a) * t[..., None]
    alpha = np.full((S, S, 1), 255, np.float32)
    return Image.fromarray(np.concatenate([rgb, alpha], -1).astype(np.uint8), "RGBA")


def rounded_triangle(points, radius: int) -> Image.Image:
    layer = Image.new("L", (S, S), 0)
    pen = ImageDraw.Draw(layer)
    pen.polygon(points, fill=255)
    pen.line(points + [points[0]], fill=255, width=radius * 2, joint="curve")
    for x, y in points:
        pen.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return layer


def shrink(points, cx: float, cy: float, by: float):
    out = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        length = (dx * dx + dy * dy) ** 0.5
        out.append((x - dx / length * by, y - dy / length * by))
    return out


def draw() -> Image.Image:
    image = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    image.paste(gradient((255, 255, 255), (238, 236, 246), diagonal=False), (0, 0), squircle())
    cx, cy, h = S * 0.535, S * 0.5, S * 0.40
    corners = [(cx - h * 0.45, cy - h / 2), (cx - h * 0.45, cy + h / 2), (cx + h * 0.55, cy)]
    radius = int(S * 0.04)
    play = rounded_triangle(shrink(corners, cx, cy, radius * 1.6), radius)
    cut = Image.new("L", (S, S), 0)
    gx = cx - h * 0.05
    ImageDraw.Draw(cut).polygon([(gx - S * 0.02, 0), (gx + S * 0.035, 0), (gx - S * 0.035, S), (gx - S * 0.09, S)], fill=255)
    play = Image.fromarray(np.minimum(np.array(play), 255 - np.array(cut)))
    image.paste(gradient((124, 92, 255), (255, 92, 140)), (0, 0), play)
    return image.resize((1024, 1024), Image.LANCZOS)


def main() -> None:
    image = draw()
    sizes = [(16, 16), (20, 20), (24, 24), (32, 32), (40, 40), (48, 48), (64, 64), (128, 128), (256, 256)]
    image.save(ROOT / "packaging" / "app.ico", sizes=sizes)
    image.resize((256, 256), Image.LANCZOS).save(ROOT / "frontend" / "public" / "favicon.png")


if __name__ == "__main__":
    main()
