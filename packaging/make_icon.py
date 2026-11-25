"""Draw the app icon: packaging/app.ico (installer, shortcuts) and frontend/public/favicon.png
(the app window). Run again only to change the design: python packaging/make_icon.py"""

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SIZE = 1024


def draw() -> Image.Image:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    # Rounded tile with a diagonal violet-to-coral gradient.
    gradient = Image.new("RGBA", (SIZE, SIZE))
    start, end = (99, 72, 230), (244, 98, 94)
    pixels = gradient.load()
    for y in range(SIZE):
        for x in range(SIZE):
            t = (x + y) / (2 * (SIZE - 1))
            pixels[x, y] = tuple(round(a + (b - a) * t) for a, b in zip(start, end)) + (255,)
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle((40, 40, SIZE - 40, SIZE - 40), radius=220, fill=255)
    image.paste(gradient, (0, 0), mask)
    pen = ImageDraw.Draw(image)
    # A vertical "reel" frame with a play triangle.
    pen.rounded_rectangle((300, 200, 724, 824), radius=70, outline=(255, 255, 255, 255), width=56)
    pen.polygon([(430, 380), (430, 644), (640, 512)], fill=(255, 255, 255, 255))
    # A small spark (the AI part) at the top right.
    cx, cy, r = 790, 250, 95
    pen.polygon(
        [(cx, cy - r), (cx + 26, cy - 26), (cx + r, cy), (cx + 26, cy + 26), (cx, cy + r), (cx - 26, cy + 26), (cx - r, cy), (cx - 26, cy - 26)],
        fill=(255, 236, 150, 255),
    )
    return image


def main() -> None:
    image = draw()
    image.save(ROOT / "packaging" / "app.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    image.resize((256, 256), Image.LANCZOS).save(ROOT / "frontend" / "public" / "favicon.png")


if __name__ == "__main__":
    main()
