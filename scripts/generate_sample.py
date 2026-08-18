from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "samples" / "sample.mp4"


def main() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required to generate samples/sample.mp4")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    font = _font()
    draw = (
        f"drawtext=fontfile='{font}':text='{{text}}':fontsize=48:fontcolor=white:"
        "x=(w-text_w)/2:y=(h-text_h)/2"
        if font
        else "null"
    )
    labels = ["Introduction", "Product demo", "Key benefit", "Conclusion"]
    colors = ["0x2563eb", "0x16a34a", "0xdc2626", "0xca8a04"]
    cmd = [ffmpeg, "-y"]
    for color in colors:
        cmd += ["-f", "lavfi", "-i", f"color=c={color}:s=1280x720:d=6:r=30"]
    cmd += ["-f", "lavfi", "-i", "sine=frequency=220:duration=24"]
    if font:
        parts = [
            f"[{i}:v]{draw.replace('{text}', label)}[v{i}]" for i, label in enumerate(labels)
        ]
        parts.append("[v0][v1][v2][v3]concat=n=4:v=1:a=0[v]")
        filter_complex = ";".join(parts)
    else:
        filter_complex = "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[v]"
    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "4:a",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(OUTPUT),
    ]
    subprocess.run(cmd, check=True)
    print(f"Wrote {OUTPUT}")


def _font() -> str | None:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return str(path).replace("\\", "/").replace(":", r"\:")
    return None


if __name__ == "__main__":
    sys.exit(main())
