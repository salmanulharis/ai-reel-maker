#!/usr/bin/env python3
"""Phase 0 trial: how fast Remotion draws a narrated video on this computer.

Makes a synthetic base video (FFmpeg test pattern), builds a timeline with a title, beat texts
and word-highlighted captions, and renders it with each worker count asked for. Prints frames
per second, the time per second of video, and the peak memory of Node + Chrome, so the chunk
size and the default number of workers can be chosen from real numbers. Results are saved to
data/remotion_benchmark.json. Nothing else on the computer is changed.

    python scripts/remotion_benchmark.py                          # 1080x1920, 20 s, workers 4 and 6
    python scripts/remotion_benchmark.py --size 1920x1080 --seconds 30 --workers 2 4 6
    python scripts/remotion_benchmark.py --prepare                # prepare the tools first (downloads)
    python scripts/remotion_benchmark.py --unload-ollama          # free Ollama's memory first
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.compose import remotion  # noqa: E402
from app.compose.render_spec import RenderSpec, SpecCaption, SpecText, SpecVideo, SpecWord  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.video.ffmpeg_runner import FFmpegRunner  # noqa: E402


class PeakMemory:
    """Peak resident memory of this process's children (Node, Chrome), sampled twice a second."""

    def __init__(self):
        self.peak = 0
        self._stop = threading.Event()

    def __enter__(self):
        threading.Thread(target=self._watch, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()

    def _watch(self):
        import psutil

        me = psutil.Process()
        while not self._stop.wait(0.5):
            total = 0
            for child in me.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.Error:
                    pass
            self.peak = max(self.peak, total)


def spec_for(width: int, height: int, fps: float, seconds: float) -> RenderSpec:
    words = "Three friends set out before sunrise not knowing the mountain had other plans".split()
    captions = []
    cursor = 3.0
    while cursor + 2.5 < seconds:
        chunk = words[: 4 if height > width else 6]
        words = words[len(chunk) :] + chunk
        timed = [SpecWord(text=w, start=round(cursor + i * 0.4, 2), end=round(cursor + (i + 1) * 0.4, 2)) for i, w in enumerate(chunk)]
        captions.append(SpecCaption(start=cursor, end=cursor + 0.4 * len(chunk), text=" ".join(chunk), words=timed))
        cursor += 0.4 * len(chunk) + 0.3
    texts = [SpecText(kind="title", text="A day on the mountain", start=0, end=2.6)]
    texts += [SpecText(kind="beat", text=f"Beat {n}", start=n * 6.0 + 0.3, end=n * 6.0 + 3.8) for n in range(1, int(seconds // 6))]
    return RenderSpec(
        width=width,
        height=height,
        fps=fps,
        duration=seconds,
        video=SpecVideo(kind="base", src="base.mp4"),
        texts=texts,
        captions=captions,
        caption_style="bold",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", default="1080x1920")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--workers", nargs="+", type=int, default=[4, 6])
    parser.add_argument("--prepare", action="store_true", help="prepare the Remotion tools first (npm install, bundle, browser)")
    parser.add_argument("--unload-ollama", action="store_true")
    args = parser.parse_args()
    width, height = (int(part) for part in args.size.lower().split("x"))
    settings = get_settings()
    if args.prepare and not remotion.ready(settings):
        print("Preparing the Remotion tools (one time)…")
        remotion._prepare(settings)
        state = remotion.status(settings)["prepare"]
        if state.get("state") == "error":
            print(state.get("error"))
            return 1
    if not remotion.ready(settings):
        print("The Remotion tools are not prepared. Use Settings → Narrated videos, or run with --prepare.")
        return 1
    if args.unload_ollama:
        from app.services.ollama_runtime import get_ollama_client

        print("Unloaded:", get_ollama_client(settings).unload_others(None) or "nothing")
    runner = FFmpegRunner()
    results = {"size": args.size, "fps": args.fps, "seconds": args.seconds, "runs": []}
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        base = folder / "base.mp4"
        runner.run(
            [
                "ffmpeg", "-hide_banner", "-y", "-f", "lavfi",
                "-i", f"testsrc2=size={width}x{height}:rate={args.fps}:duration={args.seconds}",
                "-c:v", "libx264", "-preset", "veryfast", "-g", str(int(args.fps / 2)), "-pix_fmt", "yuv420p", str(base),
            ]
        )
        spec = spec_for(width, height, args.fps, args.seconds)
        frames = int(round(args.seconds * args.fps))
        for workers in args.workers:
            settings.remotion_concurrency = workers
            shutil.rmtree(folder / "chunks", ignore_errors=True)
            began = time.perf_counter()
            with PeakMemory() as memory:
                remotion.render_picture(json.loads(spec.model_dump_json()), folder, settings, runner)
            took = time.perf_counter() - began
            row = {
                "workers": workers,
                "seconds": round(took, 1),
                "frames_per_second": round(frames / took, 1),
                "seconds_per_video_second": round(took / args.seconds, 2),
                "peak_memory_mb": round(memory.peak / 1024 / 1024),
            }
            results["runs"].append(row)
            print(
                f"{workers} workers: {frames} frames in {row['seconds']}s = {row['frames_per_second']} fps "
                f"({row['seconds_per_video_second']}s per second of video), peak {row['peak_memory_mb']} MB"
            )
    out = settings.data_dir / "remotion_benchmark.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
