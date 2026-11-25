#!/usr/bin/env python3
"""Phase 0 trial: how fast the narration voice (Kokoro) speaks on this computer.

Speaks a few sample lines with each chosen voice and prints, per voice: the real-time factor
(seconds of work per second of speech), the measured speaking rate in words per second (the
brief's word budgets assume app.brief.schema.WORDS_PER_SECOND), and the time Whisper takes to
time the words. Results are also saved to data/voice_benchmark.json.

    python scripts/voice_benchmark.py                    # af_heart
    python scripts/voice_benchmark.py --voices af_heart am_michael --speed 1.0
    python scripts/voice_benchmark.py --download         # download the voice model first
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.brief.schema import WORDS_PER_SECOND  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.voice import kokoro  # noqa: E402

LINES = [
    "Three friends set out before sunrise, not knowing the mountain had other plans.",
    "Halfway up, the trail disappears under fresh snow, and every step becomes a decision.",
    "Then the clouds open, and the whole valley lights up below them.",
    "It took nine hours, two wrong turns and one very cold lunch, but they made it.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voices", nargs="+", default=[kokoro.DEFAULT_VOICE])
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--download", action="store_true", help="download the voice model if it is missing")
    parser.add_argument("--no-align", action="store_true", help="skip the Whisper word timing")
    args = parser.parse_args()
    settings = get_settings()
    if not kokoro.runtime_available():
        print("kokoro-onnx is not installed: pip install -r requirements.txt")
        return 1
    if not kokoro.installed():
        if not args.download:
            print("The voice model is not downloaded. Run again with --download (about 340 MB).")
            return 1
        print("Downloading the voice model…")
        kokoro._MODEL.begin()
        kokoro._MODEL.download()
        if not kokoro.installed():
            print(kokoro.download_status().get("error") or "Download failed")
            return 1
    started = time.perf_counter()
    narrator = kokoro.get_narrator(int(getattr(settings, "tts_threads", 0) or 0))
    load_seconds = time.perf_counter() - started
    print(f"Voice model loaded in {load_seconds:.1f}s")
    results = {"load_seconds": round(load_seconds, 2), "speed": args.speed, "assumed_words_per_second": WORDS_PER_SECOND, "voices": {}}
    for voice in args.voices:
        voice = kokoro.valid_voice(voice)
        narrator.speak("Warming up.", voice, args.speed)
        audio_seconds = work_seconds = align_seconds = 0.0
        words = 0
        with tempfile.TemporaryDirectory() as folder:
            for number, line in enumerate(LINES):
                began = time.perf_counter()
                samples = narrator.speak(line, voice, args.speed)
                work_seconds += time.perf_counter() - began
                audio_seconds += len(samples) / kokoro.SAMPLE_RATE
                words += len(line.split())
                if not args.no_align:
                    import soundfile

                    path = Path(folder) / f"line{number}.wav"
                    soundfile.write(str(path), samples, kokoro.SAMPLE_RATE)
                    began = time.perf_counter()
                    timed = kokoro.align_words(path, line, settings)
                    align_seconds += time.perf_counter() - began
                    if not timed:
                        print(f"  (Whisper found no words in line {number + 1}; captions would use spread timings)")
        row = {
            "audio_seconds": round(audio_seconds, 2),
            "work_seconds": round(work_seconds, 2),
            "real_time_factor": round(work_seconds / max(0.01, audio_seconds), 3),
            "words_per_second": round(words / max(0.01, audio_seconds), 2),
            "align_seconds": round(align_seconds, 2),
        }
        results["voices"][voice] = row
        print(
            f"{voice}: {row['audio_seconds']}s of speech in {row['work_seconds']}s "
            f"(x{row['real_time_factor']} real time), {row['words_per_second']} words/s"
            + (f", word timing {row['align_seconds']}s" if not args.no_align else "")
        )
    out = settings.data_dir / "voice_benchmark.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {out}")
    print(f"Brief word budgets assume {WORDS_PER_SECOND} words/s (app/brief/schema.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
