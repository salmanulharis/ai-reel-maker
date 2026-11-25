"""Narration: the per-line cache, word timings and the English check (no real voice model)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.voice import kokoro


def fake_speaker(calls: list):
    def speak(text, voice, speed):
        calls.append((text, voice, speed))
        return np.zeros(int(kokoro.SAMPLE_RATE * 0.5 * len(text.split())), dtype=np.float32)

    return speak


def no_align(path, text):
    return []


def test_lines_are_spoken_once_and_reused(tmp_path):
    calls: list = []
    lines = [("b1", "Hello there friend."), ("b2", ""), ("b3", "Second line here now.")]
    first = kokoro.narrate(tmp_path, lines, voice="af_heart", speed=1.0, speak=fake_speaker(calls), align=no_align)
    assert set(first) == {"b1", "b3"} and len(calls) == 2
    assert first["b1"]["duration"] == pytest.approx(1.5)
    assert (tmp_path / kokoro.NARRATION_DIR / first["b1"]["file"]).is_file()
    # Spread timings when Whisper gives nothing: every word, in order, within the line.
    words = first["b1"]["words"]
    assert [w["text"] for w in words] == ["Hello", "there", "friend."] and words[-1]["end"] == pytest.approx(1.5)
    again = kokoro.narrate(tmp_path, lines, voice="af_heart", speed=1.0, speak=fake_speaker(calls), align=no_align)
    assert len(calls) == 2 and again == first
    changed = kokoro.narrate(
        tmp_path, [("b1", "Hello there friend."), ("b3", "A new second line.")], voice="af_heart", speed=1.0,
        speak=fake_speaker(calls), align=no_align,
    )
    assert len(calls) == 3 and changed["b1"]["key"] == first["b1"]["key"]
    folder = tmp_path / kokoro.NARRATION_DIR
    manifest = json.loads((folder / kokoro.MANIFEST).read_text(encoding="utf-8"))
    assert set(manifest) == {first["b1"]["key"], changed["b3"]["key"]}
    assert not (folder / first["b3"]["file"]).exists()


def test_voice_and_speed_change_the_key():
    key = kokoro.line_key("Same words.", "af_heart", 1.0)
    assert key == kokoro.line_key("  Same   words. ", "af_heart", 1.0)
    assert key != kokoro.line_key("Same words.", "am_michael", 1.0)
    assert key != kokoro.line_key("Same words.", "af_heart", 1.1)
    assert kokoro.valid_voice("nope") == kokoro.DEFAULT_VOICE
    assert kokoro.language_for("bf_emma") == "en-gb" and kokoro.language_for("af_heart") == "en-us"


def test_match_words_takes_heard_times_and_fills_gaps():
    heard = [("hello", 0.1, 0.4), ("big", 0.5, 0.7), ("world", 1.2, 1.6)]
    timed = kokoro.match_words("Hello, big wide world!", heard)
    assert [w["text"] for w in timed] == ["Hello,", "big", "wide", "world!"]
    assert timed[0]["start"] == 0.1 and timed[3]["start"] == 1.2
    assert 0.7 <= timed[2]["start"] < timed[2]["end"] <= 1.2
    tail = kokoro.match_words("one two three", [("one", 0.0, 0.3)])
    assert tail[-1]["end"] >= tail[-1]["start"] >= tail[0]["end"]
    assert kokoro.match_words("", heard) == []


def test_spread_words():
    words = kokoro.spread_words("a bb ccc", 3.0)
    assert words[0]["start"] == 0 and words[-1]["end"] == pytest.approx(3.0)
    assert words[2]["end"] - words[2]["start"] > words[0]["end"] - words[0]["start"]
    assert kokoro.spread_words("", 2) == [] and kokoro.spread_words("x", 0) == []


def test_speakable_only_for_latin_text():
    assert kokoro.speakable("Hello, café 42!")
    assert not kokoro.speakable("これは日本語です")
    assert not kokoro.speakable("1234 ...")


def test_download_status_lists_voices():
    status = kokoro.download_status()
    assert status["voices"]["af_heart"].startswith("Heart") and status["size_mb"] == kokoro.MODEL_DOWNLOAD_MB
