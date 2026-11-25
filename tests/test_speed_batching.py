"""Batched speech to text (sentence-sized segments) and scenes tiled into one picture for the
video model."""

from __future__ import annotations

import re
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.ai.context_fit import (
    VISION_CAPTION_TOKENS,
    VISION_TOKENS_PER_SCENE,
    fitting_scenes_per_image,
    fitting_vision_batch,
    vision_ctx_for_batch,
    vision_scene_cost,
)
from app.ai.ollama import OllamaError, OllamaTruncatedError
from app.analysis import audio as audio_module
from app.analysis import frame_grid
from app.analysis.visual import analyze_scenes_visual
from app.config import Settings


# Speech to text


def _word(text, start, end):
    return types.SimpleNamespace(word=text, start=start, end=end)


def _segment(words, start=None, end=None, text=None):
    return types.SimpleNamespace(
        start=words[0].start if start is None else start,
        end=words[-1].end if end is None else end,
        text="".join(w.word for w in words) if text is None else text,
        words=words,
    )


def _parts(segments):
    return [(round(s.start, 2), round(s.end, 2), s.text.strip()) for s in audio_module._sentence_segments(segments)]


def test_batched_pieces_are_split_after_sentence_ends():
    words = [
        _word(" Hello", 0.0, 0.4), _word(" there.", 0.5, 0.9),
        _word(" How", 1.0, 1.2), _word(" are", 1.3, 1.5), _word(" you?", 1.6, 2.0),
        _word(" Fine", 2.2, 2.6),
    ]
    assert _parts([_segment(words)]) == [
        (0.0, 0.9, "Hello there."),
        (1.0, 2.0, "How are you?"),
        (2.2, 2.6, "Fine"),
    ]


def test_batched_pieces_are_split_at_long_pauses_only():
    words = [_word(" one", 0.0, 0.5), _word(" two", 0.9, 1.3), _word(" three", 2.5, 3.0)]
    assert _parts([_segment(words)]) == [(0.0, 1.3, "one two"), (2.5, 3.0, "three")]


def test_sentence_ends_inside_quotes_and_in_other_scripts_count():
    words = [_word(' "Stop."', 0.0, 0.5), _word(" 好。", 0.6, 0.8), _word("吗？", 0.9, 1.0), _word(" नमस्ते।", 1.1, 1.5)]
    assert [text for _, _, text in _parts([_segment(words)])] == ['"Stop."', "好。", "吗？", "नमस्ते।"]


def test_words_written_without_spaces_are_joined_as_written():
    words = [_word("今日", 0.0, 0.3), _word("は", 0.3, 0.4), _word("晴れ", 0.4, 0.8)]
    assert _parts([_segment(words)]) == [(0.0, 0.8, "今日は晴れ")]


def test_a_segment_without_word_times_is_kept_whole():
    plain = types.SimpleNamespace(start=1.0, end=4.0, text=" No words here.", words=None)
    assert list(audio_module._sentence_segments([plain])) == [plain]


class FakeModel:
    def __init__(self):
        self.sequential = 0

    def transcribe(self, audio, **options):
        self.sequential += 1
        return iter([_segment([_word(" Old", 0.0, 1.0)])]), types.SimpleNamespace(language="en", language_probability=0.9)


class FakeBatched:
    def __init__(self, segments):
        self.segments = segments
        self.audio = None
        self.options: dict = {}

    def transcribe(self, audio, **options):
        self.audio = audio
        self.options = options
        return iter(self.segments), types.SimpleNamespace(language="en", language_probability=0.97)


def test_batched_mode_is_used_with_a_batch_size_and_its_pieces_become_sentences(monkeypatch):
    model = FakeModel()
    batched = FakeBatched([_segment([_word(" First.", 0.0, 1.0), _word(" Second.", 1.2, 2.0)])])
    monkeypatch.setattr(audio_module, "_get_whisper_model", lambda *a: model)
    monkeypatch.setattr(audio_module, "_batched_pipeline", lambda m: batched)
    transcript = audio_module._transcribe_with(Path("a.wav"), "small", "cpu", "int8", 4, 1, batch_size=8)
    assert model.sequential == 0
    assert batched.options["batch_size"] == 8 and batched.options["word_timestamps"] is True
    assert batched.options["vad_filter"] is True
    assert [(s.start, s.end, s.text) for s in transcript.segments] == [(0.0, 1.0, "First."), (1.2, 2.0, "Second.")]
    assert [w.word for w in transcript.words] == ["First.", "Second."]
    assert transcript.language == "en" and transcript.language_probability == pytest.approx(0.97)


def test_without_a_batch_size_the_one_at_a_time_method_is_kept(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(audio_module, "_get_whisper_model", lambda *a: model)
    monkeypatch.setattr(audio_module, "_batched_pipeline", lambda m: pytest.fail("batched mode used"))
    transcript = audio_module._transcribe_with(Path("a.wav"), "small", "cpu", "int8", 4, 1)
    assert model.sequential == 1 and [s.text for s in transcript.segments] == ["Old"]


def test_an_older_faster_whisper_without_batching_falls_back(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(audio_module, "_get_whisper_model", lambda *a: model)
    monkeypatch.setattr(audio_module, "_batched_pipeline", lambda m: None)
    audio_module._transcribe_with(Path("a.wav"), "small", "cpu", "int8", 4, 1, batch_size=8)
    assert model.sequential == 1


def test_batched_mode_continues_after_the_saved_point(monkeypatch):
    batched = FakeBatched([_segment([_word(" Again.", 0.5, 1.5), _word(" More.", 1.6, 2.0)])])
    monkeypatch.setattr(audio_module, "_get_whisper_model", lambda *a: FakeModel())
    monkeypatch.setattr(audio_module, "_batched_pipeline", lambda m: batched)
    monkeypatch.setattr(audio_module, "_audio_after", lambda path, seconds: np.zeros(16000 * 5, dtype=np.float32))
    resume = {"until": 10.0, "segments": [{"start": 1.0, "end": 10.0, "text": "Before"}], "language": "de"}
    transcript = audio_module._transcribe_with(Path("a.wav"), "small", "cpu", "int8", 4, 1, resume=resume, batch_size=4)
    assert batched.options["language"] == "de"
    assert [(s.start, s.end, s.text) for s in transcript.segments] == [
        (1.0, 10.0, "Before"),
        (10.5, 11.5, "Again."),
        (11.6, 12.0, "More."),
    ]


def test_batch_size_reaches_the_transcription(monkeypatch):
    seen = {}

    def run(*args, **kwargs):
        seen.update(kwargs)
        return audio_module.Transcript()

    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", types.SimpleNamespace(WhisperModel=object))
    monkeypatch.setattr(audio_module, "_transcribe_with", run)
    _transcript, info = audio_module.transcribe_detailed(Path("a.wav"), device="cpu", batch_size=5)
    assert info["ok"] and seen["batch_size"] == 5


# Scenes per picture: sums


def test_scenes_per_picture_shrinks_to_fit_the_context():
    assert fitting_scenes_per_image(4, 2048, 256, VISION_TOKENS_PER_SCENE) == 4
    # Prompt mode also asks for a caption per scene: longer replies.
    with_caption = VISION_TOKENS_PER_SCENE + VISION_CAPTION_TOKENS
    assert fitting_scenes_per_image(4, 2048, 256, with_caption) == 3
    assert fitting_scenes_per_image(4, 3072, 256, with_caption) == 4
    assert fitting_scenes_per_image(20, 16384, 256) == 9  # at most a 3x3 grid
    assert fitting_scenes_per_image(4, 1024, 256) == 1
    assert fitting_scenes_per_image(0, 8192, 256) == 1


def test_one_frame_per_picture_keeps_the_previous_sums():
    for batch in (1, 2, 4):
        num_ctx = vision_ctx_for_batch(batch)
        assert fitting_vision_batch(batch, num_ctx, 256) == fitting_vision_batch(batch, num_ctx, 256, scenes_per_image=1)
    # Grids need room for every scene's reply.
    assert fitting_vision_batch(4, 6144, 256) == 4
    assert fitting_vision_batch(4, 6144, 256, scenes_per_image=4) == 3
    assert vision_ctx_for_batch(1, scenes_per_image=4) == 2048


def test_scene_cost_shares_only_the_reading():
    assert vision_scene_cost(1) == pytest.approx(1.0)
    assert vision_scene_cost(4) == pytest.approx(0.4)
    assert vision_scene_cost(0) == pytest.approx(1.0)


# The grid picture


def _frame(path: Path, size=(960, 540), colour=(200, 40, 40)) -> Path:
    Image.new("RGB", size, colour).save(path, "JPEG")
    return path


def test_layout_is_square_with_few_empty_cells():
    assert frame_grid.layout(1, 512, 288) == (1, 1)
    assert frame_grid.layout(4, 512, 288) == (2, 2)
    assert frame_grid.layout(6, 512, 288) == (2, 3)
    assert frame_grid.layout(9, 512, 288) == (3, 3)
    assert frame_grid.layout(3, 512, 288) == (1, 3)
    assert frame_grid.layout(5, 512, 288) == (2, 3)
    assert frame_grid.layout(4, 288, 512) == (2, 2)
    assert frame_grid.layout(2, 288, 512) == (2, 1)


def test_grid_keeps_each_frame_at_the_size_it_is_sent_alone(tmp_path):
    frames = [(i, _frame(tmp_path / f"{i}.jpg")) for i in range(4)]
    grid = frame_grid.compose(frames, 512)
    cell_width, cell_height = 512, 288 + frame_grid.LABEL_HEIGHT
    assert grid.size == (2 * cell_width + frame_grid.GAP, 2 * cell_height + frame_grid.GAP)
    # Under a megapixel: Ollama reads it at the cost of a single frame.
    assert grid.size[0] * grid.size[1] <= 1024 * 1024
    # The frame sits below its label strip, unchanged in colour.
    assert grid.getpixel((256, frame_grid.LABEL_HEIGHT + 144))[0] > 150
    # The label strip has white text on black.
    strip = grid.crop((0, 0, cell_width, frame_grid.LABEL_HEIGHT))
    assert max(pixel[0] for pixel in strip.getdata()) > 200 and min(pixel[0] for pixel in strip.getdata()) < 30


def test_frames_of_other_shapes_are_centred_in_their_cells(tmp_path):
    frames = [(1, _frame(tmp_path / "wide.jpg")), (2, _frame(tmp_path / "tall.jpg", size=(540, 960)))]
    grid = frame_grid.compose(frames, 512)
    assert max(grid.size) <= 2 * 512 + frame_grid.GAP + frame_grid.LABEL_HEIGHT


def test_unreadable_frames_are_not_tiled(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"jpg")
    assert frame_grid.can_tile(_frame(tmp_path / "ok.jpg"))
    assert not frame_grid.can_tile(bad)
    assert not frame_grid.can_tile(tmp_path / "missing.jpg")


# The grid in visual analysis


class GridClient:
    def __init__(self, num_ctx=16384, truncate_over=None, memory_errors=0):
        self.num_ctx = num_ctx
        self.truncate_over = truncate_over
        self.memory_errors = memory_errors
        self.calls: list[dict] = []
        self.single_frames = 0

    def role_options(self, role):
        return {"num_ctx": self.num_ctx}

    def encode_image(self, path):
        self.single_frames += 1
        return "frame"

    def chat(self, model, prompt, images=None, schema=None, options=None, schema_first=True):
        ids = [int(match) for match in re.findall(r"scene_id=(\d+)", prompt)]
        self.calls.append({"images": list(images or []), "ids": ids, "prompt": prompt, "options": dict(options or {})})
        if self.memory_errors:
            self.memory_errors -= 1
            raise OllamaError("HTTP 500: model requires more system memory (5.1 GiB) than is available")
        if self.truncate_over is not None and len(ids) > self.truncate_over:
            raise OllamaTruncatedError("cut off")
        return {"scenes": [{"scene_id": scene_id, "visual_importance": 0.7, "caption": f"scene {scene_id}"} for scene_id in ids]}


def _inputs(tmp_path: Path, count: int, unreadable: tuple[int, ...] = ()):
    scenes = [{"scene_id": i, "start": i * 5.0, "end": i * 5.0 + 5.0, "duration": 5.0} for i in range(count)]
    frames = {}
    for scene in scenes:
        path = tmp_path / f"scene_{scene['scene_id']}.jpg"
        if scene["scene_id"] in unreadable:
            path.write_bytes(b"jpg")
        else:
            _frame(path, colour=(20 * scene["scene_id"] % 255, 90, 160))
        frames[str(scene["scene_id"])] = [{"timestamp": scene["start"] + 1, "image_path": str(path)}]
    return scenes, frames


def test_scenes_share_labelled_pictures(tmp_path):
    scenes, frames = _inputs(tmp_path, 6)
    client = GridClient()
    starts: list[tuple[int, int, int]] = []
    results, _errors, reasons = analyze_scenes_visual(
        client, "m", scenes, frames, 30.0, settings=Settings(ollama_vision_batch_size=1, ollama_vision_scenes_per_image=4),
        on_batch_start=lambda *args: starts.append(args),
    )
    assert reasons == {} and len(results) == 6
    assert all(results[i].visual_importance == pytest.approx(0.7) for i in range(6))
    assert [call["ids"] for call in client.calls] == [[0, 1, 2, 3], [4, 5]]
    assert [len(call["images"]) for call in client.calls] == [1, 1]
    assert client.single_frames == 0
    assert "grid of scene frames" in client.calls[0]["prompt"]
    assert starts == [(1, 2, 4), (2, 2, 2)]


def test_several_pictures_per_request_each_hold_several_scenes(tmp_path):
    scenes, frames = _inputs(tmp_path, 8)
    client = GridClient()
    analyze_scenes_visual(client, "m", scenes, frames, 40.0, settings=Settings(ollama_vision_batch_size=2, ollama_vision_scenes_per_image=4))
    assert [len(call["images"]) for call in client.calls] == [2]
    assert client.calls[0]["ids"] == list(range(8))


def test_one_scene_per_picture_keeps_the_previous_requests(tmp_path):
    scenes, frames = _inputs(tmp_path, 3)
    client = GridClient()
    analyze_scenes_visual(
        client, "m", scenes, frames, 15.0,
        settings=Settings(ollama_vision_batch_size=1, ollama_vision_scenes_per_image=1),
    )
    assert [call["ids"] for call in client.calls] == [[0], [1], [2]]
    assert client.single_frames == 3
    assert "Image order matches scene order" in client.calls[0]["prompt"]


def test_a_small_context_puts_fewer_scenes_in_a_picture(tmp_path):
    scenes, frames = _inputs(tmp_path, 6)
    client = GridClient(num_ctx=2048)
    results, _errors, _reasons = analyze_scenes_visual(
        client, "m", scenes, frames, 30.0, settings=Settings(ollama_vision_batch_size=4, ollama_vision_scenes_per_image=4), describe=True
    )
    assert [call["ids"] for call in client.calls] == [[0, 1, 2], [3, 4, 5]]
    assert results[0].caption == "scene 0"


def test_a_frame_that_cannot_be_opened_is_sent_on_its_own(tmp_path):
    scenes, frames = _inputs(tmp_path, 5, unreadable=(2,))
    client = GridClient()
    results, _errors, reasons = analyze_scenes_visual(
        client, "m", scenes, frames, 25.0, settings=Settings(ollama_vision_batch_size=1, ollama_vision_scenes_per_image=4)
    )
    assert reasons == {} and len(results) == 5
    assert [call["ids"] for call in client.calls] == [[0, 1, 3, 4], [2]]
    assert client.single_frames == 1
    assert "Image order matches scene order" in client.calls[1]["prompt"]


def test_a_cut_off_grid_reply_is_retried_as_smaller_grids(tmp_path):
    scenes, frames = _inputs(tmp_path, 4)
    client = GridClient(truncate_over=2)
    results, _errors, reasons = analyze_scenes_visual(
        client, "m", scenes, frames, 20.0, settings=Settings(ollama_vision_batch_size=1, ollama_vision_scenes_per_image=4)
    )
    assert reasons == {} and len(results) == 4
    assert [call["ids"] for call in client.calls] == [[0, 1, 2, 3], [0, 1], [2, 3]]
    assert all(len(call["images"]) == 1 for call in client.calls)


def test_out_of_memory_on_one_grid_retries_it_once_with_less_memory(tmp_path):
    scenes, frames = _inputs(tmp_path, 4)
    client = GridClient(num_ctx=16384, memory_errors=1)
    relieved: list[int] = []
    results, _errors, reasons = analyze_scenes_visual(
        client, "m", scenes, frames, 20.0, settings=Settings(ollama_vision_batch_size=1, ollama_vision_scenes_per_image=4),
        on_memory_error=lambda: relieved.append(1),
    )
    assert reasons == {} and len(results) == 4 and relieved == [1]
    assert [call["ids"] for call in client.calls] == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert client.calls[1]["options"]["num_ctx"] < 16384


# Recorded speed stays per one-frame request


def test_recorded_vision_speed_is_per_one_frame_request(monkeypatch):
    from app.services import auto_tune

    saved = {}
    monkeypatch.setattr(auto_tune, "read_history", lambda: [])
    monkeypatch.setattr(auto_tune, "write_setting", lambda key, value: saved.update({key: value}))

    class Progress:
        timings = {"visual": 100.0}
        speeds = {"vision_scenes_per_image": 4}
        # Two pictures of four scenes: 2 x (0.8 + 4 x 0.2) = 3.2 one-frame requests of 25 s.
        usage = {"vision": {"calls": 2, "prompt_tokens": 2300, "prompt_sec": 40.0, "output_sec": 40.0, "output_tokens": 600}}

    auto_tune.record_run(Progress(), {"metadata": {"duration": 60}}, Settings())
    import json

    entry = json.loads(saved[auto_tune._HISTORY_KEY])[0]
    assert entry["vision"]["seconds_per_image"] == pytest.approx(25.0)


def test_settings_accept_the_new_numbers_within_range():
    from app.services.settings_store import validate_values

    assert validate_values({"whisper_batch_size": "0", "ollama_vision_scenes_per_image": 9}, Settings()) == {
        "whisper_batch_size": 0,
        "ollama_vision_scenes_per_image": 9,
    }
    with pytest.raises(ValueError):
        validate_values({"whisper_batch_size": 33}, Settings())
    with pytest.raises(ValueError):
        validate_values({"ollama_vision_scenes_per_image": 0}, Settings())


# Translating speech that cannot be written down reliably


def test_translate_task_reaches_whisper_in_both_modes(monkeypatch):
    model = FakeModel()
    seen = {}

    def sequential(audio, **options):
        seen.update(options)
        return iter([]), types.SimpleNamespace(language="ml", language_probability=0.9)

    model.transcribe = sequential
    monkeypatch.setattr(audio_module, "_get_whisper_model", lambda *a: model)
    audio_module._transcribe_with(Path("a.wav"), "small", "cpu", "int8", 4, 1, language="ml", task="translate")
    assert seen["task"] == "translate" and seen["language"] == "ml"
    batched = FakeBatched([])
    monkeypatch.setattr(audio_module, "_batched_pipeline", lambda m: batched)
    audio_module._transcribe_with(Path("a.wav"), "small", "cpu", "int8", 4, 1, task="translate", batch_size=8)
    assert batched.options["task"] == "translate"
    audio_module._transcribe_with(Path("a.wav"), "small", "cpu", "int8", 4, 1, batch_size=8)
    assert "task" not in batched.options, "transcribing sends no task (Whisper's default)"


def test_a_translation_is_checked_as_english_text():
    from app.analysis.speech_check import check
    from app.models.schemas import Transcript, TranscriptSegment

    english = ["This phone has a large camera sensor.", "The battery lasts two full days."] * 4
    segments = [TranscriptSegment(start=i * 3.0, end=i * 3.0 + 2.5, text=t) for i, t in enumerate(english)]
    assert check(Transcript(segments=segments, language="ml", translated=True))["reliable"]
    garbled = ["ஓட்டு வேண்டும் VS1COP ஓPTIMAISD", "எந்து ஒன்று phoneுடன் RAMを足りない", "தரம்ஞஞஞஞஞஞஞஞ मकी गान"] * 4
    segments = [TranscriptSegment(start=i * 3.0, end=i * 3.0 + 2.5, text=t) for i, t in enumerate(garbled)]
    assert not check(Transcript(segments=segments, language="ml", translated=True))["reliable"]


# The processing details of a single video


def test_run_details_of_one_video_come_from_its_analysis(tmp_path):
    import json

    from app.analysis.pipeline import RUN_INFO_FILE, PipelineProgress, VideoAnalyzer
    from app.models.schemas import HighlightConfig

    analyzer = VideoAnalyzer(Settings(data_dir=tmp_path / "data", enable_repetition_analysis=False), runner=None, ollama=None)
    piece = {
        "metadata": {"duration": 30.0, "width": 640, "height": 360, "fps": 30.0, "has_audio": True},
        "candidates": [],
        "scenes": [{"scene_id": 0}, {"scene_id": 1}],
        "transcript": {"segments": [], "words": [], "language": "ml", "translated": True},
        "visual": {"0": {"score_source": "ai"}, "1": {"score_source": "ai"}},
        "speech_reliable": False,
    }
    analyzer.analyze = lambda *args, **kwargs: dict(piece)
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x")
    cache = {"speakers": {"stale": True}}
    analyzer.analyze_sources(
        tmp_path, [{"id": "v0", "path": str(source), "filename": "in.mp4"}], HighlightConfig(), PipelineProgress(lambda *_: None), cache
    )
    found = json.loads((tmp_path / RUN_INFO_FILE).read_text(encoding="utf-8"))["found"]
    assert found["speech_reliable"] is False
    assert found["scenes_ai"] == 2 and found["scenes"] == 2
    assert found["language"] == "ml" and found["speech_translated"] is True
    assert "speakers" not in cache, "data the analysis dropped is not kept from before"


def test_a_long_piece_without_punctuation_is_split_at_its_longest_pauses():
    # Whisper sometimes writes a batched piece without punctuation: 20 s with short pauses.
    words, start = [], 0.0
    for index in range(40):
        gap = 0.6 if index in (14, 29) else 0.05
        words.append(_word(f" w{index}", start, start + 0.45))
        start += 0.45 + gap
    parts = _parts([_segment(words)])
    assert [p[2].split()[0] for p in parts] == ["w0", "w15", "w30"]
    assert all(end - begin <= audio_module.MAX_SEGMENT_SECONDS for begin, end, _ in parts)


def test_a_long_piece_without_any_pause_is_kept():
    words = [_word(f" w{i}", i * 0.5, i * 0.5 + 0.5) for i in range(30)]
    assert len(_parts([_segment(words)])) == 1
