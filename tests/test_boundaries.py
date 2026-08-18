from app.analysis.boundaries import snap_window
from app.models.schemas import Transcript, TranscriptSegment, WordTimestamp


def test_snap_window_prefers_word_and_scene_boundaries():
    transcript = Transcript(
        segments=[TranscriptSegment(start=10.0, end=14.2, text="This is why the most important thing works.")],
        words=[
            WordTimestamp(word="This", start=10.0, end=10.2),
            WordTimestamp(word="works", start=13.8, end=14.2),
        ],
    )
    start, end, incomplete = snap_window(
        10.4,
        13.5,
        transcript,
        silences=[{"start": 9.4, "end": 9.9}],
        scenes=[{"start": 9.9, "end": 15.0}],
        source_duration=60,
        pad=0.2,
    )
    assert start <= 10.4
    assert end >= 13.5
    assert end <= 15.0
