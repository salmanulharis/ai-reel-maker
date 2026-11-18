"""The clips the render uses: exactly as kept on the Review page."""

from pathlib import Path

from app.workers.worker import plan_items_from_clips

SOURCE = Path("/videos/talk.mp4")


def test_clips_made_longer_by_hand_are_kept_whole():
    # A 40 s clip, far beyond any longest-clip setting: the person made it this long on purpose.
    items, skipped = plan_items_from_clips([{"start": 10, "end": 50, "reason": "r"}], {}, SOURCE)
    assert skipped == 0
    assert [(i.start, i.end) for i in items] == [(10, 50)]


def test_fields_are_carried_over():
    lookup = {"v1": Path("/videos/second.mp4")}
    clip = {
        "start": 1,
        "end": 3,
        "reason": "Funny",
        "score": 0.8,
        "transcript": "hi",
        "duration_reason": "d",
        "source_id": "v1",
        "source_name": "second.mp4",
        "short_index": 2,
    }
    item = plan_items_from_clips([clip], lookup, SOURCE)[0][0]
    assert (item.reason, item.score, item.transcript, item.short_index) == ("Funny", 0.8, "hi", 2)
    assert item.source_path == str(Path("/videos/second.mp4"))
    defaults = plan_items_from_clips([{"start": 0, "end": 1}], {}, SOURCE)[0][0]
    assert defaults.reason == "Selected moment" and defaults.source_path == str(SOURCE)


def test_clips_with_impossible_times_are_left_out_instead_of_failing():
    clips = [
        {"start": 5, "end": 2},  # end before start (typed by hand)
        {"start": 4, "end": 4},  # no length
        {"start": -1, "end": 3},
        {"start": "x", "end": 3},
        {"end": 3},
        {"start": 1, "end": 2},
    ]
    items, skipped = plan_items_from_clips(clips, {}, SOURCE)
    assert skipped == 5
    assert [(i.start, i.end) for i in items] == [(1, 2)]
