from app.analysis.visual import sample_timestamps
from app.models.schemas import Scene


def test_sample_timestamps_stay_inside_safe_media_window():
    scene = Scene(scene_id=0, start=0.0, end=30.53, duration=30.53)
    stamps = sample_timestamps(scene, source_duration=30.53)
    assert stamps
    assert min(stamps) >= 0
    assert max(stamps) <= 30.53 - 0.12
    assert 30.52 not in stamps
