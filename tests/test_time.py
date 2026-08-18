from app.utils.timefmt import (
    format_duration,
    parse_timestamp,
    clamp,
    seconds_to_timecode,
)


def test_format_duration_hours_minutes_seconds():
    assert format_duration(0) == "00:00"
    assert format_duration(27) == "00:27"
    assert format_duration(318) == "05:18"
    assert format_duration(3723) == "01:02:03"


def test_parse_timestamp_supports_common_forms():
    assert parse_timestamp("00:32") == 32.0
    assert parse_timestamp("01:24") == 84.0
    assert parse_timestamp("1:10") == 70.0
    assert parse_timestamp("00:32.5") == 32.5
    assert parse_timestamp("1:02:03") == 3723.0
    assert parse_timestamp(91.3) == 91.3


def test_seconds_to_timecode_includes_millis():
    assert seconds_to_timecode(82.4) == "00:01:22.400"


def test_clamp_bounds_values():
    assert clamp(-1, 0, 1) == 0
    assert clamp(0.5, 0, 1) == 0.5
    assert clamp(2, 0, 1) == 1
