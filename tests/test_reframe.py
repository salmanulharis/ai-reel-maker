"""Smart framing: centre 9:16 / 1:1 / 4:5 crops on the people in each clip."""

from pathlib import Path

import numpy as np
import pytest

import app.video.reframe as reframe
import app.video.renderer as renderer_module
from app.models.schemas import ClipPlan, ClipPlanItem, HighlightConfig, RenderSettings, VideoMetadata
from app.video.reframe import aspect_crop_filter, subject_center, to_canvas
from app.video.renderer import Renderer


def test_middle_crop_is_unchanged_from_earlier_versions():
    assert aspect_crop_filter("9:16") == "crop='if(gte(iw/ih,0.5625),ih*0.5625,iw)':'if(gte(iw/ih,0.5625),ih,iw/0.5625)'"
    assert aspect_crop_filter("original") is None
    assert aspect_crop_filter("") is None


def test_crop_around_a_point_stays_inside_the_picture():
    crop = aspect_crop_filter("9:16", (0.8, 0.3))
    assert crop.startswith(aspect_crop_filter("9:16") + ":")
    assert "'max(0,min(iw-ow,iw*0.8000-ow/2))'" in crop
    assert "'max(0,min(ih-oh,ih*0.3000-oh*0.4))'" in crop
    # Out-of-range points are clamped.
    assert "iw*1.0000" in aspect_crop_filter("1:1", (3, -1)) and "ih*0.0000" in aspect_crop_filter("1:1", (3, -1))


def test_to_canvas_accounts_for_letterboxing():
    # Same shape: unchanged.
    assert to_canvas((0.25, 0.5), 1280, 720, 1920, 1080) == pytest.approx((0.25, 0.5))
    # A 4:3 video padded into a 16:9 canvas: content is 1440 wide, 240 px bars on each side.
    assert to_canvas((0.0, 0.5), 1440, 1080, 1920, 1080) == pytest.approx((240 / 1920, 0.5))
    assert to_canvas((1.0, 0.5), 1440, 1080, 1920, 1080) == pytest.approx((1680 / 1920, 0.5))
    assert to_canvas((0.3, 0.4), 0, 0, 1920, 1080) == (0.3, 0.4)


def test_subject_center_weights_bigger_faces(monkeypatch):
    faces = iter([(0.2, 0.5, 0.01), None, (0.8, 0.3, 0.03)])
    monkeypatch.setattr(reframe, "largest_face", lambda image: next(faces))
    assert subject_center([1, 2, 3]) == pytest.approx((0.65, 0.35))
    monkeypatch.setattr(reframe, "largest_face", lambda image: None)
    assert subject_center([1, 2]) is None
    assert subject_center([]) is None


def test_largest_face_handles_blank_and_empty_images():
    assert reframe.largest_face(None) is None
    assert reframe.largest_face(np.zeros((0, 0, 3), dtype=np.uint8)) is None
    assert reframe.largest_face(np.full((720, 1280, 3), 128, dtype=np.uint8)) is None


def test_config_keeps_the_middle_for_existing_jobs():
    assert HighlightConfig().reframe == "center"
    assert HighlightConfig(reframe="subject").reframe == "subject"
    assert RenderSettings().reframe == "center"


def _render(tmp_path: Path, settings: RenderSettings, monkeypatch, centers):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x" * 2000)
    commands = []

    class FakeRunner:
        def run(self, cmd):
            commands.append(cmd)
            Path(cmd[-1]).write_bytes(b"x" * 2000)

        def probe(self, path):
            return VideoMetadata(duration=6, width=1920, height=1080, fps=30, has_audio=True, size_bytes=2000)

    seen = iter(centers)
    monkeypatch.setattr(renderer_module, "clip_frames", lambda *args: ["frame"])
    monkeypatch.setattr(renderer_module, "subject_center", lambda images: next(seen))
    result = Renderer(FakeRunner()).render(
        source,
        ClipPlan(
            target_duration=4,
            estimated_duration=4,
            clips=[ClipPlanItem(start=0, end=2, reason="a"), ClipPlanItem(start=4, end=6, reason="b")],
        ),
        tmp_path / "out.mp4",
        settings,
        tmp_path / "work",
    )
    cuts = [cmd for cmd in commands if Path(cmd[-1]).name.startswith("clip_")]
    return result, [cmd[cmd.index("-vf") + 1] for cmd in cuts]


def test_renderer_follows_faces_per_clip(tmp_path, monkeypatch):
    # Validation would compare against a real output; it is not what this test is about.
    monkeypatch.setattr(renderer_module, "validate_output", lambda *a, **k: {"ok": True, "error": None})
    result, filters = _render(
        tmp_path, RenderSettings(aspect_ratio="9:16", reframe="subject"), monkeypatch, [(0.7, 0.4), None]
    )
    assert "iw*0.7000-ow/2" in filters[0]
    assert "-ow/2" not in filters[1] and aspect_crop_filter("9:16") in filters[1]
    assert result["reframe"] == {"followed": 1, "clips": 2}


def test_renderer_keeps_the_middle_when_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(renderer_module, "validate_output", lambda *a, **k: {"ok": True, "error": None})
    result, filters = _render(tmp_path, RenderSettings(aspect_ratio="9:16"), monkeypatch, [])
    assert all("-ow/2" not in f for f in filters)
    assert result["reframe"] is None
    result, filters = _render(tmp_path, RenderSettings(reframe="subject"), monkeypatch, [])
    assert result["reframe"] is None
