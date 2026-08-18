from app.utils.security import is_allowed_extension, sanitize_filename, validate_job_relative
from pathlib import Path
import pytest


def test_sanitize_filename_strips_paths_and_unknown_types():
    assert sanitize_filename("../../secret.exe") == "secret.mp4"
    assert ".." not in sanitize_filename("folder/video.mov")
    assert sanitize_filename("demo.mkv").endswith(".mkv")


def test_allowed_extensions():
    assert is_allowed_extension("talk.mp4")
    assert is_allowed_extension("clip.webm")
    assert not is_allowed_extension("notes.pdf")


def test_validate_job_relative_blocks_escape(tmp_path):
    root = tmp_path / "job"
    root.mkdir()
    with pytest.raises(ValueError):
        validate_job_relative(root, "..", "etc", "passwd")
    inside = validate_job_relative(root, "outputs", "highlight.mp4")
    assert inside.is_relative_to(root) or str(root) in str(inside)
