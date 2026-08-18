from app.config import Settings
from app.models.schemas import HighlightConfig


def test_settings_derive_upload_limit_and_paths(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", max_upload_size_gb=2)
    settings.ensure_directories()
    assert settings.max_upload_bytes == 2 * 1024 * 1024 * 1024
    assert settings.jobs_dir.exists()
    assert settings.tmp_dir.exists()


def test_highlight_config_swaps_inverted_range():
    config = HighlightConfig(min_duration=40, max_duration=20)
    assert config.min_duration == 20
    assert config.max_duration == 40


def test_social_short_defaults_to_vertical():
    config = HighlightConfig(social_short=True)
    assert config.aspect_ratio == "9:16"


def test_allow_repeated_content_follows_remove_repetition():
    assert HighlightConfig().allow_repeated_content is False
    assert HighlightConfig(allow_repeated_content=True).remove_repetition is False
    assert HighlightConfig(remove_repetition=False).allow_repeated_content is True
    assert HighlightConfig().max_sequence_duration == 5.0
    assert HighlightConfig().sequence_duration_mode == "auto"
    assert HighlightConfig().max_repeated_duration == 15.0
