from app.ai.json_util import parse_model_json, clamp_unit


def test_parse_model_json_extracts_object_from_markdown_fence():
    raw = """Here you go
```json
{"importance": 0.91, "novelty": 0.78}
```
"""
    data = parse_model_json(raw)
    assert data["importance"] == 0.91
    assert data["novelty"] == 0.78


def test_parse_model_json_repairs_trailing_commas():
    raw = '{"importance": 0.5, "novelty": 0.2,}'
    data = parse_model_json(raw)
    assert data["importance"] == 0.5


def test_clamp_unit_handles_percent_and_strings():
    assert clamp_unit(1.4) == 1.0
    assert clamp_unit(-2) == 0.0
    assert clamp_unit("82") == 0.82
    assert clamp_unit("0.4") == 0.4
