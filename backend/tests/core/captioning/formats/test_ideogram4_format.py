import json
from app.core.captioning.formats import Ideogram4Format


def test_generation_prompt_includes_bbox_and_branch_rules():
    p = Ideogram4Format().build_generation_prompt()
    low = p.lower()
    assert "json" in low
    assert "compositional_deconstruction" in p
    assert "bbox" in low


def test_generation_prompt_appends_user_instructions():
    p = Ideogram4Format().build_generation_prompt("focus on the sword")
    assert "focus on the sword" in p


def test_generation_overrides_set_token_floor():
    o = Ideogram4Format().generation_overrides()
    assert o["min_new_tokens"] >= 3072
    assert o["max_tokens"] >= o["min_new_tokens"]


def test_parse_and_normalize_recovers_messy_json():
    raw = "```json\n" + json.dumps({
        "high_level_description": "x",
        "compositional_deconstruction": {"background": "b", "elements": []},
    }) + "\n```"
    data = Ideogram4Format().parse_and_normalize(raw)
    assert data["compositional_deconstruction"]["background"] == "b"


def test_parse_and_normalize_wraps_garbage_in_skeleton():
    fmt = Ideogram4Format()
    data = fmt.parse_and_normalize("this is not json at all")
    assert "compositional_deconstruction" in data
    assert data["high_level_description"].startswith("this is not json")


def test_json_schema_requires_deconstruction():
    schema = Ideogram4Format().json_schema()
    assert "compositional_deconstruction" in schema["required"]
