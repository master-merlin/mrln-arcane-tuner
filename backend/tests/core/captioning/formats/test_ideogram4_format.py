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


def test_generation_overrides_raise_ceiling_without_a_floor():
    o = Ideogram4Format().generation_overrides()
    # A high ceiling for the long structured JSON, but NO min-token floor — a
    # floor forces a finished model to keep emitting tokens -> trailing garbage.
    assert o["max_tokens"] >= 4096
    assert "min_new_tokens" not in o


def test_parse_and_normalize_recovers_messy_json():
    raw = (
        "```json\n"
        + json.dumps(
            {
                "high_level_description": "x",
                "compositional_deconstruction": {"background": "b", "elements": []},
            }
        )
        + "\n```"
    )
    data = Ideogram4Format().parse_and_normalize(raw)
    assert data["compositional_deconstruction"]["background"] == "b"


def test_ingest_generated_swaps_xfirst_bbox_to_yfirst():
    # A captioner emits x-first [x_min,y_min,x_max,y_max]; the PAGANI-text case:
    # x 386..636 (wide), y 175..225 (thin band near top).
    raw = json.dumps(
        {
            "high_level_description": "x",
            "compositional_deconstruction": {
                "background": "b",
                "elements": [
                    {
                        "type": "text",
                        "text": "PAGANI",
                        "bbox": [386, 175, 636, 225],
                        "desc": "badge",
                    }
                ],
            },
        }
    )
    data = Ideogram4Format().ingest_generated(raw)
    bb = data["compositional_deconstruction"]["elements"][0]["bbox"]
    # Stored canonical y-first: [y_min, x_min, y_max, x_max].
    assert bb == [175, 386, 225, 636]


def test_parse_and_normalize_does_not_swap_bbox():
    # Refine/editor round-trip on already-canonical y-first data must NOT swap.
    raw = json.dumps(
        {
            "high_level_description": "x",
            "compositional_deconstruction": {
                "background": "b",
                "elements": [
                    {"type": "obj", "bbox": [175, 386, 225, 636], "desc": "d"}
                ],
            },
        }
    )
    data = Ideogram4Format().parse_and_normalize(raw)
    assert data["compositional_deconstruction"]["elements"][0]["bbox"] == [
        175,
        386,
        225,
        636,
    ]


def test_parse_and_normalize_wraps_garbage_in_skeleton():
    fmt = Ideogram4Format()
    data = fmt.parse_and_normalize("this is not json at all")
    assert "compositional_deconstruction" in data
    assert data["high_level_description"].startswith("this is not json")


def test_json_schema_requires_deconstruction():
    schema = Ideogram4Format().json_schema()
    assert "compositional_deconstruction" in schema["required"]
