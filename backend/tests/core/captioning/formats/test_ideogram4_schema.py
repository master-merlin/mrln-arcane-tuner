import json

from app.core.captioning.formats.schema import ideogram4 as ix


def _doc(**over):
    base = {
        "high_level_description": "A man at a table.",
        "style_description": {
            "aesthetics": "baroque",
            "lighting": "chiaroscuro",
            "photo": "oil on canvas",
            "medium": "painting",
            "color_palette": ["#13100c", "#3b2a1e"],
        },
        "compositional_deconstruction": {
            "background": "near-black interior",
            "elements": [
                {
                    "type": "obj",
                    "bbox": [90, 180, 720, 760],
                    "desc": "the man",
                    "color_palette": ["#6e4a2e"],
                },
            ],
        },
    }
    base.update(over)
    return base


def test_detect_true_for_valid_json_with_deconstruction():
    assert ix.detect(json.dumps(_doc())) is True


def test_detect_false_for_plain_text():
    assert ix.detect("a man, a table, dramatic lighting") is False


def test_detect_false_for_json_without_deconstruction():
    assert ix.detect(json.dumps({"high_level_description": "x"})) is False


def test_normalize_uppercases_and_caps_image_palette():
    d = _doc()
    d["style_description"]["color_palette"] = [f"#{i:06x}" for i in range(20)]
    out = ix.normalize(d)
    pal = out["style_description"]["color_palette"]
    assert len(pal) == ix.MAX_IMAGE_PALETTE
    assert all(c == c.upper() for c in pal)


def test_normalize_caps_element_palette_to_five():
    d = _doc()
    d["compositional_deconstruction"]["elements"][0]["color_palette"] = [
        f"#{i:06x}" for i in range(9)
    ]
    out = ix.normalize(d)
    assert len(out["compositional_deconstruction"]["elements"][0]["color_palette"]) == 5


def test_normalize_photo_branch_drops_art_style():
    d = _doc()
    d["style_description"]["medium"] = "photograph"
    d["style_description"]["art_style"] = "stippling"  # illegal on photo branch
    out = ix.normalize(d)
    sd = out["style_description"]
    assert "art_style" not in sd
    assert "photo" in sd


def test_normalize_nonphoto_branch_migrates_photo_to_art_style():
    d = _doc()
    d["style_description"]["medium"] = "illustration"
    # only `photo` present; non-photo branch must surface it as art_style
    out = ix.normalize(d)
    sd = out["style_description"]
    assert "photo" not in sd
    assert sd["art_style"] == "oil on canvas"


def test_normalize_clamps_out_of_range_bbox():
    d = _doc()
    # Bboxes are stored y-first [y_min, x_min, y_max, x_max]; the VLM emits
    # y-first per the generation prompt.  normalize() only clamps to [0, BBOX_MAX]
    # and rounds — it does not swap axes.  Verify out-of-range values are clamped.
    d["compositional_deconstruction"]["elements"][0]["bbox"] = [1200, -5, 500, 500]
    out = ix.normalize(d)
    bb = out["compositional_deconstruction"]["elements"][0]["bbox"]
    assert all(0 <= v <= ix.BBOX_MAX for v in bb)
    # Verify a clearly in-range y-first bbox passes through unchanged.
    d2 = _doc()
    d2["compositional_deconstruction"]["elements"][0]["bbox"] = [100, 200, 300, 400]
    out2 = ix.normalize(d2)
    assert out2["compositional_deconstruction"]["elements"][0]["bbox"] == [
        100,
        200,
        300,
        400,
    ]


def test_swap_bbox_xy_swaps_pairs():
    assert ix.swap_bbox_xy([10, 20, 30, 40]) == [20, 10, 40, 30]


def test_canon_medium_maps_aliases():
    assert ix.canon_medium("Oil painting.") == "painting"
    assert ix.canon_medium("3D render") == "3d_render"
    assert ix.canon_medium("photo") == "photograph"


def test_serialize_is_compact_and_unicode():
    d = _doc(high_level_description="café scene")
    s = ix.serialize(ix.normalize(d))
    assert ", " not in s and ": " not in s  # compact separators
    assert "café" in s  # ensure_ascii=False


def test_normalize_enforces_key_order_photo_branch():
    d = _doc()
    d["style_description"]["medium"] = "photograph"
    out = ix.normalize(d)
    assert list(out["style_description"].keys()) == [
        "aesthetics",
        "lighting",
        "photo",
        "medium",
        "color_palette",
    ]


def test_normalize_enforces_top_level_key_order():
    out = ix.normalize(_doc())
    assert list(out.keys()) == [
        "high_level_description",
        "style_description",
        "compositional_deconstruction",
    ]


def test_skeleton_wraps_text_into_valid_doc():
    sk = ix.skeleton("just some words")
    assert ix.detect(ix.serialize(sk)) is True
    assert sk["high_level_description"] == "just some words"


def test_parse_extracts_json_from_fenced_block():
    raw = "Here you go:\n```json\n" + json.dumps(_doc()) + "\n```\nDone."
    parsed = ix.parse(raw)
    assert parsed is not None
    assert "compositional_deconstruction" in parsed


def test_parse_extracts_json_object_followed_by_trailing_prose():
    # Real-world failure: a model emits a valid JSON object, then keeps chatting
    # (word counts, hashtags, sign-offs, text containing stray braces). The first
    # balanced object must be extracted cleanly, not widened to a later brace.
    raw = json.dumps(_doc()) + "\n\n(Word count: 157) Done! #ready {not json}"
    parsed = ix.parse(raw)
    assert parsed is not None
    assert parsed["compositional_deconstruction"]["background"] == "near-black interior"


def test_normalize_enforces_key_order_nonphoto_branch():
    d = _doc()
    d["style_description"]["medium"] = "illustration"
    # Only `photo` present; non-photo branch surfaces it as art_style.
    out = ix.normalize(d)
    assert list(out["style_description"].keys()) == [
        "aesthetics",
        "lighting",
        "medium",
        "art_style",
        "color_palette",
    ]


def test_migrate_old_format_color_before_desc_and_titlecase_medium():
    old = {
        "high_level_description": "x",
        "style_description": {
            "aesthetics": "a",
            "lighting": "l",
            "photo": "p",
            "medium": "Painting.",
            "color_palette": ["#abc"],  # 3-digit
        },
        "compositional_deconstruction": {
            "background": "bg",
            "elements": [
                {
                    "type": "obj",
                    "color_palette": ["#fff"],
                    "bbox": [1, 2, 3, 4],
                    "desc": "d",
                }
            ],
        },
    }
    out = ix.normalize(old)
    assert out["style_description"]["medium"] == "painting"
    assert out["style_description"]["color_palette"] == ["#AABBCC"]
    el = out["compositional_deconstruction"]["elements"][0]
    assert list(el.keys()) == ["type", "bbox", "desc", "color_palette"]
