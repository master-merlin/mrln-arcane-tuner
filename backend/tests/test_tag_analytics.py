# backend/tests/test_tag_analytics.py
"""Unit tests for compute_tag_analytics — pure, no file I/O."""

from app.core.dataset.tag_analytics import compute_tag_analytics, DEFAULT_CONTRADICTION_RULES


def test_frequency_and_orphans():
    items = [
        ("a.png", "cat, dog"),
        ("b.png", "cat, bird"),
        ("c.png", "Cat, fish"),   # 'Cat' folds into 'cat'
    ]
    out = compute_tag_analytics(items, top_n=10, rules=[])
    freq = {t["tag"]: t["count"] for t in out["top_tags"]}
    assert freq["cat"] == 3
    assert freq["dog"] == 1
    assert out["total_images"] == 3
    assert out["total_tags"] == 4  # cat, dog, bird, fish
    assert set(out["orphan_tags"]) == {"dog", "bird", "fish"}


def test_cooccurrence_matrix_is_symmetric_with_freq_diagonal():
    items = [
        ("a.png", "cat, dog"),
        ("b.png", "cat, dog"),
        ("c.png", "cat"),
    ]
    out = compute_tag_analytics(items, top_n=10, rules=[])
    co = out["cooccurrence"]
    i = co["labels"].index("cat")
    j = co["labels"].index("dog")
    assert co["matrix"][i][i] == 3  # cat frequency on the diagonal
    assert co["matrix"][j][j] == 2
    assert co["matrix"][i][j] == 2  # cat+dog together twice
    assert co["matrix"][i][j] == co["matrix"][j][i]  # symmetric


def test_cooccurrence_limited_to_top_n_labels():
    items = [("x.png", "a, b, c, d, e")]
    out = compute_tag_analytics(items, top_n=2, rules=[])
    assert len(out["cooccurrence"]["labels"]) == 2
    assert len(out["cooccurrence"]["matrix"]) == 2


def test_contradiction_detection():
    items = [
        ("a.png", "day, sunny, beach"),
        ("b.png", "night, day, city"),   # both day + night → contradiction
        ("c.png", "indoor, cozy"),
    ]
    out = compute_tag_analytics(items, top_n=10, rules=[["day", "night"]])
    contradictions = out["contradictions"]
    assert len(contradictions) == 1
    c = contradictions[0]
    assert {c["a"], c["b"]} == {"day", "night"}
    assert c["count"] == 1
    assert c["images"] == ["b.png"]


def test_default_rules_present():
    assert any("day" in r and "night" in r for r in DEFAULT_CONTRADICTION_RULES)


def test_empty_dataset():
    out = compute_tag_analytics([], top_n=10, rules=[])
    assert out["total_images"] == 0
    assert out["top_tags"] == []
    assert out["cooccurrence"]["labels"] == []
    assert out["contradictions"] == []


# --- prose (natural-language) mode ------------------------------------------------

def test_comma_captions_autodetect_tags_mode():
    items = [("a.png", "cat, dog, bird"), ("b.png", "cat, fish")]
    out = compute_tag_analytics(items, top_n=10, rules=[])
    assert out["style"] == "tags"


def test_prose_autodetected_tokenized_into_terms_and_bigrams():
    items = [
        ("a.png", "a red sports car parked on a road"),
        ("b.png", "a blue sports car on a road"),
    ]
    out = compute_tag_analytics(items, top_n=30, rules=[])
    assert out["style"] == "prose"
    freq = {t["tag"]: t["count"] for t in out["top_tags"]}
    # content-word unigrams
    assert freq["car"] == 2
    assert freq["road"] == 2
    # 2-word phrase (consecutive content words)
    assert freq["sports car"] == 2
    # stopwords dropped, and the whole sentence is NOT one giant tag
    assert "a" not in freq and "on" not in freq
    assert "a red sports car parked on a road" not in freq


def test_explicit_style_overrides_autodetect():
    items = [("a.png", "a red car, a blue sky")]
    # forced tags → comma split keeps the full segments
    tags = {t["tag"] for t in compute_tag_analytics(items, rules=[], style="tags")["top_tags"]}
    assert "a red car" in tags and "a blue sky" in tags
    # forced prose → content words + phrases, never the raw segment
    prose = {t["tag"] for t in compute_tag_analytics(items, rules=[], style="prose")["top_tags"]}
    assert "red car" in prose and "blue sky" in prose
    assert "a red car" not in prose


def test_prose_cooccurrence_uses_terms():
    items = [
        ("a.png", "red car near brick wall"),
        ("b.png", "blue car near brick wall"),
    ]
    out = compute_tag_analytics(items, top_n=30, rules=[], style="prose")
    co = out["cooccurrence"]
    i = co["labels"].index("car")
    j = co["labels"].index("brick wall")
    assert co["matrix"][i][j] == 2  # car co-occurs with the phrase "brick wall" twice
