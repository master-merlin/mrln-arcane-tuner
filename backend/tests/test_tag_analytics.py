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
