# backend/app/core/dataset/tag_analytics.py
"""Pure tag-analytics computation over (image, caption) pairs.

No file I/O — callers pass already-read caption strings so this is trivially
unit-testable. Tags are comma-split, whitespace-collapsed, trimmed, and
lowercased for counting.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from typing import Any

# Mutually-exclusive tag pairs flagged when both appear on one image.
DEFAULT_CONTRADICTION_RULES: list[list[str]] = [
    ["day", "night"],
    ["indoor", "outdoor"],
    ["summer", "winter"],
    ["smiling", "frowning"],
]

_WS = re.compile(r"\s+")


def _tags(caption: str) -> list[str]:
    out: list[str] = []
    for raw in caption.split(","):
        tag = _WS.sub(" ", raw).strip().lower()
        if tag:
            out.append(tag)
    return out


def compute_tag_analytics(
    items: list[tuple[str, str]],
    top_n: int = 30,
    rules: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Compute frequency, orphans, co-occurrence (top_n), contradictions.

    ``items`` is a list of ``(image_name, caption_text)``.
    """
    if rules is None:
        rules = DEFAULT_CONTRADICTION_RULES

    freq: Counter[str] = Counter()
    per_image_tags: list[tuple[str, set[str]]] = []
    for image, caption in items:
        tags = set(_tags(caption))
        per_image_tags.append((image, tags))
        freq.update(tags)

    top_tags = [{"tag": t, "count": c} for t, c in freq.most_common()]
    orphan_tags = sorted(t for t, c in freq.items() if c == 1)

    labels = [t for t, _ in freq.most_common(top_n)]
    index = {t: i for i, t in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    for _image, tags in per_image_tags:
        present = [t for t in tags if t in index]
        for t in present:
            i = index[t]
            matrix[i][i] += 1
        for a, b in combinations(present, 2):
            i, j = index[a], index[b]
            matrix[i][j] += 1
            matrix[j][i] += 1

    contradictions: list[dict[str, Any]] = []
    for rule in rules:
        a, b = rule[0].lower(), rule[1].lower()
        images = [img for img, tags in per_image_tags if a in tags and b in tags]
        if images:
            contradictions.append({"a": a, "b": b, "count": len(images), "images": images})

    return {
        "total_images": len(items),
        "total_tags": len(freq),
        "top_tags": top_tags,
        "orphan_tags": orphan_tags,
        "cooccurrence": {"labels": labels, "matrix": matrix},
        "contradictions": contradictions,
    }
