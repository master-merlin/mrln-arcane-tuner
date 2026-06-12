# backend/app/core/dataset/tag_analytics.py
"""Pure tag-analytics computation over (image, caption) pairs.

No file I/O — callers pass already-read caption strings so this is trivially
unit-testable. Two analysis styles:

* ``tags``  — booru/SD style: comma-split, whitespace-collapsed, lowercased.
* ``prose`` — natural-language captions: content-word unigrams plus adjacent
  2-word phrases (stopwords removed), so a whole descriptive sentence doesn't
  collapse into one useless "tag".

The style is chosen by the caller (e.g. from the model's caption style) or, when
unspecified, auto-detected from the corpus's comma density.
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
# A word: alphanumerics with internal hyphens/apostrophes kept whole, so
# "two-door", "close-up", "driver's", "maroon-colored" stay as single terms.
_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

# Grammatical glue + caption boilerplate that carries no descriptive signal.
# Descriptive verbs/nouns (parked, standing, wearing, view, background…) are
# deliberately NOT here — they're meaningful for dataset analysis.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "they", "them",
    "their", "there", "here", "his", "her", "your", "my", "our",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had", "having", "do", "does", "did", "doing",
    "of", "in", "on", "at", "to", "from", "with", "by", "for", "as", "into", "onto",
    "over", "under", "near", "behind", "between", "among", "amongst", "around",
    "above", "below", "beside", "against", "within", "without", "through",
    "throughout", "during", "before", "after", "along", "across", "up", "down",
    "off", "out",
    "and", "or", "but", "nor", "so", "yet", "if", "then", "else", "because", "while",
    "not", "no",
    "which", "who", "whom", "whose", "what", "where", "when", "why", "how",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "also", "very", "more", "most", "some", "any", "all", "both", "each", "few",
    "many", "much", "such", "own", "same", "than", "too", "just", "only",
    "image", "images", "img", "photo", "photos", "photograph", "photographs",
    "picture", "pictures", "shot",
    "shows", "show", "showing", "shown", "depicts", "depict", "depicting",
    "depicted", "featuring", "feature", "features", "featured", "appears",
    "appear", "appearing", "display", "displays", "displaying", "displayed",
    "captured", "taken", "seen",
})


def _tags(caption: str) -> list[str]:
    out: list[str] = []
    for raw in caption.split(","):
        tag = _WS.sub(" ", raw).strip().lower()
        if tag:
            out.append(tag)
    return out


def _prose_terms(caption: str) -> list[str]:
    """Content-word unigrams + adjacent content-word bigrams (stopwords removed)."""
    tokens = _WORD.findall(caption.lower())
    flags = [(t, t not in _STOPWORDS and len(t) >= 2) for t in tokens]
    terms = [t for t, ok in flags if ok]
    # Bigrams only from two CONSECUTIVE content words (no stopword between) →
    # surfaces real phrases like "sports car", "brick wall", "rear view".
    for (t1, ok1), (t2, ok2) in zip(flags, flags[1:]):
        if ok1 and ok2:
            terms.append(f"{t1} {t2}")
    return terms


def _detect_style(items: list[tuple[str, str]]) -> str:
    """Auto-detect 'tags' vs 'prose' from comma density.

    A caption is tag-like when it has a comma and short comma-segments
    (<= 3 words/segment). The corpus is 'tags' if at least half qualify.
    """
    tagish = 0
    total = 0
    for _image, caption in items:
        c = caption.strip()
        if not c:
            continue
        total += 1
        commas = c.count(",")
        if commas >= 1 and len(c.split()) / (commas + 1) <= 3:
            tagish += 1
    if total == 0:
        return "tags"
    return "tags" if tagish / total >= 0.5 else "prose"


def _extract(caption: str, style: str) -> list[str]:
    return _tags(caption) if style == "tags" else _prose_terms(caption)


def compute_tag_analytics(
    items: list[tuple[str, str]],
    top_n: int = 30,
    rules: list[list[str]] | None = None,
    style: str | None = None,
) -> dict[str, Any]:
    """Compute frequency, orphans, co-occurrence (top_n), contradictions.

    ``items`` is a list of ``(image_name, caption_text)``. ``style`` is
    ``"tags"``, ``"prose"``, or ``None`` to auto-detect from the corpus.
    """
    if rules is None:
        rules = DEFAULT_CONTRADICTION_RULES
    resolved_style = style if style in ("tags", "prose") else _detect_style(items)

    freq: Counter[str] = Counter()
    per_image_tags: list[tuple[str, set[str]]] = []
    for image, caption in items:
        tags = set(_extract(caption, resolved_style))
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
        "style": resolved_style,
        "top_tags": top_tags,
        "orphan_tags": orphan_tags,
        "cooccurrence": {"labels": labels, "matrix": matrix},
        "contradictions": contradictions,
    }
