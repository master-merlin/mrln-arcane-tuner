# backend/app/core/captioning/caption_suggestions.py
"""Staging for LLM-refined caption suggestions + accept (snapshot+promote) / reject.

    {dataset_path}/suggestions/{definition_id}/{stem}.txt   # pending suggestion
Accept snapshots any existing variant to ``{variant}.bak`` then promotes the
suggestion to the live variant and removes the suggestion file.
"""

from __future__ import annotations

import os
import shutil

from app.core.captioning import caption_variants as cv

_SUGGESTIONS_SUBDIR = "suggestions"


def suggestion_dir(dataset_path: str, definition_id: str, masked: bool = False) -> str:
    base = os.path.join(dataset_path, _SUGGESTIONS_SUBDIR, definition_id)
    return os.path.join(base, "masked") if masked else base


def suggestion_path(dataset_path: str, definition_id: str, stem: str, masked: bool = False) -> str:
    return os.path.join(suggestion_dir(dataset_path, definition_id, masked), f"{stem}.txt")


def write_suggestion(dataset_path: str, definition_id: str, stem: str, text: str, masked: bool = False) -> None:
    path = suggestion_path(dataset_path, definition_id, stem, masked)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_suggestion(dataset_path: str, definition_id: str, stem: str, masked: bool = False) -> str | None:
    path = suggestion_path(dataset_path, definition_id, stem, masked)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def list_suggestion_stems(dataset_path: str, definition_id: str, masked: bool = False) -> list[str]:
    d = suggestion_dir(dataset_path, definition_id, masked)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".txt"))


def reject_suggestion(dataset_path: str, definition_id: str, stem: str, masked: bool = False) -> None:
    path = suggestion_path(dataset_path, definition_id, stem, masked)
    if os.path.exists(path):
        os.remove(path)


def accept_suggestion(dataset_path: str, definition_id: str, stem: str, masked: bool = False) -> None:
    """Snapshot any existing variant, promote the suggestion to the live variant, clear it."""
    suggestion = read_suggestion(dataset_path, definition_id, stem, masked)
    if suggestion is None:
        return
    target = cv.variant_path(dataset_path, definition_id, stem, masked)
    if os.path.exists(target):
        shutil.copyfile(target, target + ".bak")
    cv.write_variant(dataset_path, definition_id, stem, suggestion, masked)
    reject_suggestion(dataset_path, definition_id, stem, masked)
