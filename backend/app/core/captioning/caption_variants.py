# backend/app/core/captioning/caption_variants.py
"""Per-definition caption variant storage + the variant-or-general resolver.

Layout mirrors the existing ``masked/{stem}.txt`` convention:
    {dataset_path}/captions/{definition_id}/{stem}.txt   # model-specific variant
    {dataset_path}/{stem}.txt                            # general (source/fallback)
"""

from __future__ import annotations

import os
from pathlib import Path

from app.api._path_guard import validate_path_within

_VARIANTS_SUBDIR = "captions"


def variant_dir(dataset_path: str, definition_id: str, masked: bool = False) -> str:
    # definition_id is client-supplied (routed straight from the API request
    # body/query in caption_variant_routes.py); resolve through the shared
    # containment guard before it becomes a directory segment so a crafted
    # "../../evil" can't escape the dataset (raises HTTPException(403)).
    base = validate_path_within(
        Path(dataset_path) / _VARIANTS_SUBDIR / definition_id, dataset_path
    )
    return str(base / "masked") if masked else str(base)


def variant_path(dataset_path: str, definition_id: str, stem: str, masked: bool = False) -> str:
    # stem is the second client-supplied segment; guard it independently of
    # definition_id (already guarded by variant_dir above) — the resolved
    # Path returned here is what every read_variant/write_variant/has_variant
    # call actually opens.
    candidate = Path(variant_dir(dataset_path, definition_id, masked)) / f"{stem}.txt"
    return str(validate_path_within(candidate, dataset_path))


def has_variant(dataset_path: str, definition_id: str, stem: str, masked: bool = False) -> bool:
    return os.path.exists(variant_path(dataset_path, definition_id, stem, masked))


def read_variant(dataset_path: str, definition_id: str, stem: str, masked: bool = False) -> str | None:
    path = variant_path(dataset_path, definition_id, stem, masked)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_variant(dataset_path: str, definition_id: str, stem: str, text: str, masked: bool = False) -> None:
    path = variant_path(dataset_path, definition_id, stem, masked)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _read_general(dataset_path: str, stem: str) -> str | None:
    # stem reaches here client-supplied via resolve_caption <- get_caption_variant's
    # ?stem= query param — guard the same as variant_path/suggestion_path.
    for ext in (".txt", ".caption"):
        path = validate_path_within(Path(dataset_path) / f"{stem}{ext}", dataset_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return None


def _read_masked(dataset_path: str, stem: str) -> str | None:
    path = validate_path_within(
        Path(dataset_path) / "masked" / f"{stem}.txt", dataset_path
    )
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def resolve_caption(dataset_path: str, stem: str, definition_id: str | None, masked: bool = False) -> str:
    """Return the variant caption when present, else the masked/general caption, else ''.

    Original axis (masked=False): variant → general `{stem}.txt` → ''.
    Masked axis (masked=True):    masked variant → `masked/{stem}.txt`
                                  → original general `{stem}.txt` → '' .
    The masked chain falls back to the original caption (not '') when no masked
    caption exists — an unmasked caption is a better seed than nothing.
    """
    if definition_id:
        variant = read_variant(dataset_path, definition_id, stem, masked)
        if variant is not None:
            return variant
    if masked:
        masked_cap = _read_masked(dataset_path, stem)
        if masked_cap is not None:
            return masked_cap
    general = _read_general(dataset_path, stem)
    return general if general is not None else ""


def list_variant_texts(dataset_path: str, definition_id: str, masked: bool = False) -> dict[str, str]:
    """Return ``{stem: text}`` for every variant of ``definition_id`` on the
    given axis — one cheap directory scan so the grid can resolve a whole
    dataset's model-aware captions in a single request. Non-recursive, so the
    original axis never picks up the nested ``masked/`` files.
    """
    d = variant_dir(dataset_path, definition_id, masked)
    if not os.path.isdir(d):
        return {}
    out: dict[str, str] = {}
    for entry in os.listdir(d):
        if not entry.endswith(".txt"):
            continue
        path = os.path.join(d, entry)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            out[entry[: -len(".txt")]] = f.read()
    return out


def list_variant_definition_ids(dataset_path: str) -> list[str]:
    root = os.path.join(dataset_path, _VARIANTS_SUBDIR)
    if not os.path.isdir(root):
        return []
    return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
