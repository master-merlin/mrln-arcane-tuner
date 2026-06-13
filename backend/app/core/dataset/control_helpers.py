"""Control-slot helpers for paired-image (edit/kontext) datasets.

Edit datasets pair each target image with 1..3 stem-matched control
("before") images living in ``control/``, ``control_2/``, ``control_3/``
subfolders — the same stem-pairing convention as ``masks/``/``masked/``.
Captions belong to the target only.

Which physical slot plays the *logical* training target is dataset
metadata, not disk layout: ``control_info.role_order`` is a permutation
of physical slot names (``root`` + control dirs) where position 0 is the
target and the rest are controls in order. ``None`` means the default
order (root image is the target). Re-ordering never moves files, so
latent caches keyed by physical slot stay valid.
"""

from __future__ import annotations

import os
from typing import Any

from PIL import Image

# Physical control slot directories, in slot order.
CONTROL_SLOTS: tuple[str, ...] = ("control", "control_2", "control_3")

# The root image's slot name inside ``role_order``.
ROOT_SLOT = "root"

# Probe order is fixed so duplicate stems across extensions resolve
# deterministically (first match wins).
CONTROL_IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


def detect_control_slots(dataset_path: str, stem: str) -> dict[str, dict[str, Any]]:
    """Find a stem's control images across all slot directories.

    Returns an ordered ``{slot_dir: {rel_path, width, height}}`` mapping
    containing only the slots that have a matching file. Dimensions are
    (0, 0) when the file can't be read as an image.
    """
    slots: dict[str, dict[str, Any]] = {}
    for slot in CONTROL_SLOTS:
        slot_dir = os.path.join(dataset_path, slot)
        if not os.path.isdir(slot_dir):
            continue
        for ext in CONTROL_IMAGE_EXTS:
            full = os.path.join(slot_dir, f"{stem}{ext}")
            if not os.path.exists(full):
                continue
            try:
                with Image.open(full) as img:
                    w, h = img.size
            except Exception:
                w, h = 0, 0
            slots[slot] = {
                "rel_path": f"{slot}/{stem}{ext}",
                "width": w,
                "height": h,
            }
            break
    return slots


def list_control_stem_maps(dataset_path: str) -> dict[str, dict[str, str]]:
    """Scan each control slot directory once into ``{slot: {stem: filename}}``.

    Bulk variant of :func:`detect_control_slots` for the ``/pairs``
    endpoint — one ``scandir`` per slot instead of per-stem probing.
    Stems are lowercased to match the pairs keying; on duplicate stems
    across extensions the :data:`CONTROL_IMAGE_EXTS` order wins.
    """
    result: dict[str, dict[str, str]] = {}
    ext_rank = {ext: i for i, ext in enumerate(CONTROL_IMAGE_EXTS)}
    for slot in CONTROL_SLOTS:
        slot_dir = os.path.join(dataset_path, slot)
        if not os.path.isdir(slot_dir):
            continue
        stem_map: dict[str, str] = {}
        try:
            entries = os.scandir(slot_dir)
        except OSError:
            continue
        with entries:
            for entry in entries:
                if not entry.is_file(follow_symlinks=False):
                    continue
                stem, ext = os.path.splitext(entry.name.lower())
                if ext not in ext_rank:
                    continue
                current = stem_map.get(stem)
                if current is None or (
                    ext_rank[ext]
                    < ext_rank[os.path.splitext(current.lower())[1]]
                ):
                    stem_map[stem] = entry.name
        if stem_map:
            result[slot] = stem_map
    return result


def resolve_effective_roles(
    media_file: str,
    slot_files: dict[str, str],
    role_order: list[str] | None,
) -> tuple[str, list[str]]:
    """Resolve the logical (target, controls) rel-paths for one pair group.

    ``slot_files`` maps control slot dirs to their rel paths (slot order).
    ``role_order`` lists physical slot names with position 0 = target; a
    partial list has the unlisted available slots appended in default
    order. Any entry naming an unavailable slot invalidates the whole
    order (fall back to default) — health checks flag it, training treats
    the item as an incomplete pair.
    """
    available: dict[str, str] = {ROOT_SLOT: media_file, **slot_files}
    default_order = list(available.keys())

    order = default_order
    if role_order:
        if all(slot in available for slot in role_order):
            deduped = list(dict.fromkeys(role_order))
            order = deduped + [s for s in default_order if s not in deduped]
        # else: invalid reference → keep default order

    return available[order[0]], [available[s] for s in order[1:]]
