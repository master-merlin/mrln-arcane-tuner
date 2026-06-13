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


def control_slot_dir_name(slot_index: int) -> str:
    """Map a 1-based control slot index to its physical directory name."""
    if not 1 <= slot_index <= len(CONTROL_SLOTS):
        raise ValueError(f"slot must be 1..{len(CONTROL_SLOTS)}")
    return CONTROL_SLOTS[slot_index - 1]


def prepare_control_slot_path(
    dataset_path: str, slot_index: int, stem: str, ext: str
) -> str:
    """Compute (and prepare) the destination path for a control image.

    Creates the slot directory and purges any same-stem sibling files with
    other extensions so slot detection (fixed-priority ext order) stays
    deterministic after a write — mirrors the control-upload route. Returns
    the absolute destination path ``{slot_dir}/{stem}{ext}``.
    """
    ext = ext.lower()
    slot_dir = os.path.join(dataset_path, control_slot_dir_name(slot_index))
    os.makedirs(slot_dir, exist_ok=True)
    for other_ext in CONTROL_IMAGE_EXTS:
        if other_ext == ext:
            continue
        sibling = os.path.join(slot_dir, f"{stem}{other_ext}")
        if os.path.exists(sibling):
            try:
                os.remove(sibling)
            except OSError:
                pass
    return os.path.join(slot_dir, f"{stem}{ext}")


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


# Relative aspect-ratio tolerance for the dim_mismatch warning. Pairs may
# legitimately differ in size (bucketing handles it) — only a different
# *shape* is worth flagging, and 3% matches the scan's AR snapping.
_AR_TOLERANCE = 0.03


def compute_pair_health(dataset) -> dict[str, Any]:
    """On-demand pair-health report for one dataset (disk walk + metadata).

    Disk is the source of truth for which targets/controls exist; the
    scan-maintained ``media_metadata`` supplies dimensions, ``role_order``
    and the ``target_edited_at`` stamp for the warning checks. All
    findings are warnings — none block training by themselves (training
    applies its own skip/abort policy on incomplete pairs).
    """
    from app.core.dataset.media_types import MULTIMEDIA_EXTENSIONS

    path = dataset.path
    targets: dict[str, str] = {}
    if os.path.isdir(path):
        with os.scandir(path) as entries:
            for entry in entries:
                if not entry.is_file(follow_symlinks=False):
                    continue
                if entry.name.startswith((".", "~")):
                    continue
                stem, ext = os.path.splitext(entry.name.lower())
                if ext in MULTIMEDIA_EXTENSIONS:
                    targets[stem] = entry.name

    control_maps = list_control_stem_maps(path)
    active_slots = list(control_maps.keys())

    missing_by_slot: dict[str, list[str]] = {}
    if getattr(dataset, "kind", "standard") == "edit" and not active_slots:
        # An edit dataset with no controls at all: every target is unpaired.
        if targets:
            missing_by_slot[CONTROL_SLOTS[0]] = sorted(targets)
    for slot, stem_map in control_maps.items():
        missing = sorted(s for s in targets if s not in stem_map)
        if missing:
            missing_by_slot[slot] = missing

    orphans = [
        {"slot": slot, "rel_path": f"{slot}/{fname}"}
        for slot, stem_map in control_maps.items()
        for s, fname in sorted(stem_map.items())
        if s not in targets
    ]

    paired_count = sum(
        1 for s in targets
        if active_slots and all(s in control_maps[slot] for slot in active_slots)
    )
    fully_paired = (
        bool(targets) and bool(active_slots) and paired_count == len(targets)
    )

    warnings: list[dict[str, str]] = []
    for rel_path, meta in (dataset.media_metadata or {}).items():
        stem = os.path.splitext(os.path.basename(rel_path))[0].lower()
        info = meta.get("control_info") or {}
        slots = info.get("slots") or {}

        role_order = info.get("role_order")
        if role_order:
            avail = {ROOT_SLOT, *slots.keys()}
            if any(s not in avail for s in role_order):
                warnings.append({"stem": stem, "type": "role_order_invalid"})

        t_w, t_h = meta.get("width") or 0, meta.get("height") or 0
        if t_w and t_h:
            target_ar = t_w / t_h
            for slot_info in slots.values():
                c_w, c_h = slot_info.get("width") or 0, slot_info.get("height") or 0
                if c_w and c_h and (
                    abs(c_w / c_h - target_ar) / target_ar > _AR_TOLERANCE
                ):
                    warnings.append({"stem": stem, "type": "dim_mismatch"})
                    break

        stamp = info.get("target_edited_at")
        if stamp and slots:
            try:
                newest_control = max(
                    os.path.getmtime(os.path.join(path, s["rel_path"]))
                    for s in slots.values() if s.get("rel_path")
                )
            except (OSError, ValueError):
                newest_control = None
            if newest_control is not None and stamp > newest_control:
                warnings.append(
                    {"stem": stem, "type": "target_edited_after_control"}
                )

    return {
        "kind": getattr(dataset, "kind", "standard"),
        "target_count": len(targets),
        "paired_count": paired_count,
        "fully_paired": fully_paired,
        "active_slots": active_slots,
        "missing_by_slot": missing_by_slot,
        "orphans": orphans,
        "warnings": warnings,
    }
