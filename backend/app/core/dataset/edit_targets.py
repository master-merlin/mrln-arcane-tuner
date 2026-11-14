"""Every filesystem path a destructive pixel edit of one image touches.

A crop takes its ``relative_path`` from the client. The source image and each
file derived from it are resolved here, once, through
:func:`validate_path_within`, BEFORE any filesystem call: one location that
leaves the dataset root (``..``, an absolute path, a link) refuses the whole
edit with the guard's 403, so a refusal always comes before a mutation.

The returned paths are the ones the edit must use. Re-deriving a location
with ``os.path.join`` after the check would make the check decorative: the
joined string is not the one that was validated. Temporary siblings are
members too, because an existing ``.tmp`` link can leave the root while its
destination stays inside.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.api._path_guard import validate_path_within
from app.core.dataset import thumbnails


@dataclass(frozen=True)
class EditTargets:
    """Resolved, contained paths for one edit of one image."""

    source: Path
    #: ``masks/<stem>.png``, ``masked/<stem>.jpg``, ``masked/<stem>.txt``.
    mask_files: tuple[Path, ...]
    #: Every thumbnail rendition to unlink, current and legacy layout.
    thumbnails: tuple[Path, ...]
    #: The default-size rendition regenerated after the edit, its temporary
    #: sibling, and the legacy flat name the regeneration may adopt from.
    thumbnail_dst: Path
    thumbnail_tmp: Path
    thumbnail_legacy: Path
    overlays_json: Path
    overlays_json_tmp: Path
    overlay_png: Path


def resolve_edit_targets(dataset_path: str | Path, relative_path: str) -> EditTargets:
    """Resolve every target of an edit of *relative_path* inside *dataset_path*.

    Raises the guard's ``HTTPException(403)`` if any one of them leaves the
    dataset root. The source is resolved first.
    """
    root = Path(dataset_path)

    def contained(p: str | Path) -> Path:
        return validate_path_within(p, dataset_path)

    source = contained(root / relative_path)

    mask_stem = os.path.splitext(relative_path)[0]
    rel = relative_path.replace(os.sep, "/")
    overlay_stem = os.path.splitext(os.path.basename(rel))[0]
    ds = str(dataset_path)
    dst = thumbnails.thumbnail_path_for(ds, relative_path)

    return EditTargets(
        source=source,
        mask_files=tuple(contained(p) for p in (
            root / "masks" / f"{mask_stem}.png",
            root / "masked" / f"{mask_stem}.jpg",
            root / "masked" / f"{mask_stem}.txt",
        )),
        thumbnails=tuple(
            contained(p) for p in thumbnails.thumbnail_paths_for(ds, relative_path)
        ),
        thumbnail_dst=contained(dst),
        thumbnail_tmp=contained(dst.with_suffix(dst.suffix + ".tmp")),
        thumbnail_legacy=contained(thumbnails._legacy_path_for(ds, relative_path)),
        overlays_json=contained(root / "overlays.json"),
        overlays_json_tmp=contained(root / "overlays.json.tmp"),
        overlay_png=contained(root / "overlays" / f"{overlay_stem}.png"),
    )
