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
from app.core.dataset.media_helpers import OVERLAYS_JSON, mask_file_paths, overlay_png_path


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
        # The whole path must resolve inside (a leaf link pointing out is
        # refused), but the edit acts on the NAME: its resolved directory plus
        # its own final component. Deleting or replacing a leaf that is a link
        # inside the dataset then removes the link, never the other image's
        # file it points at.
        p = Path(p)
        validate_path_within(p, dataset_path)
        return validate_path_within(p.parent, dataset_path) / p.name

    source = contained(root / relative_path)

    ds = str(dataset_path)
    dst = thumbnails.thumbnail_path_for(ds, relative_path)
    mask_stem = os.path.splitext(relative_path)[0]

    return EditTargets(
        source=source,
        mask_files=tuple(contained(p) for p in mask_file_paths(ds, mask_stem)),
        thumbnails=tuple(
            contained(p) for p in thumbnails.thumbnail_paths_for(ds, relative_path)
        ),
        thumbnail_dst=contained(dst),
        thumbnail_tmp=contained(dst.with_suffix(dst.suffix + ".tmp")),
        thumbnail_legacy=contained(thumbnails._legacy_path_for(ds, relative_path)),
        overlays_json=contained(root / OVERLAYS_JSON),
        overlays_json_tmp=contained(root / f"{OVERLAYS_JSON}.tmp"),
        overlay_png=contained(overlay_png_path(ds, relative_path)),
    )
