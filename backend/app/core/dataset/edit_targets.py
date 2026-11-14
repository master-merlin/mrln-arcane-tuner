"""Every filesystem path a destructive pixel edit of one image touches.

A crop takes its ``relative_path`` from the client. The source image and each
file derived from it are resolved here, once, through
:func:`validate_path_within`, BEFORE any filesystem call: one location that
leaves the dataset root (``..``, an absolute path, a link) refuses the whole
edit with the guard's 403, so a refusal always comes before a mutation.

The returned paths are the ones the edit must use. Re-deriving a location
with ``os.path.join`` after the check would make the check decorative: the
joined string is not the one that was validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.api._path_guard import validate_path_within


@dataclass(frozen=True)
class EditTargets:
    """Resolved, contained paths for one edit."""

    source: Path


def resolve_edit_targets(dataset_path: str | Path, relative_path: str) -> EditTargets:
    """Resolve every target of an edit of *relative_path* inside *dataset_path*.

    Raises the guard's ``HTTPException(403)`` if any one of them leaves the
    dataset root.
    """
    return EditTargets(
        source=validate_path_within(Path(dataset_path) / relative_path, dataset_path),
    )
