"""Shared safe zip build/extract for portable archives.

Pure filesystem + zip helpers: build an archive (manifest first, then files
under a root, skipping regenerable dirs) and extract it safely (reject ``..``
and absolute paths; stream large members to disk so big videos never load
fully into RAM).
"""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.core.portable.envelope import MANIFEST_NAME, ManifestError


def write_zip(
    root: Path, manifest: dict[str, Any], *, skip_dirs: Iterable[str] = ()
) -> io.BytesIO:
    """Build an in-memory zip: ``manifest.json`` first, then every file under *root*.

    Any file whose path contains a path segment in *skip_dirs* is omitted.
    Returns a seeked-to-zero ``BytesIO``.
    """
    root = Path(root)
    skip = set(skip_dirs)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
        for file_path in root.rglob("*"):
            if any(part in skip for part in file_path.parts):
                continue
            if file_path.is_file():
                arc_name = file_path.relative_to(root).as_posix()
                zf.write(file_path, arc_name)
    buf.seek(0)
    return buf


def write_manifest_zip(manifest: dict[str, Any]) -> io.BytesIO:
    """Build a manifest-only archive (no file tree).

    Used by template archives (which carry no on-disk files) and by the
    project archive's top-level envelope. Returns a seeked-to-zero ``BytesIO``.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
    buf.seek(0)
    return buf


def safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract every entry except ``manifest.json`` into *dest*, safely.

    Rejects absolute paths and ``..`` traversal (raises ``ManifestError``).
    Symlink members are not a redirect risk here: ``zf.open`` + ``copyfileobj``
    writes the link's target *text* as an inert regular file, never restoring
    a real symlink, so a later member cannot be redirected outside *dest*.
    """
    dest = Path(dest).resolve()
    for member in zf.infolist():
        name = member.filename
        if name == MANIFEST_NAME or name.endswith("/"):
            continue
        target = (dest / name).resolve()
        if not target.is_relative_to(dest):
            raise ManifestError(f"Unsafe path in archive: {name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
