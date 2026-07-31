"""Shared safe zip build/extract for portable archives.

Pure filesystem + zip helpers: build an archive (manifest first, then files
under a root, skipping regenerable dirs) and extract it safely (reject ``..``
and absolute paths; stream large members to disk so big videos never load
fully into RAM).
"""

from __future__ import annotations

import io
import json
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
            # is_file() FOLLOWS symlinks, so a link inside the dataset pulled
            # its target's bytes into the archive even when that target lives
            # outside the root — the export equivalent of the traversal the
            # read routes now reject with 403.
            if file_path.is_symlink():
                continue
            if file_path.is_file():
                arc_name = file_path.relative_to(root).as_posix()
                zf.write(file_path, arc_name)
    buf.seek(0)
    return buf


def write_zip_to_path(
    dest: Path, root: Path, manifest: dict[str, Any], *, skip_dirs: Iterable[str] = ()
) -> None:
    """Like :func:`write_zip` but streams straight to *dest* on disk instead
    of building the archive in an in-memory buffer first.

    ``zipfile.ZipFile.write`` copies each source file in chunks (not a full
    read into RAM), so this is the version to use when *root* may contain
    multi-GB media (e.g. a video dataset) — the caller only ever holds one
    file's worth of buffered I/O at a time, never the whole archive.
    """
    root = Path(root)
    skip = set(skip_dirs)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
        for file_path in root.rglob("*"):
            if any(part in skip for part in file_path.parts):
                continue
            # is_file() FOLLOWS symlinks, so a link inside the dataset pulled
            # its target's bytes into the archive even when that target lives
            # outside the root — the export equivalent of the traversal the
            # read routes now reject with 403.
            if file_path.is_symlink():
                continue
            if file_path.is_file():
                arc_name = file_path.relative_to(root).as_posix()
                zf.write(file_path, arc_name)


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


def write_bundle_zip(
    manifest: dict[str, Any], entries: dict[str, bytes]
) -> io.BytesIO:
    """Build a bundle archive: ``manifest.json`` first, then each named entry
    (``arcname -> bytes``). Used for the project archive, whose entries are
    nested template/dataset ``.zip`` payloads. Returns a seeked-to-zero buffer.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
        for arcname, payload in entries.items():
            zf.writestr(arcname, payload)
    buf.seek(0)
    return buf


def write_bundle_zip_to_path(
    dest: Path, manifest: dict[str, Any], entries: dict[str, bytes | Path]
) -> None:
    """Like :func:`write_bundle_zip` but streams straight to *dest* on disk.

    Each entry may be raw ``bytes`` (small payloads, e.g. a template archive)
    or a ``Path`` to a file already on disk (e.g. a nested dataset archive
    written via :func:`write_zip_to_path`) — the latter is copied into the
    bundle via ``zf.write`` (chunked), never read fully into memory first.
    Used by the project export route so a project bundling multi-GB embedded
    dataset archives is never held fully in RAM.
    """
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
        for arcname, payload in entries.items():
            if isinstance(payload, Path):
                zf.write(payload, arcname)
            else:
                zf.writestr(arcname, payload)


#: Ceiling on the total uncompressed bytes one archive may expand to. An import
#: is an authenticated local action, so this is a guard against a malformed or
#: maliciously-crafted archive filling the disk (a few MB of zeros deflates
#: ~1000:1), not a security boundary. Generous enough for a real multi-GB video
#: dataset export.
MAX_EXTRACT_BYTES = 512 * 1024**3  # 512 GiB


def safe_extract(
    zf: zipfile.ZipFile, dest: Path, *, max_total_bytes: int = MAX_EXTRACT_BYTES
) -> None:
    """Extract every entry except ``manifest.json`` into *dest*, safely.

    Rejects absolute paths and ``..`` traversal (raises ``ManifestError``).
    Symlink members are not a redirect risk here: ``zf.open`` + ``copyfileobj``
    writes the link's target *text* as an inert regular file, never restoring
    a real symlink, so a later member cannot be redirected outside *dest*.

    Expansion is capped at *max_total_bytes*. The declared sizes are checked
    first (cheap) and the running total is checked again while copying, because
    a crafted header can under-report ``file_size``.
    """
    dest = Path(dest).resolve()

    declared = sum(
        m.file_size
        for m in zf.infolist()
        if m.filename != MANIFEST_NAME and not m.filename.endswith("/")
    )
    if declared > max_total_bytes:
        raise ManifestError(
            f"Archive expands to {declared} bytes, over the "
            f"{max_total_bytes}-byte import limit."
        )

    written = 0
    for member in zf.infolist():
        name = member.filename
        if name == MANIFEST_NAME or name.endswith("/"):
            continue
        target = (dest / name).resolve()
        if not target.is_relative_to(dest):
            raise ManifestError(f"Unsafe path in archive: {name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, open(target, "wb") as out:
            while chunk := src.read(1024 * 1024):
                written += len(chunk)
                if written > max_total_bytes:
                    raise ManifestError(
                        "Archive expanded past the "
                        f"{max_total_bytes}-byte import limit "
                        "(declared sizes under-reported)."
                    )
                out.write(chunk)
