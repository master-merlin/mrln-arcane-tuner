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


#: Absolute backstop on total uncompressed bytes, used only when the free space
#: on the destination volume cannot be determined. A fixed ceiling is the wrong
#: shape on its own: on a rented 100 GB volume a 512 GiB allowance is no limit
#: at all, so the effective limit is derived from the volume at import time --
#: see :func:`resolve_extract_limit`. An import is an authenticated local
#: action, so this guards against a malformed or hostile archive filling the
#: disk, not against an attacker who already has credentials.
MAX_EXTRACT_BYTES = 512 * 1024**3  # 512 GiB

#: Fraction of currently-free space an import may consume. The invariant is
#: that a completed import must not leave the volume with no room to operate --
#: the database still needs to write, and a full disk fails in far more places
#: than the import itself.
EXTRACT_FREE_SPACE_FRACTION = 0.9

#: Absolute free-space floor. Below this an import is refused before it starts
#: rather than part-way through, because a half-extracted dataset is worse than
#: a refused one.
MIN_FREE_BYTES = 256 * 1024**2  # 256 MiB

#: Cap on member count. A zip bomb need not be large: an archive of many
#: thousands of tiny members exhausts inodes and wall-clock without ever
#: approaching a byte ceiling. Generous for a real dataset, where each item
#: contributes an image plus a caption sidecar.
MAX_MEMBERS = 100_000

#: Cap on the overall compression ratio (uncompressed / compressed). Dataset
#: exports are dominated by already-compressed media and sit near 1:1; text
#: sidecars reach single digits. A zip bomb is three orders of magnitude above
#: this. The cap rejects BEFORE any expansion, from header data alone.
MAX_COMPRESSION_RATIO = 200

#: The ratio check is only meaningful once an archive is big enough for the
#: ratio to mean anything -- a manifest-only template export is a few hundred
#: bytes of JSON and can legitimately exceed any ratio.
RATIO_CHECK_FLOOR_BYTES = 64 * 1024**2  # 64 MiB


def resolve_extract_limit(dest: Path, absolute_cap: int = MAX_EXTRACT_BYTES) -> int:
    """Largest expansion permitted for an import into *dest*.

    Volume-relative, not fixed: the limit that matters is how much room the
    destination actually has, and that is a property of the deployment, not of
    the machine this constant was written on.

    Falls back to *absolute_cap* if free space cannot be read (an unusual
    filesystem, a path that does not exist yet) -- degrading to the old
    behaviour is better than refusing every import.
    """
    try:
        free = shutil.disk_usage(_nearest_existing(dest)).free
    except OSError:
        return absolute_cap
    return min(absolute_cap, int(free * EXTRACT_FREE_SPACE_FRACTION))


def _nearest_existing(path: Path) -> Path:
    """Walk up to the first directory that exists, for a disk_usage probe.

    The destination is usually created by the import itself, and
    ``disk_usage`` on a non-existent path raises.
    """
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def _declared_total(zf: zipfile.ZipFile) -> int:
    """Total uncompressed size the archive's own headers claim.

    Advisory only -- a crafted header can under-report it, which is why the
    running total during the copy re-checks. Extracted as a function so that
    branch is reachable in tests: with an honest archive the declared size and
    the real size are equal, so the copy-loop guard can never fire through the
    public entry point, and an untestable guard is one nobody notices breaking.
    """
    return sum(
        m.file_size
        for m in zf.infolist()
        if m.filename != MANIFEST_NAME and not m.filename.endswith("/")
    )


def _reject_before_expansion(zf: zipfile.ZipFile, declared: int) -> None:
    """Cheap header-only checks that must run before a single byte is written."""
    members = [
        m
        for m in zf.infolist()
        if m.filename != MANIFEST_NAME and not m.filename.endswith("/")
    ]

    if len(members) > MAX_MEMBERS:
        raise ManifestError(
            f"Archive contains {len(members)} members, over the "
            f"{MAX_MEMBERS} limit."
        )

    compressed = sum(m.compress_size for m in members)
    if declared > RATIO_CHECK_FLOOR_BYTES and compressed > 0:
        ratio = declared / compressed
        if ratio > MAX_COMPRESSION_RATIO:
            raise ManifestError(
                f"Archive compression ratio {ratio:.0f}:1 exceeds the "
                f"{MAX_COMPRESSION_RATIO}:1 limit; refusing to expand it."
            )


def safe_extract(
    zf: zipfile.ZipFile, dest: Path, *, max_total_bytes: int | None = None
) -> None:
    """Extract every entry except ``manifest.json`` into *dest*, safely.

    Rejects absolute paths and ``..`` traversal (raises ``ManifestError``).
    Symlink members are not a redirect risk here: ``zf.open`` + ``copyfileobj``
    writes the link's target *text* as an inert regular file, never restoring
    a real symlink, so a later member cannot be redirected outside *dest*.

    Four limits, in increasing cost order so a hostile archive is refused as
    cheaply as possible:

    1. member count and compression ratio, from headers alone;
    2. declared total size against the limit;
    3. free space on the destination volume;
    4. the running total while copying -- because a crafted header can
       under-report ``file_size``, so the declared check alone is advisory.

    *max_total_bytes* defaults to :func:`resolve_extract_limit`, i.e. relative
    to the destination volume rather than a number fixed at authoring time.
    """
    dest = Path(dest).resolve()

    if max_total_bytes is None:
        max_total_bytes = resolve_extract_limit(dest)

    declared = _declared_total(zf)

    _reject_before_expansion(zf, declared)

    if declared > max_total_bytes:
        raise ManifestError(
            f"Archive expands to {declared} bytes, over the "
            f"{max_total_bytes}-byte import limit for this volume."
        )

    # Headroom: refuse up front rather than half-way through. Checked against
    # the declared size, which is the best estimate available before writing.
    try:
        free = shutil.disk_usage(_nearest_existing(dest)).free
    except OSError:
        free = None
    if free is not None and free - declared < MIN_FREE_BYTES:
        raise ManifestError(
            f"Not enough free space to import: {declared} bytes needed, "
            f"{free} available, and at least {MIN_FREE_BYTES} must remain free."
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
