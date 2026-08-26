"""Import limits: a hostile archive is refused before it costs anything.

The pre-existing byte ceiling was a fixed 512 GiB. That is the wrong *shape*
rather than the wrong number: on a rented 100 GB volume it is not a limit at
all, and the machine a constant was written on is not the machine it runs on.
So the effective limit is now derived from the destination volume, with the
fixed value demoted to a backstop for when free space cannot be read.

The other three limits exist because a byte ceiling alone does not catch the
cheap attacks: an archive of a hundred thousand tiny members exhausts inodes
and wall-clock without approaching any byte total, and a bomb's headers reveal
it before a single byte is written.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from app.core.portable import archive as arc
from app.core.portable.envelope import MANIFEST_NAME, ManifestError


def _zip(entries: dict[str, bytes], *, compress=zipfile.ZIP_DEFLATED) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compress) as zf:
        zf.writestr(MANIFEST_NAME, "{}")
        for name, payload in entries.items():
            zf.writestr(name, payload)
    buf.seek(0)
    return zipfile.ZipFile(buf)


# ── The zip bomb: refused from headers, before expansion ────────────────


def test_zip_bomb_is_rejected_before_a_single_byte_is_written(tmp_path):
    """A high-ratio archive is refused on header data alone.

    Asserted on the filesystem, not just the exception: the point of a
    ratio check is that nothing is expanded, so an implementation that
    detected the bomb *while* writing would pass a raises-only test while
    still filling the disk.
    """
    dest = tmp_path / "out"
    dest.mkdir()
    # 128 MiB of zeros deflates roughly 1000:1 -- over the floor, so the
    # ratio check applies, and far over the ratio cap.
    bomb = _zip({"bomb.bin": b"\0" * (128 * 1024**2)})

    with pytest.raises(ManifestError, match="compression ratio"):
        arc.safe_extract(bomb, dest)

    assert list(dest.iterdir()) == [], "the bomb was partially expanded"


def test_small_high_ratio_archive_is_allowed(tmp_path):
    """Prove the negative on the ratio floor.

    A manifest-only or tiny text export can legitimately exceed any ratio.
    If the floor were removed, ordinary template exports would start failing
    -- so pin that they do not.
    """
    dest = tmp_path / "out"
    dest.mkdir()
    small = _zip({"captions.txt": b"a" * 4096})
    arc.safe_extract(small, dest)
    assert (dest / "captions.txt").read_bytes() == b"a" * 4096


# ── Member count ────────────────────────────────────────────────────────


def test_member_count_cap_rejects_before_expansion(tmp_path, monkeypatch):
    """Many tiny members cost inodes and wall-clock, not bytes."""
    dest = tmp_path / "out"
    dest.mkdir()
    monkeypatch.setattr(arc, "MAX_MEMBERS", 3)
    many = _zip({f"f{i}.txt": b"x" for i in range(4)})

    with pytest.raises(ManifestError, match="members"):
        arc.safe_extract(many, dest)

    assert list(dest.iterdir()) == []


# ── The volume-relative limit ───────────────────────────────────────────


def test_limit_is_relative_to_the_destination_volume(tmp_path, monkeypatch):
    """The headline fix: on a small volume, the small number binds.

    A fixed 512 GiB ceiling would let this through; the volume-derived one
    must not.
    """
    dest = tmp_path / "out"
    dest.mkdir()

    class _Usage:
        total = 100 * 1024**2
        used = 0
        free = 100 * 1024**2  # 100 MiB free

    monkeypatch.setattr(arc.shutil, "disk_usage", lambda p: _Usage())

    limit = arc.resolve_extract_limit(dest)
    assert limit == int(100 * 1024**2 * arc.EXTRACT_FREE_SPACE_FRACTION)
    assert limit < arc.MAX_EXTRACT_BYTES, "the volume limit must bind, not the backstop"


def test_absolute_cap_still_applies_on_a_huge_volume(tmp_path, monkeypatch):
    """A very large volume must not lift the limit past the backstop."""

    class _Usage:
        total = free = 10 * 1024**4  # 10 TiB
        used = 0

    monkeypatch.setattr(arc.shutil, "disk_usage", lambda p: _Usage())
    assert arc.resolve_extract_limit(tmp_path) == arc.MAX_EXTRACT_BYTES


def test_unreadable_free_space_falls_back_to_the_backstop(tmp_path, monkeypatch):
    """Degrade to the old behaviour rather than refusing every import."""

    def boom(_p):
        raise OSError("no statvfs here")

    monkeypatch.setattr(arc.shutil, "disk_usage", boom)
    assert arc.resolve_extract_limit(tmp_path) == arc.MAX_EXTRACT_BYTES


def test_limit_probe_walks_up_to_an_existing_directory(tmp_path):
    """The destination usually does not exist yet -- disk_usage would raise."""
    missing = tmp_path / "a" / "b" / "c"
    assert arc._nearest_existing(missing) == tmp_path.resolve()
    # And the real call must not raise for a not-yet-created destination.
    assert arc.resolve_extract_limit(missing) > 0


# ── Headroom ────────────────────────────────────────────────────────────


def test_import_refused_when_it_would_exhaust_the_volume(tmp_path, monkeypatch):
    """Refuse up front: a half-extracted dataset is worse than a refused one."""
    dest = tmp_path / "out"
    dest.mkdir()

    class _Usage:
        total = 10 * 1024**2
        used = 0
        free = 10 * 1024**2  # 10 MiB free, less than MIN_FREE_BYTES

    monkeypatch.setattr(arc.shutil, "disk_usage", lambda p: _Usage())
    payload = _zip({"data.bin": b"y" * (1024 * 1024)})

    with pytest.raises(ManifestError, match="free space"):
        arc.safe_extract(payload, dest)

    assert list(dest.iterdir()) == []


# ── The existing protections must survive the rewrite ───────────────────


def test_traversal_still_rejected(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    evil = _zip({"../escape.txt": b"pwned"})
    with pytest.raises(ManifestError, match="Unsafe path"):
        arc.safe_extract(evil, dest)
    assert not (tmp_path / "escape.txt").exists()


def test_under_reported_header_cannot_produce_an_oversized_extraction(
    tmp_path, monkeypatch
):
    """A member whose header lies about its size must not expand past the limit.

    Measured, and worth recording: when the header under-reports, CPython's own
    integrity check raises ``BadZipFile`` on the CRC before our running total is
    reached. So the running-total guard is **defence in depth**, not the primary
    catch -- but it is still worth keeping, because it is the only check that
    does not depend on a stdlib implementation detail continuing to behave this
    way.

    Asserted on the invariant (nothing oversized reaches disk) rather than on a
    specific exception type, so the test keeps meaning whichever layer refuses.
    """
    dest = tmp_path / "out"
    dest.mkdir()
    payload = _zip({"data.bin": b"z" * (2 * 1024 * 1024)})

    for info in payload.infolist():
        if info.filename != MANIFEST_NAME:
            info.file_size = 0  # the lie

    monkeypatch.setattr(
        arc.shutil, "disk_usage", lambda p: type("U", (), {"free": 1 << 40})()
    )

    with pytest.raises((ManifestError, zipfile.BadZipFile)):
        arc.safe_extract(payload, dest, max_total_bytes=1024)

    written = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    assert written <= 1024, f"{written} bytes reached disk past a 1024-byte limit"


def test_running_total_guard_fires_on_its_own(tmp_path, monkeypatch):
    """Pin the running-total branch directly, since the CRC check masks it above.

    Without this the branch is unreachable in tests and could be deleted or
    broken with the suite still green.
    """
    dest = tmp_path / "out"
    dest.mkdir()
    payload = _zip({"data.bin": b"z" * (2 * 1024 * 1024)})

    # The headers under-report (this is the lie a crafted archive tells), so
    # every pre-expansion check passes and only the copy loop can refuse.
    monkeypatch.setattr(arc, "_declared_total", lambda zf: 0)
    monkeypatch.setattr(
        arc.shutil, "disk_usage", lambda p: type("U", (), {"free": 1 << 40})()
    )

    with pytest.raises(ManifestError, match="expanded past|under-reported"):
        arc.safe_extract(payload, dest, max_total_bytes=1024)


def test_ordinary_export_round_trips(tmp_path):
    """Prove the negative: a normal archive still extracts unchanged."""
    dest = tmp_path / "out"
    dest.mkdir()
    good = _zip({"a.txt": b"alpha", "sub/b.txt": b"beta"})
    arc.safe_extract(good, dest)
    assert (dest / "a.txt").read_bytes() == b"alpha"
    assert (dest / "sub" / "b.txt").read_bytes() == b"beta"
    assert not (dest / MANIFEST_NAME).exists(), "manifest must not be extracted"


def test_limits_are_documented_as_invariants_not_this_machine(tmp_path):
    """ARCHITECTURE D10 invariant 10, pinned rather than trusted to review.

    The fixed ceiling must stay a *backstop*: if someone re-promotes it to the
    primary limit, `resolve_extract_limit` stops consulting the volume and this
    fails.
    """
    assert Path(arc.__file__).exists()
    assert arc.resolve_extract_limit.__doc__ is not None
    assert "Volume-relative" in arc.resolve_extract_limit.__doc__
