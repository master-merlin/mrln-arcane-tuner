"""Guards for the pinned library cover.

Before this, `preview_image` was pure derived data: every scan overwrote it
with whatever non-audio file the scanner enumerated first, and the user had no
say. The pin adds exactly one piece of state (`preview_pinned`) and the
interesting behaviour is all about what a SCAN does to it — a pin that a rescan
silently reverts is the same defect as having no pin at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.core.dataset_manager import dataset_manager


@pytest.fixture
def ds(tmp_path, monkeypatch):
    """A three-image dataset registered with the manager, cleaned up after."""
    root = tmp_path / "datasets"
    root.mkdir()
    monkeypatch.setattr(dataset_manager, "default_root", str(root))

    path = root / "pin_ds"
    path.mkdir()
    for name in ("a_first.jpg", "b_second.jpg", "c_third.jpg"):
        Image.new("RGB", (64, 48), "blue").save(path / name)

    dataset_manager.create_dataset("pin_ds", path=str(path))
    dataset_manager.scan_dataset("pin_ds")
    yield dataset_manager.datasets["pin_ds"]
    dataset_manager.delete_dataset("pin_ds", delete_files=True)


def test_unpinned_cover_is_the_scanner_s_choice(ds):
    assert ds.preview_pinned is False
    assert ds.preview_image == "a_first.jpg"


def test_pinning_replaces_the_cover(ds):
    result = dataset_manager.set_preview_image("pin_ds", "c_third.jpg")

    assert result.preview_image == "c_third.jpg"
    assert result.preview_pinned is True


def test_a_pin_survives_a_rescan(ds):
    """The whole point. A rescan re-elects the first file for everyone else."""
    dataset_manager.set_preview_image("pin_ds", "c_third.jpg")

    rescanned = dataset_manager.scan_dataset("pin_ds")

    assert rescanned.preview_image == "c_third.jpg"
    assert rescanned.preview_pinned is True


def test_a_pin_survives_a_restart(ds):
    """It is persisted state, not in-memory state — read it back from the DB."""
    dataset_manager.set_preview_image("pin_ds", "b_second.jpg")

    row = dataset_manager._dataset_repo.get_by_name("pin_ds")

    assert row is not None
    assert row["preview_image"] == "b_second.jpg"
    assert row["preview_pinned"] == 1


def test_unpinning_re_elects_immediately(ds):
    """Not "at the next scan" — the user must see the result of their click."""
    dataset_manager.set_preview_image("pin_ds", "c_third.jpg")

    result = dataset_manager.set_preview_image("pin_ds", None)

    assert result.preview_pinned is False
    assert result.preview_image == "a_first.jpg"


def test_a_pin_whose_file_is_deleted_heals_on_the_next_scan(ds):
    """Self-healing: a dangling pin must never strand the card with no way back."""
    dataset_manager.set_preview_image("pin_ds", "c_third.jpg")
    (Path(ds.path) / "c_third.jpg").unlink()

    rescanned = dataset_manager.scan_dataset("pin_ds")

    assert rescanned.preview_pinned is False
    assert rescanned.preview_image == "a_first.jpg"


# ── Rejected inputs ──────────────────────────────────────────────────────


def test_a_missing_file_cannot_be_pinned(ds):
    with pytest.raises(ValueError):
        dataset_manager.set_preview_image("pin_ds", "ghost.jpg")


def test_a_path_outside_the_dataset_cannot_be_pinned(ds):
    """The cover is served back out over /media — an escape here is traversal."""
    with pytest.raises(Exception):  # noqa: B017 - path guard raises HTTPException
        dataset_manager.set_preview_image("pin_ds", "../../etc/passwd")


def test_audio_cannot_be_pinned(ds):
    """Audio has no renderable frame; pinning it would blank the card."""
    (Path(ds.path) / "track.wav").write_bytes(b"RIFF0000WAVE")

    with pytest.raises(ValueError):
        dataset_manager.set_preview_image("pin_ds", "track.wav")


def test_pinning_an_unknown_dataset_raises(ds):
    with pytest.raises(ValueError):
        dataset_manager.set_preview_image("no_such_dataset", "a_first.jpg")


# ── Listing order independence (LANE-115) ───────────────────────────────


def test_scan_cover_does_not_depend_on_listing_order(tmp_path, monkeypatch):
    """`os.scandir` order is filesystem-dependent (alphabetical on NTFS, not
    guaranteed on Linux). The scan's election must not depend on it — it must
    agree with the sorted-order election `_auto_preview_candidate` uses for
    the unpin path, whatever order the raw listing comes back in.
    """
    root = tmp_path / "datasets"
    root.mkdir()
    monkeypatch.setattr(dataset_manager, "default_root", str(root))

    path = root / "order_ds"
    path.mkdir()
    names = ("a_first.jpg", "b_second.jpg", "c_third.jpg")
    for name in names:
        Image.new("RGB", (64, 48), "blue").save(path / name)

    real_scandir = __import__("os").scandir

    class _ReversedScandir:
        """A drop-in for `os.scandir` usable both as an iterator and as the
        context manager some callers (e.g. thumbnail cleanup) use it as —
        real `os.scandir` supports both.
        """

        def __init__(self, scan_path):
            with real_scandir(scan_path) as it:
                entries = sorted(it, key=lambda e: e.name, reverse=True)
            self._it = iter(entries)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._it)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        "app.core.dataset_manager.os.scandir", _ReversedScandir,
    )

    dataset_manager.create_dataset("order_ds", path=str(path))
    scanned = dataset_manager.scan_dataset("order_ds")

    assert scanned.preview_image == "a_first.jpg", (
        "scan cover depends on listing order"
    )

    dataset_manager.set_preview_image("order_ds", "c_third.jpg")
    unpinned = dataset_manager.set_preview_image("order_ds", None)

    assert unpinned.preview_image == scanned.preview_image, (
        "scan cover depends on listing order"
    )

    dataset_manager.delete_dataset("order_ds", delete_files=True)
