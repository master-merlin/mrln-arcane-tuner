"""The hash path must be able to read every format the scanner accepts.

LANE-85 (a), found by the user in UAT-9.4. An ``.avif`` in a real dataset had
``solid_hash = NULL``: ``cv2.imdecode`` returned ``None`` because this OpenCV
build reports ``AVIF: NO``, the caller logged ``hash_calculation_failed`` into
a server log nobody reads, and the item sat in the grid looking perfectly
healthy while being **unable to match anything at any threshold**.

The sharp part is that AVIF is a format this product deliberately supports:
``pillow-avif-plugin`` is a pinned dependency, registered in
``app/core/dataset/thumbnails.py``, and the THUMBNAIL path renders AVIF fine.
Two libraries disagreed about what the app supports and nothing reconciled
them, so an image could be visible and unhashable at the same time.

These tests pin the reconciliation, and the constraint that makes it safe:
**cv2 stays the primary decoder.** Every hash already in every user's database
was computed by cv2, and a perceptual hash is a PERSISTED format -- if the
fallback ever started handling files cv2 can read, stored hashes would stop
matching freshly computed ones and duplicate detection would silently degrade
across every existing dataset. So the fallback must be reachable ONLY when cv2
fails, and `test_cv2_stays_the_primary_decoder` is what holds that line.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from app.core import image_hash
from app.core.image_hash import solide_hash_robust

#: 48x48 bits of D-hash, hex-encoded: (48 * 48) / 4 == 576 characters.
EXPECTED_HEX_LEN = 576


def _busy_rgb(w: int = 96, h: int = 72) -> Image.Image:
    """An image with real structure.

    A flat or purely gradient image hashes to all zeros (every adjacent-pixel
    difference is equal), which would let a broken decoder look like a working
    one -- the same vacuity the stability suite excludes `gradient` for.
    """
    rng = np.random.default_rng(20260904)
    arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    arr[h // 3 : 2 * h // 3, w // 4 : 3 * w // 4] = 240  # a solid block, so there ARE edges
    return Image.fromarray(arr, mode="RGB")


def _write_avif(path):
    try:
        import pillow_avif  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"pillow-avif-plugin unavailable: {exc}")
    try:
        _busy_rgb().save(path, format="AVIF")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"this Pillow cannot WRITE avif: {exc}")
    return path


def test_an_avif_image_gets_a_hash(tmp_path):
    """The defect itself: an AVIF must hash, not come back empty."""
    p = _write_avif(tmp_path / "car.avif")

    result = solide_hash_robust(str(p))

    assert isinstance(result, str)
    assert len(result) == EXPECTED_HEX_LEN, f"expected a 48x48 hash, got {len(result)} hex chars"
    int(result, 16)  # parses as hex
    assert set(result) != {"0"}, "an all-zero hash would pass this test while meaning nothing"


def test_the_avif_case_is_not_vacuous(tmp_path):
    """Control: prove cv2 alone really cannot read that file on this build.

    Without this, a future OpenCV that gains AVIF would make the test above
    pass for a reason unrelated to the fallback, and the fallback could rot
    unnoticed. This test SKIPS rather than fails in that world, and says so.
    """
    p = _write_avif(tmp_path / "car.avif")

    raw = np.fromfile(str(p), dtype=np.uint8)
    if cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE) is not None:
        pytest.skip(
            "this OpenCV decodes AVIF, so the Pillow fallback is no longer exercised "
            "by the AVIF case -- find another format cv2 cannot read, or retire it"
        )
    assert cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE) is None


def test_cv2_stays_the_primary_decoder(tmp_path, monkeypatch):
    """A format cv2 CAN read must never reach the fallback.

    Every hash in every existing database came from cv2. If the fallback ever
    handled those files too, stored hashes would stop matching recomputed ones.
    Proven by making the fallback fatal and checking a PNG still hashes.
    """
    p = tmp_path / "car.png"
    _busy_rgb().save(p, format="PNG")
    expected = solide_hash_robust(str(p))

    def _explode(*a, **k):
        raise AssertionError("the Pillow fallback ran for a file cv2 can read")

    monkeypatch.setattr(image_hash.Image, "open", _explode)

    assert solide_hash_robust(str(p)) == expected


def test_a_file_no_decoder_can_read_still_raises(tmp_path):
    """Prove the negative: undecodable input must raise, never return a hash.

    "Could not decode" must not become "here is a hash" -- a bogus hash is
    worse than none, because it would compare equal to other bogus hashes and
    invent duplicates.
    """
    p = tmp_path / "not-an-image.avif"
    p.write_bytes(b"\x00\x01\x02 this is not an image at all \xff\xfe")

    with pytest.raises(Exception):
        solide_hash_robust(str(p))


def test_an_empty_file_raises(tmp_path):
    p = tmp_path / "empty.png"
    p.write_bytes(b"")

    with pytest.raises(Exception):
        solide_hash_robust(str(p))
