"""Perceptual hashes must stay comparable across a dependency bump.

`solide_hash_robust` values are PERSISTED per image and compared against each
other for duplicate and near-duplicate detection, so they are a stored format,
not a transient computation. If an image-library upgrade moves them far enough,
rows written before the upgrade stop matching rows written after it and the
corpus silently splits: duplicates go unnoticed, and nothing errors.

Nothing caught that. The rest of `test_image_hash.py` asserts properties that
hold under any drift — determinism within a run, different images differing,
rotations staying similar — so opencv could change every value and stay green.

Measured on the 2026-09-03 opencv 4.13 -> 5.0.0 bump: on byte-identical inputs
three of five fixtures moved, by 6 to 14 bits out of 2304, and the product's
own similarity between the old and new hash of the SAME image stayed at
0.9970 or better against a 0.9 duplicate threshold. So that bump was safe --
by a wide margin, and now by a measured one.

This test pins that margin rather than the exact digits. Exact values would
fail on any platform whose opencv build rounds differently, which is noise; a
similarity floor fails only when a change is big enough to actually reclassify
images, which is the thing worth knowing. When it fires, the question is not
"update the constant" -- it is whether existing datasets need a rehash.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.core.image_hash import (
    _hex_to_bool_grid,
    similarity_from_grids,
    solide_hash_robust,
)

#: Below this, two hashes of the SAME image would start to be treated as
#: different images by anything using the harmonization default (0.9). The
#: floor is deliberately far above that: the observed worst case across an
#: opencv MAJOR bump was 0.9970, so 0.99 flags a regression an order of
#: magnitude worse than the one we accepted while ignoring rounding noise.
SIMILARITY_FLOOR = 0.99

#: Recorded on opencv 5.0.0 / numpy 2.3.5 (2026-09-03), the versions
#: backend/requirements.txt pins. Regenerate ONLY together with a decision
#: about rehashing existing datasets -- see the module docstring.
#:
#: `gradient` is deliberately ABSENT. A pure horizontal ramp hashes to all
#: zeros (every adjacent-pixel difference is equal after the background strip
#: and resize), and two all-zero hashes compare as identical no matter what the
#: library does -- so pinning it would add a case that can never fail. It stays
#: in the within-a-run test, where "all zeros, every time" is still a real
#: assertion, and is excluded here rather than quietly passing.
REFERENCE: dict[str, str] = {
    "noise_rgb": (
        "7251d609496399a062b29ae64eedb5c4b4c52748eca5658ad98a6e8cb71224ca"
        "55c5433552ab7bc4966a2d249dcc8a55c0d679c9d8e2e71c78c728bef2ad3e9b"
        "59b5566a5652c935ab6cb965645672b43563669a0a229956b51a921048589a64"
        "304c0c5b48a15525553657746e952ac761a52b652a6645a92d4b2cc94a95c5b2"
        "aad730c946556152a1af852a694a389ce89d4ccf1959635d9529254b499563496"
        "693d5976c6d6b2ab4990692dadb4d579554d96953265632552a5d22af2275249"
        "6fb4db568d51ae132955b4a24b42a93c62eb6ad6b236695b2ad72cac96633655"
        "199b08f496955ab47364a94b6755db195269aa3453c26b15a85743e2d525e956"
        "bb75a6426cc0ddfaacb2f2848e77594b1a99b636b34caa5869dd3694b8e774e"
    ),
    "tiny": (
        "07f83f01617e07f83f01617e07ba7f03617e5e06bf01617e5e06bf01617e5e06"
        "bc0341fe5f862c0f41fe5f862c0f41fe07860c0f01fe07860f0d01fa07860f0d"
        "01fa07860f2d017807828bfd007807828bfd00780780abcd06000780abcd0600"
        "0780abcd060017803f0d1e0017803f0d1e0007e03f0d1e0001f80f0c1f8601f8"
        "0f0c1f8601fa0f0c1f86007e0f3c1f82007e0f3c1f82407e833c1f80601ec23c"
        "0780601ec23c0780601e8c0d41e0581ebc0d41e0581ebc0d41e01e00bc0f7078"
        "1e00bc0f78781e00fc0f783c1e00fc307e1e1e00fc307e1e1e02fc307e1e1e06"
        "f8007e1e1e06f8007e1e1e02fc007e021f80bc0078021f80bc0078025fc0bc01"
        "78005fe0bc0378005fe0bc0378007fa0fc005a007fa0fc107e007fa0fc107e00"
    ),
    "wide": (
        "4a29c4c6aaed1728cb948a45b5326b4acb56a4ad4b62ca56aeab6cb54a55ab53"
        "6b9534d932d2cadcb4bb66b6998cb59a26b6831e1a52b0666a115b3ad0662a2b"
        "5b2bd852b52b6971db50b4a96bd95353892ca359f631487e555dae258c92565e"
        "ae6d8c3552d6a64c4d2552d362e04d2152c35a615343d64b526593434b4b92f5"
        "b4da6d4b90d4c4c9355bb2d2d455097cb1da9956491633578b7655162b13bbac"
        "55181b2291ed1a9b5976d6d52a8d59564f56ae294a454d16ad6956794f568db9"
        "5a9306560cb25ab3a6550b345a932f55d92bdcd34e5d912b88d35c6db529ad9a"
        "495d4d132d3af3494cd36b53514b4c153243156aa54532c72d32d64d3653edb6"
        "d56a4652e4b751664a5a246b5ac50b5cb4ea2ac1b149d4a22ed035cbe4966cd2"
    ),
}


def _fixture(name: str, tmp_path):
    """Deterministic images, seeded so any machine produces the same bytes."""
    rng = np.random.default_rng(20260903)
    specs = {
        "noise_rgb": lambda: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "gradient": lambda: np.tile(
            np.linspace(0, 255, 320, dtype=np.uint8)[None, :, None], (240, 1, 3)
        ),
        "tiny": lambda: rng.integers(0, 256, (17, 23, 3), dtype=np.uint8),
        "wide": lambda: rng.integers(0, 256, (64, 512, 3), dtype=np.uint8),
    }
    # Draw every spec in a fixed order so the shared rng yields stable bytes.
    arrays = {k: fn() for k, fn in specs.items()}
    path = tmp_path / f"{name}.png"
    Image.fromarray(arrays[name]).save(path)
    return path


@pytest.mark.parametrize("name", ["noise_rgb", "gradient", "tiny", "wide"])
def test_hash_is_stable_within_a_run(name, tmp_path):
    """Baseline the floor test depends on: the same file hashes identically."""
    path = _fixture(name, tmp_path)
    assert solide_hash_robust(str(path)) == solide_hash_robust(str(path))


@pytest.mark.parametrize("name", ["noise_rgb", "gradient", "tiny", "wide"])
def test_hash_still_matches_the_recorded_reference(name, tmp_path):
    """The stored-format guard: today's hash must still READ AS the same image
    as the hash recorded when these dependencies were last accepted."""
    if name not in REFERENCE:
        pytest.skip(f"{name} is excluded from REFERENCE on purpose — see its note")
    fresh = solide_hash_robust(str(_fixture(name, tmp_path)))
    sim = similarity_from_grids(
        _hex_to_bool_grid(REFERENCE[name]), _hex_to_bool_grid(fresh)
    )
    assert sim >= SIMILARITY_FLOOR, (
        f"{name}: a stored hash and a freshly computed one now agree at only "
        f"{sim:.6f} (floor {SIMILARITY_FLOOR}). An image library upgrade has "
        "moved perceptual hashes far enough to change duplicate detection; "
        "existing datasets may need a rehash before this is accepted."
    )


def test_the_floor_would_actually_catch_a_reclassification():
    """Negative control: a hash corrupted far enough must fail the floor, or
    the assertion above is decoration."""
    good = "f0" * 288
    bad = ("f0" * 144) + ("0f" * 144)
    sim = similarity_from_grids(_hex_to_bool_grid(good), _hex_to_bool_grid(bad))
    assert sim < SIMILARITY_FLOOR, (
        f"a half-inverted hash scored {sim:.6f}, at or above the floor — the "
        "floor is too low to catch anything"
    )
