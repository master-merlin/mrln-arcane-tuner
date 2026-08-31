"""LANE-53 — a flat rendition is ADOPTED on first read, not regenerated.

``a5003618`` moved every rendition to ``.thumbnails/<edge>/<stem>.webp``.
`ensure_thumbnail` only ever stats that path, so a dataset still in the flat
layout has **no readable cache at all**: every tile is decoded from the
full-size source on first view. LANE-40 (`609475ca`) surveyed those files as
"dead bytes with no read path" and shipped a sweep whose whole remedy is
``unlink`` — correct about the pixels, wrong about the cost.

**Measured, quiet machine, `backend/datasets/Acura_NSX_Type_S_2022` (25 media,
621 MB of source):** regenerating its 25 tiles at edge 256 takes **49.80 s
total, 1.99 s mean, 3.93 s worst** against **2.82 ms** for the same 25 cache
hits. 87 of the user's 95 cached datasets have no ``256/`` directory at all,
so that cost is live for almost the whole library.

**The edge is recoverable, in two steps, and the second one is not optional.**
`_legacy_path_for` composes the one name this source and size was written to —
the same composition `_legacy_thumbnail_paths` already uses for invalidation,
so a rendition can never be adopted after the edit that should have retired
it. But composing a name does not settle whose bytes are in it:
``foo@1024.webp`` is what the old scheme called BOTH ``foo.png``'s 1024
rendition and ``foo@1024.png``'s default one, which is why `a5003618` moved
the size into a directory. So adoption also proves provenance by pixel size —
a default rendition is capped at 256 and can never be 1024px. That gate was
added because the first implementation here shipped without it and DID serve
one source's tile as another's cover; the test meant to catch it passed
anyway, because it read the two sources in the order that consumed the
contested file first.

Adoption is gated on ``rendition mtime >= source mtime``. The app's own
invalidation already deletes the legacy names whenever the pixels change
(`thumbnail_paths_for` appends them), so a stale flat rendition can only come
from an out-of-band edit — and 27 of the user's 3180 flat renditions are
exactly that, so the gate is neither a no-op nor a formality.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from app.core.dataset import thumbnails

BLUE = (0, 0, 255)
RED = (255, 0, 0)


# ── Helpers ──────────────────────────────────────────────────────────────


def _source(path: Path, colour: tuple[int, int, int] = BLUE, size: int = 900) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), colour).save(path)
    return path


def _webp(path: Path, colour: tuple[int, int, int] = RED, size: int = 256) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), colour).save(path, "WEBP", quality=90)
    return path


def _age(path: Path, seconds: float) -> None:
    """Backdate *path* by *seconds* so mtime ordering is deterministic.

    Filesystem timestamp granularity is coarse enough that two writes in the
    same test can land on the identical mtime; every ordering assertion here
    sets the times explicitly rather than relying on write order.
    """
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns - int(seconds * 1e9)))


def _pixel(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as im:
        return im.convert("RGB").getpixel((5, 5))


def _near(actual: tuple[int, int, int], expected: tuple[int, int, int]) -> bool:
    return all(abs(a - e) <= 12 for a, e in zip(actual, expected, strict=True))


def _flat_dataset(root: Path, stem: str = "foo") -> Path:
    """A dataset exactly as an upgraded install finds it.

    The rendition is painted a DIFFERENT colour from the source purely as a
    tracer: it is the only way a test can tell "these bytes were reused" from
    "these bytes were re-derived". It is not a claim that the two disagree —
    on disk they do not (160 of 160 real flat renditions matched the
    dimensions of the ``256/`` file the app regenerated beside them).
    """
    root.mkdir(parents=True, exist_ok=True)
    src = _source(root / f"{stem}.png", BLUE)
    _age(src, 60)
    _webp(root / ".thumbnails" / f"{stem}.webp", RED, 256)
    return root


# ── The defect, stated as observable output ──────────────────────────────


def test_first_view_reuses_the_flat_rendition_instead_of_decoding_the_source(
    tmp_path, monkeypatch,
):
    """The whole lane in one assertion: the source is never decoded.

    Asserted on the observable outcome — ``generate_thumbnail`` is not called
    and the served bytes are the flat rendition's — rather than on a timing,
    which is what makes it a guard instead of a benchmark. Before this lane
    the source was decoded for every tile of every unmigrated dataset.
    """
    ds = _flat_dataset(tmp_path / "ds")

    calls: list[str] = []
    real = thumbnails.generate_thumbnail
    monkeypatch.setattr(
        thumbnails,
        "generate_thumbnail",
        lambda src, dst, edge=256: calls.append(src) or real(src, dst, edge),
    )

    served = thumbnails.ensure_thumbnail(str(ds), "foo.png")

    assert served == ds / ".thumbnails" / "256" / "foo.webp"
    assert calls == [], "the flat rendition was on disk — nothing should decode"
    assert _near(_pixel(served), RED), "served bytes are the adopted rendition"


def test_adoption_moves_the_bytes_rather_than_copying_them(tmp_path):
    """A rename, not a duplicate: the orphan stops existing.

    LANE-40 promised the user those bytes go away. Adoption keeps that
    promise by relocating them — if this copied, the migration survey would
    keep offering to reclaim a file the cache now depends on.
    """
    ds = _flat_dataset(tmp_path / "ds")
    flat = ds / ".thumbnails" / "foo.webp"
    before = flat.read_bytes()

    served = thumbnails.ensure_thumbnail(str(ds), "foo.png")

    assert not flat.exists(), "the flat orphan must be gone, not duplicated"
    assert served.read_bytes() == before
    assert thumbnails.legacy_layout_survey(str(ds)) == (0, 0)


def test_the_flat_path_is_still_never_served_directly(tmp_path):
    """Adoption does not resurrect the old layout as an addressable path.

    The collision `a5003618` fixed (``foo.png``@512 and ``foo@512.png``@256
    naming the same file) comes back the moment a flat name is servable, so
    the returned path must always be under ``<edge>/``.
    """
    ds = _flat_dataset(tmp_path / "ds")

    served = thumbnails.ensure_thumbnail(str(ds), "foo.png")

    assert served.parent.name == "256"
    assert served.parent.parent.name == ".thumbnails"


# ── Which rendition is adopted — derived, never guessed ───────────────────


def test_each_size_adopts_only_its_own_flat_name(tmp_path):
    """256 takes ``<stem>.webp``; 1024 takes ``<stem>@1024.webp``.

    Pinned per size because the flat scheme encoded the default size by its
    ABSENCE, so an off-by-one in that branch would silently serve a 256 tile
    as a 1024 cover.
    """
    ds = tmp_path / "ds"
    ds.mkdir()
    src = _source(ds / "foo.png", BLUE)
    _age(src, 60)
    _webp(ds / ".thumbnails" / "foo.webp", RED, 256)
    _webp(ds / ".thumbnails" / "foo@1024.webp", (0, 255, 0), 1024)

    small = thumbnails.ensure_thumbnail(str(ds), "foo.png", 256)
    large = thumbnails.ensure_thumbnail(str(ds), "foo.png", 1024)

    assert _near(_pixel(small), RED)
    assert _near(_pixel(large), (0, 255, 0))
    assert Image.open(large).size == (1024, 1024)


def test_a_source_whose_own_name_contains_a_size_suffix_is_not_confused(
    tmp_path,
):
    """``foo@1024.webp`` names TWO different renditions. Pixel size decides.

    The old scheme gave that one filename to both ``foo.png``'s 1024
    rendition and ``foo@1024.png``'s default one — the collision that forced
    the relayout. Adoption reads those bytes back, so it inherits the
    ambiguity and has to settle it: a default rendition is capped at 256 and
    can never be 1024px, so the long edge is proof of provenance.

    **``foo.png`` is read FIRST, and that ordering is the point.** An earlier
    version of this test read ``foo@1024.png`` first, which adopted the
    contested file out of the way before ``foo.png`` could ask for it — so it
    passed against an implementation that did serve the wrong pixels
    (verified: it returned a 256px red tile as ``foo.png``'s 1024 cover).
    A test whose subject is a collision must make the collision reachable.
    """
    ds = tmp_path / "ds"
    ds.mkdir()
    # Larger than the 1024 rendition so the size assertion below is a real
    # downscale — a 900px source would come back 900px and prove nothing.
    for name in ("foo.png", "foo@1024.png"):
        _age(_source(ds / name, BLUE, size=1200), 60)
    # The flat rendition of the ODD source, at the default size.
    _webp(ds / ".thumbnails" / "foo@1024.webp", RED, 256)

    plain_large = thumbnails.ensure_thumbnail(str(ds), "foo.png", 1024)
    odd = thumbnails.ensure_thumbnail(str(ds), "foo@1024.png", 256)

    assert _near(_pixel(plain_large), BLUE), (
        "foo.png's 1024 cover must be re-derived from foo.png, never taken "
        "from the file that happens to be named foo@1024.webp"
    )
    assert Image.open(plain_large).size == (1024, 1024)
    assert _near(_pixel(odd), RED), (
        "and the contested file must still be adopted by the source it "
        "really belongs to — refusing both would be a regression, not a fix"
    )


def test_a_mis_sized_rendition_is_refused_at_a_non_default_edge(tmp_path):
    """Long edge must EQUAL the requested one, not merely fit inside it.

    ``<=`` would be enough to keep a 256px file out of a 1024 slot only by
    accident; a 512px one would still slip through and paint a soft cover.
    Prove the negative with a rendition that is under the cap and still wrong.
    """
    ds = tmp_path / "ds"
    ds.mkdir()
    _age(_source(ds / "foo.png", BLUE, size=1200), 60)
    _webp(ds / ".thumbnails" / "foo@1024.webp", RED, 512)

    served = thumbnails.ensure_thumbnail(str(ds), "foo.png", 1024)

    assert _near(_pixel(served), BLUE)
    assert Image.open(served).size == (1024, 1024)


def test_a_genuine_large_rendition_is_still_adopted(tmp_path):
    """The positive control for the size gate — it must not refuse everything.

    785 of the ``@``-suffixed renditions measured in the user's library are
    exactly 1024px, i.e. genuine covers. A gate that rejected them too would
    make every test above pass while adopting nothing.
    """
    ds = tmp_path / "ds"
    ds.mkdir()
    _age(_source(ds / "foo.png", BLUE, size=1200), 60)
    _webp(ds / ".thumbnails" / "foo@1024.webp", RED, 1024)

    served = thumbnails.ensure_thumbnail(str(ds), "foo.png", 1024)

    assert _near(_pixel(served), RED)
    assert served == ds / ".thumbnails" / "1024" / "foo.webp"


def test_a_subdirectory_source_adopts_through_the_same_flattened_stem(tmp_path):
    """``control/img.png`` wrote ``control__img.webp`` — adopt that, not
    ``img.webp``, or a control image inherits the root image's tile."""
    ds = tmp_path / "ds"
    ds.mkdir()
    _age(_source(ds / "control" / "img.png", BLUE), 60)
    _webp(ds / ".thumbnails" / "control__img.webp", RED, 256)
    _webp(ds / ".thumbnails" / "img.webp", (0, 255, 0), 256)

    served = thumbnails.ensure_thumbnail(str(ds), "control/img.png")

    assert served == ds / ".thumbnails" / "256" / "control__img.webp"
    assert _near(_pixel(served), RED)


# ── Prove the negative ───────────────────────────────────────────────────


def test_a_rendition_older_than_its_source_is_regenerated_not_adopted(tmp_path):
    """An out-of-band edit must not be papered over with pre-edit pixels.

    Invalidation already deletes the legacy names when the app changes a
    source (`thumbnail_paths_for` appends them), so this only fires for edits
    that bypassed the app — 27 of the 3180 flat renditions in the user's
    library are exactly that case.
    """
    ds = tmp_path / "ds"
    ds.mkdir()
    _webp(ds / ".thumbnails" / "foo.webp", RED, 256)
    _age(ds / ".thumbnails" / "foo.webp", 600)
    _source(ds / "foo.png", BLUE)  # written last => newer than the rendition

    served = thumbnails.ensure_thumbnail(str(ds), "foo.png")

    assert _near(_pixel(served), BLUE), "stale rendition must not be adopted"
    assert served == ds / ".thumbnails" / "256" / "foo.webp"


def test_an_already_migrated_dataset_is_untouched(tmp_path):
    """The control group: a live ``<edge>/`` rendition wins over everything."""
    ds = tmp_path / "ds"
    ds.mkdir()
    _age(_source(ds / "foo.png", BLUE), 60)
    live = _webp(ds / ".thumbnails" / "256" / "foo.webp", (0, 255, 0), 256)
    _webp(ds / ".thumbnails" / "foo.webp", RED, 256)
    before = live.read_bytes()

    served = thumbnails.ensure_thumbnail(str(ds), "foo.png")

    assert served.read_bytes() == before
    assert (ds / ".thumbnails" / "foo.webp").exists(), (
        "an unused orphan is left for the migration sweep, not silently "
        "promoted over a live rendition"
    )


def test_a_missing_source_is_still_a_miss_even_with_a_flat_rendition(tmp_path):
    """Adoption must not make a deleted source look present.

    `ensure_thumbnail` returns None for a missing source; serving an orphan
    rendition would resurrect a tile for a file the dataset no longer has.
    """
    ds = tmp_path / "ds"
    ds.mkdir()
    _webp(ds / ".thumbnails" / "gone.webp", RED, 256)

    assert thumbnails.ensure_thumbnail(str(ds), "gone.png") is None
