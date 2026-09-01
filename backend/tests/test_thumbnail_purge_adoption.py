"""LANE-53 — the migration sweep adopts before it unlinks.

The read path learned to adopt a flat rendition (``test_thumbnail_adoption``).
:func:`purge_legacy_layout` did not, and it is the path the user reaches
FIRST: LANE-40's banner offers to "reclaim" the flat layout, so a user who is
told about dead bytes presses the button before he ever opens the dataset.
On the user's own library that button destroyed **3153 adoptable renditions**
and guaranteed him the 38.69 s first view (25 media, 621 MB of source,
MEASURED on a copy) that adoption exists to prevent. **A remedy that makes the
reported problem permanent is not a remedy** — that is the defect this file
pins, and the first test states it as the sweep's output.

Three properties are asserted here that the read path cannot assert alone,
because the sweep visits files no reader ever asks for:

* **Order-independence.** ``foo@1024.webp`` is the name the old scheme gave
  BOTH ``foo.png``'s 1024 rendition and ``foo@1024.png``'s default one — the
  collision `a5003618` existed to fix. The sweep offers that one file to both
  claimants, so which of them asks first is part of the fixture and not an
  incidental: the previous revision of this lane shipped a collision test that
  passed only because it read the two sources in the order that consumed the
  file first. Every collision test below runs in BOTH orders.
* **Uncertainty is not a licence to delete.** A rendition the sweep cannot
  classify is left where it is. Keeping a file costs its bytes; deleting it
  costs the decode this whole lane is about.
* **A live rendition is never overwritten.** ``os.replace`` is happy to move a
  year-old flat file over the ``<edge>/`` one the app wrote this morning, and
  the sweep — unlike a read — never checks the destination on its way in.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from app.core.dataset import thumbnails

BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


# ── Helpers ──────────────────────────────────────────────────────────────


def _source(path: Path, colour: tuple[int, int, int] = BLUE, size: int = 2000) -> Path:
    """A source big enough that every allowed edge is a genuine downscale.

    2000 px matters: thumbnails never upscale, so a 900 px source's "1024"
    rendition is 900 px and the provenance gate would refuse it for a reason
    that has nothing to do with what these tests are about.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), colour).save(path)
    return path


def _webp(path: Path, colour: tuple[int, int, int] = RED, size: int = 256) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), colour).save(path, "WEBP", quality=90)
    return path


def _age(path: Path, seconds: float) -> None:
    """Backdate *path* so mtime ordering is explicit, never write-order luck."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns - int(seconds * 1e9)))


def _pixel(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as im:
        return im.convert("RGB").getpixel((5, 5))


def _near(actual: tuple[int, int, int], expected: tuple[int, int, int]) -> bool:
    """WebP is lossy; a solid fill still round-trips within a few units."""
    return all(abs(a - e) <= 12 for a, e in zip(actual, expected, strict=True))


def _flat_dataset(root: Path, stem: str = "foo") -> Path:
    """A dataset as an upgraded install finds it: a source and its flat tile.

    The rendition is a different colour from the source purely as a tracer —
    the only way a test can tell "these bytes were reused" from "these bytes
    were re-derived".
    """
    root.mkdir(parents=True, exist_ok=True)
    _age(_source(root / f"{stem}.png", BLUE), 60)
    _webp(root / ".thumbnails" / f"{stem}.webp", RED, 256)
    return root


def _force_source_order(monkeypatch, order: tuple[str, ...]) -> None:
    """Pin the order the sweep enumerates sources in.

    The order is a fixture variable, not an accident of ``scandir``: a
    collision is only reachable when the two claimants are tried in a
    particular order, and a test that leaves that to the filesystem tests one
    of the two cases at random.
    """
    monkeypatch.setattr(
        thumbnails, "_iter_thumbnailable_sources", lambda _p: list(order),
    )


ORDERS = [("foo.png", "foo@1024.png"), ("foo@1024.png", "foo.png")]


# ── The defect, stated as observable output ──────────────────────────────


def test_the_sweep_adopts_the_rendition_it_used_to_delete(tmp_path, monkeypatch):
    """The whole lane in one assertion: cleaning up must not cost a decode.

    Before this change the sweep unlinked the flat file, and the first view
    that followed decoded the full-size source. Now the sweep is what makes
    that first view free — asserted on observable output (``generate_thumbnail``
    is never called and the served bytes are the flat rendition's), never on a
    timing.
    """
    ds = _flat_dataset(tmp_path / "ds")
    tracer = (ds / ".thumbnails" / "foo.webp").read_bytes()

    outcome = thumbnails.purge_legacy_layout(str(ds))

    assert outcome.adopted == 1, "the sweep destroyed an adoptable rendition"
    assert outcome.removed == 0

    calls: list[str] = []
    monkeypatch.setattr(
        thumbnails,
        "generate_thumbnail",
        lambda src, dst, edge=256: calls.append(src) or False,
    )
    served = thumbnails.ensure_thumbnail(str(ds), "foo.png")

    assert served == ds / ".thumbnails" / "256" / "foo.webp"
    assert served.read_bytes() == tracer, "the first view re-derived the tile"
    assert calls == [], f"the source was decoded after the sweep: {calls}"


def test_the_sweep_reports_an_adoption_as_an_adoption(tmp_path):
    """What happened to the bytes is what the user must be told.

    LANE-40's banner sells the sweep as freed disk space. If the sweep MOVED
    the file, "reclaimed" is the wrong word for it, and a caller that only
    receives a removal count cannot say anything else.
    """
    ds = _flat_dataset(tmp_path / "ds")
    surveyed, _ = thumbnails.legacy_layout_survey(str(ds))

    outcome = thumbnails.purge_legacy_layout(str(ds))

    assert (outcome.adopted, outcome.removed, outcome.kept) == (1, 0, 0)
    assert outcome.adopted + outcome.removed + outcome.kept == surveyed


# ── The collision, reachable in both orders ──────────────────────────────


def _collision_dataset(root: Path, rendition_edge: int) -> Path:
    """Two sources, ONE flat name, and the pixels decide which owns it.

    ``foo.png``'s 1024 rendition and ``foo@1024.png``'s default rendition were
    both written to ``.thumbnails/foo@1024.webp``. Only one file exists here,
    and *rendition_edge* is who it really belongs to: 256 makes it
    ``foo@1024.png``'s, 1024 makes it ``foo.png``'s.
    """
    root.mkdir(parents=True, exist_ok=True)
    _age(_source(root / "foo.png", BLUE), 60)
    _age(_source(root / "foo@1024.png", BLUE), 60)
    _webp(root / ".thumbnails" / "foo@1024.webp", GREEN, rendition_edge)
    return root


@pytest.mark.parametrize("order", ORDERS, ids=["plain-first", "suffixed-first"])
def test_a_256px_flat_file_goes_to_the_source_whose_name_carries_the_suffix(
    tmp_path, monkeypatch, order,
):
    """A 256 px file cannot be anyone's 1024 rendition. Both orders.

    This is the mutation that caught the previous revision: with the pixel-size
    gate removed, whichever source was enumerated first took the file, and
    ``foo.png`` served a 256 px tile as its 1024 cover.
    """
    ds = _collision_dataset(tmp_path / "ds", rendition_edge=256)
    _force_source_order(monkeypatch, order)

    outcome = thumbnails.purge_legacy_layout(str(ds))

    mine = ds / ".thumbnails" / "256" / "foo@1024.webp"
    theirs = ds / ".thumbnails" / "1024" / "foo.webp"
    assert mine.exists(), f"the 256px rendition was not adopted by its owner ({order})"
    assert _near(_pixel(mine), GREEN)
    assert Image.open(mine).size == (256, 256)
    assert not theirs.exists(), (
        f"a 256px tile was adopted as a 1024 cover ({order}) — the collision "
        "a5003618 fixed is open again"
    )
    assert outcome.adopted == 1
    assert outcome.removed == 0


@pytest.mark.parametrize("order", ORDERS, ids=["plain-first", "suffixed-first"])
def test_a_1024px_flat_file_goes_to_the_source_whose_rendition_it_is(
    tmp_path, monkeypatch, order,
):
    """The same name, the other owner. Both orders.

    The positive control for the gate: it must not refuse everything ambiguous,
    or the sweep silently stops adopting and every other test here still
    passes.
    """
    ds = _collision_dataset(tmp_path / "ds", rendition_edge=1024)
    _force_source_order(monkeypatch, order)

    outcome = thumbnails.purge_legacy_layout(str(ds))

    mine = ds / ".thumbnails" / "1024" / "foo.webp"
    theirs = ds / ".thumbnails" / "256" / "foo@1024.webp"
    assert mine.exists(), f"the 1024px rendition was not adopted by its owner ({order})"
    assert _near(_pixel(mine), GREEN)
    assert Image.open(mine).size == (1024, 1024)
    assert not theirs.exists(), (
        f"a 1024px cover was adopted as a 256px tile ({order})"
    )
    assert outcome.adopted == 1


# ── What may be deleted, and what may not ────────────────────────────────


def test_a_stale_rendition_is_removed_rather_than_adopted(tmp_path):
    """Adopting stale pixels would be worse than deleting them.

    A flat rendition older than its source can only come from an edit that
    bypassed the app (invalidation already deletes the legacy names) — 27 of
    the user's 3180 are exactly that. Adopting one makes pre-edit pixels
    READABLE, which is the one outcome worse than a regeneration.
    """
    ds = _flat_dataset(tmp_path / "ds")
    stale = ds / ".thumbnails" / "foo.webp"
    _age(stale, 3600)

    outcome = thumbnails.purge_legacy_layout(str(ds))

    assert (outcome.adopted, outcome.removed, outcome.kept) == (0, 1, 0)
    assert not stale.exists()
    assert not (ds / ".thumbnails" / "256" / "foo.webp").exists(), (
        "a stale rendition was promoted into the live cache"
    )


def test_an_orphan_of_a_deleted_source_is_removed(tmp_path):
    """The reclaim still reclaims: no source composes this name.

    The dataset root is enumerated by the same rule the scan uses, and the name
    carries no subdirectory marker, so it can only have come from a root-level
    source — and that source is gone.
    """
    ds = _flat_dataset(tmp_path / "ds")
    orphan = _webp(ds / ".thumbnails" / "deleted.webp", RED, 256)

    outcome = thumbnails.purge_legacy_layout(str(ds))

    assert not orphan.exists(), "a genuine orphan survived the reclaim"
    assert (outcome.adopted, outcome.removed, outcome.kept) == (1, 1, 0)


def test_a_subdirectory_sources_rendition_is_kept_not_deleted(tmp_path):
    """Uncertain is not a licence to delete — the core rule of this lane.

    ``control/img.png`` flattens to ``control__img``, and nothing here walks
    below the dataset root, so the sweep cannot say whether that source still
    exists. Deleting it costs a full decode of a control image; keeping it
    costs its bytes. LANE-40 chose the first without noticing there was a
    choice.
    """
    ds = _flat_dataset(tmp_path / "ds")
    _age(_source(ds / "control" / "img.png", BLUE), 60)
    unknown = _webp(ds / ".thumbnails" / "control__img.webp", GREEN, 256)

    outcome = thumbnails.purge_legacy_layout(str(ds))

    assert unknown.exists(), (
        "a subdirectory source's rendition was deleted on a guess"
    )
    assert (outcome.adopted, outcome.removed, outcome.kept) == (1, 0, 1)


def test_a_root_source_that_owns_a_double_underscore_name_is_still_judged(tmp_path):
    """Knowing beats guessing: the ``__`` rule is a fallback, not a veto.

    ``my__file.png`` is a perfectly ordinary root-level filename, and its flat
    rendition's name is indistinguishable from a subdirectory source's. It is
    enumerated, so the sweep KNOWS whose it is and the gates decide — here they
    refuse it as stale, and refusing means deleting. If the ``__`` check ran
    first, every dataset with a double underscore in a filename would keep its
    dead renditions forever and the sweep would quietly stop working for it.
    """
    ds = tmp_path / "ds"
    ds.mkdir()
    _age(_source(ds / "my__file.png", BLUE), 60)
    stale = _webp(ds / ".thumbnails" / "my__file.webp", RED, 256)
    _age(stale, 3600)

    outcome = thumbnails.purge_legacy_layout(str(ds))

    assert not stale.exists(), "a known source's stale rendition was kept on a guess"
    assert (outcome.adopted, outcome.removed, outcome.kept) == (0, 1, 0)


def test_interrupted_write_residue_is_removed_whatever_its_name(tmp_path):
    """A ``.webp.tmp`` is never a complete rendition, so it is never adoptable.

    Including one that carries the subdirectory marker: the "keep what you
    cannot classify" rule protects renditions, and a half-written temp file is
    not one.
    """
    ds = _flat_dataset(tmp_path / "ds")
    residue = [
        _webp(ds / ".thumbnails" / "gone.webp.tmp", RED, 256),
        _webp(ds / ".thumbnails" / "control__img.webp.tmp", RED, 256),
    ]

    outcome = thumbnails.purge_legacy_layout(str(ds))

    assert [p for p in residue if p.exists()] == []
    assert (outcome.adopted, outcome.removed, outcome.kept) == (1, 2, 0)


def test_the_sweep_never_overwrites_a_live_rendition(tmp_path):
    """A cache must not go backwards in time.

    ``os.replace`` will happily move a year-old flat file over the ``<edge>/``
    rendition the app wrote this morning. A read cannot reach that case (it
    returns on the destination's existence first); the sweep visits sources
    nobody read, so the check has to live next to the write.
    """
    ds = _flat_dataset(tmp_path / "ds")
    live = _webp(ds / ".thumbnails" / "256" / "foo.webp", BLUE, 256)
    flat = ds / ".thumbnails" / "foo.webp"
    _age(flat, 5)

    outcome = thumbnails.purge_legacy_layout(str(ds))

    assert _near(_pixel(live), BLUE), "the sweep overwrote a live rendition"
    assert not flat.exists(), "the superseded flat file was not reclaimed"
    assert (outcome.adopted, outcome.removed, outcome.kept) == (0, 1, 0)


# ── Accounting, and the cheapness the survey depends on ──────────────────


def test_every_surveyed_file_is_accounted_for(tmp_path):
    """SUPERSEDES ``test_survey_and_purge_share_one_predicate``'s arithmetic.

    That test asserted ``surveyed == removed``, which was true and is now the
    wrong contract: it can only hold if the sweep deletes everything it sees.
    What must hold is that nothing the user was SHOWN goes unexplained — every
    surveyed file is adopted, removed, or knowingly kept.
    """
    ds = _flat_dataset(tmp_path / "ds")
    _webp(ds / ".thumbnails" / "foo@1024.webp", RED, 1024)      # adopted
    _webp(ds / ".thumbnails" / "deleted.webp", RED, 256)        # orphaned
    _webp(ds / ".thumbnails" / "control__img.webp", RED, 256)   # kept
    _webp(ds / ".thumbnails" / "half.webp.tmp", RED, 256)       # residue
    surveyed, _ = thumbnails.legacy_layout_survey(str(ds))

    outcome = thumbnails.purge_legacy_layout(str(ds))

    assert surveyed == 5
    assert outcome.adopted + outcome.removed + outcome.kept == surveyed
    assert (outcome.adopted, outcome.removed, outcome.kept) == (2, 2, 1)
    # Idempotent: a second pass has nothing left to adopt or delete.
    assert thumbnails.purge_legacy_layout(str(ds)) == (0, 0, 1)


def test_a_migrated_dataset_is_never_walked(tmp_path, monkeypatch):
    """The sweep runs over the whole library, so the no-op case must stay free.

    Adoption needs the dataset's source list, and reading it for a dataset that
    has no flat renditions would turn a one-directory check into two for every
    dataset the user owns. Pinned by counting, because "cheap" is otherwise a
    claim nobody can fail.
    """
    ds = tmp_path / "ds"
    _source(ds / "baz.png", BLUE)
    _webp(ds / ".thumbnails" / "256" / "baz.webp", BLUE, 256)

    seen: list[str] = []
    real = os.scandir
    monkeypatch.setattr(
        os, "scandir", lambda path=".": (seen.append(str(path)), real(path))[1],
    )
    outcome = thumbnails.purge_legacy_layout(str(ds))

    assert outcome == (0, 0, 0)
    assert seen == [str(ds / ".thumbnails")], (
        f"a migrated dataset cost more than one directory read: {seen}"
    )


def test_a_dataset_root_that_cannot_be_read_deletes_nothing(tmp_path, monkeypatch):
    """The failure direction is "keep", not "delete".

    Simulated where it really happens — the ``scandir`` of the dataset root
    raises, the ``.thumbnails/`` listing is left real — because the seam under
    test is exactly the one that decides what is an orphan. An implementation
    that read "the root listed no sources" off a FAILED listing would wipe the
    whole cache of any dataset whose root momentarily refused to list, which is
    an ordinary thing for a Windows folder under an antivirus scan to do.
    """
    ds = _flat_dataset(tmp_path / "ds")
    _webp(ds / ".thumbnails" / "bar@512.webp", RED, 512)

    real = os.scandir

    def refusing_scandir(path="."):
        if Path(path) == ds:
            raise PermissionError(13, "Access is denied", str(path))
        return real(path)

    monkeypatch.setattr(os, "scandir", refusing_scandir)
    outcome = thumbnails.purge_legacy_layout(str(ds))

    assert outcome.removed == 0, "an unreadable dataset root cost the user files"
    assert (outcome.adopted, outcome.kept) == (0, 2)
    assert (ds / ".thumbnails" / "foo.webp").exists()


def test_an_emptied_dataset_still_reclaims_its_orphans(tmp_path):
    """The other half of that distinction, and the positive control for it.

    A dataset whose media were all deleted legitimately owns none of its flat
    renditions. If "unreadable" and "empty" were collapsed into one answer, the
    safe one would make the reclaim a permanent no-op for exactly the datasets
    that have the most to reclaim.
    """
    ds = tmp_path / "ds"
    ds.mkdir()
    _webp(ds / ".thumbnails" / "deleted.webp", RED, 256)
    _webp(ds / ".thumbnails" / "also_deleted@1024.webp", RED, 1024)

    assert thumbnails.purge_legacy_layout(str(ds)) == (0, 2, 0)
    assert thumbnails.legacy_layout_survey(str(ds)) == (0, 0)
