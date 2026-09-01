"""LANE-40 — migrating datasets off the pre-``<edge>/`` flat thumbnail layout.

`a5003618` moved every rendition from ``.thumbnails/<stem>@<edge>.webp`` to
``.thumbnails/<edge>/<stem>.webp`` and wired :func:`purge_legacy_layout` into
``_prepare_scan``. The lane row (and the UAT-4.2 sign-off) recorded the
consequence as *"every already-scanned dataset keeps serving the old layout
until someone rescans it"*.

**That premise is wrong, and the first test in this file is what proves it.**
`ensure_thumbnail` derives its path only from `thumbnail_path_for`, which knows
only the new layout, so a flat file was *unreachable*: the thumbnail
regenerated from source and the served pixels were correct with no rescan at
all.

**LANE-53 amends what that costs.** LANE-40 read "unreachable" as "harmless
bytes" and priced the remedy as `unlink`. Regenerating is not free: MEASURED
at **49.80 s to rebuild one 25-image dataset's tiles**, on a library where 87
of 95 cached datasets had no ``256/`` directory at all. So a first view now
ADOPTS the flat rendition — a rename into ``<edge>/`` — and the sweep is what
clears whatever no read ever claimed. See ``test_thumbnail_adoption.py``; the
survey/purge contract pinned below is unchanged, including the negative (a
dataset already in the new layout is neither offered nor touched).
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from app.core.dataset import thumbnails
from app.core.dataset_manager import Dataset

BLUE = (0, 0, 255)
RED = (255, 0, 0)


# ── Helpers ──────────────────────────────────────────────────────────────


def _source(path: Path, colour: tuple[int, int, int] = BLUE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), colour).save(path)


def _webp(path: Path, colour: tuple[int, int, int] = RED) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), colour).save(path, "WEBP", quality=90)


def _legacy_dataset(root: Path, stems: tuple[str, ...] = ("foo", "bar")) -> Path:
    """A dataset as an upgraded install finds it: sources plus flat renditions.

    The flat names are the pre-`<edge>/` scheme verbatim — ``<stem>.webp`` for
    the default size and ``<stem>@<edge>.webp`` for the others.
    """
    root.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        _source(root / f"{stem}.png", BLUE)
        _webp(root / ".thumbnails" / f"{stem}.webp", RED)
        _webp(root / ".thumbnails" / f"{stem}@1024.webp", RED)
    return root


def _migrated_dataset(root: Path, stems: tuple[str, ...] = ("baz",)) -> Path:
    """A dataset already on the new layout — the control group."""
    root.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        _source(root / f"{stem}.png", BLUE)
        _webp(root / ".thumbnails" / "256" / f"{stem}.webp", BLUE)
        _webp(root / ".thumbnails" / "1024" / f"{stem}.webp", BLUE)
    return root


def _flat_webp_bytes(root: Path) -> int:
    d = root / ".thumbnails"
    return sum(f.stat().st_size for f in d.iterdir() if f.is_file()) if d.is_dir() else 0


def _pixel(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as im:
        return im.convert("RGB").getpixel((5, 5))


def _near(actual: tuple[int, int, int], expected: tuple[int, int, int]) -> bool:
    """WebP is lossy; a solid fill still round-trips within a few units."""
    return all(abs(a - e) <= 12 for a, e in zip(actual, expected, strict=True))


def _make_dataset(path: Path, name: str) -> Dataset:
    return Dataset(id=name, name=name, path=str(path), created_at=0.0)


# ── The defect, stated as observable output ──────────────────────────────


def test_a_read_now_reclaims_the_flat_rendition_by_adopting_it(tmp_path):
    """SUPERSEDED BY LANE-53 — was ``..._unreachable_but_never_reclaimed``.

    The original asserted the opposite of the last line below: that a read
    "did not, and cannot" reclaim the flat bytes, and that a blue source must
    come back blue even with a red flat rendition beside it. Both were true of
    the code, and both encoded the wrong contract — the price of "cannot
    reclaim" is that the source is decoded instead, MEASURED at 49.80 s for
    one 25-image dataset (see ``test_thumbnail_adoption.py``).

    What survives from the original, and must: the served PATH is the
    ``<edge>/`` one. Only the bytes' provenance changed, so the ``foo@512``
    collision `a5003618` fixed stays fixed — pinned directly by
    ``test_a_source_whose_own_name_contains_a_size_suffix_is_not_confused``.
    """
    ds = _legacy_dataset(tmp_path / "ds", stems=("foo",))
    assert _flat_webp_bytes(ds) > 0, "fixture must contain flat renditions"
    adopted = (ds / ".thumbnails" / "foo.webp").read_bytes()

    served = thumbnails.ensure_thumbnail(str(ds), "foo.png")

    assert served == ds / ".thumbnails" / "256" / "foo.webp"
    assert served.read_bytes() == adopted, (
        "the flat rendition's own bytes must be reused, not re-derived"
    )
    # Reclaimed by the read itself — the orphan is gone, not duplicated.
    assert not (ds / ".thumbnails" / "foo.webp").exists()


# ── Detection ────────────────────────────────────────────────────────────


def test_survey_reports_flat_renditions_with_their_byte_cost(tmp_path):
    ds = _legacy_dataset(tmp_path / "ds", stems=("foo", "bar"))
    expected_bytes = _flat_webp_bytes(ds)

    files, size = thumbnails.legacy_layout_survey(str(ds))

    assert files == 4  # 2 stems x (default + @1024)
    assert size == expected_bytes


def test_survey_ignores_a_dataset_already_on_the_new_layout(tmp_path):
    """Prove the negative: the control group is not offered for migration."""
    ds = _migrated_dataset(tmp_path / "ds")

    assert thumbnails.legacy_layout_survey(str(ds)) == (0, 0)


def test_survey_is_zero_without_a_thumbnail_dir(tmp_path):
    (tmp_path / "ds").mkdir()
    assert thumbnails.legacy_layout_survey(str(tmp_path / "ds")) == (0, 0)


def test_survey_does_not_walk_the_dataset_tree(tmp_path, monkeypatch):
    """Detection must be cheap: ONE scandir of ``.thumbnails/``, never a walk.

    Pinned by counting the scandir calls, because "cheap" is otherwise a claim
    nobody can fail. A dataset with subdirectories full of media must not add
    a single directory read.
    """
    ds = _legacy_dataset(tmp_path / "ds", stems=("foo",))
    for sub in ("control", "masks", "extra"):
        _source(ds / sub / "img.png")
    (ds / ".thumbnails" / "256").mkdir(parents=True, exist_ok=True)
    _webp(ds / ".thumbnails" / "256" / "foo.webp", BLUE)

    import os as _os

    seen: list[str] = []
    real = _os.scandir

    def counting_scandir(path="."):
        seen.append(str(path))
        return real(path)

    monkeypatch.setattr(_os, "scandir", counting_scandir)
    thumbnails.legacy_layout_survey(str(ds))

    assert len(seen) == 1, f"expected one scandir of .thumbnails/, saw {seen}"
    assert Path(seen[0]) == ds / ".thumbnails"


def test_survey_and_purge_share_one_predicate(tmp_path):
    """The count the user is shown and the count the sweep acts on are the same
    number, produced by the same enumerator — a survey that drifts from the
    sweep would promise a migration it does not perform.

    **AMENDED BY LANE-53.** This asserted ``surveyed == removed``, which was
    true and was the wrong contract: it can only hold if the sweep deletes
    everything it sees, which is the defect. Every surveyed file is still
    accounted for, but now by one of three outcomes — here the four real
    renditions are ADOPTED (the 64 px sources make ``@1024`` a downscale to
    64 px, so only the default size proves out) and only the temp residue is
    deleted. The arithmetic is pinned exhaustively in
    ``test_thumbnail_purge_adoption.py``.
    """
    ds = _legacy_dataset(tmp_path / "ds", stems=("foo", "bar"))
    _webp(ds / ".thumbnails" / "stale.webp.tmp", RED)

    surveyed, _ = thumbnails.legacy_layout_survey(str(ds))
    outcome = thumbnails.purge_legacy_layout(str(ds))

    assert surveyed == 5
    assert outcome.adopted + outcome.removed + outcome.kept == surveyed
    assert outcome.adopted == 2, "the sweep destroyed renditions it could adopt"
    assert thumbnails.legacy_layout_survey(str(ds)) == (0, 0)


def test_purge_spares_live_renditions_of_a_migrated_dataset(tmp_path):
    ds = _migrated_dataset(tmp_path / "ds")
    live = ds / ".thumbnails" / "256" / "baz.webp"
    before = live.read_bytes()

    assert thumbnails.purge_legacy_layout(str(ds)) == (0, 0, 0)
    assert live.read_bytes() == before


# ── The API surface ──────────────────────────────────────────────────────


def _install_datasets(monkeypatch, datasets: dict[str, Dataset]) -> None:
    """Point the routes' dataset registry at a fixture set."""
    from app.api.dataset import thumbnail_routes

    monkeypatch.setattr(
        thumbnail_routes.dataset_manager, "datasets", datasets, raising=False,
    )


def test_survey_route_reports_only_the_unmigrated_datasets(
    client, tmp_path, monkeypatch,
):
    legacy_a = _legacy_dataset(tmp_path / "a", stems=("foo",))
    legacy_b = _legacy_dataset(tmp_path / "b", stems=("foo", "bar"))
    migrated = _migrated_dataset(tmp_path / "c")
    _install_datasets(monkeypatch, {
        "a": _make_dataset(legacy_a, "a"),
        "b": _make_dataset(legacy_b, "b"),
        "c": _make_dataset(migrated, "c"),
    })

    body = client.get("/api/datasets/thumbnails/legacy").json()

    assert [d["name"] for d in body["datasets"]] == ["a", "b"], (
        "the already-migrated dataset must not be offered"
    )
    assert body["dataset_count"] == 2
    assert body["total_files"] == 2 + 4
    assert body["total_bytes"] == _flat_webp_bytes(legacy_a) + _flat_webp_bytes(legacy_b)


def test_survey_route_skips_a_dataset_whose_path_is_gone(
    client, tmp_path, monkeypatch,
):
    _install_datasets(monkeypatch, {
        "ghost": _make_dataset(tmp_path / "nope", "ghost"),
    })

    body = client.get("/api/datasets/thumbnails/legacy").json()

    assert body["datasets"] == []
    assert body["total_files"] == 0


def test_survey_route_is_not_shadowed_by_the_dataset_name_parameter(
    client, tmp_path, monkeypatch,
):
    """Endpoint-identity pin for the literal-vs-parameter path collision.

    ``crud_router`` owns ``/datasets/{name}/...``. Nothing there matches these
    two literals *today* — measured: reversing the two `include_router` lines
    in ``app/api/dataset/__init__.py`` leaves this whole file green — so this
    pins the RESOLUTION rather than the mount order, and does it through a
    response field no other handler in the app can produce. Both directions
    are asserted, because the collision has two: the literal route must
    survive, and a dataset literally NAMED ``thumbnails`` must still serve its
    own pixels through ``/datasets/{name}/thumbnail``.
    """
    ds = _legacy_dataset(tmp_path / "thumbnails", stems=("foo",))
    # A second, ALREADY-migrated item carries the routing assertion. The
    # original read `foo.png` and asserted blue, which silently depended on
    # the flat rendition being unreachable; LANE-53 adopts it instead, and a
    # routing test must not turn red over a cache-layout change it is not
    # about. `foo`'s flat files still make the survey half meaningful.
    _source(ds / "live.png", BLUE)
    _webp(ds / ".thumbnails" / "256" / "live.webp", BLUE)
    _install_datasets(monkeypatch, {"thumbnails": _make_dataset(ds, "thumbnails")})

    survey = client.get("/api/datasets/thumbnails/legacy")
    assert survey.status_code == 200
    assert survey.json()["dataset_count"] == 1
    # The body IS the endpoint identity: `dataset_count` exists on exactly one
    # response model in the app (ThumbnailLegacySurveyResponse), so a shadowing
    # `/datasets/{name}/legacy` could not produce it. Asserting only `200`
    # here would be a tick — a shadowing handler answers 200 too.
    assert "dataset_count" in survey.json()

    # ... and the same-named dataset's own thumbnail route still resolves.
    own = client.get(
        "/api/datasets/thumbnails/thumbnail",
        params={"image_rel_path": "live.png", "max_edge": 256},
    )
    assert own.status_code == 200
    assert _near(_pixel_of_response(own.content), BLUE)


def _pixel_of_response(payload: bytes) -> tuple[int, int, int]:
    with Image.open(io.BytesIO(payload)) as im:
        return im.convert("RGB").getpixel((5, 5))


def test_migrate_route_moves_what_it_can_and_reclaims_the_rest(
    client, tmp_path, monkeypatch,
):
    """End to end on observable output: the flat layout is gone from the legacy
    dataset, the already-migrated one is byte-identical, and the tiles the
    sweep could prove are now SERVING rather than deleted.

    **RENAMED AND REWRITTEN BY LANE-53** — was
    ``..._reclaims_the_bytes_and_leaves_migrated_data_alone``, whose last line
    asserted ``_near(..., BLUE)``: that the tile served after the sweep is the
    SOURCE's colour, i.e. that the sweep had thrown the cached rendition away
    and the next view re-derived it. It passed for two months and it is the
    defect stated as the contract — the button the user was offered to fix his
    slow first view was the thing guaranteeing it. The tracer is RED here for
    exactly one reason: only adopted bytes can be red.
    """
    from app.core.tasks.task_manager import task_manager

    legacy = _legacy_dataset(tmp_path / "a", stems=("foo", "bar"))
    migrated = _migrated_dataset(tmp_path / "c")
    live = migrated / ".thumbnails" / "256" / "baz.webp"
    live_before = live.read_bytes()
    _install_datasets(monkeypatch, {
        "a": _make_dataset(legacy, "a"),
        "c": _make_dataset(migrated, "c"),
    })

    started = client.post("/api/datasets/thumbnails/migrate")
    assert started.status_code == 200
    body = started.json()
    assert body["dataset_count"] == 1
    assert body["files"] == 4

    task_manager.join_lane("background", timeout=30)
    task = task_manager.get(body["task_id"])
    assert task is not None
    assert task.status.value == "completed"
    assert task.ok == 4

    assert _flat_webp_bytes(legacy) == 0
    assert live.read_bytes() == live_before, "a migrated dataset must be untouched"

    for stem in ("foo", "bar"):
        served = thumbnails.ensure_thumbnail(str(legacy), f"{stem}.png")
        assert served == legacy / ".thumbnails" / "256" / f"{stem}.webp"
        assert _near(_pixel(served), RED), (
            f"{stem}'s tile is the source's colour — the sweep destroyed the "
            "rendition and the view re-derived it, which is LANE-53's defect"
        )


def test_migrate_route_is_single_flight(client, tmp_path, monkeypatch):
    """Rate limit: the shared background lane cannot be flooded by a user who
    clicks twice. The second POST is refused, not queued."""
    import threading

    from app.core.tasks.task_manager import task_manager

    legacy = _legacy_dataset(tmp_path / "a", stems=("foo",))
    _install_datasets(monkeypatch, {"a": _make_dataset(legacy, "a")})

    gate = threading.Event()
    real_purge = thumbnails.purge_legacy_layout

    def blocking_purge(path):
        gate.wait(10)
        return real_purge(path)

    monkeypatch.setattr(thumbnails, "purge_legacy_layout", blocking_purge)

    first = client.post("/api/datasets/thumbnails/migrate")
    assert first.status_code == 200
    try:
        second = client.post("/api/datasets/thumbnails/migrate")
        assert second.status_code == 409
    finally:
        gate.set()
        task_manager.join_lane("background", timeout=30)


def test_migrate_route_refuses_when_nothing_needs_migrating(
    client, tmp_path, monkeypatch,
):
    migrated = _migrated_dataset(tmp_path / "c")
    _install_datasets(monkeypatch, {"c": _make_dataset(migrated, "c")})

    resp = client.post("/api/datasets/thumbnails/migrate")

    assert resp.status_code == 409
    assert "nothing" in resp.json()["detail"].lower()


def test_migration_worker_stops_on_cancel_and_leaves_the_rest_intact(tmp_path):
    """Cancellable: a cancel between datasets stops the sweep, and the
    datasets it had not reached still hold every one of their bytes."""
    from app.api.dataset.thumbnail_routes import run_thumbnail_migration
    from app.core.tasks.task_manager import task_manager

    first = _legacy_dataset(tmp_path / "a", stems=("foo",))
    second = _legacy_dataset(tmp_path / "b", stems=("foo",))
    first_bytes = _flat_webp_bytes(first)
    second_bytes = _flat_webp_bytes(second)
    assert first_bytes > 0 and second_bytes > 0

    task = task_manager.create(type="thumbnail_migration", title="t", total=2)
    task_manager.start(task.id)
    task_manager.cancel(task.id)

    run_thumbnail_migration(
        task.id, [_make_dataset(first, "a"), _make_dataset(second, "b")],
    )

    # Bytes first, deliberately: the status is what a cancel *says*, the bytes
    # are what it *did*, and only the second one is the guarantee.
    assert _flat_webp_bytes(first) == first_bytes, (
        "a cancel must stop the sweep, not merely stop reporting it"
    )
    assert _flat_webp_bytes(second) == second_bytes
    assert task_manager.get(task.id).status.value == "cancelled"
