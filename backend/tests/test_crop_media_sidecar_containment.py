"""A crop's derived files are contained too, and the contained paths are the
ones used.

Cropping ``a.jpg`` also deletes ``masks/a.png``, ``masked/a.jpg`` and
``masked/a.txt``, rewrites the thumbnail renditions, and re-renders or drops
the overlay (``overlays/a.png``, ``overlays.json``). A valid ``a.jpg`` whose
sidecar directory or file is a link out of the dataset used to delete or write
outside it. ``resolve_edit_targets`` now resolves every one of those paths
before any filesystem call, so a sidecar that leaves the root refuses the
whole crop with 403 before the source is saved, and each helper uses the
returned paths instead of joining its own.

The coverage tests register the dataset through a link and make every sidecar
directory a link to a directory inside it, so a helper that re-derives a path
passes a different string than the one the guard returned.
"""

from __future__ import annotations

import builtins
import contextlib
import dataclasses
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from PIL import Image

import app.core.dataset_manager as dm_mod
from app.core.dataset import media_helpers, overlay_recipe, thumbnails
from tests.support.crop_fixtures import bare_manager, image
from tests.support.dir_links import link_dir, link_name_to

TARGET = (16, 16)
SOURCE_SIZE = (40, 30)
DENIED = "Access denied: path is outside the allowed directory."
RECIPE_OPS = [{"type": "vignette", "enabled": True, "params": {"amount": 0.3}}]


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def manager(tmp_path):
    return bare_manager(tmp_path)


def _recipe(path: Path) -> None:
    path.write_text(json.dumps({"a.jpg": {
        "overlay_file": "overlays/a.png", "created_at": "x", "operations": RECIPE_OPS,
    }}), encoding="utf-8")


def _new_dataset(manager, path: Path | None = None):
    ds = manager.create_dataset("foo", path=None if path is None else str(path))
    image(Path(ds.path) / "a.jpg", SOURCE_SIZE, "red")
    ds.media_metadata["a.jpg"] = {
        "width": SOURCE_SIZE[0], "height": SOURCE_SIZE[1], "size_bytes": 1,
        "enabled": True, "has_mask": True, "has_overlay": True,
    }
    return ds


@pytest.fixture()
def dataset(manager):
    return _new_dataset(manager)


def _tree(root: Path) -> dict[str, bytes]:
    """Every file under *root* and its bytes."""
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


# ── The seven ways a sidecar leaves the root ───────────────────────────────
# Each one builds its link and returns the outside directory whose files must
# stay exactly as they are.


def _masks_out(ds: Path, out: Path) -> Path:
    image(out / "a.png")
    link_dir(ds / "masks", out)
    return out


def _masked_out(ds: Path, out: Path) -> Path:
    image(out / "a.jpg")
    (out / "a.txt").write_text("outside caption", encoding="utf-8")
    link_dir(ds / "masked", out)
    return out


def _thumbnails_out(ds: Path, out: Path) -> Path:
    image(out / "256" / "a.webp")
    image(out / "a.webp")
    link_dir(ds / ".thumbnails", out)
    return out


def _overlays_out(ds: Path, out: Path) -> Path:
    image(out / "a.png")
    _recipe(ds / "overlays.json")
    link_dir(ds / "overlays", out)
    return out


def _overlays_json_out(ds: Path, out: Path) -> Path:
    out.mkdir(parents=True)
    _recipe(out / "overlays.json")
    link_name_to(ds / "overlays.json", out / "overlays.json")
    return out


def _thumbnail_tmp_out(ds: Path, out: Path) -> Path:
    image(out / "victim.webp")
    (ds / ".thumbnails" / "256").mkdir(parents=True)
    link_name_to(ds / ".thumbnails" / "256" / "a.webp.tmp", out / "victim.webp")
    return out


def _overlays_json_tmp_out(ds: Path, out: Path) -> Path:
    out.mkdir(parents=True)
    (out / "victim.json").write_text('{"keep": true}', encoding="utf-8")
    _recipe(ds / "overlays.json")
    link_name_to(ds / "overlays.json.tmp", out / "victim.json")
    return out


SIDECAR_LINKS = {
    "masks": _masks_out,
    "masked": _masked_out,
    "thumbnails": _thumbnails_out,
    "overlays": _overlays_out,
    "overlays_json": _overlays_json_out,
}
# A ``.tmp`` name is never opened (every write goes to a fresh temp file), so
# one linked outside is no refusal: the crop runs and leaves it alone.
TEMP_LINKS = {
    "thumbnail_tmp": _thumbnail_tmp_out,
    "overlays_json_tmp": _overlays_json_tmp_out,
}


def _assert_refused_and_outside_unchanged(manager, dataset, tmp_path, kind: str) -> None:
    out = SIDECAR_LINKS[kind](Path(dataset.path), tmp_path / f"outside_{kind}")
    before = _tree(out)
    assert before, "the outside directory must hold real files"

    with pytest.raises(HTTPException) as excinfo:
        manager.crop_media("foo", "a.jpg", *TARGET)

    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == DENIED
    assert _tree(out) == before


def test_crop_with_masks_dir_linked_outside_is_refused_and_deletes_nothing(
    manager, dataset, tmp_path,
):
    _assert_refused_and_outside_unchanged(manager, dataset, tmp_path, "masks")


def test_crop_with_masked_dir_linked_outside_is_refused_and_deletes_nothing(
    manager, dataset, tmp_path,
):
    _assert_refused_and_outside_unchanged(manager, dataset, tmp_path, "masked")


def test_crop_with_thumbnails_dir_linked_outside_is_refused_and_writes_nothing(
    manager, dataset, tmp_path,
):
    _assert_refused_and_outside_unchanged(manager, dataset, tmp_path, "thumbnails")


def test_crop_with_overlays_dir_linked_outside_is_refused_and_writes_nothing(
    manager, dataset, tmp_path,
):
    _assert_refused_and_outside_unchanged(manager, dataset, tmp_path, "overlays")


def test_crop_with_overlays_json_linked_outside_is_refused_and_leaves_it(
    manager, dataset, tmp_path,
):
    _assert_refused_and_outside_unchanged(manager, dataset, tmp_path, "overlays_json")


@pytest.mark.parametrize("kind", sorted(TEMP_LINKS))
def test_crop_with_a_temp_name_linked_outside_writes_nothing_through_it(
    manager, dataset, tmp_path, kind,
):
    ds = Path(dataset.path)
    image(ds / "overlays" / "a.png", SOURCE_SIZE)
    out = TEMP_LINKS[kind](ds, tmp_path / f"outside_{kind}")
    _recipe(ds / "overlays.json")
    before = _tree(out)
    assert before, "the outside directory must hold real files"

    assert manager.crop_media("foo", "a.jpg", *TARGET) is True

    assert _tree(out) == before
    with Image.open(thumbnails.thumbnail_path_for(str(ds), "a.jpg")) as im:
        assert im.size == TARGET
    data = json.loads((ds / "overlays.json").read_text(encoding="utf-8"))
    assert data["a.jpg"]["created_at"] != "x"


@pytest.mark.parametrize("kind", sorted(SIDECAR_LINKS))
def test_a_sidecar_refusal_leaves_the_source_image_uncropped(
    manager, dataset, tmp_path, kind,
):
    """The refusal comes before the save: the source keeps its bytes."""
    source = Path(dataset.path) / "a.jpg"
    SIDECAR_LINKS[kind](Path(dataset.path), tmp_path / f"outside_{kind}")
    before = source.read_bytes()

    with pytest.raises(HTTPException):
        manager.crop_media("foo", "a.jpg", *TARGET)

    assert source.read_bytes() == before
    with Image.open(source) as im:
        assert im.size == SOURCE_SIZE
    assert dataset.media_metadata["a.jpg"]["width"] == SOURCE_SIZE[0]
    manager._media_repo.update.assert_not_called()


def test_crop_route_answers_403_for_a_linked_sidecar_dir(
    manager, dataset, tmp_path, client, monkeypatch,
):
    import app.api.dataset.crop_routes as crop_routes

    monkeypatch.setattr(crop_routes, "dataset_manager", manager)
    out = _masks_out(Path(dataset.path), tmp_path / "outside_route")
    before = _tree(out)

    resp = client.post("/api/datasets/foo/crop", json={
        "path": "a.jpg", "target_width": TARGET[0], "target_height": TARGET[1],
    })

    assert resp.status_code == 403
    assert resp.json()["detail"] == DENIED
    assert tmp_path.name not in resp.text
    assert _tree(out) == before


# ── Positive controls ──────────────────────────────────────────────────────


def test_crop_still_deletes_contained_masks_and_regenerates_the_thumbnail(
    manager, dataset,
):
    ds = Path(dataset.path)
    sidecars = [ds / "masks" / "a.png", ds / "masked" / "a.jpg"]
    for p in sidecars:
        image(p)
    (ds / "masked" / "a.txt").write_text("c", encoding="utf-8")
    sidecars.append(ds / "masked" / "a.txt")
    image(ds / ".thumbnails" / "512" / "a.webp")
    image(ds / ".thumbnails" / "a.webp")  # legacy flat rendition

    assert manager.crop_media("foo", "a.jpg", *TARGET) is True

    for p in sidecars:
        assert not p.exists(), p
    assert not (ds / ".thumbnails" / "512" / "a.webp").exists()
    assert not (ds / ".thumbnails" / "a.webp").exists()
    with Image.open(thumbnails.thumbnail_path_for(str(ds), "a.jpg")) as im:
        assert im.size == TARGET
    assert not list((ds / ".thumbnails").rglob("*.tmp"))
    assert dataset.media_metadata["a.jpg"]["has_mask"] is False


def test_crop_removes_a_linked_sidecar_name_not_what_it_points_to(manager, dataset):
    """A sidecar NAME that is a link inside the dataset is removed as a name:
    the file (or directory) it points at belongs to another image and stays."""
    ds = Path(dataset.path)
    image(ds / "masks" / "other.png")
    mask_link = ds / "masks" / "a.png"
    mask_target = link_name_to(mask_link, ds / "masks" / "other.png")
    image(ds / ".thumbnails" / "512" / "other.webp")
    thumb_link = ds / ".thumbnails" / "512" / "a.webp"
    thumb_target = link_name_to(thumb_link, ds / ".thumbnails" / "512" / "other.webp")

    assert manager.crop_media("foo", "a.jpg", *TARGET) is True

    for link, target in ((mask_link, mask_target), (thumb_link, thumb_target)):
        assert not os.path.lexists(link), link
        assert target.exists(), target
    assert (ds / "masks" / "other.png").exists()
    assert (ds / ".thumbnails" / "512" / "other.webp").exists()


# ── No write goes through an existing name ─────────────────────────────────
# Each output of a crop is a NAME that may already exist as a file link (a
# symlink, or a hard link, which no path check can see) to another file of
# the dataset. Every output is written to a freshly created temp file and
# moved onto its name, so the linked-to file keeps its bytes and the output
# ends up a regular file of its own.


def _make_link(how: str, name: Path, victim: Path) -> None:
    name.parent.mkdir(parents=True, exist_ok=True)
    if how == "hardlink":
        os.link(victim, name)
        return
    try:
        os.symlink(victim, name)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"this OS refuses a file symlink here: {e}")


def _victim_text(ds: Path) -> Path:
    victim = ds / "notes.txt"
    victim.write_text("unrelated dataset file", encoding="utf-8")
    return victim


def _victim_image(ds: Path) -> Path:
    victim = ds / "b.jpg"
    image(victim, SOURCE_SIZE, "green")
    return victim


def _victim_recipe(ds: Path) -> Path:
    victim = ds / "recipe_copy.json"
    _recipe(victim)
    return victim


# name of the output -> (its path, the victim it is linked to)
WRITE_TARGETS = {
    "source": (lambda ds: ds / "a.jpg", _victim_image),
    "thumbnail_tmp": (lambda ds: ds / ".thumbnails" / "256" / "a.webp.tmp", _victim_text),
    "thumbnail": (lambda ds: ds / ".thumbnails" / "256" / "a.webp", _victim_image),
    "overlay_png": (lambda ds: ds / "overlays" / "a.png", _victim_image),
    "overlays_json": (lambda ds: ds / "overlays.json", _victim_recipe),
    "overlays_json_tmp": (lambda ds: ds / "overlays.json.tmp", _victim_text),
}


def _is_own_regular_file(p: Path) -> bool:
    return p.is_file() and not p.is_symlink() and os.stat(p).st_nlink == 1


@pytest.mark.parametrize("how", ["symlink", "hardlink"])
@pytest.mark.parametrize("target", sorted(WRITE_TARGETS))
def test_a_crop_never_writes_through_an_existing_name(manager, dataset, how, target):
    ds = Path(dataset.path)
    if target != "overlay_png":
        image(ds / "overlays" / "a.png", SOURCE_SIZE)
    if target != "overlays_json":
        _recipe(ds / "overlays.json")
    name_of, make_victim = WRITE_TARGETS[target]
    name = name_of(ds)
    if target == "source":
        name.unlink()  # the fixture's a.jpg; the link takes its name
    victim = make_victim(ds)
    _make_link(how, name, victim)
    before = victim.read_bytes()

    assert manager.crop_media("foo", "a.jpg", *TARGET) is True

    assert victim.read_bytes() == before, f"{target}: wrote through the {how}"
    for output in (ds / "a.jpg", ds / ".thumbnails" / "256" / "a.webp",
                   ds / "overlays" / "a.png", ds / "overlays.json"):
        assert _is_own_regular_file(output), output
    for output in (ds / "a.jpg", ds / ".thumbnails" / "256" / "a.webp", ds / "overlays" / "a.png"):
        with Image.open(output) as im:
            assert im.size == TARGET, output
    data = json.loads((ds / "overlays.json").read_text(encoding="utf-8"))
    assert data["a.jpg"]["operations"] == RECIPE_OPS
    assert data["a.jpg"]["created_at"] != "x"
    assert not [p.name for p in ds.rglob("*.tmp") if p != name]


def test_crop_still_rerenders_a_contained_overlay(manager, dataset):
    ds = Path(dataset.path)
    image(ds / "overlays" / "a.png", SOURCE_SIZE)
    _recipe(ds / "overlays.json")

    manager.crop_media("foo", "a.jpg", *TARGET)

    with Image.open(ds / "overlays" / "a.png") as im:
        assert im.size == TARGET
    data = json.loads((ds / "overlays.json").read_text(encoding="utf-8"))
    assert data["a.jpg"]["operations"] == RECIPE_OPS
    assert data["a.jpg"]["created_at"] != "x"
    assert not list(ds.rglob("*.tmp"))
    meta = dataset.media_metadata["a.jpg"]
    assert meta["has_overlay"] is True
    assert meta["overlay_dimensions"] == list(TARGET)


def test_helpers_keep_their_behaviour_without_targets(tmp_path):
    """The other callers (adjust, upscale, delete) pass no targets."""
    ds = tmp_path / "plain"
    image(ds / "a.jpg", SOURCE_SIZE)
    image(ds / "masks" / "a.png")
    image(ds / "masked" / "a.jpg")
    (ds / "masked" / "a.txt").write_text("c", encoding="utf-8")
    image(ds / "overlays" / "a.png", SOURCE_SIZE)
    _recipe(ds / "overlays.json")

    media_helpers.invalidate_mask_files(str(ds), "a", reason="test")
    assert not (ds / "masks" / "a.png").exists()
    assert not (ds / "masked" / "a.jpg").exists()
    assert not (ds / "masked" / "a.txt").exists()

    meta = {"a.jpg": {"width": 1, "height": 1, "size_bytes": 1}}
    media_helpers.refresh_media_metadata_after_change(
        meta, "a.jpg", str(ds / "a.jpg"), dataset_path=str(ds),
    )
    assert meta["a.jpg"]["size_bytes"] == (ds / "a.jpg").stat().st_size
    assert thumbnails.thumbnail_path_for(str(ds), "a.jpg").exists()
    thumbnails.invalidate_thumbnail(str(ds), "a.jpg")
    assert not thumbnails.thumbnail_path_for(str(ds), "a.jpg").exists()

    result = overlay_recipe.rerender_overlay_from_recipe(str(ds), "a.jpg")
    assert result is not None and result[0] == SOURCE_SIZE

    assert media_helpers.invalidate_overlay_files(str(ds), "a.jpg") is True
    assert not (ds / "overlays" / "a.png").exists()
    assert json.loads((ds / "overlays.json").read_text(encoding="utf-8")) == {}

    media_helpers.write_overlays_json(str(ds / "overlays.json"), {"k": 1})
    assert json.loads((ds / "overlays.json").read_text(encoding="utf-8")) == {"k": 1}
    assert not list(ds.rglob("*.tmp"))


# ── Coverage: every IO receives a path from EditTargets ─────────────────────


@pytest.fixture()
def linked(manager, tmp_path):
    """A dataset registered through a link, every sidecar dir a link inside it.

    Holds every sidecar of ``a.jpg``, so each branch of the crop runs.
    """
    real = tmp_path / "real_ds"
    real.mkdir()
    via = tmp_path / "ds_link"
    link_dir(via, real)
    for name, inner in (("masks", "_m"), ("masked", "_md"),
                        (".thumbnails", "_t"), ("overlays", "_o")):
        link_dir(real / name, real / inner)
    ds = _new_dataset(manager, via)
    image(via / "masks" / "a.png")
    image(via / "masked" / "a.jpg")
    (via / "masked" / "a.txt").write_text("c", encoding="utf-8")
    for edge in thumbnails.ALLOWED_MAX_EDGES:
        image(via / ".thumbnails" / str(edge) / "a.webp")
    image(via / ".thumbnails" / "a.webp")
    image(via / "overlays" / "a.png", SOURCE_SIZE)
    _recipe(via / "overlays.json")
    return ds, via, real.resolve()


# Creating a member's directory (and ``Path.mkdir``'s own stat of it) names
# the member's parent, never the member itself.
_DIR_OPS = {"mkdir", "makedirs", "stat"}


@contextlib.contextmanager
def _record_io():
    """Record the path ARGUMENT of every filesystem call, as given.

    Calls made while ``resolve_edit_targets`` runs are not recorded (resolving
    is what it is for); its results are collected instead.
    """
    calls: list[tuple[str, object]] = []
    resolved: list = []
    state = {"quiet": False}

    def spy(op, real, positions=(0,), quiet_inside=False):
        def wrapper(*args, **kwargs):
            if state["quiet"]:
                return real(*args, **kwargs)
            for i in positions:
                if i < len(args) and isinstance(args[i], (str, os.PathLike)):
                    calls.append((op, args[i]))
            if not quiet_inside:
                return real(*args, **kwargs)
            state["quiet"] = True
            try:
                return real(*args, **kwargs)
            finally:
                state["quiet"] = False
        return wrapper

    real_resolve = dm_mod.resolve_edit_targets

    def resolving(*args, **kwargs):
        state["quiet"] = True
        try:
            targets = real_resolve(*args, **kwargs)
        finally:
            state["quiet"] = False
        resolved.append(targets)
        return targets

    patches = [
        (builtins, "open", spy("open", builtins.open)),
        (Image, "open", spy("Image.open", Image.open)),
        (Image.Image, "save", spy("Image.save", Image.Image.save, (1,))),
        (os, "remove", spy("remove", os.remove)),
        (os, "unlink", spy("unlink", os.unlink)),
        (os, "replace", spy("replace", os.replace, (0, 1))),
        (os, "rename", spy("rename", os.rename, (0, 1))),
        (os, "mkdir", spy("mkdir", os.mkdir)),
        # os.makedirs probes the parent of its argument: record only the call.
        (os, "makedirs", spy("makedirs", os.makedirs, quiet_inside=True)),
        (os, "stat", spy("stat", os.stat)),
        (os.path, "exists", spy("exists", os.path.exists)),
        (os.path, "getsize", spy("getsize", os.path.getsize)),
        (dm_mod, "resolve_edit_targets", resolving),
    ]
    with contextlib.ExitStack() as stack:
        for owner, name, value in patches:
            stack.enter_context(patch.object(owner, name, value))
        yield calls, resolved


def _members(targets) -> set[Path]:
    out: set[Path] = set()
    for f in dataclasses.fields(targets):
        value = getattr(targets, f.name)
        out.update(value if isinstance(value, tuple) else (value,))
    return out


def _fresh_temp_of(p: Path, members: set[Path]) -> Path | None:
    """The member *p* is a fresh temp file of (``<name>.<random>.tmp`` beside it)."""
    for m in members:
        if p.parent == m.parent and p.name.startswith(f"{m.name}.") and p.suffix == ".tmp":
            return m
    return None


def _assert_every_io_from_targets(calls, resolved, via: Path, real: Path) -> set[Path]:
    """Every recorded path under the dataset is a member, a member's directory
    (for creating it), or a fresh temp file beside a member. Returns the
    members reached, a fresh temp counting for its member."""
    assert len(resolved) == 1
    members = _members(resolved[0])
    parents = {m.parent for m in members}
    seen: set[Path] = set()
    stray = []
    for op, arg in calls:
        p = Path(arg)
        if not (p.is_relative_to(via) or p.is_relative_to(real)):
            continue
        if p in members or (op in _DIR_OPS and p in parents):
            seen.add(p)
            continue
        if (owner := _fresh_temp_of(p, members)) is not None:
            seen.add(owner)
            continue
        stray.append((op, str(arg)))
    assert not stray, f"IO on a path not returned by resolve_edit_targets: {stray}"
    return seen


def _fresh_temps(calls, real: Path) -> set[Path]:
    """The members a fresh temp file was moved onto (``os.replace`` src -> dst)."""
    return {
        Path(dst) for (op, src), (_, dst) in zip(calls, calls[1:])
        if op == "replace" and Path(src).suffix == ".tmp" and Path(dst).is_relative_to(real)
    }


def test_every_io_of_a_crop_receives_a_path_from_edit_targets(manager, linked):
    ds, via, real = linked

    with _record_io() as (calls, resolved):
        assert manager.crop_media("foo", "a.jpg", *TARGET) is True

    seen = _assert_every_io_from_targets(calls, resolved, via, real)
    t = resolved[0]
    # Positive control: every branch really ran through the recorded calls,
    # and every output was written as a fresh file moved onto its name.
    for expected in (t.source, *t.mask_files, t.thumbnail_dst, t.overlays_json, t.overlay_png):
        assert expected in seen, expected
    assert _fresh_temps(calls, real) == {t.source, t.thumbnail_dst, t.overlays_json, t.overlay_png}
    # Every member went through the guard: none is still the joined string.
    unresolved = [m for m in _members(t) if m.is_relative_to(via)]
    assert not unresolved, unresolved
    with Image.open(real / "_o" / "a.png") as im:
        assert im.size == TARGET


def test_every_io_of_the_overlay_fallback_receives_a_path_from_edit_targets(
    manager, linked, monkeypatch,
):
    """The re-render raises, so ``invalidate_overlay_files`` drops the overlay."""
    from app.core.image_processing import pipeline

    ds, via, real = linked

    def boom(*a, **k):
        raise RuntimeError("render failed")

    monkeypatch.setattr(pipeline, "execute_pipeline", boom)

    with _record_io() as (calls, resolved):
        manager.crop_media("foo", "a.jpg", *TARGET)

    seen = _assert_every_io_from_targets(calls, resolved, via, real)
    t = resolved[0]
    assert t.overlay_png in seen and t.overlays_json in _fresh_temps(calls, real)
    assert not (real / "_o" / "a.png").exists()
    assert json.loads((real / "overlays.json").read_text(encoding="utf-8")) == {}


def test_every_temp_file_of_a_crop_comes_from_edit_targets(manager, linked):
    """The old fixed temp names exist as links to elsewhere INSIDE the
    dataset. No write goes near them: every temp file is a fresh one beside
    its validated output, and what the old names point at is left alone."""
    ds, via, real = linked
    image(real / "_tt" / "t.webp")
    (real / "_tj").mkdir()
    (real / "_tj" / "t.json").write_text("{}", encoding="utf-8")
    link_name_to(via / ".thumbnails" / "256" / "a.webp.tmp", real / "_tt" / "t.webp")
    link_name_to(via / "overlays.json.tmp", real / "_tj" / "t.json")

    with _record_io() as (calls, resolved):
        manager.crop_media("foo", "a.jpg", *TARGET)

    _assert_every_io_from_targets(calls, resolved, via, real)
    touched = {Path(a).name for _, a in calls}
    assert "a.webp.tmp" not in touched and "overlays.json.tmp" not in touched
    assert (real / "_tj" / "t.json").read_text(encoding="utf-8") == "{}"
    assert (real / "_tt" / "t.webp").is_file()


def test_the_thumbnail_is_generated_from_the_resolved_source(manager, linked, monkeypatch):
    ds, via, real = linked
    sources: dict[str, list] = {"generate": [], "adopt": []}
    real_generate = thumbnails.generate_thumbnail
    real_adopt = thumbnails._adopt_legacy_rendition

    def generate(src_path, *a, **k):
        sources["generate"].append(src_path)
        return real_generate(src_path, *a, **k)

    def adopt(dataset_path, rel_path, max_edge, src_path, *a, **k):
        sources["adopt"].append(src_path)
        return real_adopt(dataset_path, rel_path, max_edge, src_path, *a, **k)

    monkeypatch.setattr(thumbnails, "generate_thumbnail", generate)
    monkeypatch.setattr(thumbnails, "_adopt_legacy_rendition", adopt)

    with _record_io() as (_, resolved):
        manager.crop_media("foo", "a.jpg", *TARGET)

    source = resolved[0].source
    assert source != via / "a.jpg"
    assert [Path(s) for s in sources["adopt"]] == [source]
    assert [Path(s) for s in sources["generate"]] == [source]
