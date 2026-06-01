"""Tests for cache_routes._aggregate_cache_stats.

The DATASETS KPI corner needs the full on-disk folder size (images + captions
+ masks + .cache/ + everything), not just the .cache/ subset. These tests pin
the contract: `dataset_root_bytes` aggregates EVERY byte under each dataset
root, while `total_bytes` still aggregates only `.cache/<model>/<version>/<type>/`
subtrees.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from app.api.cache_routes import _aggregate_cache_stats, _purge_cache
from app.core.dataset_manager import Dataset


def _make_dataset(path: Path, name: str = "ds") -> Dataset:
    """Build a minimal Dataset pointing at *path* — the only fields the
    aggregator touches are `path`. Other required fields get filler values."""
    return Dataset(id=name, name=name, path=str(path), created_at=0.0)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_dataset_root_bytes_sums_root_files_and_cache(tmp_path):
    """`dataset_root_bytes` covers root files AND .cache/, while
    `total_bytes` covers only the .cache/<model>/<version>/<type>/ subset."""
    ds_dir = tmp_path / "ds_a"

    # Non-cache root files — images, captions, masks. These count toward
    # dataset_root_bytes but NOT total_bytes.
    _write(ds_dir / "img1.png", b"x" * 1000)
    _write(ds_dir / "img1.txt", b"a caption" + b"\n")            # 10 bytes
    _write(ds_dir / "masks" / "img1.png", b"y" * 500)

    # Cache subtree — counts toward BOTH dataset_root_bytes and total_bytes.
    _write(ds_dir / ".cache" / "sdxl" / "v1" / "latents" / "orig" / "1024x1024" / "img1.npy", b"L" * 2000)
    _write(ds_dir / ".cache" / "sdxl" / "v1" / "te1" / "orig" / "img1.npy", b"E" * 700)

    ds = _make_dataset(ds_dir)

    with patch("app.api.cache_routes.dataset_manager") as mgr:
        mgr.list_datasets.return_value = [ds]
        stats = _aggregate_cache_stats()

    # Cache-only totals (current/unchanged contract).
    assert stats["total_bytes"] == 2000 + 700
    assert stats["latent_bytes"] == 2000
    assert stats["embedding_bytes"] == 700
    assert stats["cached_datasets"] == 1

    # Full-folder total — must include the non-cache root files.
    # Root files: 1000 (img1.png) + 10 (img1.txt) + 500 (masks/img1.png)
    # Cache files: 2000 + 700
    expected_root = 1000 + 10 + 500 + 2000 + 700
    assert stats["dataset_root_bytes"] == expected_root

    # And the two MUST differ when there's anything outside .cache/.
    assert stats["dataset_root_bytes"] != stats["total_bytes"]


def test_dataset_root_bytes_without_cache(tmp_path):
    """A dataset with no .cache/ still contributes to dataset_root_bytes."""
    ds_dir = tmp_path / "ds_b"
    _write(ds_dir / "img1.png", b"z" * 1234)

    ds = _make_dataset(ds_dir, name="ds_b")

    with patch("app.api.cache_routes.dataset_manager") as mgr:
        mgr.list_datasets.return_value = [ds]
        stats = _aggregate_cache_stats()

    assert stats["cached_datasets"] == 0
    assert stats["total_bytes"] == 0
    assert stats["dataset_root_bytes"] == 1234


def test_dataset_root_bytes_aggregates_across_datasets(tmp_path):
    """dataset_root_bytes sums every dataset's full folder."""
    ds1 = tmp_path / "ds1"
    ds2 = tmp_path / "ds2"
    _write(ds1 / "a.bin", b"1" * 100)
    _write(ds1 / ".cache" / "m" / "v" / "latents" / "orig" / "1024x1024" / "a.npy", b"L" * 50)
    _write(ds2 / "b.bin", b"2" * 200)

    with patch("app.api.cache_routes.dataset_manager") as mgr:
        mgr.list_datasets.return_value = [_make_dataset(ds1, "ds1"), _make_dataset(ds2, "ds2")]
        stats = _aggregate_cache_stats()

    assert stats["dataset_root_bytes"] == 100 + 50 + 200
    assert stats["total_bytes"] == 50
    assert stats["latent_bytes"] == 50
    assert stats["cached_datasets"] == 1


def test_dataset_root_bytes_skips_missing_path(tmp_path):
    """A dataset whose `path` doesn't exist on disk contributes 0."""
    missing = tmp_path / "gone"
    # Don't create `missing/`.

    with patch("app.api.cache_routes.dataset_manager") as mgr:
        mgr.list_datasets.return_value = [_make_dataset(missing, "gone")]
        stats = _aggregate_cache_stats()

    assert stats["dataset_root_bytes"] == 0
    assert stats["total_bytes"] == 0
    assert stats["cached_datasets"] == 0


# ── _purge_cache: per model / version / type filtering ──────────────────────

def _seed_cache(root: Path) -> None:
    """Two models, sdxl with two versions, each version with latents + te1."""
    _write(root / "sdxl" / "1.0.0" / "latents" / "orig" / "1024x1024" / "a.npy", b"L" * 100)
    _write(root / "sdxl" / "1.0.0" / "te1" / "orig" / "a.npy", b"E" * 50)
    _write(root / "sdxl" / "2.0.0" / "latents" / "orig" / "1024x1024" / "a.npy", b"L" * 100)
    _write(root / "sdxl" / "2.0.0" / "te1" / "orig" / "a.npy", b"E" * 50)
    _write(root / "flux" / "1.0.0" / "latents" / "orig" / "1024x1024" / "a.npy", b"L" * 100)


def test_purge_by_model_only(tmp_path):
    """models=[sdxl] removes both sdxl versions, leaves flux untouched."""
    root = tmp_path / ".cache"
    _seed_cache(root)
    res = _purge_cache(root, models=["sdxl"], types=None, variants=None)
    assert res["deleted"] > 0
    assert not (root / "sdxl").exists()
    assert (root / "flux").exists()


def test_purge_by_model_and_version(tmp_path):
    """versions=[1.0.0] for sdxl removes only that version; 2.0.0 survives."""
    root = tmp_path / ".cache"
    _seed_cache(root)
    _purge_cache(root, models=["sdxl"], types=None, variants=None, versions=["1.0.0"])
    assert not (root / "sdxl" / "1.0.0").exists()
    assert (root / "sdxl" / "2.0.0").exists()
    assert (root / "flux" / "1.0.0").exists()


def test_purge_by_model_version_and_type(tmp_path):
    """models+versions+types removes just that leaf — sibling type + other
    version + other model all survive."""
    root = tmp_path / ".cache"
    _seed_cache(root)
    _purge_cache(root, models=["sdxl"], types=["te1"], variants=None, versions=["1.0.0"])
    assert not (root / "sdxl" / "1.0.0" / "te1").exists()
    assert (root / "sdxl" / "1.0.0" / "latents").exists()  # sibling type kept
    assert (root / "sdxl" / "2.0.0" / "te1").exists()       # other version kept
    assert (root / "flux").exists()                          # other model kept


def test_purge_version_filter_applies_across_models(tmp_path):
    """versions filter with no models targets that version in every model."""
    root = tmp_path / ".cache"
    _seed_cache(root)
    _purge_cache(root, models=None, types=None, variants=None, versions=["1.0.0"])
    assert not (root / "sdxl" / "1.0.0").exists()
    assert (root / "sdxl" / "2.0.0").exists()
    assert not (root / "flux" / "1.0.0").exists()
