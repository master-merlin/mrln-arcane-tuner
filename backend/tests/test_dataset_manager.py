"""
Tests for DatasetManager — covers CRUD, scanning, pairs, versioning,
caption coverage, harmonization, and the Dataset Pydantic model.
"""

import os

import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from app.core.dataset_manager import Dataset, DatasetManager


# ── Helpers ──────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_settings():
    """Patch SettingsManager so DatasetManager never touches real disk."""
    mock_instance = MagicMock()
    mock_instance.get_module_settings.return_value = {}
    mock_instance.update_module_settings = MagicMock()
    with patch("app.core.dataset_manager.get_settings_manager", return_value=mock_instance):
        yield mock_instance


@pytest.fixture()
def manager(tmp_path, mock_settings):
    """Create a DatasetManager rooted in tmp_path."""
    storage_file = str(tmp_path / "dataset_locations.json")
    default_root = str(tmp_path / "datasets")
    os.makedirs(default_root, exist_ok=True)

    with patch.object(DatasetManager, "__init__", lambda self, **kw: None):
        mgr = DatasetManager()

    mgr.root_dir = str(tmp_path)
    mgr.storage_file = storage_file
    mgr.default_root = default_root
    mgr.settings_manager = mock_settings
    mgr.datasets = {}
    mgr._loop = None
    # Mock DB layer so save/persist don't touch real SQLite
    mgr._db = MagicMock()
    mgr._dataset_repo = MagicMock()
    mgr._media_repo = MagicMock()
    return mgr


def _create_image(path: str, width: int = 100, height: int = 100):
    """Write a tiny PIL image to *path*."""
    img = Image.new("RGB", (width, height), "red")
    img.save(path)


def _create_caption(path: str, text: str = "a test caption"):
    """Write a text file at *path*."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ── Dataset Pydantic Model ───────────────────────────────────────────────


class TestDatasetModel:
    """Tests for the Dataset Pydantic model."""

    def test_required_fields(self):
        """id, name, path, created_at are required."""
        ds = Dataset(id="1", name="test", path="/tmp/test", created_at=1000.0)
        assert ds.id == "1"
        assert ds.name == "test"

    def test_default_values(self):
        """Optional fields should default correctly."""
        ds = Dataset(id="1", name="test", path="/tmp/test", created_at=1000.0)
        assert ds.file_count == 0
        assert ds.caption_coverage is False
        assert ds.missing is False
        assert ds.version == "1.0.0"
        assert ds.media_metadata == {}

    def test_serialization_round_trip(self):
        """model_dump / model_validate should produce identical objects."""
        ds = Dataset(id="1", name="test", path="/p", created_at=1.0, version="2.1.0")
        data = ds.model_dump()
        ds2 = Dataset.model_validate(data)
        assert ds == ds2


# ── Create / List / Get / Delete ─────────────────────────────────────────


class TestDatasetCRUD:
    """Tests for create, list, get, delete."""

    def test_create_dataset(self, manager):
        """create_dataset should register and create directory."""
        ds = manager.create_dataset("myds", description="test")
        assert ds.name == "myds"
        assert ds.description == "test"
        assert os.path.isdir(ds.path)
        assert "myds" in manager.datasets

    def test_create_duplicate_raises(self, manager):
        """Creating a dataset with the same name should raise ValueError."""
        manager.create_dataset("dup")
        with pytest.raises(ValueError, match="already exists"):
            manager.create_dataset("dup")

    def test_create_with_custom_path(self, manager, tmp_path):
        """Custom path should be used instead of default_root."""
        custom = str(tmp_path / "custom_dir")
        ds = manager.create_dataset("custom", path=custom)
        assert ds.path == custom
        assert os.path.isdir(custom)

    def test_list_datasets(self, manager):
        """list_datasets returns all registered datasets."""
        manager.create_dataset("a")
        manager.create_dataset("b")
        result = manager.list_datasets()
        assert len(result) == 2
        names = {ds.name for ds in result}
        assert names == {"a", "b"}

    def test_get_dataset_found(self, manager):
        """get_dataset returns the correct dataset."""
        manager.create_dataset("found")
        ds = manager.get_dataset("found")
        assert ds is not None
        assert ds.name == "found"

    def test_get_dataset_not_found(self, manager):
        """get_dataset returns None for unknown name."""
        assert manager.get_dataset("ghost") is None

    def test_delete_dataset(self, manager):
        """delete_dataset removes the dataset from registry."""
        manager.create_dataset("doomed")
        manager.delete_dataset("doomed")
        assert "doomed" not in manager.datasets

    def test_delete_dataset_with_files(self, manager):
        """delete_dataset with delete_files=True removes the directory."""
        ds = manager.create_dataset("files")
        _create_image(os.path.join(ds.path, "img.png"))
        manager.delete_dataset("files", delete_files=True)
        assert not os.path.exists(ds.path)

    def test_delete_nonexistent_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.delete_dataset("ghost")


# ── Update Dataset ───────────────────────────────────────────────────────


class TestUpdateDataset:
    """Tests for update_dataset (rename + description change)."""

    def test_update_description_only(self, manager):
        """Changing description without renaming should work."""
        manager.create_dataset("orig")
        ds = manager.update_dataset("orig", "orig", "new desc")
        assert ds.description == "new desc"

    def test_rename_dataset(self, manager):
        """Renaming should update dictionary key and path."""
        manager.create_dataset("old_name")
        ds = manager.update_dataset("old_name", "new_name", "desc")
        assert ds.name == "new_name"
        assert "new_name" in manager.datasets
        assert "old_name" not in manager.datasets

    def test_rename_to_existing_raises(self, manager):
        """Renaming to an existing name should raise."""
        manager.create_dataset("a")
        manager.create_dataset("b")
        with pytest.raises(ValueError, match="already exists"):
            manager.update_dataset("a", "b", "")

    def test_rename_to_empty_raises(self, manager):
        """Renaming to empty string should raise."""
        manager.create_dataset("x")
        with pytest.raises(ValueError, match="cannot be empty"):
            manager.update_dataset("x", "  ", "")

    def test_update_nonexistent_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.update_dataset("nope", "nope", "")


# ── Scan Dataset ─────────────────────────────────────────────────────────


class TestScanDataset:
    """Tests for scan_dataset."""

    @patch("app.core.dataset_manager.solide_hash_robust", return_value="deadbeef" * 4)
    def test_scan_counts_files(self, mock_hash, manager):
        """scan should count multimedia and caption files."""
        ds = manager.create_dataset("scanme")
        _create_image(os.path.join(ds.path, "img1.png"))
        _create_image(os.path.join(ds.path, "img2.jpg"), 200, 100)
        _create_caption(os.path.join(ds.path, "img1.txt"))

        result = manager.scan_dataset("scanme")
        assert result.multimedia_count == 2
        assert result.caption_count == 1
        assert result.file_count == 3
        assert result.last_scanned_at is not None

    @patch("app.core.dataset_manager.solide_hash_robust", return_value="abcd1234" * 4)
    def test_scan_caption_coverage(self, mock_hash, manager):
        """caption_coverage should be True when every image has a caption."""
        ds = manager.create_dataset("covered")
        _create_image(os.path.join(ds.path, "a.png"))
        _create_caption(os.path.join(ds.path, "a.txt"))

        result = manager.scan_dataset("covered")
        assert result.caption_coverage is True

    @patch("app.core.dataset_manager.solide_hash_robust", return_value="abcd1234" * 4)
    def test_scan_incomplete_coverage(self, mock_hash, manager):
        """caption_coverage should be False when some images lack captions."""
        ds = manager.create_dataset("partial")
        _create_image(os.path.join(ds.path, "a.png"))
        _create_image(os.path.join(ds.path, "b.png"))
        _create_caption(os.path.join(ds.path, "a.txt"))

        result = manager.scan_dataset("partial")
        assert result.caption_coverage is False

    @patch("app.core.dataset_manager.solide_hash_robust", return_value="abcd1234" * 4)
    def test_scan_sets_preview_image(self, mock_hash, manager):
        """First multimedia file should become preview_image."""
        ds = manager.create_dataset("prev")
        _create_image(os.path.join(ds.path, "first.png"))

        result = manager.scan_dataset("prev")
        assert result.preview_image is not None

    @patch("app.core.dataset_manager.solide_hash_robust", return_value="abcd1234" * 4)
    def test_scan_extracts_dimensions(self, mock_hash, manager):
        """Media metadata should include width/height/aspect_ratio."""
        ds = manager.create_dataset("dims")
        _create_image(os.path.join(ds.path, "wide.png"), 400, 200)

        result = manager.scan_dataset("dims")
        meta = result.media_metadata.get("wide.png", {})
        assert meta["width"] == 400
        assert meta["height"] == 200
        assert meta["orientation"] == "landscape"

    def test_scan_missing_path_raises(self, manager):
        """Scanning a dataset whose path doesn't exist should raise."""
        ds = manager.create_dataset("vanish")
        os.rmdir(ds.path)  # delete the dir
        with pytest.raises(FileNotFoundError):
            manager.scan_dataset("vanish")

    def test_scan_nonexistent_dataset_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.scan_dataset("ghost")

    @patch("app.core.dataset_manager.solide_hash_robust", return_value="abcd1234" * 4)
    def test_scan_skips_hidden_files(self, mock_hash, manager):
        """Files starting with '.' or '~' should be ignored."""
        ds = manager.create_dataset("hidden")
        _create_image(os.path.join(ds.path, ".hidden.png"))
        _create_image(os.path.join(ds.path, "~temp.png"))
        _create_image(os.path.join(ds.path, "visible.png"))

        result = manager.scan_dataset("hidden")
        assert result.multimedia_count == 1
        assert result.file_count == 1


# ── Versioning ───────────────────────────────────────────────────────────


class TestVersioning:
    """Tests for version bumping."""

    def test_bump_patch(self, manager):
        """Patch bump should increment Z in X.Y.Z."""
        ds = manager.create_dataset("ver")
        ds.version = "1.2.3"
        manager._bump_version(ds, "patch")
        assert ds.version == "1.2.4"

    def test_bump_minor(self, manager):
        """Minor bump should increment Y and reset Z."""
        ds = manager.create_dataset("ver2")
        ds.version = "1.2.3"
        manager._bump_version(ds, "minor")
        assert ds.version == "1.3.0"

    def test_bump_major(self, manager):
        """Major bump should increment X and reset Y, Z."""
        ds = manager.create_dataset("ver3")
        ds.version = "1.2.3"
        manager._bump_version(ds, "major")
        assert ds.version == "2.0.0"

    def test_bump_invalid_version_resets(self, manager):
        """Invalid version string should reset to 1.0.0."""
        ds = manager.create_dataset("bad_ver")
        ds.version = "not_semver"
        manager._bump_version(ds, "patch")
        assert ds.version == "1.0.1"

    def test_public_bump_returns_version(self, manager):
        """bump_dataset_version should return the new version string."""
        manager.create_dataset("pub")
        result = manager.bump_dataset_version("pub", "patch")
        assert result == "1.0.1"

    def test_public_bump_nonexistent_returns_none(self, manager):
        assert manager.bump_dataset_version("ghost", "patch") is None

    @patch("app.core.dataset_manager.solide_hash_robust", return_value="abcd1234" * 4)
    def test_scan_bumps_minor_on_file_count_change(self, mock_hash, manager):
        """Second scan with different file count should bump minor version."""
        ds = manager.create_dataset("evolving")
        _create_image(os.path.join(ds.path, "a.png"))
        manager.scan_dataset("evolving")  # first scan

        _create_image(os.path.join(ds.path, "b.png"))
        result = manager.scan_dataset("evolving")  # second scan
        # Minor bump because multimedia_count changed
        assert result.version == "1.1.0"


# ── Pairs ────────────────────────────────────────────────────────────────


class TestGetPairs:
    """Tests for get_dataset_pairs."""

    def test_paired_media_and_caption(self, manager):
        """Image + caption with same stem should produce a paired entry."""
        ds = manager.create_dataset("pairs")
        _create_image(os.path.join(ds.path, "photo.png"))
        _create_caption(os.path.join(ds.path, "photo.txt"), "a dog")

        pairs = manager.get_dataset_pairs("pairs")
        assert len(pairs) >= 1
        pair = next(p for p in pairs if p["stem"] == "photo")
        assert pair["media_file"] is not None
        assert pair["caption_file"] is not None

    def test_unpaired_media(self, manager):
        """Image without caption should still appear (caption_file=None)."""
        ds = manager.create_dataset("unpaired")
        _create_image(os.path.join(ds.path, "solo.jpg"))

        pairs = manager.get_dataset_pairs("unpaired")
        pair = next(p for p in pairs if p["stem"] == "solo")
        assert pair["media_file"] is not None
        assert pair["caption_file"] is None

    def test_nonexistent_dataset_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.get_dataset_pairs("ghost")

    def test_missing_path_returns_empty(self, manager):
        """If dataset path is gone, return empty list."""
        ds = manager.create_dataset("gone")
        os.rmdir(ds.path)
        pairs = manager.get_dataset_pairs("gone")
        assert pairs == []


# ── Target Dimensions ────────────────────────────────────────────────────


class TestCalculateTargetDims:
    """Tests for calculate_target_dims."""

    def test_landscape_calculation(self, manager):
        """Landscape: long side is width, short derived from AR."""
        w, h = manager.calculate_target_dims(1024, 1.5, "landscape")
        assert w == 1024
        # h ≈ 1024 / 1.5 ≈ 682.67 → rounded
        assert h > 0

    def test_portrait_calculation(self, manager):
        """Portrait: long side is height, width derived from AR."""
        w, h = manager.calculate_target_dims(1024, 0.75, "portrait")
        assert h == 1024
        assert w > 0

    def test_square_aspect_ratio(self, manager):
        """AR=1.0 should give equal width and height."""
        w, h = manager.calculate_target_dims(1024, 1.0, "landscape")
        assert w == 1024
        assert h == 1024


# ── AR Display ───────────────────────────────────────────────────────────


class TestArToDisplay:
    """Tests for _ar_to_display."""

    def test_square(self):
        assert DatasetManager._ar_to_display(1.0, "squared") == "1:1"

    def test_landscape(self):
        result = DatasetManager._ar_to_display(1.5, "landscape")
        assert ":" in result

    def test_portrait_swaps_ratio(self):
        result = DatasetManager._ar_to_display(0.75, "portrait")
        # portrait display should put H:W
        parts = result.split(":")
        assert len(parts) == 2


# ── Thumbnail Invalidation ───────────────────────────────────────────────


class TestThumbnailInvalidation:
    """Mutations that call update_metadata_after_edit must refresh the thumbnail."""

    def test_crop_regenerates_thumbnail(self, manager, tmp_path):
        from app.core.dataset import thumbnails

        ds_path = tmp_path / "datasets" / "tn_crop"
        ds_path.mkdir(parents=True)
        img_path = ds_path / "a.jpg"
        _create_image(str(img_path), 400, 300)

        ds = manager.create_dataset("tn_crop", path=str(ds_path))
        manager.scan_dataset("tn_crop")

        thumb = thumbnails.thumbnail_path_for(str(ds_path), "a.jpg")
        assert thumb.exists(), "scan should have generated the initial thumbnail"
        first_mtime = thumb.stat().st_mtime_ns

        # Ensure mtime granularity does not collide on Windows (100 ns)
        import time as _time
        _time.sleep(0.05)

        manager.crop_media("tn_crop", "a.jpg", target_w=200, target_h=150)

        assert thumb.exists()
        assert thumb.stat().st_mtime_ns > first_mtime, \
            "thumbnail mtime should advance after crop"

    def test_delete_media_pair_removes_thumbnail(self, manager, tmp_path):
        from app.core.dataset import thumbnails

        ds_path = tmp_path / "datasets" / "tn_del"
        ds_path.mkdir(parents=True)
        _create_image(str(ds_path / "a.jpg"))

        manager.create_dataset("tn_del", path=str(ds_path))
        manager.scan_dataset("tn_del")

        thumb = thumbnails.thumbnail_path_for(str(ds_path), "a.jpg")
        assert thumb.exists()

        manager.delete_media_pair("tn_del", "a.jpg")

        assert not thumb.exists()
