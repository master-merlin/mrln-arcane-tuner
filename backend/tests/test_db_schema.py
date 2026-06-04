"""
Tests for database schema, migrations, and repository patterns.

Covers:
- V2 migration (boolean flags: has_mask, has_masked, has_masked_caption)
- MediaItemRepository column handling and boolean coercion
- _with_conn transactional patterns for atomic operations
- DatasetRepository upsert_with_conn shared-transaction pattern
"""

import time
import uuid

import pytest
from unittest.mock import patch

from app.core.db.engine import DatabaseEngine
from app.core.db.migrations import run_migrations, _migrate_v1, _migrate_v2
from app.core.db.repositories.media_item_repo import MediaItemRepository
from app.core.db.repositories.dataset_repo import DatasetRepository


# ── Helpers ──────────────────────────────────────────────────────────────


@pytest.fixture()
def db_engine(tmp_path):
    """Create an isolated DatabaseEngine backed by a temp SQLite file."""
    db_path = str(tmp_path / "test.db")
    engine = DatabaseEngine(db_path)
    return engine


@pytest.fixture()
def migrated_engine(db_engine):
    """DatabaseEngine with all migrations applied."""
    run_migrations(db_engine)
    return db_engine


@pytest.fixture()
def media_repo(migrated_engine):
    """MediaItemRepository that uses our test engine via patched get_db."""
    repo = MediaItemRepository()
    with patch("app.core.db.repositories.media_item_repo.get_db", return_value=migrated_engine):
        yield repo


@pytest.fixture()
def dataset_repo(migrated_engine):
    """DatasetRepository that uses our test engine via patched get_db."""
    repo = DatasetRepository()
    with patch("app.core.db.repositories.dataset_repo.get_db", return_value=migrated_engine):
        yield repo


def _bootstrap_v1(engine: DatabaseEngine):
    """Run V1 migration with schema_version tracking (for isolated migration tests)."""
    with engine.write() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            )
        """)
        _migrate_v1(conn)
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")


def _insert_v1_dataset(engine: DatabaseEngine, ds_id: str):
    """Insert a minimal dataset row for FK satisfaction in migration tests."""
    with engine.write() as conn:
        conn.execute("""
            INSERT INTO datasets (id, name, path, created_at)
            VALUES (?, ?, ?, ?)
        """, (ds_id, f"ds_{ds_id[:8]}", "/tmp/test", time.time()))


def _make_dataset_row(name: str = "test_ds") -> dict:
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "path": f"/tmp/{name}",
        "description": "",
        "created_at": time.time(),
        "last_scanned_at": None,
        "file_count": 0,
        "total_size_bytes": 0,
        "multimedia_count": 0,
        "caption_count": 0,
        "mask_count": 0,
        "caption_coverage": False,
        "missing": False,
        "preview_image": None,
        "majority_ar": 1.0,
        "harmonization_score": 0.0,
        "classifier": "",
        "version": "1.0.0",
        "has_cache": False,
    }


def _make_media_item(dataset_id: str, rel_path: str = "img.png", **overrides) -> dict:
    """Build a media item dict compatible with the DB schema.

    Note: 'id' is omitted — it's INTEGER AUTOINCREMENT.
    Note: frame_count defaults to 0 (NOT NULL in schema).
    """
    defaults = {
        "dataset_id": dataset_id,
        "rel_path": rel_path,
        "width": 100,
        "height": 100,
        "aspect_ratio": 1.0,
        "orientation": "squared",
        "size_bytes": 1024,
        "solid_hash": "deadbeef" * 4,
        "is_majority_ar": False,
        "target_width": 100,
        "target_height": 100,
        "has_mask": False,
        "has_masked": False,
        "has_masked_caption": False,
        "mask_info": None,
        "has_caption": False,
        "is_video": False,
        "frame_count": 0,
        "tags": None,
        "notes": None,
        "quality_score": None,
        "added_at": time.time(),
    }
    defaults.update(overrides)
    return defaults


# ── V2 Migration ─────────────────────────────────────────────────────────


class TestV2Migration:
    """Tests for the V2 schema migration (boolean flag columns)."""

    def test_v2_adds_boolean_columns(self, db_engine):
        """V2 migration should add has_mask, has_masked, has_masked_caption."""
        _bootstrap_v1(db_engine)
        ds_id = "test-ds-1"
        _insert_v1_dataset(db_engine, ds_id)

        with db_engine.write() as conn:
            _migrate_v2(conn)

        # Verify columns exist by inserting a row with the new columns
        with db_engine.write() as conn:
            conn.execute("""
                INSERT INTO media_items (
                    dataset_id, rel_path, width, height,
                    aspect_ratio, orientation, size_bytes, has_mask,
                    has_masked, has_masked_caption
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ds_id, "img.png", 100, 100,
                  1.0, "squared", 1024, 1, 0, 1))

        with db_engine.connection() as conn:
            row = conn.execute(
                "SELECT has_mask, has_masked, has_masked_caption FROM media_items LIMIT 1"
            ).fetchone()
            assert row["has_mask"] == 1
            assert row["has_masked"] == 0
            assert row["has_masked_caption"] == 1

    def test_v2_backfills_from_path_columns(self, db_engine):
        """V2 migration should backfill booleans from existing path columns."""
        _bootstrap_v1(db_engine)
        ds_id = "test-ds-2"
        _insert_v1_dataset(db_engine, ds_id)

        # Insert rows with old-style path columns (pre-V2)
        with db_engine.write() as conn:
            conn.execute("""
                INSERT INTO media_items (
                    dataset_id, rel_path, width, height,
                    aspect_ratio, orientation, size_bytes,
                    mask_file, masked_file, masked_caption_file
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ds_id, "with_mask.png", 100, 100,
                  1.0, "squared", 1024,
                  "masks/with_mask.png", "masked/with_mask.jpg", "masked/with_mask.txt"))
            conn.execute("""
                INSERT INTO media_items (
                    dataset_id, rel_path, width, height,
                    aspect_ratio, orientation, size_bytes,
                    mask_file, masked_file, masked_caption_file
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ds_id, "no_mask.png", 100, 100,
                  1.0, "squared", 1024,
                  None, None, None))

        # Run V2 migration
        with db_engine.write() as conn:
            _migrate_v2(conn)

        with db_engine.connection() as conn:
            rows = conn.execute(
                "SELECT rel_path, has_mask, has_masked, has_masked_caption "
                "FROM media_items ORDER BY rel_path"
            ).fetchall()

        masked_row = next(r for r in rows if r["rel_path"] == "with_mask.png")
        clean_row = next(r for r in rows if r["rel_path"] == "no_mask.png")

        assert masked_row["has_mask"] == 1
        assert masked_row["has_masked"] == 1
        assert masked_row["has_masked_caption"] == 1

        assert clean_row["has_mask"] == 0
        assert clean_row["has_masked"] == 0
        assert clean_row["has_masked_caption"] == 0

    def test_v2_is_idempotent(self, db_engine):
        """Running V2 migration twice should not fail."""
        _bootstrap_v1(db_engine)

        with db_engine.write() as conn:
            _migrate_v2(conn)
        # Should not raise
        with db_engine.write() as conn:
            _migrate_v2(conn)

    def test_run_migrations_reaches_latest(self, db_engine):
        """Full run_migrations should bring schema to the latest version."""
        run_migrations(db_engine)
        with db_engine.connection() as conn:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            assert row["version"] == 10


# ── V10 Migration: persisted pending-queue priority ──────────────────────


class TestV10Migration:
    """V10 adds a persisted ``priority`` column to ``job_history`` so a manual
    pending-queue reorder survives a backend restart."""

    def test_v10_adds_priority_column(self, db_engine):
        run_migrations(db_engine)
        with db_engine.write() as conn:
            conn.execute(
                "INSERT INTO job_history (id, definition_id, status, created_at, priority) "
                "VALUES (?, ?, ?, ?, ?)",
                ("job-pri", "flux/dev", "pending", time.time(), 3),
            )
        with db_engine.connection() as conn:
            row = conn.execute(
                "SELECT priority FROM job_history WHERE id = ?", ("job-pri",)
            ).fetchone()
            assert row["priority"] == 3

    def test_v10_priority_defaults_zero(self, db_engine):
        run_migrations(db_engine)
        with db_engine.write() as conn:
            conn.execute(
                "INSERT INTO job_history (id, definition_id, status, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("job-nopri", "flux/dev", "pending", time.time()),
            )
        with db_engine.connection() as conn:
            row = conn.execute(
                "SELECT priority FROM job_history WHERE id = ?", ("job-nopri",)
            ).fetchone()
            assert row["priority"] == 0


# ── Media Item Repository ────────────────────────────────────────────────


class TestMediaItemRepoBooleans:
    """Tests for boolean flag handling in MediaItemRepository."""

    def test_columns_include_boolean_flags(self):
        """_COLUMNS should include has_mask, has_masked, has_masked_caption."""
        cols = MediaItemRepository._COLUMNS
        assert "has_mask" in cols
        assert "has_masked" in cols
        assert "has_masked_caption" in cols

    def test_columns_exclude_old_path_fields(self):
        """_COLUMNS should NOT include the old path columns."""
        cols = MediaItemRepository._COLUMNS
        assert "mask_file" not in cols
        assert "masked_file" not in cols
        assert "masked_caption_file" not in cols
        assert "caption_file" not in cols

    def test_prepare_coerces_booleans_to_int(self):
        """_prepare should convert boolean flags to int for SQLite storage."""
        data = {
            "has_mask": True,
            "has_masked": False,
            "has_masked_caption": True,
            "is_majority_ar": True,
            "has_caption": False,
            "is_video": False,
        }
        prepared = MediaItemRepository._prepare(data)
        assert prepared["has_mask"] == 1
        assert prepared["has_masked"] == 0
        assert prepared["has_masked_caption"] == 1
        assert prepared["is_majority_ar"] == 1
        assert prepared["has_caption"] == 0
        assert prepared["is_video"] == 0

    def test_to_metadata_dict_converts_ints_to_bools(self, media_repo, dataset_repo):
        """to_metadata_dict should convert int booleans back to Python bools."""
        ds = _make_dataset_row("bool_test")
        dataset_repo.upsert(ds)

        item = _make_media_item(ds["id"], "test.png",
                                has_mask=True, has_masked=False,
                                has_masked_caption=True)
        media_repo.upsert(item)

        result = media_repo.to_metadata_dict(ds["id"])
        assert "test.png" in result
        meta = result["test.png"]
        assert meta["has_mask"] is True
        assert meta["has_masked"] is False
        assert meta["has_masked_caption"] is True
        assert isinstance(meta["has_mask"], bool)

    def test_upsert_and_retrieve(self, media_repo, dataset_repo):
        """Round-trip: upsert a media item with booleans, then retrieve."""
        ds = _make_dataset_row("upsert_test")
        dataset_repo.upsert(ds)

        item = _make_media_item(ds["id"], "photo.jpg",
                                has_mask=True, has_masked=True,
                                has_masked_caption=False,
                                has_caption=True)
        media_repo.upsert(item)

        rows = media_repo.get_by_dataset(ds["id"])
        assert len(rows) == 1
        row = rows[0]
        assert row["has_mask"] == 1  # Raw SQLite int
        assert row["has_masked"] == 1
        assert row["has_masked_caption"] == 0
        assert row["has_caption"] == 1


# ── _with_conn Transaction Patterns ──────────────────────────────────────


class TestWithConnTransactions:
    """Tests for the _with_conn shared-transaction pattern."""

    def test_update_with_conn_shares_transaction(self, media_repo, dataset_repo, migrated_engine):
        """update_with_conn should use the caller's connection (single commit)."""
        ds = _make_dataset_row("txn_update")
        dataset_repo.upsert(ds)
        item = _make_media_item(ds["id"], "txn.png", has_mask=False)
        media_repo.upsert(item)

        # Update within a shared transaction
        with migrated_engine.write() as conn:
            media_repo.update_with_conn(
                conn, ds["id"], "txn.png",
                {"has_mask": True, "has_masked": True}
            )
            # Before commit, verify we can read the update in the same txn
            row = conn.execute(
                "SELECT has_mask, has_masked FROM media_items WHERE rel_path = ?",
                ("txn.png",)
            ).fetchone()
            assert row["has_mask"] == 1
            assert row["has_masked"] == 1

        # Verify persisted after commit
        result = media_repo.to_metadata_dict(ds["id"])
        assert result["txn.png"]["has_mask"] is True
        assert result["txn.png"]["has_masked"] is True

    def test_delete_with_conn_shares_transaction(self, media_repo, dataset_repo, migrated_engine):
        """delete_with_conn should use the caller's connection."""
        ds = _make_dataset_row("txn_delete")
        dataset_repo.upsert(ds)
        item = _make_media_item(ds["id"], "doomed.png")
        media_repo.upsert(item)

        with migrated_engine.write() as conn:
            media_repo.delete_with_conn(conn, ds["id"], "doomed.png")
            # Verify deleted within the transaction
            count = conn.execute(
                "SELECT COUNT(*) as cnt FROM media_items WHERE rel_path = ?",
                ("doomed.png",)
            ).fetchone()["cnt"]
            assert count == 0

        # Verify persisted after commit
        rows = media_repo.get_by_dataset(ds["id"])
        assert len(rows) == 0

    def test_atomic_delete_and_update(self, media_repo, dataset_repo, migrated_engine):
        """Deleting a media item and updating dataset counters atomically."""
        ds = _make_dataset_row("atomic_op")
        ds["multimedia_count"] = 2
        ds["mask_count"] = 1
        dataset_repo.upsert(ds)

        item1 = _make_media_item(ds["id"], "keep.png", has_mask=True)
        item2 = _make_media_item(ds["id"], "remove.png", has_mask=True)
        media_repo.upsert(item1)
        media_repo.upsert(item2)

        # Atomic: delete item + decrement counters in one transaction
        with migrated_engine.write() as conn:
            media_repo.delete_with_conn(conn, ds["id"], "remove.png")
            dataset_repo.upsert_with_conn(conn, {
                **ds,
                "multimedia_count": 1,
                "mask_count": 0,
            })

        # Verify both changes committed together
        rows = media_repo.get_by_dataset(ds["id"])
        assert len(rows) == 1
        assert rows[0]["rel_path"] == "keep.png"

    def test_bulk_upsert_with_conn(self, media_repo, dataset_repo, migrated_engine):
        """bulk_upsert_with_conn should insert/update multiple items in one txn."""
        ds = _make_dataset_row("bulk_test")
        dataset_repo.upsert(ds)

        items = [
            _make_media_item(ds["id"], f"img{i}.png", has_mask=(i % 2 == 0))
            for i in range(5)
        ]

        with migrated_engine.write() as conn:
            media_repo.bulk_upsert_with_conn(conn, ds["id"], items)

        rows = media_repo.get_by_dataset(ds["id"])
        assert len(rows) == 5

        result = media_repo.to_metadata_dict(ds["id"])
        assert result["img0.png"]["has_mask"] is True
        assert result["img1.png"]["has_mask"] is False
        assert result["img2.png"]["has_mask"] is True


# ── Dataset Repository ───────────────────────────────────────────────────


class TestDatasetRepoUpsert:
    """Tests for dataset_repo upsert and upsert_with_conn."""

    def test_upsert_insert_and_update(self, dataset_repo, migrated_engine):
        """upsert should insert new row, then update existing."""
        ds = _make_dataset_row("upsert_ds")
        dataset_repo.upsert(ds)

        # Verify inserted
        with migrated_engine.connection() as conn:
            row = conn.execute(
                "SELECT name FROM datasets WHERE id = ?", (ds["id"],)
            ).fetchone()
            assert row["name"] == "upsert_ds"

        # Update description
        ds["description"] = "updated"
        dataset_repo.upsert(ds)
        with migrated_engine.connection() as conn:
            row = conn.execute(
                "SELECT description FROM datasets WHERE id = ?", (ds["id"],)
            ).fetchone()
            assert row["description"] == "updated"

    def test_upsert_with_conn_uses_provided_connection(self, dataset_repo, migrated_engine):
        """upsert_with_conn should not open its own transaction."""
        ds = _make_dataset_row("shared_txn")

        with migrated_engine.write() as conn:
            dataset_repo.upsert_with_conn(conn, ds)
            # Verify within same transaction
            row = conn.execute(
                "SELECT name FROM datasets WHERE id = ?", (ds["id"],)
            ).fetchone()
            assert row["name"] == "shared_txn"

    def test_upsert_conflicting_name_doesnt_crash(self, dataset_repo):
        """Upserting with same name but different ID should handle conflict."""
        ds1 = _make_dataset_row("conflict")
        dataset_repo.upsert(ds1)

        ds2 = _make_dataset_row("conflict")
        ds2["id"] = str(uuid.uuid4())  # different ID, same name
        # Should not raise — INSERT OR REPLACE handles this
        dataset_repo.upsert(ds2)
