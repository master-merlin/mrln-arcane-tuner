"""Schema migrations for arcane_tuner.db.

Uses a simple integer-versioned migration system:
- ``schema_version`` table tracks current version
- Each ``_migrate_vN`` function is idempotent
- Migrations run sequentially on startup
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from .engine import DatabaseEngine

logger = structlog.get_logger(__name__)


# ── Public entry point ──────────────────────────────────────────────────

def run_migrations(engine: DatabaseEngine) -> None:
    """Apply all pending migrations."""
    with engine.write() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            )
        """)
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        current = row["version"] if row else 0

    migrations = [
        _migrate_v1,
        _migrate_v2,
        _migrate_v3,
    ]

    for i, migrate_fn in enumerate(migrations, start=1):
        if i > current:
            logger.info("running_migration", version=i)
            with engine.write() as conn:
                migrate_fn(conn)
                if current == 0:
                    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (i,))
                else:
                    conn.execute("UPDATE schema_version SET version = ?", (i,))
            current = i
            logger.info("migration_complete", version=i)

    logger.info("schema_up_to_date", version=current)


# ── V1: Initial schema ─────────────────────────────────────────────────

def _migrate_v1(conn) -> None:
    """Create all initial tables."""

    # ── datasets ────────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL UNIQUE,
            path            TEXT NOT NULL,
            description     TEXT NOT NULL DEFAULT '',
            created_at      REAL NOT NULL,
            last_scanned_at REAL,
            file_count      INTEGER NOT NULL DEFAULT 0,
            total_size_bytes INTEGER NOT NULL DEFAULT 0,
            multimedia_count INTEGER NOT NULL DEFAULT 0,
            caption_count   INTEGER NOT NULL DEFAULT 0,
            mask_count      INTEGER NOT NULL DEFAULT 0,
            caption_coverage INTEGER NOT NULL DEFAULT 0,
            missing         INTEGER NOT NULL DEFAULT 0,
            preview_image   TEXT,
            majority_ar     REAL,
            harmonization_score REAL NOT NULL DEFAULT 0.0,
            classifier      TEXT NOT NULL DEFAULT '',
            version         TEXT NOT NULL DEFAULT '1.0.0',
            has_cache       INTEGER NOT NULL DEFAULT 0,
            source_type     TEXT NOT NULL DEFAULT 'local',
            license         TEXT NOT NULL DEFAULT '',
            updated_at      REAL
        )
    """)

    # ── media_items ─────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS media_items (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id      TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
            rel_path        TEXT NOT NULL,
            width           INTEGER NOT NULL DEFAULT 0,
            height          INTEGER NOT NULL DEFAULT 0,
            aspect_ratio    REAL NOT NULL DEFAULT 0.0,
            orientation     TEXT NOT NULL DEFAULT '',
            size_bytes      INTEGER NOT NULL DEFAULT 0,
            solid_hash      TEXT NOT NULL DEFAULT '',
            is_majority_ar  INTEGER NOT NULL DEFAULT 0,
            target_width    INTEGER NOT NULL DEFAULT 0,
            target_height   INTEGER NOT NULL DEFAULT 0,
            mask_file       TEXT,
            masked_file     TEXT,
            masked_caption_file TEXT,
            mask_info       TEXT,
            caption_file    TEXT,
            has_caption     INTEGER NOT NULL DEFAULT 0,
            is_video        INTEGER NOT NULL DEFAULT 0,
            frame_count     INTEGER NOT NULL DEFAULT 0,
            tags            TEXT,
            notes           TEXT,
            quality_score   REAL,
            added_at        REAL
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_media_items_dataset_path
        ON media_items(dataset_id, rel_path)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_media_items_orientation
        ON media_items(dataset_id, orientation)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_media_items_majority_ar
        ON media_items(dataset_id, is_majority_ar)
    """)

    # ── job_history ─────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_history (
            id                  TEXT PRIMARY KEY,
            lora_name           TEXT NOT NULL DEFAULT '',
            definition_id       TEXT NOT NULL DEFAULT '',
            status              TEXT NOT NULL DEFAULT 'pending',
            config              TEXT NOT NULL DEFAULT '{}',
            created_at          REAL NOT NULL,
            started_at          REAL,
            finished_at         REAL,
            duration_seconds    REAL,
            training_seconds    REAL,
            error               TEXT,
            output_dir          TEXT,
            final_checkpoint    TEXT,
            final_lora_file     TEXT,
            final_lora_size_bytes INTEGER,
            total_steps         INTEGER NOT NULL DEFAULT 0,
            completed_steps     INTEGER NOT NULL DEFAULT 0,
            config_version      TEXT,
            config_schema_version INTEGER,
            resumed_from        TEXT,
            datasets_used       TEXT,
            network_rank        INTEGER,
            network_alpha       INTEGER,
            optimizer_type      TEXT,
            learning_rate       REAL,
            lr_scheduler        TEXT,
            timestep_sampling   TEXT,
            batch_size          INTEGER,
            grad_accum          INTEGER,
            avg_loss            REAL,
            min_loss            REAL,
            loss_history        TEXT,
            avg_step_time       REAL,
            avg_save_time       REAL,
            targeted_layers     TEXT,
            tags                TEXT,
            notes               TEXT,
            parent_job_id       TEXT REFERENCES job_history(id),
            quantization        TEXT,
            mixed_precision     TEXT,
            ema_enabled         INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_job_history_definition
        ON job_history(definition_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_job_history_status
        ON job_history(status)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_job_history_created
        ON job_history(created_at DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_job_history_lora
        ON job_history(lora_name)
    """)

    # ── checkpoints ─────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id          TEXT NOT NULL REFERENCES job_history(id) ON DELETE CASCADE,
            step            INTEGER NOT NULL,
            path            TEXT NOT NULL,
            lora_file       TEXT,
            lora_size_bytes INTEGER,
            created_at      REAL NOT NULL,
            loss_at_step    REAL,
            lr_at_step      REAL,
            is_final        INTEGER NOT NULL DEFAULT 0,
            is_deleted      INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_checkpoints_job_step
        ON checkpoints(job_id, step)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_checkpoints_path
        ON checkpoints(path)
    """)

    # ── templates ───────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS templates (
            id              TEXT PRIMARY KEY,
            category        TEXT NOT NULL,
            name            TEXT NOT NULL,
            definition_id   TEXT,
            model_id        TEXT,
            is_default      INTEGER NOT NULL DEFAULT 0,
            readonly        INTEGER NOT NULL DEFAULT 0,
            system_prompt   TEXT,
            config          TEXT NOT NULL DEFAULT '{}',
            created_at      REAL NOT NULL,
            updated_at      REAL,
            used_count      INTEGER NOT NULL DEFAULT 0,
            last_used_at    REAL
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_templates_category_name
        ON templates(category, name)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_templates_category_def
        ON templates(category, definition_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_templates_category_model
        ON templates(category, model_id)
    """)

    # ── sample_images ───────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sample_images (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id      TEXT NOT NULL REFERENCES job_history(id) ON DELETE CASCADE,
            step        INTEGER NOT NULL,
            prompt      TEXT NOT NULL DEFAULT '',
            seed        INTEGER,
            path        TEXT NOT NULL,
            width       INTEGER NOT NULL DEFAULT 0,
            height      INTEGER NOT NULL DEFAULT 0,
            created_at  REAL NOT NULL
        )
    """)

    # ── step_metrics ────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS step_metrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id          TEXT NOT NULL REFERENCES job_history(id) ON DELETE CASCADE,
            step            INTEGER NOT NULL,
            loss            REAL,
            lr              REAL,
            grad_norm       REAL,
            timestep_mean   REAL,
            epoch           REAL,
            created_at      REAL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_step_metrics_job_step
        ON step_metrics(job_id, step)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_step_metrics_job_loss
        ON step_metrics(job_id, loss)
    """)

    # ── job_datasets (many-to-many) ─────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_datasets (
            job_id          TEXT NOT NULL REFERENCES job_history(id) ON DELETE CASCADE,
            dataset_id      TEXT REFERENCES datasets(id) ON DELETE SET NULL,
            dataset_name    TEXT NOT NULL,
            dataset_version TEXT NOT NULL DEFAULT '1.0.0',
            num_repeats     INTEGER NOT NULL DEFAULT 1,
            masking_enabled INTEGER NOT NULL DEFAULT 0,
            caption_dropout REAL NOT NULL DEFAULT 0.0,
            PRIMARY KEY (job_id, dataset_name)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_job_datasets_job
        ON job_datasets(job_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_job_datasets_dataset
        ON job_datasets(dataset_id)
    """)


# ── V2: Boolean flags for mask/masked/masked_caption ───────────────

def _migrate_v2(conn) -> None:
    """Add boolean flag columns and backfill from path columns.

    Path columns are kept for backward compat but no longer written.
    """
    # Add boolean columns (idempotent via IF NOT EXISTS isn't available
    # for ALTER TABLE, so we try/except for already-added columns)
    for col in ("has_mask", "has_masked", "has_masked_caption"):
        try:
            conn.execute(
                f"ALTER TABLE media_items ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0"
            )
        except Exception:
            pass  # Column already exists

    # Backfill from existing path columns
    conn.execute("""
        UPDATE media_items SET has_mask = 1
        WHERE mask_file IS NOT NULL AND mask_file != ''
    """)
    conn.execute("""
        UPDATE media_items SET has_masked = 1
        WHERE masked_file IS NOT NULL AND masked_file != ''
    """)
    conn.execute("""
        UPDATE media_items SET has_masked_caption = 1
        WHERE masked_caption_file IS NOT NULL AND masked_caption_file != ''
    """)


# ── V3: Overlay tracking column ────────────────────────────────────

def _migrate_v3(conn) -> None:
    """Add has_overlay boolean column to media_items."""
    try:
        conn.execute(
            "ALTER TABLE media_items ADD COLUMN has_overlay INTEGER NOT NULL DEFAULT 0"
        )
    except Exception:
        pass  # Column already exists
