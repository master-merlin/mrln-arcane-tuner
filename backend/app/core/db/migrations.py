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
        _migrate_v4,
        _migrate_v5,
        _migrate_v6,
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


# ── V4: Project-driven settings architecture ─────────────────────────

def _migrate_v4(conn) -> None:
    """Create project-driven architecture tables.

    Introduces:
    - ``projects`` — top-level project entity
    - ``captioning_templates`` — domain-specific captioning templates
    - ``masking_templates`` — domain-specific masking templates
    - ``training_templates`` — domain-specific training templates
    - ``project_preferences`` — active selections per project
    - ``project_datasets`` — M:N project ↔ dataset association
    - ``saved_concepts`` — masking concepts with flexible global/project scope

    Also:
    - Adds ``project_id`` FK to ``job_history``
    - Drops legacy ``templates`` table
    - Strips training/captioning/masking keys from ``settings.json``
    """
    import json as _json
    import time as _time
    import uuid as _uuid

    # ── projects ─────────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL DEFAULT '',
            color       TEXT NOT NULL DEFAULT '#6366f1',
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL
        )
    """)

    # ── captioning_templates ─────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS captioning_templates (
            id            TEXT PRIMARY KEY,
            project_id    TEXT REFERENCES projects(id) ON DELETE CASCADE,
            model_id      TEXT NOT NULL,
            name          TEXT NOT NULL,
            is_default    INTEGER NOT NULL DEFAULT 0,
            readonly      INTEGER NOT NULL DEFAULT 0,
            system_prompt TEXT NOT NULL DEFAULT 'Describe this image in detail.',
            config        TEXT NOT NULL DEFAULT '{}',
            created_at    REAL NOT NULL,
            updated_at    REAL,
            used_count    INTEGER NOT NULL DEFAULT 0,
            last_used_at  REAL,
            branched_from TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_cap_tpl_project
        ON captioning_templates(project_id, model_id)
    """)

    # ── masking_templates ────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS masking_templates (
            id            TEXT PRIMARY KEY,
            project_id    TEXT REFERENCES projects(id) ON DELETE CASCADE,
            model_id      TEXT NOT NULL,
            name          TEXT NOT NULL,
            is_default    INTEGER NOT NULL DEFAULT 0,
            readonly      INTEGER NOT NULL DEFAULT 0,
            config        TEXT NOT NULL DEFAULT '{}',
            created_at    REAL NOT NULL,
            updated_at    REAL,
            used_count    INTEGER NOT NULL DEFAULT 0,
            last_used_at  REAL,
            branched_from TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_mask_tpl_project
        ON masking_templates(project_id, model_id)
    """)

    # ── training_templates ───────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_templates (
            id            TEXT PRIMARY KEY,
            project_id    TEXT REFERENCES projects(id) ON DELETE CASCADE,
            definition_id TEXT NOT NULL,
            name          TEXT NOT NULL,
            is_default    INTEGER NOT NULL DEFAULT 0,
            readonly      INTEGER NOT NULL DEFAULT 0,
            config        TEXT NOT NULL DEFAULT '{}',
            created_at    REAL NOT NULL,
            updated_at    REAL,
            used_count    INTEGER NOT NULL DEFAULT 0,
            last_used_at  REAL,
            branched_from TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_train_tpl_project
        ON training_templates(project_id, definition_id)
    """)

    # ── project_preferences ──────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS project_preferences (
            id                        TEXT PRIMARY KEY,
            project_id                TEXT UNIQUE REFERENCES projects(id) ON DELETE CASCADE,
            selected_caption_model    TEXT DEFAULT 'florence-2',
            active_caption_template   TEXT,
            qwen3_variant             TEXT DEFAULT '4B-Instruct',
            selected_mask_model       TEXT DEFAULT 'sam3',
            active_mask_template      TEXT,
            training_selections       TEXT NOT NULL DEFAULT '{}'
        )
    """)

    # ── project_datasets (M:N) ───────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS project_datasets (
            project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            dataset_id  TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
            added_at    REAL,
            PRIMARY KEY (project_id, dataset_id)
        )
    """)

    # ── saved_concepts ───────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS saved_concepts (
            id          TEXT PRIMARY KEY,
            project_id  TEXT REFERENCES projects(id) ON DELETE SET NULL,
            name        TEXT NOT NULL,
            points      TEXT NOT NULL DEFAULT '[]',
            model_id    TEXT NOT NULL DEFAULT 'sam3',
            created_at  REAL NOT NULL,
            updated_at  REAL
        )
    """)

    # ── Amend job_history with project_id ────────────────────────────
    try:
        conn.execute(
            "ALTER TABLE job_history ADD COLUMN project_id TEXT "
            "REFERENCES projects(id) ON DELETE SET NULL"
        )
    except Exception:
        pass  # Column already exists
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_job_project
        ON job_history(project_id)
    """)

    # ── Drop legacy templates table ──────────────────────────────────
    conn.execute("DROP TABLE IF EXISTS templates")
    conn.execute("DROP INDEX IF EXISTS idx_templates_category_name")
    conn.execute("DROP INDEX IF EXISTS idx_templates_category_def")
    conn.execute("DROP INDEX IF EXISTS idx_templates_category_model")

    # ── Seed Global default templates ────────────────────────────────
    now = _time.time()

    # --- Captioning defaults (per model, project_id = NULL = General) ---
    cap_defaults = [
        {
            "id": "cap_default_florence2",
            "model_id": "florence-2",
            "name": "Default",
            "system_prompt": "Describe this image in detail.",
            "config": _json.dumps({
                "task_type": "Detailed Caption",
                "max_tokens": 512,
                "num_beams": 3,
            }),
        },
        {
            "id": "cap_default_qwen3vl",
            "model_id": "qwen3-vl",
            "name": "Default",
            "system_prompt": "Describe this image in detail.",
            "config": _json.dumps({
                "max_long_side": 1280,
                "temperature": 0.7,
                "top_p": 0.8,
                "num_beams": 1,
                "repetition_penalty": 1.2,
                "max_tokens": 512,
                "frames": 16,
            }),
        },
        {
            "id": "cap_default_joycaption",
            "model_id": "joycaption",
            "name": "Default",
            "system_prompt": "Describe this image in detail.",
            "config": _json.dumps({
                "caption_type": "Descriptive",
                "caption_length": "long",
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": 512,
                "name_input": "",
                "refer_character_name": False,
                "exclude_people_info": False,
                "include_lighting": False,
                "include_camera_angle": False,
                "include_watermark": False,
                "include_jpeg_artifacts": False,
                "include_exif": False,
                "exclude_sexual": False,
                "exclude_resolution": False,
                "include_aesthetic_quality": False,
                "include_composition": False,
                "exclude_text": False,
                "specify_depth_field": False,
                "specify_lighting_sources": False,
                "no_ambiguous_language": False,
                "include_nsfw_rating": False,
                "only_important_elements": False,
                "exclude_artist_name": False,
                "identify_orientation": False,
                "use_profanity": False,
                "no_euphemisms": False,
                "include_character_age": False,
                "include_shot_type": False,
                "exclude_mood": False,
                "include_vantage_height": False,
                "mention_watermark": False,
                "avoid_meta_phrases": False,
            }),
        },
        {
            "id": "cap_default_youtuvl",
            "model_id": "youtu-vl",
            "name": "Default",
            "system_prompt": "Describe this image in detail.",
            "config": _json.dumps({
                "max_long_side": 768,
                "max_num_patches": 256,
                "temperature": 0.1,
                "top_p": 0.001,
                "repetition_penalty": 1.05,
                "max_tokens": 512,
            }),
        },
    ]
    for cd in cap_defaults:
        conn.execute(
            "INSERT OR IGNORE INTO captioning_templates "
            "(id, project_id, model_id, name, is_default, readonly, "
            " system_prompt, config, created_at, updated_at) "
            "VALUES (?, NULL, ?, ?, 1, 1, ?, ?, ?, ?)",
            (cd["id"], cd["model_id"], cd["name"],
             cd["system_prompt"], cd["config"], now, now),
        )

    # --- Masking defaults ---
    mask_defaults = [
        {
            "id": "mask_default_sam3",
            "model_id": "sam3",
            "name": "Default",
            "config": _json.dumps({
                "text_prompt": "subject",
                "multimask_output": True,
                "max_hole_area": 0,
                "max_sprinkle_area": 0,
            }),
        },
    ]
    for md in mask_defaults:
        conn.execute(
            "INSERT OR IGNORE INTO masking_templates "
            "(id, project_id, model_id, name, is_default, readonly, "
            " config, created_at, updated_at) "
            "VALUES (?, NULL, ?, ?, 1, 1, ?, ?, ?)",
            (md["id"], md["model_id"], md["name"], md["config"], now, now),
        )

    # ── Seed General preferences (project_id = NULL) ─────────────────
    conn.execute(
        "INSERT OR IGNORE INTO project_preferences "
        "(id, project_id, selected_caption_model, qwen3_variant, "
        " selected_mask_model, training_selections) "
        "VALUES (?, NULL, 'florence-2', '4B-Instruct', 'sam3', '{}')",
        (str(_uuid.uuid4()),),
    )

    # ── Strip training/captioning/masking from settings.json ─────────
    _strip_settings_json()

    logger.info(
        "v4_migration_complete",
        msg="Project-driven architecture tables created, legacy templates dropped",
    )


def _strip_settings_json() -> None:
    """Remove training, captioning, masking keys from settings.json.

    Called as part of V4 migration. Leaves only ``application`` (and any
    other unknown keys for forward-compat).
    """
    import json as _json
    import os as _os

    # migrations.py is at backend/app/core/db/migrations.py
    # → 4 dirname calls to reach backend/
    root = _os.path.dirname(
        _os.path.dirname(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        )
    )
    settings_path = _os.path.join(root, "settings.json")

    if not _os.path.exists(settings_path):
        return

    try:
        with open(settings_path, "r") as f:
            data = _json.load(f)
    except Exception:
        return

    changed = False
    for key in ("training", "captioning", "masking"):
        if key in data:
            del data[key]
            changed = True

    if changed:
        with open(settings_path, "w") as f:
            _json.dump(data, f, indent=2)
        logger.info("settings_json_stripped", removed=["training", "captioning", "masking"])

# ── V5: PID tracking for job recovery ──────────────────────────────

def _migrate_v5(conn) -> None:
    """Add ``pid`` column to ``job_history`` for process-liveness checks."""
    try:
        conn.execute(
            "ALTER TABLE job_history ADD COLUMN pid INTEGER"
        )
    except Exception:
        pass  # Column already exists

# ── V6: Virtual Epoch Tracking ─────────────────────────────────────

def _migrate_v6(conn) -> None:
    """Add ``completed_epochs`` column to ``job_history`` to record virtual final epoch."""
    try:
        conn.execute(
            "ALTER TABLE job_history ADD COLUMN completed_epochs REAL"
        )
    except Exception:
        pass  # Column already exists

