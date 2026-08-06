"""Schema migrations for arcane_tuner.db.

Uses a simple integer-versioned migration system:
- ``schema_version`` table tracks current version
- Each ``_migrate_vN`` function is idempotent
- Migrations run sequentially on startup
"""

from __future__ import annotations

import os
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
        _migrate_v7,
        _migrate_v8,
        _migrate_v9,
        _migrate_v10,
        _migrate_v11,
        _migrate_v12,
        _migrate_v13,
        _migrate_v14,
        _migrate_v15,
        _migrate_v16,
        _migrate_v17,
        _migrate_v18,
        _migrate_v19,
        _migrate_v20,
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

# ── V7: Dataset-level LoRA metadata ────────────────────────────────

def _migrate_v7(conn) -> None:
    """Add ``trigger_word``, ``tags`` (comma-joined) and ``notes`` to ``datasets``."""
    for ddl in (
        "ALTER TABLE datasets ADD COLUMN trigger_word TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE datasets ADD COLUMN tags         TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE datasets ADD COLUMN notes        TEXT NOT NULL DEFAULT ''",
    ):
        try:
            conn.execute(ddl)
        except Exception:
            pass  # Column already exists


# ── V8: Per-image enabled flag (exclude-from-training) ─────────────

def _migrate_v8(conn) -> None:
    """Add ``enabled`` boolean column to ``media_items``.

    Persists the per-image "exclude from training" toggle. Existing rows
    backfill to ``1`` (enabled) via the column DEFAULT, matching the scan
    default (``build_media_entry`` seeds ``enabled=True``), so legacy data
    is unaffected and only an explicit toggle marks an image disabled.
    """
    try:
        conn.execute(
            "ALTER TABLE media_items ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1"
        )
    except Exception:
        pass  # Column already exists


# ── V9: Per-definition training statistics (estimation wall) ───────

def _migrate_v9(conn) -> None:
    """Create ``definition_stats`` + add measured-cost columns to ``job_history``.

    Powers the data-calibrated estimation wall: ``definition_stats`` is a
    local, fully-recomputable cache of per-``definition_id`` calibration
    coefficients (median ``actual / cost_model`` over completed runs). The
    new ``job_history`` columns persist measured run costs (recovered from
    ``step_metrics`` + on-disk manifests during backfill) so estimates and
    VRAM predictions calibrate against reality, not just the analytic model.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS definition_stats (
            definition_id  TEXT PRIMARY KEY,
            run_count      INTEGER NOT NULL DEFAULT 0,
            stats          TEXT NOT NULL DEFAULT '{}',
            updated_at     REAL,
            source_version INTEGER NOT NULL DEFAULT 1
        )
    """)

    for ddl in (
        "ALTER TABLE job_history ADD COLUMN peak_vram_train_mb REAL",
        "ALTER TABLE job_history ADD COLUMN peak_vram_cache_mb REAL",
        "ALTER TABLE job_history ADD COLUMN total_run_bytes INTEGER",
    ):
        try:
            conn.execute(ddl)
        except Exception:
            pass  # Column already exists


# ── V10: Persisted pending-queue priority ──────────────────────────

def _migrate_v10(conn) -> None:
    """Add ``priority`` to ``job_history`` so a manual pending-queue reorder
    survives a backend restart.

    Priority is the primary run-order key (lower = sooner); ``created_at`` is
    only the FIFO tiebreaker. Without persistence a restart reloads every job
    at the column DEFAULT of 0, silently reverting the queue to creation order.
    """
    try:
        conn.execute(
            "ALTER TABLE job_history ADD COLUMN priority INTEGER NOT NULL DEFAULT 0"
        )
    except Exception:
        pass  # Column already exists


def _migrate_v11(conn) -> None:
    """Add ``wildcard`` to ``captioning_templates``.

    The wildcard is a per-template runtime value substituted into every
    ``{wildcard}`` token of the system prompt before captioning, so a prompt
    that reuses a name in several places can be kept clean (the name lives in
    one field, not duplicated through the prompt text).
    """
    try:
        conn.execute(
            "ALTER TABLE captioning_templates "
            "ADD COLUMN wildcard TEXT NOT NULL DEFAULT ''"
        )
    except Exception:
        pass  # Column already exists


# ── V12: Drop dead saved_concepts table ────────────────────────────

def _migrate_v12(conn) -> None:
    """Drop the orphaned ``saved_concepts`` table.

    The saved-concepts feature (masking concepts with global/project scope)
    was removed in the post-overhaul dead-code sweep — its HTTP routes and
    repository are gone and nothing in the codebase reads or writes the table.
    No other table FK-references it, so the drop is self-contained.

    Destructive + irreversible: any rows are permanently lost. This is a
    forward-only migration (the system has no down-migrations), so it runs
    exactly once on each DB already at v11.
    """
    conn.execute("DROP TABLE IF EXISTS saved_concepts")


# ── V13: API captioning provider default templates ──────────────────────

def _migrate_v13(conn) -> None:
    """Seed readonly General default templates for api-* caption providers."""
    import json as _json
    import time as _time

    now = _time.time()
    base = {"temperature": 0.7, "top_p": 1.0, "max_tokens": 512,
            "max_long_side": 1024}
    defaults = [
        ("cap_default_api_openai", "api-openai", "gpt-4o"),
        ("cap_default_api_anthropic", "api-anthropic", "claude-sonnet-4-6"),
        ("cap_default_api_gemini", "api-gemini", "gemini-2.5-flash"),
        ("cap_default_api_openrouter", "api-openrouter", ""),
        ("cap_default_api_custom", "api-custom", ""),
    ]
    for tid, model_id, default_model in defaults:
        conn.execute(
            "INSERT OR IGNORE INTO captioning_templates "
            "(id, project_id, model_id, name, is_default, readonly, "
            " system_prompt, config, created_at, updated_at) "
            "VALUES (?, NULL, ?, 'Default', 1, 1, ?, ?, ?, ?)",
            (tid, model_id, "Describe this image in detail.",
             _json.dumps({**base, "model": default_model}), now, now),
        )


# ── V14: RemBG masking default + readonly hardening ─────────────────────

def _migrate_v14(conn) -> None:
    """Seed the missing RemBG masking default; harden editable defaults.

    v4 seeded a masking default only for sam3, so RemBG had no template at
    all — the settings panel silently dropped edits (no active template) and
    the copy-on-edit flow never ran. Config mirrors the frontend's RemBG code
    defaults. Also flips ``readonly`` on any General row still marked
    ``is_default`` but editable, so "system default" templates can never be
    written through again.
    """
    import json as _json
    import time as _time

    now = _time.time()
    conn.execute(
        "INSERT OR IGNORE INTO masking_templates "
        "(id, project_id, model_id, name, is_default, readonly, "
        " config, created_at, updated_at) "
        "VALUES ('mask_default_rembg', NULL, 'rembg', 'Default', 1, 1, ?, ?, ?)",
        (_json.dumps({
            "model_name": "birefnet-general",
            "post_process_mask": True,
            "alpha_matting": False,
            "alpha_matting_foreground_threshold": 240,
            "alpha_matting_background_threshold": 10,
            "alpha_matting_erode_size": 10,
        }), now, now),
    )
    conn.execute(
        "UPDATE masking_templates SET readonly = 1 "
        "WHERE is_default = 1 AND project_id IS NULL AND readonly = 0"
    )


# ── V15: Paired-image (edit/kontext) dataset support ────────────────────

def _migrate_v15(conn) -> None:
    """Add dataset ``kind`` + per-item control metadata for edit datasets.

    ``datasets.kind`` is an open string enum (``standard`` | ``edit``;
    ``video``/``mixed`` reserved) — it gates pair UI/validation but never
    deletes files when switched. ``media_items.control_count`` /
    ``control_info`` mirror the ``mask_info`` pattern: per-slot rel paths
    and dimensions for stem-matched ``control*/`` images, plus the logical
    ``role_order`` (which physical slot is the training target) and the
    ``target_edited_at`` staleness stamp.
    """
    for ddl in (
        "ALTER TABLE datasets ADD COLUMN kind TEXT NOT NULL DEFAULT 'standard'",
        "ALTER TABLE media_items ADD COLUMN control_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE media_items ADD COLUMN control_info TEXT",
    ):
        try:
            conn.execute(ddl)
        except Exception:
            pass  # Column already exists


# ── V16: Per-clip video metadata (video-LoRA ingest) ────────────────────

def _migrate_v16(conn) -> None:
    """Add per-clip video metadata columns to ``media_items``.

    Foundation for video-LoRA training: the scanner probes each trainable
    clip (mp4/webm/mkv/avi — NOT animated ``.gif``) with PyAV and persists
    its framerate, duration, audio presence and codec so later phases can
    build trim/clip-health on top. ``is_video`` / ``frame_count`` already
    exist (v1) and are NOT re-added. ``clip_warnings`` is a JSON string
    mirroring the ``mask_info`` pattern — populated in a later phase, left
    NULL here. ``trim_start_s`` / ``trim_end_s`` carry user trim bounds.
    ``frame_count_estimated`` flags a frame count derived from
    ``duration × fps`` (the container had no exact frame count).
    """
    for ddl in (
        "ALTER TABLE media_items ADD COLUMN fps REAL",
        "ALTER TABLE media_items ADD COLUMN duration_s REAL",
        "ALTER TABLE media_items ADD COLUMN has_audio INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE media_items ADD COLUMN video_codec TEXT",
        "ALTER TABLE media_items ADD COLUMN trim_start_s REAL",
        "ALTER TABLE media_items ADD COLUMN trim_end_s REAL",
        "ALTER TABLE media_items ADD COLUMN frame_count_estimated INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE media_items ADD COLUMN clip_warnings TEXT",
    ):
        try:
            conn.execute(ddl)
        except Exception:
            pass  # Column already exists


# ── V17: Repair legacy 'standard' definition_id placeholders ────────────

def _migrate_v17(conn) -> None:
    """One-time repair of legacy ``job_history`` rows whose ``definition_id``
    was persisted as the plugin_id placeholder ``'standard'`` instead of the
    real model ID.

    This mutation used to run on *every* ``GET /jobs/history/stats`` request
    (a write inside an idempotent GET). Moving it here makes it run exactly
    once per database, on the upgrade to v17, and keeps the stats endpoint
    read-only. The ``config`` snapshot still carries the true ``definition_id``,
    so we recover it via ``json_extract``.

    Idempotent: the WHERE clause only matches rows still at the placeholder
    with a usable id in config, so re-running (or running on a fresh/already
    -repaired DB) is a no-op.
    """
    conn.execute("""
        UPDATE job_history
        SET definition_id = json_extract(config, '$.definition_id')
        WHERE definition_id = 'standard'
          AND json_extract(config, '$.definition_id') IS NOT NULL
          AND json_extract(config, '$.definition_id') != ''
    """)


# ── V18: Repair ema_enabled from the config snapshot ────────────────────

def _migrate_v18(conn) -> None:
    """One-time repair of ``job_history.ema_enabled`` from the ``config``
    JSON snapshot.

    The live writer populated the column from ``config['use_ema']`` — a key
    that never existed (the real training-config key is ``ema``) — so every
    pre-fix row carries ``ema_enabled = 0`` regardless of what the run used.
    The config snapshot has the truth; recover it once. Rows whose snapshot
    lacks the key keep their current value.

    Idempotent: recomputes the same values on re-run.
    """
    conn.execute("""
        UPDATE job_history
        SET ema_enabled = CASE
            WHEN json_extract(config, '$.ema') IN (1, '1', 'true') THEN 1
            ELSE 0
        END
        WHERE json_extract(config, '$.ema') IS NOT NULL
    """)


# ── V19: Persist lora_on_disk (retires the per-request isfile() sweep) ──

def _migrate_v19(conn) -> None:
    """Add ``job_history.lora_on_disk``, backfilled by one disk sweep.

    ``get_stats`` used to run an ``os.path.isfile()`` check over every
    completed job's ``final_lora_file`` on EVERY request (a filesystem sweep
    inside a read-only GET). Persisting the flag here — computed once, at
    migration time — lets ``get_stats`` become pure SQL (``SUM(lora_on_disk)``).
    Nullable: rows with no ``final_lora_file`` at all are left NULL (no file
    to check), matching ``get_stats``'s existing ``final_lora_file IS NOT
    NULL`` filter. Kept fresh afterward at the two points a job's on-disk
    state actually changes: run completion (``pipeline_train.py``) and the
    ``stats/backfill.py`` reconcile pass (which already walks the disk).

    Idempotent: the ``ADD COLUMN`` is guarded (SQLite has no
    ``IF NOT EXISTS`` for columns) and the backfill recomputes the same
    values from disk on re-run.
    """
    try:
        conn.execute("ALTER TABLE job_history ADD COLUMN lora_on_disk INTEGER")
    except Exception:
        pass  # Column already exists

    rows = conn.execute("""
        SELECT id, final_lora_file FROM job_history
        WHERE status = 'completed' AND final_lora_file IS NOT NULL
    """).fetchall()
    for row in rows:
        on_disk = 1 if os.path.isfile(row["final_lora_file"]) else 0
        conn.execute(
            "UPDATE job_history SET lora_on_disk = ? WHERE id = ?",
            (on_disk, row["id"]),
        )


# ── V20: active_layers on step_metrics (adaptive layer-targeting replay) ──

def _migrate_v20(conn) -> None:
    """Add ``step_metrics.active_layers``, populated per step by the adaptive
    layer-targeting controller's live ``adaptive_active`` count.

    Nullable, and left NULL (never backfilled to 0) for every existing row
    and every future step where the feature is off: 0 is itself a meaningful
    value here ("every remaining layer just froze"), so it must stay
    distinguishable from "adaptive targeting was never enabled for this run"
    — a stats replay chart that coerced absence to 0 would draw a fake
    narrowing-to-zero staircase for runs that never used the feature.

    Idempotent: the ``ADD COLUMN`` is guarded (SQLite has no
    ``IF NOT EXISTS`` for columns), matching every prior single-column
    migration in this file.
    """
    try:
        conn.execute("ALTER TABLE step_metrics ADD COLUMN active_layers INTEGER")
    except Exception:
        pass  # Column already exists
