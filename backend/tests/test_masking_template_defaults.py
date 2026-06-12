"""Tests for seeded masking default templates (v14).

Every masking model the frontend offers must have a readonly General default
template — RemBG had none (only sam3 was seeded in v4), so switching to RemBG
left the user with no template: edits were silently dropped and the
copy-on-edit flow never triggered. v14 also hardens older rows where
``is_default=1`` but ``readonly=0`` (editable "system defaults").
"""

import json
import sqlite3


def _make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE masking_templates (
            id TEXT PRIMARY KEY, project_id TEXT, model_id TEXT, name TEXT,
            is_default INTEGER, readonly INTEGER,
            config TEXT, created_at REAL, updated_at REAL,
            used_count INTEGER DEFAULT 0, last_used_at REAL, branched_from TEXT
        )
    """)
    return conn


def test_migrate_v14_seeds_rembg_default():
    from app.core.db.migrations import _migrate_v14

    conn = _make_conn()
    _migrate_v14(conn)

    row = conn.execute(
        "SELECT * FROM masking_templates WHERE model_id = 'rembg'"
    ).fetchone()
    assert row is not None
    assert row["is_default"] == 1 and row["readonly"] == 1
    assert row["project_id"] is None
    cfg = json.loads(row["config"])
    # Mirrors the frontend's RemBG code defaults.
    assert cfg["model_name"] == "birefnet-general"
    assert cfg["post_process_mask"] is True
    assert cfg["alpha_matting"] is False
    assert {
        "alpha_matting_foreground_threshold",
        "alpha_matting_background_threshold",
        "alpha_matting_erode_size",
    } <= set(cfg)


def test_migrate_v14_is_idempotent():
    from app.core.db.migrations import _migrate_v14

    conn = _make_conn()
    _migrate_v14(conn)
    _migrate_v14(conn)
    n = conn.execute(
        "SELECT COUNT(*) FROM masking_templates WHERE model_id = 'rembg'"
    ).fetchone()[0]
    assert n == 1


def test_migrate_v14_hardens_editable_general_defaults():
    from app.core.db.migrations import _migrate_v14

    conn = _make_conn()
    # An older "system default" left editable (is_default=1, readonly=0).
    conn.execute(
        "INSERT INTO masking_templates (id, project_id, model_id, name, "
        "is_default, readonly, config, created_at) "
        "VALUES ('mask_default_sam3', NULL, 'sam3', 'Default', 1, 0, '{}', 0)"
    )
    _migrate_v14(conn)
    row = conn.execute(
        "SELECT readonly FROM masking_templates WHERE id = 'mask_default_sam3'"
    ).fetchone()
    assert row["readonly"] == 1
