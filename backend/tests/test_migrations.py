"""Tests for the schema migration system.

Focus: the v17 one-time data repair that rewrites legacy
``job_history.definition_id = 'standard'`` rows from their config, replacing
the per-request mutation that used to live inside ``GET /jobs/history/stats``.
"""

from __future__ import annotations

from app.core.db import migrations
from app.core.db.engine import DatabaseEngine


def _rows(eng: DatabaseEngine) -> dict[str, str]:
    return {
        r["id"]: r["definition_id"]
        for r in eng.connection().execute(
            "SELECT id, definition_id FROM job_history"
        )
    }


def test_v17_repairs_legacy_standard_definition_id(tmp_path):
    """v17 rewrites definition_id='standard' from config.definition_id, and is
    idempotent (safe to run against a fresh AND an already-repaired DB)."""
    eng = DatabaseEngine(db_path=str(tmp_path / "mig.db"))
    with eng.write() as conn:
        migrations._migrate_v1(conn)  # create the schema

    with eng.write() as conn:
        # Legacy placeholder + real id in config → repaired to 'flux'.
        conn.execute(
            "INSERT INTO job_history (id, definition_id, config, created_at) "
            "VALUES ('j1', 'standard', ?, 1.0)",
            ('{"definition_id": "flux"}',),
        )
        # Legacy placeholder but no usable id in config → left as 'standard'.
        conn.execute(
            "INSERT INTO job_history (id, definition_id, config, created_at) "
            "VALUES ('j2', 'standard', '{}', 1.0)",
        )
        # Already-correct row → untouched.
        conn.execute(
            "INSERT INTO job_history (id, definition_id, config, created_at) "
            "VALUES ('j3', 'sdxl', '{}', 1.0)",
        )

    with eng.write() as conn:
        migrations._migrate_v17(conn)

    expected = {"j1": "flux", "j2": "standard", "j3": "sdxl"}
    assert _rows(eng) == expected

    # Idempotent: a second run changes nothing.
    with eng.write() as conn:
        migrations._migrate_v17(conn)
    assert _rows(eng) == expected

    eng.close()


def test_v17_repair_runs_once_per_db(tmp_path):
    """The repair is a versioned migration, so it does NOT re-run on an
    already-migrated DB — unlike the old per-request GET mutation. A legacy
    row inserted after migrations have run stays untouched until an
    explicit new migration would target it."""
    eng = DatabaseEngine(db_path=str(tmp_path / "mig2.db"))
    eng.initialize()  # runs every migration incl. v17

    with eng.write() as conn:
        conn.execute(
            "INSERT INTO job_history (id, definition_id, config, created_at) "
            "VALUES ('late', 'standard', ?, 1.0)",
            ('{"definition_id": "flux"}',),
        )

    # Re-running migrations is a no-op — schema_version is already at head.
    migrations.run_migrations(eng)

    val = eng.connection().execute(
        "SELECT definition_id FROM job_history WHERE id = 'late'"
    ).fetchone()["definition_id"]
    assert val == "standard"

    eng.close()
