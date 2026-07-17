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


def test_v18_backfills_ema_from_config(tmp_path):
    """v18 repairs ema_enabled from the config snapshot: the live writer read
    config['use_ema'] (a key that never existed; real key is 'ema'), so every
    pre-fix row carries ema_enabled=0 regardless of truth. Idempotent."""
    eng = DatabaseEngine(db_path=str(tmp_path / "mig18.db"))
    with eng.write() as conn:
        migrations._migrate_v1(conn)  # create the schema

    with eng.write() as conn:
        # EMA actually on (JSON true) but recorded as 0 → repaired to 1.
        conn.execute(
            "INSERT INTO job_history (id, definition_id, config, created_at, ema_enabled) "
            "VALUES ('e1', 'flux', ?, 1.0, 0)",
            ('{"ema": true}',),
        )
        # EMA off (JSON false) → stays/normalizes to 0.
        conn.execute(
            "INSERT INTO job_history (id, definition_id, config, created_at, ema_enabled) "
            "VALUES ('e2', 'flux', ?, 1.0, 0)",
            ('{"ema": false}',),
        )
        # Config lacks the key → row untouched (keeps whatever it has).
        conn.execute(
            "INSERT INTO job_history (id, definition_id, config, created_at, ema_enabled) "
            "VALUES ('e3', 'flux', '{}', 1.0, 1)",
        )
        # Integer-style snapshot value → treated as on.
        conn.execute(
            "INSERT INTO job_history (id, definition_id, config, created_at, ema_enabled) "
            "VALUES ('e4', 'flux', ?, 1.0, 0)",
            ('{"ema": 1}',),
        )

    with eng.write() as conn:
        migrations._migrate_v18(conn)

    def ema_rows():
        return {
            r["id"]: r["ema_enabled"]
            for r in eng.connection().execute(
                "SELECT id, ema_enabled FROM job_history"
            )
        }

    expected = {"e1": 1, "e2": 0, "e3": 1, "e4": 1}
    assert ema_rows() == expected

    # Idempotent: a second run changes nothing.
    with eng.write() as conn:
        migrations._migrate_v18(conn)
    assert ema_rows() == expected

    eng.close()
