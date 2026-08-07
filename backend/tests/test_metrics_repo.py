"""Tests for MetricsRepository — bulk insert/query for step_metrics.

Covers the V20 migration adding ``active_layers`` (adaptive layer-targeting
staircase replay data — see docs/superpowers/plans for the
adaptive-layer-targeting feature): the column must round-trip a real value
AND, critically, must stay NULL (never coerce to 0) when the feature is off,
since 0 is itself a meaningful "zero layers active" fact distinct from "no
data was recorded".
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.core.db.engine import DatabaseEngine
from app.core.db.migrations import run_migrations
from app.core.db.repositories.metrics_repo import MetricsRepository


@pytest.fixture()
def db_engine(tmp_path):
    """Isolated DatabaseEngine backed by a temp SQLite file, fully migrated."""
    engine = DatabaseEngine(db_path=str(tmp_path / "metrics_test.db"))
    run_migrations(engine)
    return engine


@pytest.fixture()
def metrics_repo(db_engine):
    """MetricsRepository wired to the isolated test engine via patched get_db."""
    repo = MetricsRepository()
    with patch("app.core.db.repositories.metrics_repo.get_db", return_value=db_engine):
        yield repo


def _insert_job(db_engine, job_id: str) -> None:
    """Minimal job_history row so step_metrics' FK is satisfiable."""
    with db_engine.write() as conn:
        conn.execute(
            "INSERT INTO job_history (id, definition_id, created_at) VALUES (?, ?, ?)",
            (job_id, "flux", 0.0),
        )


class TestActiveLayersColumn:
    def test_batch_insert_and_read_active_layers(self, db_engine, metrics_repo):
        """A real active_layers value round-trips through get_loss_curve."""
        _insert_job(db_engine, "j1")
        metrics_repo.batch_insert("j1", [
            {"step": 1, "loss": 0.5, "lr": 1e-4, "grad_norm": 0.1,
             "timestep_mean": 500.0, "epoch": 0.1, "active_layers": 248},
        ])
        rows = metrics_repo.get_loss_curve("j1")
        assert len(rows) == 1
        assert rows[0]["active_layers"] == 248

    def test_absent_active_layers_is_null_not_zero(self, db_engine, metrics_repo):
        """D1: a feature-off run (no adaptive_active key at all) must persist
        NULL, never 0 — 0 is a distinct, meaningful value ("every layer got
        frozen") that a NULL-as-0 coercion would make indistinguishable from
        "adaptive targeting was never enabled for this run"."""
        _insert_job(db_engine, "j2")
        metrics_repo.batch_insert("j2", [
            {"step": 1, "loss": 0.4, "lr": 1e-4, "grad_norm": 0.1,
             "timestep_mean": 500.0, "epoch": 0.1},
        ])
        rows = metrics_repo.get_loss_curve("j2")
        assert len(rows) == 1
        assert rows[0]["active_layers"] is None

    def test_explicit_zero_active_layers_stays_zero(self, db_engine, metrics_repo):
        """The real "every remaining layer just froze" case must not be
        conflated with the absent-key NULL case above."""
        _insert_job(db_engine, "j3")
        metrics_repo.batch_insert("j3", [
            {"step": 1, "loss": 0.3, "lr": 1e-4, "active_layers": 0},
        ])
        rows = metrics_repo.get_loss_curve("j3")
        assert rows[0]["active_layers"] == 0

    def test_mixed_batch_preserves_per_row_value(self, db_engine, metrics_repo):
        """Mirrors the brief's illustrative batch: one row with a real
        active_layers value, one row without — the two must not bleed
        into each other within a single batch_insert call."""
        _insert_job(db_engine, "j4")
        metrics_repo.batch_insert("j4", [
            {"step": 1, "loss": 0.5, "lr": 1e-4, "grad_norm": 0.1,
             "timestep_mean": 500.0, "epoch": 0.1, "active_layers": 248},
            {"step": 2, "loss": 0.4, "lr": 1e-4, "grad_norm": 0.1,
             "timestep_mean": 500.0, "epoch": 0.1, "active_layers": None},
        ])
        rows = metrics_repo.get_loss_curve("j4")
        assert rows[0]["active_layers"] == 248
        assert rows[1]["active_layers"] is None


class TestV20MigrationIdempotence:
    def test_v20_adds_active_layers_column(self, tmp_path):
        from app.core.db import migrations

        engine = DatabaseEngine(db_path=str(tmp_path / "mig20.db"))
        with engine.write() as conn:
            migrations._migrate_v1(conn)
        with engine.write() as conn:
            migrations._migrate_v20(conn)
        with engine.connection() as conn:
            cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(step_metrics)").fetchall()
            }
        assert "active_layers" in cols
        engine.close()

    def test_v20_is_idempotent(self, tmp_path):
        from app.core.db import migrations

        engine = DatabaseEngine(db_path=str(tmp_path / "mig20b.db"))
        with engine.write() as conn:
            migrations._migrate_v1(conn)
        with engine.write() as conn:
            migrations._migrate_v20(conn)
        # Second run against an already-migrated schema must not raise.
        with engine.write() as conn:
            migrations._migrate_v20(conn)
        engine.close()
