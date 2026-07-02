"""
E2E tests for api/training/history_routes.py — job history, checkpoints, samples, metrics, rerun.
"""

import contextlib
from unittest.mock import MagicMock, patch


_HIST_MODULE = "app.api.training.history_routes"
_JOB_REPO = "app.core.db.repositories.job_repo.JobHistoryRepository"
_CP_REPO = "app.core.db.repositories.checkpoint_repo.CheckpointRepository"
_SAMPLE_REPO = "app.core.db.repositories.sample_repo.SampleImageRepository"
_METRICS_REPO = "app.core.db.repositories.metrics_repo.MetricsRepository"


@patch(_JOB_REPO)
def test_list_job_history(MockRepo, client):
    MockRepo.return_value.list_recent.return_value = []
    response = client.get("/api/jobs/history")
    assert response.status_code == 200
    assert response.json() == []


@patch(_JOB_REPO)
def test_list_job_history_full_payload(MockRepo, client):
    """P3c pin: the open JobHistoryRow model (extra=allow) must not strip or
    add keys to a realistic row — including a nested `config` blob and
    columns added by later ALTER TABLE migrations (project_id, pid, etc.)."""
    row = {
        "id": "job-1", "lora_name": "my_lora", "definition_id": "flux1-schnell",
        "status": "completed", "config": {"lr": 1e-4, "steps": 100},
        "created_at": 100.0, "started_at": 101.0, "finished_at": 200.0,
        "duration_seconds": 99.0, "training_seconds": 90.0,
        "avg_loss": 0.3, "min_loss": 0.1, "tags": ["a", "b"],
        "datasets_used": ["ds1"], "project_id": None, "pid": None,
        "completed_epochs": 2.5, "priority": 0, "ema_enabled": False,
    }
    MockRepo.return_value.list_recent.return_value = [row]
    response = client.get("/api/jobs/history")
    assert response.status_code == 200
    assert response.json() == [row]


@patch(_SAMPLE_REPO)
@patch(_CP_REPO)
@patch(_JOB_REPO)
def test_get_job_history_detail_found(MockJobRepo, MockCpRepo, MockSampleRepo, client):
    # Realistic job_history row (lora_name/definition_id/created_at are
    # NOT-NULL columns the JobHistoryRow response_model requires).
    mock_job = {
        "id": "job-1", "lora_name": "my_lora", "definition_id": "flux1-schnell",
        "status": "completed", "created_at": 0.0,
    }
    MockJobRepo.return_value.get_by_id.return_value = mock_job
    MockJobRepo.return_value.get_datasets_for_job.return_value = []
    MockCpRepo.return_value.list_by_job.return_value = []
    MockSampleRepo.return_value.list_by_job.return_value = []
    response = client.get("/api/jobs/history/job-1")
    assert response.status_code == 200
    assert response.json()["id"] == "job-1"


@patch(_SAMPLE_REPO)
@patch(_CP_REPO)
@patch(_JOB_REPO)
def test_get_job_history_detail_full_payload(
    MockJobRepo, MockCpRepo, MockSampleRepo, client,
):
    """P3c pin: detail response = full job row (open, extra=allow) +
    typed checkpoints/samples + open datasets_linkage rows, byte for byte."""
    job_row = {
        "id": "job-1", "lora_name": "my_lora", "definition_id": "flux1-schnell",
        "status": "completed", "config": {"lr": 1e-4}, "created_at": 100.0,
        "output_dir": "/runs/job-1",
    }
    checkpoint = {
        "id": 1, "job_id": "job-1", "step": 100,
        "path": "/runs/job-1/lora_000100.safetensors", "created_at": 0.0,
    }
    sample = {
        "id": 1, "job_id": "job-1", "step": 100, "path": "/runs/job-1/s1.png",
        "created_at": 0.0,
    }
    linkage = {"job_id": "job-1", "dataset_id": "ds-1", "dataset_name": "myds",
               "dataset_version": "1.0.0", "num_repeats": 1,
               "masking_enabled": 0, "caption_dropout": 0.0}
    MockJobRepo.return_value.get_by_id.return_value = job_row
    MockJobRepo.return_value.get_datasets_for_job.return_value = [linkage]
    MockCpRepo.return_value.list_by_job.return_value = [checkpoint]
    MockSampleRepo.return_value.list_by_job.return_value = [sample]

    response = client.get("/api/jobs/history/job-1")
    assert response.status_code == 200
    # Checkpoint/SampleImage are pre-existing typed models (already used by
    # GET .../checkpoints and .../samples) — their optional fields serialize
    # with defaults filled in, so the expected shape includes them explicitly.
    expected_checkpoint = {
        **checkpoint, "lora_file": None, "lora_size_bytes": None,
        "loss_at_step": None, "lr_at_step": None, "is_final": False,
        "is_deleted": False,
    }
    expected_sample = {
        **sample, "prompt": "", "seed": None, "width": 0, "height": 0,
    }
    assert response.json() == {
        **job_row,
        "checkpoints": [expected_checkpoint],
        "samples": [expected_sample],
        "datasets_linkage": [linkage],
    }


@patch(_JOB_REPO)
def test_get_job_history_detail_not_found(MockRepo, client):
    MockRepo.return_value.get_by_id.return_value = None
    response = client.get("/api/jobs/history/ghost")
    assert response.status_code == 404


@patch(_CP_REPO)
def test_get_job_checkpoints(MockRepo, client):
    # Realistic checkpoint row (the response_model declares the NOT-NULL columns
    # id/job_id/step/path/created_at as required).
    MockRepo.return_value.list_by_job.return_value = [{
        "id": 1, "job_id": "job-1", "step": 100,
        "path": "/runs/job-1/lora_000100.safetensors", "created_at": 0.0,
    }]
    response = client.get("/api/jobs/history/job-1/checkpoints")
    assert response.status_code == 200
    assert len(response.json()) == 1


@patch(_SAMPLE_REPO)
def test_get_job_samples(MockRepo, client):
    MockRepo.return_value.list_by_job.return_value = []
    response = client.get("/api/jobs/history/job-1/samples")
    assert response.status_code == 200


@patch(_METRICS_REPO)
def test_get_job_metrics(MockRepo, client):
    MockRepo.return_value.get_loss_curve.return_value = []
    MockRepo.return_value.get_summary.return_value = {}
    response = client.get("/api/jobs/history/job-1/metrics")
    assert response.status_code == 200
    assert "curve" in response.json()
    assert "summary" in response.json()


@patch(_JOB_REPO)
def test_get_job_replay_not_found(MockRepo, client):
    MockRepo.return_value.get_by_id.return_value = None
    response = client.get("/api/jobs/history/ghost/replay")
    assert response.status_code == 404


@patch(_JOB_REPO)
def test_get_job_replay_from_disk(MockRepo, client, tmp_path):
    import json

    history = [{"step": 1, "loss": 0.9, "lr": 1e-4}, {"step": 2, "loss": 0.8, "lr": 1e-4}]
    (tmp_path / "loss_history.json").write_text(json.dumps(history), encoding="utf-8")
    MockRepo.return_value.get_by_id.return_value = {"id": "job-1", "output_dir": str(tmp_path)}

    response = client.get("/api/jobs/history/job-1/replay")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["source"] == "disk"
    assert body["loss"] == history


@patch(_METRICS_REPO)
@patch(_JOB_REPO)
def test_get_job_replay_db_fallback(MockJobRepo, MockMetricsRepo, client, tmp_path):
    # Output dir exists but has no loss_history.json -> fall back to DB curve.
    MockJobRepo.return_value.get_by_id.return_value = {"id": "job-1", "output_dir": str(tmp_path)}
    curve = [{"step": 1, "loss": 0.5, "lr": 1e-4}]
    MockMetricsRepo.return_value.get_loss_curve.return_value = curve

    response = client.get("/api/jobs/history/job-1/replay")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["source"] == "db"
    assert body["loss"] == curve


@patch(_METRICS_REPO)
@patch(_JOB_REPO)
def test_get_job_replay_no_data(MockJobRepo, MockMetricsRepo, client):
    # Missing output dir + empty DB curve -> available False, source none.
    MockJobRepo.return_value.get_by_id.return_value = {"id": "job-1", "output_dir": None}
    MockMetricsRepo.return_value.get_loss_curve.return_value = []

    response = client.get("/api/jobs/history/job-1/replay")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["source"] == "none"
    assert body["loss"] == []


@patch(_JOB_REPO)
def test_get_rerun_config_found(MockRepo, client):
    MockRepo.return_value.get_config_for_rerun.return_value = {"lr": 1e-4}
    response = client.get("/api/jobs/history/job-1/rerun-config")
    assert response.status_code == 200
    assert response.json()["lr"] == 1e-4


@patch(_JOB_REPO)
def test_get_rerun_config_full_payload(MockRepo, client):
    """P3c pin: dict[str, Any] passthrough must not drop/coerce a nested,
    plugin-schema-driven training config."""
    config = {
        "lr": 1e-4, "steps": 100, "nested": {"lora_rank": 16},
        "tags": ["a", "b"], "enabled": False, "note": None,
    }
    MockRepo.return_value.get_config_for_rerun.return_value = config
    response = client.get("/api/jobs/history/job-1/rerun-config")
    assert response.status_code == 200
    assert response.json() == config


@patch(_JOB_REPO)
def test_get_rerun_config_not_found(MockRepo, client):
    MockRepo.return_value.get_config_for_rerun.return_value = None
    response = client.get("/api/jobs/history/ghost/rerun-config")
    assert response.status_code == 404


@patch(_JOB_REPO)
@patch("app.core.dataset_manager.dataset_manager")
def test_get_dataset_jobs_found(mock_dm, MockRepo, client):
    mock_ds = MagicMock()
    mock_ds.id = "ds-id"
    mock_dm.get_dataset.return_value = mock_ds
    MockRepo.return_value.get_by_dataset.return_value = []
    response = client.get("/api/datasets/myds/jobs")
    assert response.status_code == 200


@patch(_JOB_REPO)
@patch("app.core.dataset_manager.dataset_manager")
def test_get_dataset_jobs_full_payload(mock_dm, MockRepo, client):
    """P3c pin: reuses JobHistoryRow (open model) — full row survives."""
    mock_ds = MagicMock()
    mock_ds.id = "ds-id"
    mock_dm.get_dataset.return_value = mock_ds
    row = {
        "id": "job-1", "lora_name": "my_lora", "definition_id": "flux1-schnell",
        "status": "completed", "config": {"lr": 1e-4}, "created_at": 100.0,
    }
    MockRepo.return_value.get_by_dataset.return_value = [row]
    response = client.get("/api/datasets/myds/jobs")
    assert response.status_code == 200
    assert response.json() == [row]


@patch("app.core.dataset_manager.dataset_manager")
def test_get_dataset_jobs_not_found(mock_dm, client):
    mock_dm.get_dataset.return_value = None
    response = client.get("/api/datasets/ghost/jobs")
    assert response.status_code == 404


# ── GET /jobs/history/stats — read-only + byte-identical payload ─────────


@contextlib.contextmanager
def _isolated_db(tmp_path):
    """Swap the DatabaseEngine singleton for a throwaway DB so the stats
    aggregates are computed against a known seed, not whatever other tests
    left in the shared session DB."""
    from app.core.db.engine import DatabaseEngine

    prev = DatabaseEngine._instance
    eng = DatabaseEngine(db_path=str(tmp_path / "stats.db"))
    eng.initialize()
    DatabaseEngine._instance = eng
    try:
        yield eng
    finally:
        eng.close()
        DatabaseEngine._instance = prev


def _seed_stats_db(eng) -> None:
    jobs = [
        # id, lora, def_id, status, created_at, steps, dur, train, avg_loss,
        # min_loss, step_time, optimizer
        ("jA", "a", "flux", "completed", 100.0, 100, 60.0, 50.0, 0.2, 0.1, 0.5, "adamw"),
        ("jB", "b", "flux", "completed", 200.0, 200, 120.0, 100.0, 0.4, 0.2, 0.7, "adamw"),
        ("jC", "c", "sdxl", "failed", 300.0, 0, None, None, None, None, None, None),
    ]
    with eng.write() as conn:
        for j in jobs:
            conn.execute(
                "INSERT INTO job_history "
                "(id, lora_name, definition_id, status, config, created_at, "
                " completed_steps, duration_seconds, training_seconds, avg_loss, "
                " min_loss, avg_step_time, optimizer_type) "
                "VALUES (?, ?, ?, ?, '{}', ?, ?, ?, ?, ?, ?, ?, ?)",
                j,
            )
        conn.execute(
            "INSERT INTO job_datasets (job_id, dataset_name) VALUES ('jA', 'ds1')"
        )
        conn.execute(
            "INSERT INTO job_datasets (job_id, dataset_name) VALUES ('jB', 'ds2')"
        )


_EXPECTED_STATS = {
    "total_jobs": 3,
    "completed": 2,
    "failed": 1,
    "stopped": 0,
    "running": 0,
    "paused": 0,
    "success_rate": 66.7,
    "total_steps": 300,
    "total_runtime_sec": 180.0,
    "total_training_sec": 150.0,
    "avg_steps": 150,
    "avg_loss": 0.3,
    "avg_min_loss": 0.15,
    "avg_step_time_sec": 0.6,
    "avg_runtime_sec": 90.0,
    "model_families": [
        {"id": "flux", "count": 2},
        {"id": "sdxl", "count": 1},
    ],
    "optimizers": [{"name": "adamw", "count": 2}],
    "unique_datasets": 2,
    "last_job": {
        "lora_name": "c",
        "definition_id": "sdxl",
        "status": "failed",
        "created_at": 300.0,
    },
}


def test_stats_payload_byte_identical(client, tmp_path):
    """Pin the exact stats payload (keys + aggregation semantics) the
    frontend consumes; must survive the repo-extraction refactor."""
    with _isolated_db(tmp_path) as eng:
        _seed_stats_db(eng)
        response = client.get("/api/jobs/history/stats")
    assert response.status_code == 200
    assert response.json() == _EXPECTED_STATS


def test_stats_get_is_write_free(client, tmp_path):
    """GET /jobs/history/stats must never open a write transaction."""
    from app.core.db.engine import DatabaseEngine

    with _isolated_db(tmp_path) as eng:
        _seed_stats_db(eng)
        with patch.object(
            DatabaseEngine,
            "write",
            side_effect=AssertionError("GET /jobs/history/stats must not write"),
        ):
            response = client.get("/api/jobs/history/stats")
    assert response.status_code == 200
    assert response.json() == _EXPECTED_STATS
