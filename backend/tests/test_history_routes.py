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
        # min_loss, step_time, optimizer, rank, lora_bytes, resumed_from
        ("jA", "a", "flux", "completed", 100.0, 100, 60.0, 50.0, 0.2, 0.1, 0.5, "adamw", 16, 1000, None),
        ("jB", "b", "flux", "completed", 200.0, 200, 120.0, 100.0, 0.4, 0.2, 0.7, "adamw", 32, 2000, "jA"),
        ("jC", "c", "sdxl", "failed", 300.0, 0, None, None, None, None, None, None, None, None, None),
    ]
    with eng.write() as conn:
        for j in jobs:
            conn.execute(
                "INSERT INTO job_history "
                "(id, lora_name, definition_id, status, config, created_at, "
                " completed_steps, duration_seconds, training_seconds, avg_loss, "
                " min_loss, avg_step_time, optimizer_type, network_rank, "
                " final_lora_size_bytes, resumed_from) "
                "VALUES (?, ?, ?, ?, '{}', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                j,
            )
        conn.execute(
            "INSERT INTO job_datasets (job_id, dataset_name) VALUES ('jA', 'ds1')"
        )
        conn.execute(
            "INSERT INTO job_datasets (job_id, dataset_name) VALUES ('jB', 'ds2')"
        )
        conn.execute(
            "INSERT INTO checkpoints (job_id, step, path, created_at, is_final, is_deleted) "
            "VALUES ('jA', 50, 'x/ckpt-50', 0, 0, 0)"
        )
        conn.execute(
            "INSERT INTO checkpoints (job_id, step, path, created_at, is_final, is_deleted) "
            "VALUES ('jA', 60, 'x/ckpt-60', 0, 0, 1)"
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
    "optimizers": [{"name": "adamw", "count": 2}],
    "unique_datasets": 2,
    "last_job": {
        "lora_name": "c",
        "definition_id": "sdxl",
        "status": "failed",
        "created_at": 300.0,
    },
    "activity": [
        {"week_start": "1969-12-29", "completed": 2, "failed": 1, "stopped": 0, "other": 0},
    ],
    "gpu_hours": 0.04,
    "overhead_pct": 16.7,
    "lora_count": 2,
    "lora_bytes": 3000,
    "lora_on_disk": 0,
    "lora_size_known": 2,
    "checkpoint_count": 1,
    "families": [
        {"id": "flux", "count": 2, "completed": 2, "success_rate": 100.0,
         "avg_step_time": 0.6, "best_loss": 0.1},
        {"id": "sdxl", "count": 1, "completed": 0, "success_rate": 0.0,
         "avg_step_time": None, "best_loss": None},
    ],
    "loss_histogram": {
        "edges": [0.1, 0.108333, 0.116667, 0.125, 0.133333, 0.141667, 0.15,
                  0.158333, 0.166667, 0.175, 0.183333, 0.191667, 0.2],
        "counts": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    },
    "hyperparams": {
        "optimizer_type": [{"value": "adamw", "count": 2}],
        "network_rank": [{"value": "16", "count": 1}, {"value": "32", "count": 1}],
        "lr_scheduler": [],
        "timestep_sampling": [],
        "quantization": [],
        "mixed_precision": [],
        "ema_enabled": [{"value": "off", "count": 3}],
        "batch_size": [],
    },
    "resume_rate": 33.3,
    "top_datasets": [{"name": "ds1", "count": 1}, {"name": "ds2", "count": 1}],
    "records": {
        "longest_run": {"job_id": "jB", "lora_name": "b", "definition_id": "flux", "value": 120.0},
        "most_steps": {"job_id": "jB", "lora_name": "b", "definition_id": "flux", "value": 200.0},
        "best_loss": {"job_id": "jA", "lora_name": "a", "definition_id": "flux", "value": 0.1},
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


def test_stats_histogram_degenerate_cases(client, tmp_path):
    """< 2 loss values → empty histogram; identical values → single bin."""
    from app.core.db.repositories.job_repo import _histogram

    assert _histogram([], bins=12) == {"edges": [], "counts": []}
    assert _histogram([0.5], bins=12) == {"edges": [], "counts": []}
    assert _histogram([0.5, 0.5, 0.5], bins=12) == {"edges": [0.5, 0.5], "counts": [3]}


def test_stats_zero_jobs_shape(client, tmp_path):
    """Empty DB returns the full shape with zeros/empties — never 404/500."""
    with _isolated_db(tmp_path):
        response = client.get("/api/jobs/history/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["total_jobs"] == 0
    assert body["activity"] == []
    assert body["families"] == []
    assert body["loss_histogram"] == {"edges": [], "counts": []}
    assert body["records"] == {"longest_run": None, "most_steps": None, "best_loss": None}
    assert body["gpu_hours"] == 0.0
    assert body["overhead_pct"] == 0.0
    assert body["resume_rate"] == 0.0


def test_stats_project_filter_new_fields(client, tmp_path):
    """The Task-1 filter also narrows the new aggregates."""
    with _isolated_db(tmp_path) as eng:
        _seed_stats_db(eng)
        _tag_projects(eng)
        response = client.get("/api/jobs/history/stats?project_id=p1")
    body = response.json()
    assert body["lora_count"] == 1            # jA only
    assert body["checkpoint_count"] == 1      # jA's live ckpt
    assert body["families"] == [
        {"id": "flux", "count": 1, "completed": 1, "success_rate": 100.0,
         "avg_step_time": 0.5, "best_loss": 0.1},
        {"id": "sdxl", "count": 1, "completed": 0, "success_rate": 0.0,
         "avg_step_time": None, "best_loss": None},
    ]
    assert body["resume_rate"] == 0.0         # jB (the resumed one) is in p2
    assert body["top_datasets"] == [{"name": "ds1", "count": 1}]


def _tag_projects(eng) -> None:
    """Assign seeded jobs to projects (jA, jC → p1; jB → p2)."""
    with eng.write() as conn:
        for pid, name in (("p1", "P1"), ("p2", "P2")):
            conn.execute(
                "INSERT INTO projects (id, name, created_at, updated_at) "
                "VALUES (?, ?, 0, 0)",
                (pid, name),
            )
        conn.execute("UPDATE job_history SET project_id = 'p1' WHERE id IN ('jA','jC')")
        conn.execute("UPDATE job_history SET project_id = 'p2' WHERE id = 'jB'")


def test_stats_project_filter_narrows_every_aggregate(client, tmp_path):
    with _isolated_db(tmp_path) as eng:
        _seed_stats_db(eng)
        _tag_projects(eng)
        response = client.get("/api/jobs/history/stats?project_id=p1")
    assert response.status_code == 200
    body = response.json()
    assert body["total_jobs"] == 2          # jA + jC
    assert body["completed"] == 1           # jA only
    assert body["failed"] == 1              # jC
    assert body["success_rate"] == 50.0
    assert body["total_steps"] == 100       # jA
    assert body["unique_datasets"] == 1     # ds1 (jA); ds2 belongs to p2's jB
    assert body["optimizers"] == [{"name": "adamw", "count": 1}]
    assert body["last_job"]["lora_name"] == "c"  # jC is newest in p1


def test_stats_project_filter_all_is_global(client, tmp_path):
    with _isolated_db(tmp_path) as eng:
        _seed_stats_db(eng)
        _tag_projects(eng)
        response = client.get("/api/jobs/history/stats?project_id=all")
    assert response.status_code == 200
    assert response.json()["total_jobs"] == 3


def test_stats_lora_semantics(client, tmp_path):
    """lora_count = completed runs (produced), lora_on_disk = final files
    verified present, lora_size_known = byte-sum coverage.

    lora_on_disk is persisted at write time (run completion / the backfill
    reconcile), NOT probed live by get_stats — so the seed here sets it
    explicitly, exactly as those write paths would have."""
    with _isolated_db(tmp_path) as eng:
        _seed_stats_db(eng)
        real = tmp_path / "a_final.safetensors"
        real.write_bytes(b"x" * 10)
        with eng.write() as conn:
            conn.execute(
                "UPDATE job_history SET final_lora_file = ?, lora_on_disk = 1 "
                "WHERE id = 'jA'",
                (str(real),),
            )
            conn.execute(
                "UPDATE job_history SET final_lora_file = ?, lora_on_disk = 0 "
                "WHERE id = 'jB'",
                (str(tmp_path / "deleted.safetensors"),),
            )
            # Completed run with NO recorded size/file still counts as produced.
            conn.execute(
                "INSERT INTO job_history "
                "(id, lora_name, definition_id, status, config, created_at) "
                "VALUES ('jD', 'd', 'flux', 'completed', '{}', 400.0)"
            )
        body = client.get("/api/jobs/history/stats").json()
    assert body["lora_count"] == 3       # completed runs, sized or not
    assert body["lora_on_disk"] == 1     # only jA's persisted flag is 1
    assert body["lora_size_known"] == 2  # jA + jB carry sizes
    assert body["lora_bytes"] == 3000


def test_stats_get_never_touches_filesystem(client, tmp_path):
    """GET /jobs/history/stats reads lora_on_disk from the DB — it must NOT
    probe the filesystem at all. Poisoning os.path.isfile to raise proves
    the per-request sweep this task retired is really gone."""
    with _isolated_db(tmp_path) as eng:
        _seed_stats_db(eng)
        with eng.write() as conn:
            conn.execute(
                "UPDATE job_history SET final_lora_file = ?, lora_on_disk = 1 "
                "WHERE id = 'jA'",
                (str(tmp_path / "a_final.safetensors"),),
            )
        with patch(
            "os.path.isfile",
            side_effect=AssertionError(
                "GET /jobs/history/stats must not touch the filesystem"
            ),
        ):
            response = client.get("/api/jobs/history/stats")
    assert response.status_code == 200
    assert response.json()["lora_on_disk"] == 1
