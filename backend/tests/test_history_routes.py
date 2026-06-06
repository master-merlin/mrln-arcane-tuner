"""
E2E tests for api/training/history_routes.py — job history, checkpoints, samples, metrics, rerun.
"""

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


@patch(_SAMPLE_REPO)
@patch(_CP_REPO)
@patch(_JOB_REPO)
def test_get_job_history_detail_found(MockJobRepo, MockCpRepo, MockSampleRepo, client):
    mock_job = {"id": "job-1", "status": "completed"}
    MockJobRepo.return_value.get_by_id.return_value = mock_job
    MockJobRepo.return_value.get_datasets_for_job.return_value = []
    MockCpRepo.return_value.list_by_job.return_value = []
    MockSampleRepo.return_value.list_by_job.return_value = []
    response = client.get("/api/jobs/history/job-1")
    assert response.status_code == 200
    assert response.json()["id"] == "job-1"


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


@patch("app.core.dataset_manager.dataset_manager")
def test_get_dataset_jobs_not_found(mock_dm, client):
    mock_dm.get_dataset.return_value = None
    response = client.get("/api/datasets/ghost/jobs")
    assert response.status_code == 404
