from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Plugin Routes ────────────────────────────────────────────────────────


@patch("app.api.training.plugin_routes.plugin_manager")
def test_list_plugins(mock_plugin_manager, client):
    mock_plugin_manager.list_plugins.return_value = [{"id": "std", "name": "Standard"}]
    response = client.get("/api/plugins")
    assert response.status_code == 200
    assert response.json() == [{"id": "std", "name": "Standard"}]


@patch("app.api.training.plugin_routes.plugin_manager")
@patch("app.api.training.plugin_routes.asyncio.to_thread")
def test_get_plugin_schema_success(mock_to_thread, mock_pm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_plugin = MagicMock()
    mock_config_schema = MagicMock()
    mock_config_schema.model_json_schema.return_value = {"type": "object", "properties": {}}
    mock_plugin.get_config_schema.return_value = mock_config_schema
    mock_plugin.enrich_schema.return_value = {"type": "object", "properties": {}}
    mock_pm.get_plugin.return_value = mock_plugin
    response = client.get("/api/plugins/std/schema")
    assert response.status_code == 200
    assert "type" in response.json()
    # Global scope → enrich_schema called with no project (None) so all datasets show.
    args, kwargs = mock_plugin.enrich_schema.call_args
    assert (len(args) > 1 and args[1] is None) or kwargs.get("project_id") is None


@patch("app.api.training.plugin_routes.plugin_manager")
@patch("app.api.training.plugin_routes.asyncio.to_thread")
def test_get_plugin_schema_passes_project_id(mock_to_thread, mock_pm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_plugin = MagicMock()
    mock_config_schema = MagicMock()
    mock_config_schema.model_json_schema.return_value = {"type": "object", "properties": {}}
    mock_plugin.get_config_schema.return_value = mock_config_schema
    mock_plugin.enrich_schema.return_value = {"type": "object", "properties": {}}
    mock_pm.get_plugin.return_value = mock_plugin
    response = client.get("/api/plugins/std/schema?project_id=proj-1")
    assert response.status_code == 200
    # The project scope MUST reach enrich_schema so the dataset_name enum is
    # filtered to the project's datasets (the dropdown-not-scoped bug).
    args, kwargs = mock_plugin.enrich_schema.call_args
    assert "proj-1" in args or kwargs.get("project_id") == "proj-1"


@patch("app.api.training.plugin_routes.plugin_manager")
@patch("app.api.training.plugin_routes.asyncio.to_thread")
def test_get_plugin_schema_not_found(mock_to_thread, mock_pm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_pm.get_plugin.return_value = None
    response = client.get("/api/plugins/nonexistent/schema")
    assert response.status_code == 404


# ── Checkpoint Routes ────────────────────────────────────────────────────


@patch("app.api.training.checkpoint_routes.inspect_checkpoint")
@patch("app.api.training.checkpoint_routes.asyncio.to_thread")
def test_inspect_checkpoint_success(mock_to_thread, mock_inspect, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_inspect.return_value = {"status": "ok", "components": []}
    import app.api.training.checkpoint_routes as ckpt_mod
    orig = ckpt_mod._ALLOWED_ROOTS
    ckpt_mod._ALLOWED_ROOTS = [Path("/").resolve()]
    try:
        with patch.object(Path, 'exists', return_value=True):
            response = client.get("/api/checkpoints/inspect?path=/some/checkpoint")
    finally:
        ckpt_mod._ALLOWED_ROOTS = orig
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── Model Definition Routes ─────────────────────────────────────────────


@patch("app.engine.models.registry.ModelRegistry._definitions", new_callable=lambda: property(lambda self: {}))
def test_list_model_definitions(mock_defs, client):
    mock_def = MagicMock()
    mock_def.model_dump.return_value = {"id": "sdxl", "family": "sdxl", "name": "SDXL", "components": {}}
    with patch.dict("app.engine.models.registry.registry._definitions", {"sdxl": mock_def}, clear=True):
        response = client.get("/api/models/definitions")
    assert response.status_code == 200


@patch("app.api.training.definition_routes.asyncio.to_thread")
def test_create_definition_duplicate(mock_to_thread, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync

    with patch("app.engine.models.registry.ModelRegistry.get_definition", return_value=MagicMock()):
        response = client.post("/api/models/definitions", json={
            "id": "dup", "family": "sdxl", "name": "Dup"
        })
        assert response.status_code == 409


@patch("app.api.training.definition_routes.asyncio.to_thread")
def test_update_definition_not_found(mock_to_thread, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync

    with patch("app.engine.models.registry.ModelRegistry.get_definition", return_value=None):
        response = client.put("/api/models/definitions/ghost", json={"name": "New"})
        assert response.status_code == 404


@patch("app.api.training.definition_routes.asyncio.to_thread")
def test_delete_definition_not_found(mock_to_thread, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync

    with patch("app.engine.models.registry.ModelRegistry.get_definition", return_value=None):
        response = client.delete("/api/models/definitions/ghost")
        assert response.status_code == 404


@patch("app.api.training.definition_routes.asyncio.to_thread")
def test_delete_definition_success(mock_to_thread, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync

    mock_def = MagicMock()
    mock_path = MagicMock()
    mock_path.exists.return_value = False
    with patch("app.engine.models.registry.ModelRegistry.get_definition", return_value=mock_def), \
         patch.dict("app.engine.models.registry.registry._paths", {"to_del": "/some/path.yaml"}, clear=False), \
         patch.dict("app.engine.models.registry.registry._definitions", {"to_del": mock_def}, clear=False), \
         patch("app.api.training.definition_routes.Path", return_value=mock_path):
        response = client.delete("/api/models/definitions/to_del")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"


# ── Job Routes ───────────────────────────────────────────────────────────


@patch("app.api.training.job_routes.job_manager")
def test_list_jobs(mock_job_manager, client):
    mock_job_manager.list_jobs.return_value = []
    response = client.get("/api/jobs")
    assert response.status_code == 200
    assert response.json() == []


@patch("app.api.training.job_routes.job_manager")
def test_create_job(mock_job_manager, client):
    mock_job_manager.create_job.return_value = {
        "id": "job-123", "plugin_id": "std", "status": "pending",
        "config": {}, "created_at": 123456789.0, "logs": []
    }
    response = client.post("/api/jobs", json={"plugin_id": "std", "config": {}})
    assert response.status_code == 200
    assert response.json()["id"] == "job-123"


@patch("app.api.training.job_routes.job_manager")
def test_create_job_advances_queue(mock_job_manager, client):
    """Creating a job must trigger queue advancement so a single pending job
    auto-starts when auto-queue is on and the GPU is idle — without waiting
    for a prior job to finish (regression: only-job-in-queue stalled because
    nothing kicked the queue on creation)."""
    mock_job_manager.create_job.return_value = {
        "id": "job-123", "plugin_id": "std", "status": "pending",
        "config": {}, "created_at": 123456789.0, "logs": []
    }
    response = client.post("/api/jobs", json={"plugin_id": "std", "config": {}})
    assert response.status_code == 200
    mock_job_manager.schedule_advance_queue.assert_called_once()


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_create_job_invalid_plugin(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_jm.create_job.side_effect = ValueError("Unknown plugin")
    response = client.post("/api/jobs", json={"plugin_id": "bad", "config": {}})
    assert response.status_code == 400


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_start_job_success(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    response = client.post("/api/jobs/job-1/start")
    assert response.status_code == 200
    assert response.json()["status"] == "started"


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_start_job_bad_request(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_jm.start_job.side_effect = ValueError("Not pending")
    response = client.post("/api/jobs/job-1/start")
    assert response.status_code == 400


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_start_job_server_error(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_jm.start_job.side_effect = RuntimeError("GPU busy")
    response = client.post("/api/jobs/job-1/start")
    assert response.status_code == 500


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_stop_job_success(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    response = client.post("/api/jobs/job-1/stop")
    assert response.status_code == 200
    assert response.json()["status"] == "stopped"


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_stop_job_bad_request(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_jm.stop_job.side_effect = ValueError("Not running")
    response = client.post("/api/jobs/job-1/stop")
    assert response.status_code == 400


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_pause_job_success(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    response = client.post("/api/jobs/job-1/pause")
    assert response.status_code == 200
    assert response.json()["status"] == "paused"


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_resume_job_success(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    response = client.post("/api/jobs/job-1/resume")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_soft_stop_job_success(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    response = client.post("/api/jobs/job-1/soft-stop")
    assert response.status_code == 200
    assert response.json()["status"] == "soft_stopping"


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_restart_job_success(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    response = client.post("/api/jobs/job-1/restart")
    assert response.status_code == 200
    assert response.json()["status"] == "restarted"


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_restart_job_server_error(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_jm.restart_job.side_effect = RuntimeError("Resource lock")
    response = client.post("/api/jobs/job-1/restart")
    assert response.status_code == 500


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_resume_from_checkpoint_success(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_jm.resume_from_checkpoint.return_value = None
    mock_jm.get_job.return_value = {
        "id": "job-1", "plugin_id": "std", "status": "pending",
        "config": {}, "created_at": 1.0, "logs": [],
    }
    response = client.post(
        "/api/jobs/job-1/resume-from-checkpoint",
        json={"checkpoint_dir": "checkpoint-000500"},
    )
    assert response.status_code == 200
    assert response.json()["id"] == "job-1"
    mock_jm.resume_from_checkpoint.assert_called_once_with("job-1", "checkpoint-000500")


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_resume_from_checkpoint_bad_request(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_jm.resume_from_checkpoint.side_effect = ValueError("Checkpoint is not resumable")
    response = client.post(
        "/api/jobs/job-1/resume-from-checkpoint",
        json={"checkpoint_dir": "checkpoint-000100"},
    )
    assert response.status_code == 400


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_get_job_logs_success(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_job = MagicMock()
    mock_job.logs = ["line1", "line2"]
    mock_jm.get_job.return_value = mock_job
    response = client.get("/api/jobs/job-1/logs")
    assert response.status_code == 200
    assert response.json() == ["line1", "line2"]


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_get_job_logs_not_found(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_jm.get_job.return_value = None
    response = client.get("/api/jobs/nonexistent/logs")
    assert response.status_code == 404


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_delete_job(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    response = client.delete("/api/jobs/job-1")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"


# ── Sample Image Routes ─────────────────────────────────────────────────


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_list_job_samples_job_not_found(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_jm.get_job.return_value = None
    response = client.get("/api/jobs/ghost/samples")
    assert response.status_code == 404


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_get_sample_image_not_found(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_job = MagicMock()
    mock_job.config = {"output_dir": "/tmp/out", "lora_name": "test", "definition_id": "model/sdxl"}
    mock_jm.get_job.return_value = mock_job
    response = client.get("/api/jobs/job-1/samples/nonexistent.png")
    assert response.status_code == 404


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_list_job_samples_includes_video(mock_to_thread, mock_jm, client, tmp_path):
    """Video (.mp4) samples must be listed too — the old .png-only filter hid
    every LTX-2/WAN video sample so the gallery looked empty."""
    from app.core.naming import model_part_from_definition_id

    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync

    defn = "ltx2-3-base"
    sdir = tmp_path / f"test_{model_part_from_definition_id(defn)}" / "samples"
    sdir.mkdir(parents=True)
    (sdir / "sample_00_step000000.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    (sdir / "sample_00_step000250.png").write_bytes(b"\x89PNG\r\n")
    (sdir / "notes.txt").write_text("ignored")  # non-media is skipped

    mock_job = MagicMock()
    mock_job.config = {"output_dir": str(tmp_path), "lora_name": "test", "definition_id": defn}
    mock_jm.get_job.return_value = mock_job

    response = client.get("/api/jobs/job-1/samples")
    assert response.status_code == 200
    names = {s["filename"] for s in response.json()}
    assert names == {"sample_00_step000000.mp4", "sample_00_step000250.png"}


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_list_job_samples_final_video_sorts_first(mock_to_thread, mock_jm, client, tmp_path):
    """The end-of-run final VIDEO sample (sample_NN_final.mp4) must be listed
    and parsed as final (step 999999 → sorts to the top), same as image
    families' sample_NN_final.png."""
    from app.core.naming import model_part_from_definition_id

    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync

    defn = "ltx2-3-base"
    sdir = tmp_path / f"test_{model_part_from_definition_id(defn)}" / "samples"
    sdir.mkdir(parents=True)
    (sdir / "sample_00_step000050.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    (sdir / "sample_00_final.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")

    mock_job = MagicMock()
    mock_job.config = {"output_dir": str(tmp_path), "lora_name": "test", "definition_id": defn}
    mock_jm.get_job.return_value = mock_job

    response = client.get("/api/jobs/job-1/samples")
    assert response.status_code == 200
    samples = response.json()
    by_name = {s["filename"]: s for s in samples}
    assert "sample_00_final.mp4" in by_name
    assert by_name["sample_00_final.mp4"]["step"] == 999999  # final → sorts first
    # Sorted newest/final-first.
    assert samples[0]["filename"] == "sample_00_final.mp4"


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_get_sample_image_serves_video_content_type(mock_to_thread, mock_jm, client, tmp_path):
    """An .mp4 sample is served as video/mp4, not the old hard-coded image/png."""
    from app.core.naming import model_part_from_definition_id

    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync

    defn = "ltx2-3-base"
    sdir = tmp_path / f"test_{model_part_from_definition_id(defn)}" / "samples"
    sdir.mkdir(parents=True)
    (sdir / "sample_00_step000000.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")

    mock_job = MagicMock()
    mock_job.config = {"output_dir": str(tmp_path), "lora_name": "test", "definition_id": defn}
    mock_jm.get_job.return_value = mock_job

    response = client.get("/api/jobs/job-1/samples/sample_00_step000000.mp4")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/mp4")


# ── Resumable checkpoint .zip download ──────────────────────────────────


def _make_checkpoint_dir(run_dir, folder="checkpoint-000100"):
    ck = run_dir / folder
    (ck / "unet").mkdir(parents=True)
    (ck / "training_state.json").write_text('{"global_step": 100}')
    (ck / "optimizer.pt").write_bytes(b"x" * 32)
    (ck / "unet" / "adapter_model.safetensors").write_bytes(b"y" * 16)
    return ck


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_download_checkpoint_zip_bundles_resumable_state(mock_to_thread, mock_jm, client, tmp_path):
    import io
    import zipfile

    async def run_sync(func, *a, **k):
        return func(*a, **k)
    mock_to_thread.side_effect = run_sync
    mock_jm.get_job.return_value = MagicMock()
    _make_checkpoint_dir(tmp_path)

    with patch("app.api.training.job_routes._resolve_run_dir", return_value=tmp_path):
        resp = client.get("/api/jobs/job-1/checkpoints/checkpoint-000100/zip")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert ".zip" in resp.headers.get("content-disposition", "")
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = set(zf.namelist())
    assert "training_state.json" in names
    assert "optimizer.pt" in names
    # Nested adapter, POSIX-separated so it extracts on a Linux pod.
    assert "unet/adapter_model.safetensors" in names


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_download_checkpoint_zip_rejects_bad_folder(mock_to_thread, mock_jm, client):
    async def run_sync(func, *a, **k):
        return func(*a, **k)
    mock_to_thread.side_effect = run_sync
    mock_jm.get_job.return_value = MagicMock()
    # Not a checkpoint-NNNNNN / final folder name → 400, never touches disk.
    resp = client.get("/api/jobs/job-1/checkpoints/secrets/zip")
    assert resp.status_code == 400


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_download_checkpoint_zip_missing_dir_404(mock_to_thread, mock_jm, client, tmp_path):
    async def run_sync(func, *a, **k):
        return func(*a, **k)
    mock_to_thread.side_effect = run_sync
    mock_jm.get_job.return_value = MagicMock()
    with patch("app.api.training.job_routes._resolve_run_dir", return_value=tmp_path):
        resp = client.get("/api/jobs/job-1/checkpoints/checkpoint-000999/zip")
    assert resp.status_code == 404


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_list_checkpoints_flags_resumable(mock_to_thread, mock_jm, client, tmp_path):
    """A distribution LoRA with a matching training-state folder is resumable;
    one without (folder pruned) is not."""
    async def run_sync(func, *a, **k):
        return func(*a, **k)
    mock_to_thread.side_effect = run_sync
    mock_jm.get_job.return_value = MagicMock()
    # Step 100 has a checkpoint dir; step 200 only has the LoRA (pruned state).
    (tmp_path / "mylora_000100.safetensors").write_bytes(b"a" * 8)
    (tmp_path / "mylora_000200.safetensors").write_bytes(b"b" * 8)
    _make_checkpoint_dir(tmp_path, "checkpoint-000100")

    with patch("app.api.training.job_routes._resolve_run_dir", return_value=tmp_path):
        resp = client.get("/api/jobs/job-1/checkpoints")

    assert resp.status_code == 200
    by_step = {c["step"]: c for c in resp.json()}
    assert by_step[100]["resumable"] is True
    assert by_step[100]["checkpoint_dir"] == "checkpoint-000100"
    assert by_step[200]["resumable"] is False
    assert by_step[200]["checkpoint_dir"] is None


# ── LoRA Tooling Routes ─────────────────────────────────────────────────


@patch("app.api.training.lora_routes._check_lora_path", side_effect=lambda p: Path(p))
@patch("app.engine.utils.lora_tools.inspect_lora")
@patch("app.api.training.lora_routes.asyncio.to_thread")
def test_inspect_lora_success(mock_to_thread, mock_inspect, mock_check, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_inspect.return_value = {"rank": 16, "alpha": 16.0, "keys": 100}
    response = client.get("/api/tools/lora/inspect?path=/some/lora.safetensors")
    assert response.status_code == 200
    assert response.json()["rank"] == 16


@patch("app.api.training.lora_routes._check_lora_path", side_effect=lambda p: Path(p))
@patch("app.engine.utils.lora_tools.inspect_lora")
@patch("app.api.training.lora_routes.asyncio.to_thread")
def test_inspect_lora_not_found(mock_to_thread, mock_inspect, mock_check, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_inspect.side_effect = FileNotFoundError()
    response = client.get("/api/tools/lora/inspect?path=/missing.safetensors")
    assert response.status_code == 404


def test_inspect_lora_outside_allowed_lists_allowed_roots(client):
    # A path outside every allowed root is rejected (correct) AND the error must
    # name the allowed directories so the user can move the file to fix it.
    from app.api.training import lora_routes
    outside = "/definitely/not/allowed/lora.safetensors"
    response = client.get(f"/api/tools/lora/inspect?path={outside}")
    assert response.status_code == 403
    detail = response.json()["detail"]
    assert "Allowed:" in detail
    for root in lora_routes._ALLOWED_ROOTS:
        assert str(root) in detail


@patch("app.api.training.lora_routes._check_lora_path", side_effect=lambda p: Path(p))
@patch("app.engine.utils.lora_tools.resize_lora")
@patch("app.api.training.lora_routes.asyncio.to_thread")
def test_resize_lora_success(mock_to_thread, mock_resize, mock_check, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_resize.return_value = {"status": "ok", "new_rank": 8}
    response = client.post("/api/tools/lora/resize", json={
        "input_path": "/a.safetensors",
        "output_path": "/b.safetensors",
        "new_rank": 8
    })
    assert response.status_code == 200
    assert response.json()["new_rank"] == 8


@patch("app.api.training.lora_routes._check_lora_path", side_effect=lambda p: Path(p))
@patch("app.engine.utils.lora_tools.resize_lora")
@patch("app.api.training.lora_routes.asyncio.to_thread")
def test_resize_lora_not_found(mock_to_thread, mock_resize, mock_check, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_resize.side_effect = FileNotFoundError()
    response = client.post("/api/tools/lora/resize", json={
        "input_path": "/missing.safetensors",
        "output_path": "/b.safetensors",
        "new_rank": 8
    })
    assert response.status_code == 404


@patch("app.api.training.lora_routes._check_lora_path", side_effect=lambda p: Path(p))
@patch("app.engine.utils.lora_tools.resize_lora")
@patch("app.api.training.lora_routes.asyncio.to_thread")
def test_resize_lora_bad_value(mock_to_thread, mock_resize, mock_check, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_resize.side_effect = ValueError("Rank must be > 0")
    response = client.post("/api/tools/lora/resize", json={
        "input_path": "/a.safetensors",
        "output_path": "/b.safetensors",
        "new_rank": -1
    })
    assert response.status_code == 400


@patch("app.api.training.lora_routes._check_lora_path", side_effect=lambda p: Path(p))
@patch("app.engine.utils.lora_tools.resize_lora")
@patch("app.api.training.lora_routes.asyncio.to_thread")
def test_resize_lora_server_error(mock_to_thread, mock_resize, mock_check, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_resize.side_effect = RuntimeError("OOM")
    response = client.post("/api/tools/lora/resize", json={
        "input_path": "/a.safetensors",
        "output_path": "/b.safetensors",
        "new_rank": 8
    })
    assert response.status_code == 500


# ── Filesystem Browse ────────────────────────────────────────────────────


def test_browse_filesystem_success(client, tmp_path):
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "child").mkdir()
    import app.api.filesystem_routes as fs_mod
    orig = fs_mod._ALLOWED_ROOTS
    fs_mod._ALLOWED_ROOTS = [tmp_path]
    try:
        response = client.get(f"/api/filesystem/browse?path={tmp_path}")
    finally:
        fs_mod._ALLOWED_ROOTS = orig
    assert response.status_code == 200
    assert len(response.json()["entries"]) >= 1


def test_browse_filesystem_not_found(client):
    import app.api.filesystem_routes as fs_mod
    orig = fs_mod._ALLOWED_ROOTS
    fs_mod._ALLOWED_ROOTS = [Path("/").resolve()]
    try:
        response = client.get("/api/filesystem/browse?path=/some/nonexistent/dir")
    finally:
        fs_mod._ALLOWED_ROOTS = orig
    assert response.status_code == 404


def test_browse_filesystem_checkpoint_detection(client, tmp_path):
    ckpt = tmp_path / "ckpt_dir"
    ckpt.mkdir()
    (ckpt / "training_state.json").write_text("{}")
    import app.api.filesystem_routes as fs_mod
    orig = fs_mod._ALLOWED_ROOTS
    fs_mod._ALLOWED_ROOTS = [tmp_path]
    try:
        response = client.get(f"/api/filesystem/browse?path={tmp_path}")
    finally:
        fs_mod._ALLOWED_ROOTS = orig
    entries = response.json()["entries"]
    ckpt_entry = next(e for e in entries if e["name"] == "ckpt_dir")
    assert ckpt_entry["type"] == "checkpoint"


# ── Sampling Control ────────────────────────────────────────────────────


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_pause_sampling(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    response = client.post("/api/jobs/job-1/pause-sampling")
    assert response.status_code == 200
    assert response.json()["status"] == "sampling_paused"


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_resume_sampling(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    response = client.post("/api/jobs/job-1/resume-sampling")
    assert response.status_code == 200
    assert response.json()["status"] == "sampling_resumed"


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_get_sampling_status(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_jm.is_sampling_paused.return_value = True
    response = client.get("/api/jobs/job-1/sampling-status")
    assert response.status_code == 200
    assert response.json()["sampling_paused"] is True


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_set_sampling_cadence(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    response = client.post("/api/jobs/job-1/sampling-cadence", json={"interval": 50})
    assert response.status_code == 200
    assert response.json()["status"] == "cadence_set"


@patch("app.api.training.job_routes.job_manager")
@patch("app.api.training.job_routes.asyncio.to_thread")
def test_get_sampling_cadence(mock_to_thread, mock_jm, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_job = MagicMock()
    mock_job.config = {"sample_every_n_steps": 100}
    mock_jm.get_job.return_value = mock_job
    mock_jm.get_sampling_cadence.return_value = 50
    response = client.get("/api/jobs/job-1/sampling-cadence")
    assert response.status_code == 200
    assert "interval" in response.json()

