"""Job-history writer tests for PipelineTrainMixin.

The live pipeline must (a) read the REAL ema config key and (b) persist LoRA
artifact fields (file + size) at checkpoint- and completion-time —
historically only the disk backfill wrote those (2026-07-16 stats rework).
"""
from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.engine.core.pipeline.pipeline_train import PipelineTrainMixin


def _pipeline_skeleton(tmp_path: Path) -> PipelineTrainMixin:
    """Bare mixin instance with just the attrs the job-history helpers read."""
    p = object.__new__(PipelineTrainMixin)
    p._job_history_id = "job-1"
    p.checkpoint_manager = types.SimpleNamespace(
        output_dir=str(tmp_path), last_lora_path=None
    )
    p.logger_component = types.SimpleNamespace(
        _loss_history=[{"loss": 0.1, "lr": 1e-4}],
        get_total_elapsed=lambda: 100.0,
        _total_save_time=10.0,
        avg_save_time=2.0,
    )
    p.config = {"definition_id": "flux"}
    p._steps_per_epoch = 0
    # Instance attrs shadow the methods — keeps the skeleton disk-free.
    p._measure_run_disk_bytes = lambda: None
    p._write_vram_measured = lambda _m: None
    return p


def test_init_job_history_reads_real_ema_key(tmp_path):
    """ema_enabled must come from config['ema'] (the real key), not 'use_ema'."""
    p = _pipeline_skeleton(tmp_path)
    p.config = {"job_id": "job-1", "ema": True, "definition_id": "flux"}
    with patch(
        "app.core.db.DatabaseEngine.get_instance", return_value=MagicMock()
    ), patch(
        "app.core.db.repositories.job_repo.JobHistoryRepository.update_status"
    ) as upd:
        p._init_job_history(max_steps=10, grad_accum=1)
    assert upd.call_args.kwargs["ema_enabled"] is True


def test_record_checkpoint_persists_lora_file_and_size(tmp_path):
    p = _pipeline_skeleton(tmp_path)
    lora = tmp_path / "x_000100.safetensors"
    lora.write_bytes(b"y" * 123)
    p.checkpoint_manager.last_lora_path = str(lora)
    with patch(
        "app.core.db.repositories.checkpoint_repo.CheckpointRepository.add"
    ) as add:
        p._record_checkpoint(step=100)
    data = add.call_args[0][0]
    assert data["lora_file"] == str(lora)
    assert data["lora_size_bytes"] == 123


def test_record_checkpoint_without_lora_records_nulls(tmp_path):
    p = _pipeline_skeleton(tmp_path)  # last_lora_path stays None
    with patch(
        "app.core.db.repositories.checkpoint_repo.CheckpointRepository.add"
    ) as add:
        p._record_checkpoint(step=100)
    data = add.call_args[0][0]
    assert data["lora_file"] is None
    assert data["lora_size_bytes"] is None


def test_complete_job_history_persists_final_lora(tmp_path):
    p = _pipeline_skeleton(tmp_path)
    lora = tmp_path / "x_final.safetensors"
    lora.write_bytes(b"z" * 456)
    p.checkpoint_manager.last_lora_path = str(lora)
    with patch(
        "app.core.db.repositories.job_repo.JobHistoryRepository.complete"
    ) as done, patch("app.core.stats.definition_stats_service.recompute"):
        p._complete_job_history(max_steps=100)
    kwargs = done.call_args.kwargs
    assert kwargs["final_lora_file"] == str(lora)
    assert kwargs["final_lora_size_bytes"] == 456
    # W5.T9: lora_on_disk is persisted here (run completion) — the ONE point
    # _current_lora_artifact() just confirmed the file via os.path.getsize(),
    # so get_stats' SUM(lora_on_disk) never needs to re-probe the filesystem.
    assert kwargs["lora_on_disk"] == 1


def test_complete_job_history_omits_artifact_when_unknown(tmp_path):
    """No located file → don't write NULLs over whatever the row has."""
    p = _pipeline_skeleton(tmp_path)
    with patch(
        "app.core.db.repositories.job_repo.JobHistoryRepository.complete"
    ) as done, patch("app.core.stats.definition_stats_service.recompute"):
        p._complete_job_history(max_steps=100)
    kwargs = done.call_args.kwargs
    assert "final_lora_file" not in kwargs
    assert "final_lora_size_bytes" not in kwargs
    assert "lora_on_disk" not in kwargs


def test_checkpoint_manager_tracks_last_lora_path(tmp_path):
    """save_checkpoint records the dist LoRA path on success and clears it
    when there is no saver."""
    from app.engine.components.checkpoints import CheckpointManager

    class _Saver:
        def save(self, components, path, metadata=None):
            Path(path).write_bytes(b"w")

    mgr = CheckpointManager(output_dir=str(tmp_path), saver_impl=_Saver())
    assert mgr.last_lora_path is None
    with patch.object(mgr, "_save_train_state"), patch.object(
        mgr, "_write_training_log"
    ):
        mgr.save_checkpoint(step=100, components={},
                            config={"lora_name": "x"}, is_final=False)
    assert mgr.last_lora_path is not None
    assert mgr.last_lora_path.endswith("_000100.safetensors")
    assert Path(mgr.last_lora_path).is_file()

    mgr.saver = None
    with patch.object(mgr, "_save_train_state"), patch.object(
        mgr, "_write_training_log"
    ):
        mgr.save_checkpoint(step=200, components={},
                            config={"lora_name": "x"}, is_final=False)
    assert mgr.last_lora_path is None
