"""A soft stop and the resume after it train every step exactly once.

Drives the real ``PipelineTrainMixin.train()`` on a bare CPU trainer (the pattern of
``test_nan_window_skip.py``). The stop signal arrives at the top of a chosen step; the
checkpoint manager records each save. The resume is what ``_resume_if_needed`` leaves
behind: ``global_step`` set to the saved step. Every step the two runs train is recorded.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from app.engine.core.pipeline.pipeline_train import PipelineTrainMixin

MAX_STEPS = 8
SAVE_EVERY = 3


class _Signals:
    """``handle_signals`` answers "soft_stop" at the top of ``stop_at``, and nothing else."""

    def __init__(self, trainer: "_Trainer", stop_at: int | None) -> None:
        self.trainer = trainer
        self.stop_at = stop_at

    def handle_signals(self):
        if self.stop_at is not None and self.trainer.global_step == self.stop_at:
            return "soft_stop"
        return None


class _Trainer(PipelineTrainMixin):
    def __init__(self, tmp_path, *, global_step: int, stop_at: int | None, trained: list[int]):
        self.logger = MagicMock()
        self.config = {
            "max_train_steps": MAX_STEPS,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "cache_latents": False,
            "save_every_n_steps": SAVE_EVERY,
            "sample_before_training": False,
            "noise_offset": 0.0,
        }
        self.inventory = [{"id": 0}]
        self.device = torch.device("cpu")
        self.autocast_dtype = torch.float32
        self.use_amp = False
        self.scaler = MagicMock()
        self.scaler.is_enabled.return_value = False
        self._model = torch.nn.Linear(4, 4)
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr=1e-3)
        self.lr_scheduler = MagicMock()
        self.ema_handler = None
        self.global_step = global_step
        self.sampler = None
        self._log_writer = None
        self._aug_h_flip = False
        self._aug_v_flip = False
        self.checkpoint_manager = MagicMock()
        self.checkpoint_manager.output_dir = str(tmp_path)
        self.logger_component = MagicMock()
        self.logger_component.last_step_time = time.time()
        self.latent_manager = MagicMock()
        self.latent_manager.encode_and_cache_batch.return_value = torch.randn(1, 4)
        self._trained = trained
        self._stop_at = stop_at

    def _get_primary_model(self):
        return self._model

    def _iter_training_batches(self, batch_size):
        while True:
            yield [{"id": 0}]

    def _get_batch(self, batch_items, decode_pixels=True):
        return {"images": torch.randn(1, 3, 8, 8), "ids": [0], "paths": ["a.png"], "captions": ["a"]}

    def _load_control_latents(self, batch):
        pass

    def _attach_conditioning(self, batch, latents):
        pass

    def encode_text(self, captions, dtype, batch=None):
        return torch.zeros(1, 4)

    def prepare_latents_for_training(self, latents):
        return latents

    def prepare_noise_for_training(self, noise):
        return noise

    def sample_timesteps(self, batch_size, latents):
        return torch.zeros(batch_size)

    def add_noise(self, latents, noise, timesteps):
        return latents

    def forward_pass(self, noisy_input, timesteps, text_emb, batch):
        self._trained.append(self.global_step)
        return self._model(noisy_input)

    def compute_target(self, latents, noise, timesteps):
        return torch.zeros_like(latents)

    def _compute_step_loss(self, pred, target, timesteps, batch, grad_accum):
        return (pred - target).pow(2).mean()

    def on_epoch_end(self, epoch_num):
        pass

    def _build_trainable_components(self):
        return {}

    def get_te_cache(self):
        return None

    def _build_cache_manifest(self):
        return None


def _train(t: _Trainer) -> None:
    from app.engine.components import signal_manager

    signals = _Signals(t, t._stop_at)
    with (
        patch("app.core.db.DatabaseEngine.get_instance", side_effect=RuntimeError("no db in test")),
        patch.object(signal_manager, "TrainingSignalManager", return_value=signals),
    ):
        asyncio.run(t.train())


def _resumable_saves(t: _Trainer) -> list[int]:
    """The step of every checkpoint a resume could start from, in the order saved."""
    return [
        c.kwargs["step"]
        for c in t.checkpoint_manager.save_checkpoint.call_args_list
        if not c.kwargs.get("is_final", False)
    ]


def _stop_then_resume(tmp_path, stop_at: int, *, start: int = 0) -> tuple[list[int], _Trainer]:
    trained: list[int] = []
    first = _Trainer(tmp_path / "a", global_step=start, stop_at=stop_at, trained=trained)
    _train(first)
    saved = _resumable_saves(first)[-1]

    resumed = _Trainer(tmp_path / "b", global_step=saved, stop_at=None, trained=trained)
    _train(resumed)
    return trained, first


@pytest.mark.parametrize("stop_at", [5, SAVE_EVERY + 1, 2 * SAVE_EVERY])
def test_a_soft_stop_and_its_resume_train_every_step_once(tmp_path, stop_at):
    trained, _ = _stop_then_resume(tmp_path, stop_at)

    assert sorted(trained) == list(range(MAX_STEPS))


def test_a_soft_stop_right_after_a_resume_loses_no_step(tmp_path):
    """Resumed from step 3, stopped before step 4 runs: the next resume starts at 4."""
    trained, first = _stop_then_resume(tmp_path, stop_at=4, start=3)

    assert _resumable_saves(first) == [3]
    assert sorted(trained) == list(range(4, MAX_STEPS))


def test_a_soft_stop_writes_no_final_checkpoint(tmp_path):
    """A stopped run is unfinished: no ``final`` save, no completed job record."""
    trained: list[int] = []
    t = _Trainer(tmp_path, global_step=0, stop_at=5, trained=trained)
    with (
        patch.object(PipelineTrainMixin, "_complete_job_history", autospec=True) as complete,
        patch.object(PipelineTrainMixin, "_record_checkpoint", autospec=True) as record,
    ):
        _train(t)

    finals = [c for c in t.checkpoint_manager.save_checkpoint.call_args_list if c.kwargs.get("is_final")]
    assert finals == []
    complete.assert_not_called()
    assert [c for c in record.call_args_list if c.kwargs.get("is_final")] == []
    assert t._soft_stop_exit is True
    t.logger_component.save_loss_history.assert_called_once()
    assert sorted(trained) == [0, 1, 2, 3, 4]


def test_a_run_that_trains_to_the_end_is_no_soft_stop(tmp_path):
    t = _Trainer(tmp_path, global_step=0, stop_at=None, trained=[])
    _train(t)

    assert t._soft_stop_exit is False
    finals = [c for c in t.checkpoint_manager.save_checkpoint.call_args_list if c.kwargs.get("is_final")]
    assert len(finals) == 1
