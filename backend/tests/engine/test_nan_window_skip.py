"""W2.T6: an all-NaN gradient-accumulation window must not step the
optimizer/scheduler/EMA on zero grads.

Drives the REAL ``PipelineTrainMixin.train()`` coroutine end-to-end (not a
re-implementation of the accumulation loop) against a minimal fake trainer.
Every family-specific hook (encode_text, forward_pass, ...) and every
DB/VRAM/checkpoint side-effect is stubbed to a trivial, deterministic value —
the only thing under test is the control flow around ``_compute_step_loss``.

``_compute_step_loss`` returns tensors from a scripted ``_loss_plan`` list so
each test can dictate exactly which micro-steps are NaN vs real, proving:

- ALL-NaN window (test 1): ``optimizer.step`` / ``lr_scheduler.step`` /
  ``ema_handler.step`` are never called, a ``nan_window_skipped`` warning is
  logged, and grads are zeroed for the next window.
- ALL-NaN window under AMP (test 2): the scaler's ``unscale_`` / ``step`` /
  ``update`` are ALSO skipped (no preceding ``scale().backward()`` to unscale).
- Partial window — one NaN + one real loss (test 3, regression guard): the
  step still fires normally. Proves the fix counts SUCCESSES, not just
  "any NaN seen", so a normal accumulation window is untouched.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from app.engine.core.pipeline.pipeline_train import PipelineTrainMixin


class _ScriptedTrainer(PipelineTrainMixin):
    """Bare trainer: only the attrs/hooks ``train()`` touches. All
    family/base-mixin hooks are stubbed to trivial CPU tensors shaped so a
    real ``nn.Linear`` can produce a genuine ``grad_fn`` for the "real loss"
    micro-steps (needed so ``loss.backward()`` doesn't raise)."""

    def __init__(self, tmp_path, *, grad_accum: int, loss_plan, use_amp: bool = False):
        self.logger = MagicMock()
        self.config = {
            "max_train_steps": 1,
            "train_batch_size": 1,
            "gradient_accumulation_steps": grad_accum,
            "cache_latents": False,
            "save_every_n_steps": 0,
            "sample_before_training": False,
            "noise_offset": 0.0,
        }
        self.inventory = [{"id": 0}]  # only len() is read before the loop
        self.device = torch.device("cpu")
        self.autocast_dtype = torch.float32
        self.use_amp = use_amp
        self.scaler = MagicMock()
        self.scaler.is_enabled.return_value = use_amp
        self.optimizer = MagicMock()
        self.optimizer.param_groups = [{"lr": 1e-4}]  # real dict: no "d" key
        self.lr_scheduler = MagicMock()
        self.ema_handler = MagicMock()
        self.global_step = 0
        self.sampler = None  # skips step-0 + final sampling blocks entirely
        self._log_writer = None
        self._aug_h_flip = False
        self._aug_v_flip = False

        self.checkpoint_manager = MagicMock()
        self.checkpoint_manager.output_dir = str(tmp_path)
        self.logger_component = MagicMock()
        self.logger_component.last_step_time = time.time()

        self.latent_manager = MagicMock()
        self.latent_manager.encode_and_cache_batch.return_value = torch.randn(1, 4)

        # Fixed model instance (not re-created per call) so the grads
        # produced by backward() are visible to the SAME parameters the
        # optimizer-step region clips/steps.
        self._model = torch.nn.Linear(4, 4)

        self._loss_plan = list(loss_plan)
        assert len(self._loss_plan) == grad_accum

    # ── stubbed family/base-mixin hooks (not under test) ──

    def _get_primary_model(self):
        return self._model

    def _iter_training_batches(self, batch_size):
        while True:
            yield [{"id": 0}]

    def _get_batch(self, batch_items, decode_pixels=True):
        return {
            "images": torch.randn(1, 3, 8, 8),
            "ids": [0],
            "paths": ["dummy.png"],
            "captions": ["a cat"],
        }

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
        return self._model(noisy_input)

    def compute_target(self, latents, noise, timesteps):
        return torch.zeros_like(latents)

    def _compute_step_loss(self, pred, target, timesteps, batch, grad_accum):
        fn = self._loss_plan.pop(0)
        return fn(pred, target)

    def _build_trainable_components(self):
        return {}

    def get_te_cache(self):
        return None

    def _build_cache_manifest(self):
        return None


def _nan(pred, target):
    return torch.tensor(float("nan"))


def _real(pred, target):
    return (pred - target).pow(2).mean()


def _run(trainer: _ScriptedTrainer) -> None:
    # Never touch the real DB — _init_job_history/_record_checkpoint/
    # _complete_job_history all catch this and no-op (job_history_id stays
    # None), matching tests/test_pipeline_artifact_persistence.py's pattern.
    with patch(
        "app.core.db.DatabaseEngine.get_instance",
        side_effect=RuntimeError("no db in test"),
    ):
        asyncio.run(trainer.train())


# ── Test 1: ALL-NaN window — the RED/GREEN case from the brief ────────────


def test_all_nan_window_skips_optimizer_scheduler_ema(tmp_path):
    t = _ScriptedTrainer(tmp_path, grad_accum=2, loss_plan=[_nan, _nan])
    _run(t)

    t.optimizer.step.assert_not_called()
    t.lr_scheduler.step.assert_not_called()
    t.ema_handler.step.assert_not_called()
    # Grads still zeroed (set_to_none) so the NEXT window starts clean.
    t.optimizer.zero_grad.assert_any_call(set_to_none=True)

    warnings = [c.args for c in t.logger.warning.call_args_list]
    assert any(a and a[0] == "nan_window_skipped" for a in warnings), (
        f"expected a nan_window_skipped warning, got: {warnings}"
    )


# ── Test 4: fix-wave — observability must not go silent (review finding) ──


def test_all_nan_window_still_advances_job_progress(tmp_path):
    """The step counter that drives the Jobs screen 'Step X/Y' KPI must not
    freeze during a NaN storm — ``_update_job_progress`` is the sole writer
    of ``completed_steps`` and must fire even when the window is fully
    skipped."""
    t = _ScriptedTrainer(tmp_path, grad_accum=2, loss_plan=[_nan, _nan])
    with patch.object(
        PipelineTrainMixin, "_update_job_progress", autospec=True
    ) as mock_progress:
        _run(t)

    mock_progress.assert_any_call(t, 0)

    # The optimizer/scheduler/EMA-not-stepped property must remain intact —
    # this is what the fix must NOT regress.
    t.optimizer.step.assert_not_called()
    t.lr_scheduler.step.assert_not_called()
    t.ema_handler.step.assert_not_called()


def test_all_nan_window_emits_marker_log_step(tmp_path):
    """A NaN-skipped window must still reach the real Jobs-screen log
    channel (``logger_component.log_step``, which feeds job_log.jsonl) with
    a clear marker — not just the structured ``self.logger.warning`` that
    never reaches the UI. The loss must NOT be a real/fabricated number:
    ``loss=None`` is the contract that keeps the chart/DB honest (see
    training_logger.log_step docstring)."""
    t = _ScriptedTrainer(tmp_path, grad_accum=2, loss_plan=[_nan, _nan])
    _run(t)

    t.logger_component.log_step.assert_called_once()
    args, kwargs = t.logger_component.log_step.call_args
    step_arg = args[0]
    loss_arg = args[1]
    assert step_arg == 0
    assert loss_arg is None, (
        f"expected loss=None for a fully-NaN window, got {loss_arg!r}"
    )
    extra = kwargs.get("extra") or (args[3] if len(args) > 3 else {})
    assert extra.get("nan_window_skipped") is True

    # Optimizer/scheduler/EMA still untouched.
    t.optimizer.step.assert_not_called()
    t.lr_scheduler.step.assert_not_called()
    t.ema_handler.step.assert_not_called()


# ── Test 2: ALL-NaN window under AMP — scaler must be skipped too ─────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast(cuda, enabled=True) needs a CUDA device")
def test_all_nan_window_under_amp_skips_scaler_too(tmp_path):
    t = _ScriptedTrainer(
        tmp_path, grad_accum=2, loss_plan=[_nan, _nan], use_amp=True
    )
    _run(t)

    t.scaler.unscale_.assert_not_called()
    t.scaler.step.assert_not_called()
    t.scaler.update.assert_not_called()
    t.optimizer.step.assert_not_called()
    t.lr_scheduler.step.assert_not_called()
    t.ema_handler.step.assert_not_called()


# ── Test 3: partial window (1 NaN + 1 real) — regression guard ────────────


def test_partial_nan_window_still_steps_optimizer(tmp_path):
    """Sanity check that the fix counts SUCCESSES rather than reacting to
    "any NaN seen" — a window with at least one real backward must still
    step optimizer/scheduler/EMA exactly as before."""
    t = _ScriptedTrainer(tmp_path, grad_accum=2, loss_plan=[_nan, _real])
    _run(t)

    t.optimizer.step.assert_called_once()
    t.lr_scheduler.step.assert_called_once()
    t.ema_handler.step.assert_called_once()

    warnings = [c.args for c in t.logger.warning.call_args_list]
    assert not any(a and a[0] == "nan_window_skipped" for a in warnings), (
        f"partial window must NOT be treated as a fully-skipped one: {warnings}"
    )
