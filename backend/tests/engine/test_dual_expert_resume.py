"""Dual-expert checkpoint resume restores BOTH experts' adapters.

Deferred from Wave 2 (same failure class W2.T4 closed for the ALL-adapters-
missing case), no dedicated brief.

``_resume_if_needed`` (:class:`PipelineOptimizationMixin`) used to build
``peft_comps`` from ONLY the ACTIVE expert (``self._get_primary_model()``) —
for a wan22/bernini_r dual-expert (``both``) run the saver writes
``unet_high`` AND ``unet_low`` independently (see e.g.
``Wan22Trainer._build_trainable_components``), so on resume the INACTIVE
expert's saved LoRA was never requested: it silently continued training from
a freshly-initialized adapter (``lora_B`` zeroed) while the active expert
resumed correctly. The W2.T4 all-adapters-missing guard in
``CheckpointManager.load_checkpoint`` never caught this — it only fires when
NONE of the requested components are found, and here exactly one (``unet``)
always was.

This module pins the fix:

* Both ``unet_high`` / ``unet_low`` PEFT adapters are requested and restored
  on a ``both``-mode resume — identity-deduped against ``unet`` (which always
  aliases whichever expert is currently active, so the same object is never
  requested twice under two names).
* An OLD checkpoint that predates a second expert (only ``unet`` on disk)
  does NOT raise — a legitimate, if degraded, resume — but DOES log a loud
  ``resume_partial_adapter_restore`` warning naming exactly what stayed
  fresh/unrestored.
* A single-expert trainer's resume (``_build_trainable_components`` returns
  only ``unet``) is byte-identical to before: no extra request, no warning.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

from app.engine.components.checkpoints import CheckpointManager
from app.engine.core.pipeline.pipeline_optimization import PipelineOptimizationMixin


# ── Tiny real PeftModel helpers ────────────────────────────────────────────


def _tiny_peft(fill: float, seed: int = 0) -> PeftModel:
    """A tiny real PEFT-wrapped ``nn.Linear`` with ``lora_B`` manually set to
    ``fill`` — a stand-in for "this adapter was actually trained for N
    steps". PEFT zero-inits ``lora_B``, so a fresh/never-restored adapter is
    observably distinct from a restored one (same pattern as
    ``test_checkpoint_resume_guard.py``'s ``_tiny_peft_model``, extended to
    make the weights themselves assertable).
    """
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 8))
    cfg = LoraConfig(r=4, lora_alpha=4, target_modules=["0"])
    pm = get_peft_model(model, cfg)
    with torch.no_grad():
        pm.base_model.model[0].lora_B["default"].weight.fill_(fill)
    return pm


def _lora_b(model: PeftModel) -> torch.Tensor:
    return model.base_model.model[0].lora_B["default"].weight.detach().clone()


# ── Fakes ───────────────────────────────────────────────────────────────────


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict]] = []
        self.infos: list[tuple[str, dict]] = []
        self.errors: list[tuple[str, dict]] = []

    def warning(self, event, **kw):
        self.warnings.append((event, kw))

    def info(self, event, **kw):
        self.infos.append((event, kw))

    def error(self, event, **kw):
        self.errors.append((event, kw))


class _FakeLoggerComponent:
    elapsed_offset = 0.0
    step_offset = 0


class _FakeTrainer(PipelineOptimizationMixin):
    """Exercises ONLY ``_resume_if_needed`` — no real weights, loader, or
    ``GenericTrainingPipeline`` machinery touched.

    ``trainable`` mirrors what a family's ``_build_trainable_components``
    returns: for a dual-expert trainer this is ``{"unet": active,
    "unet_high": ..., "unet_low": ...}`` (``unet`` always aliases whichever
    expert is active); for a single-expert trainer it's just ``{"unet":
    model}``.
    """

    def __init__(
        self,
        primary: PeftModel,
        trainable: dict[str, Any],
        checkpoint_manager: CheckpointManager,
        resume_path: str,
    ) -> None:
        self.config: dict[str, Any] = {"resume_from_checkpoint": resume_path}
        self.logger = _RecordingLogger()
        self._primary = primary
        self._trainable = trainable
        self.checkpoint_manager = checkpoint_manager
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        self.ema_handler = None
        self.global_step = 0
        self.logger_component = _FakeLoggerComponent()
        self._te_cache: dict = {}

    def _get_primary_model(self):
        return self._primary

    def _get_text_encoders(self):
        return {}

    def _build_trainable_components(self):
        return dict(self._trainable)

    def get_te_cache(self):
        return self._te_cache

    def set_te_cache(self, caches):
        self._te_cache = caches


# ── Both experts restore from a checkpoint that saved both ─────────────────


class TestDualExpertResumeRestoresBothAdapters:
    def test_both_experts_restore(self, tmp_path):
        """A checkpoint that saved BOTH experts must restore BOTH on resume
        — not just the active one."""
        mgr = CheckpointManager(str(tmp_path))
        saved_high = _tiny_peft(fill=3.0, seed=1)
        saved_low = _tiny_peft(fill=7.0, seed=2)
        ckpt_path = mgr.save_checkpoint(
            step=10,
            components={"unet": saved_high, "unet_low": saved_low},
            config={"lora_name": "test"},
        )

        fresh_high = _tiny_peft(fill=0.0, seed=3)
        fresh_low = _tiny_peft(fill=0.0, seed=4)
        t = _FakeTrainer(
            fresh_high,
            {"unet": fresh_high, "unet_high": fresh_high, "unet_low": fresh_low},
            mgr,
            ckpt_path,
        )

        t._resume_if_needed()

        assert torch.equal(_lora_b(fresh_high), _lora_b(saved_high)), (
            "active expert (unet) failed to restore"
        )
        assert torch.equal(_lora_b(fresh_low), _lora_b(saved_low)), (
            "inactive expert (unet_low) was never restored — the W3 bug"
        )
        assert not any("partial" in ev for ev, _ in t.logger.warnings), (
            t.logger.warnings
        )

    def test_active_low_expert_still_pulls_in_the_other(self, tmp_path):
        """Symmetry check: when LOW is the active expert (``unet`` aliases
        ``unet_low``), ``unet_high`` must still be requested and restored."""
        mgr = CheckpointManager(str(tmp_path))
        saved_high = _tiny_peft(fill=5.0, seed=1)
        saved_low = _tiny_peft(fill=9.0, seed=2)
        ckpt_path = mgr.save_checkpoint(
            step=10,
            components={"unet": saved_low, "unet_high": saved_high},
            config={"lora_name": "test"},
        )

        fresh_high = _tiny_peft(fill=0.0, seed=3)
        fresh_low = _tiny_peft(fill=0.0, seed=4)
        t = _FakeTrainer(
            fresh_low,  # active == low
            {"unet": fresh_low, "unet_high": fresh_high, "unet_low": fresh_low},
            mgr,
            ckpt_path,
        )

        t._resume_if_needed()

        assert torch.equal(_lora_b(fresh_low), _lora_b(saved_low))
        assert torch.equal(_lora_b(fresh_high), _lora_b(saved_high))


# ── Old checkpoint predating a second expert: warn, don't raise ────────────


class TestOldCheckpointMissingSecondExpert:
    def test_warns_not_raises(self, tmp_path):
        """A checkpoint saved BEFORE a second expert existed (only "unet" on
        disk) must NOT raise — a legitimate, if degraded, resume — but MUST
        warn loudly naming what stayed unrestored."""
        mgr = CheckpointManager(str(tmp_path))
        saved_high = _tiny_peft(fill=3.0, seed=1)
        ckpt_path = mgr.save_checkpoint(
            step=10,
            components={"unet": saved_high},  # no unet_low on disk
            config={"lora_name": "test"},
        )

        fresh_high = _tiny_peft(fill=0.0, seed=3)
        marker = -1.0  # distinguishes "never touched" from peft's own 0.0 init
        fresh_low = _tiny_peft(fill=marker, seed=4)
        t = _FakeTrainer(
            fresh_high,
            {"unet": fresh_high, "unet_high": fresh_high, "unet_low": fresh_low},
            mgr,
            ckpt_path,
        )

        t._resume_if_needed()  # must not raise

        assert torch.equal(_lora_b(fresh_high), _lora_b(saved_high))
        assert torch.equal(
            _lora_b(fresh_low), torch.full_like(_lora_b(fresh_low), marker)
        ), "the missing expert's adapter must be left exactly as-is (fresh)"

        events = [ev for ev, _ in t.logger.warnings]
        assert "resume_partial_adapter_restore" in events
        kw = dict(t.logger.warnings)["resume_partial_adapter_restore"]
        assert "unet_low" in kw["missing"]
        assert "unet" not in kw["missing"]

    def test_all_missing_still_raises(self, tmp_path):
        """Sanity: the pre-existing ALL-missing guard (W2.T4) is untouched —
        if NEITHER expert's adapter is on disk, load_checkpoint still raises
        rather than silently degrading to the partial-warn path."""
        mgr = CheckpointManager(str(tmp_path))
        ckpt_path = mgr.save_checkpoint(
            step=10,
            components={},
            config={"lora_name": "test"},
        )  # no adapters saved at all

        fresh_high = _tiny_peft(fill=0.0, seed=3)
        fresh_low = _tiny_peft(fill=0.0, seed=4)
        t = _FakeTrainer(
            fresh_high,
            {"unet": fresh_high, "unet_high": fresh_high, "unet_low": fresh_low},
            mgr,
            ckpt_path,
        )

        with pytest.raises(RuntimeError, match="adapter"):
            t._resume_if_needed()


# ── Single-expert resume is unaffected ──────────────────────────────────────


class TestSingleExpertResumeUnaffected:
    def test_no_extra_request_no_warning(self, tmp_path):
        """A trainer whose ``_build_trainable_components`` returns ONLY
        "unet" (single-expert / non-dual family) behaves exactly as before —
        no extra PEFT component requested, no partial-restore warning."""
        mgr = CheckpointManager(str(tmp_path))
        saved = _tiny_peft(fill=3.0, seed=1)
        ckpt_path = mgr.save_checkpoint(
            step=10,
            components={"unet": saved},
            config={"lora_name": "test"},
        )

        fresh = _tiny_peft(fill=0.0, seed=2)
        t = _FakeTrainer(fresh, {"unet": fresh}, mgr, ckpt_path)

        t._resume_if_needed()

        assert torch.equal(_lora_b(fresh), _lora_b(saved))
        assert t.logger.warnings == []
