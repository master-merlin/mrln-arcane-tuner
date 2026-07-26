"""Resume-integrity guard (W2.T4): silent fresh-weights resume closed.

Two silent failures used to compose into a ruined-but-plausible training run:

1. ``_save_train_state`` caught a failed PEFT adapter ``save_pretrained()``
   call, logged it, and carried on — yet still wrote ``training_state.json``.
   The checkpoint directory LOOKED valid (metadata + manifest present) but
   held zero adapter weights.
2. ``load_checkpoint`` only loaded an adapter when its ``adapter_config.json``
   existed; when absent (e.g. because of (1)) it silently skipped that
   component. A resume from such a checkpoint starts training at step N with
   a FRESHLY-INITIALIZED adapter (``lora_B`` zeroed) — the model behaves like
   the base model, the loss curve still looks plausible, and there is no
   error, warning, or signal that the run is ruined.

This module pins the fix: a failed adapter save must propagate out of
``save_checkpoint`` (never produce a "valid" checkpoint with no weights),
and ``load_checkpoint`` must raise ``RuntimeError`` when PEFT components were
requested but ZERO adapters were found on disk. A dual-expert (or any
multi-adapter) run legitimately loads a SUBSET of requested adapters — only
an all-zero load is treated as invalid.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

import app.engine.components.checkpoints as checkpoints_module
from app.engine.components.checkpoints import CheckpointManager


# ── Fixtures ─────────────────────────────────────────────────────────────


def _tiny_peft_model() -> PeftModel:
    """Minimal real PEFT-wrapped model — same pattern as
    ``test_generic_lora_saver_failloud.py`` / the saver test suite: a plain
    ``nn.Linear`` behind ``get_peft_model`` gives a genuine ``PeftModel``
    instance (not a hand-rolled fake), so monkeypatching
    ``PeftModel.save_pretrained`` at the class level actually takes effect.
    """
    model = nn.Sequential(nn.Linear(8, 8))
    cfg = LoraConfig(r=4, lora_alpha=4, target_modules=["0"])
    return get_peft_model(model, cfg)


def _make_manager(tmp_path) -> CheckpointManager:
    return CheckpointManager(str(tmp_path))


def _raise_oserror(self, *args, **kwargs):
    raise OSError("disk full")


# ── Load side: raise when PEFT requested but nothing loaded ─────────────


class TestResumeMissingAdapterRaises:
    def test_missing_adapter_dir_raises(self, tmp_path):
        """training_state.json present, but no adapter subdir for the
        requested PEFT component (e.g. from a prior silently-failed save,
        or a stale/wrong resume path) — must fail loud, not resume with
        fresh random adapters."""
        mgr = _make_manager(tmp_path)
        ckpt_path = mgr.save_checkpoint(
            step=10,
            components={},
            config={"lora_name": "test"},
        )  # no PEFT components saved -> no adapter dirs on disk

        with pytest.raises(RuntimeError, match="adapter"):
            mgr.load_checkpoint(
                ckpt_path,
                peft_components={"unet": _tiny_peft_model()},
            )

    def test_error_message_names_checkpoint_path(self, tmp_path):
        mgr = _make_manager(tmp_path)
        ckpt_path = mgr.save_checkpoint(
            step=10,
            components={},
            config={"lora_name": "test"},
        )

        with pytest.raises(RuntimeError, match="fresh random adapters"):
            mgr.load_checkpoint(
                ckpt_path,
                peft_components={"unet": _tiny_peft_model()},
            )

    def test_partial_load_does_not_raise(self, tmp_path):
        """Dual-expert (or any multi-adapter) run: ONE of several requested
        adapters missing is a legitimate partial state (e.g. a
        single-expert checkpoint being resumed by a dual-expert run's
        driver), not a hard failure — only an ALL-missing load is invalid.
        """
        mgr = _make_manager(tmp_path)
        ckpt_path = mgr.save_checkpoint(
            step=10,
            components={"unet_high": _tiny_peft_model()},
            config={"lora_name": "test"},
        )  # only unet_high/ exists on disk; unet_low/ was never saved

        state = mgr.load_checkpoint(
            ckpt_path,
            peft_components={
                "unet_high": _tiny_peft_model(),
                "unet_low": _tiny_peft_model(),
            },
        )

        assert "unet_high" in state.adapters_loaded
        assert "unet_low" not in state.adapters_loaded

    def test_no_peft_requested_does_not_raise(self, tmp_path):
        """A resume that doesn't request any PEFT components (e.g. a
        non-LoRA full fine-tune, or a caller with nothing PEFT-wrapped)
        must not be penalized by the guard."""
        mgr = _make_manager(tmp_path)
        ckpt_path = mgr.save_checkpoint(
            step=10,
            components={},
            config={"lora_name": "test"},
        )

        state = mgr.load_checkpoint(ckpt_path)  # peft_components=None

        assert state.global_step == 10


# ── Save side: adapter-save failure must propagate ───────────────────────


class TestAdapterSaveFailurePropagates:
    def test_adapter_save_failure_propagates_on_periodic_save(
        self,
        tmp_path,
        monkeypatch,
    ):
        """A failed PEFT adapter save must fail the checkpoint outright —
        even for a periodic (non-final) save. Unlike the distribution-LoRA
        save (which is best-effort on periodic saves — see
        test_periodic_lora_save_failure_does_not_raise), this IS the
        resume-state artifact; a later resume would load from exactly this
        checkpoint, so a silent partial write here is never safe to
        tolerate."""
        mgr = _make_manager(tmp_path)
        monkeypatch.setattr(PeftModel, "save_pretrained", _raise_oserror)

        with pytest.raises(OSError, match="disk full"):
            mgr.save_checkpoint(
                step=10,
                components={"unet": _tiny_peft_model()},
                config={"lora_name": "test"},
                is_final=False,
            )

    def test_adapter_save_failure_propagates_on_final_save(
        self,
        tmp_path,
        monkeypatch,
    ):
        mgr = _make_manager(tmp_path)
        monkeypatch.setattr(PeftModel, "save_pretrained", _raise_oserror)

        with pytest.raises(OSError, match="disk full"):
            mgr.save_checkpoint(
                step=1000,
                components={"unet": _tiny_peft_model()},
                config={"lora_name": "test"},
                is_final=True,
            )

    def test_adapter_save_failure_leaves_no_training_state(
        self,
        tmp_path,
        monkeypatch,
    ):
        """When the adapter save fails, training_state.json must NOT exist
        for that checkpoint directory — no half-valid, "looks resumable"
        checkpoint left behind."""
        mgr = _make_manager(tmp_path)
        monkeypatch.setattr(PeftModel, "save_pretrained", _raise_oserror)

        with pytest.raises(OSError):
            mgr.save_checkpoint(
                step=10,
                components={"unet": _tiny_peft_model()},
                config={"lora_name": "test"},
            )

        state_path = tmp_path / "checkpoint-000010" / "training_state.json"
        assert not state_path.exists()


# ── Optimizer-absent on resume: warn, don't raise, don't stay silent ────


class TestOptimizerAbsentWarns:
    def test_optimizer_absent_on_resume_warns(self, tmp_path, monkeypatch):
        """An optimizer-less resume is a legitimate (if degraded) scenario
        — momentum/Adam moments restart from scratch. It must not raise,
        but it must not be silent either: warn so the user has signal."""
        mgr = _make_manager(tmp_path)
        ckpt_path = mgr.save_checkpoint(
            step=10,
            components={},
            config={"lora_name": "test"},
        )  # no optimizer saved -> optimizer.pt absent

        warn_mock = MagicMock()
        monkeypatch.setattr(checkpoints_module.logger, "warning", warn_mock)

        optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.01)
        state = mgr.load_checkpoint(ckpt_path, optimizer=optimizer)

        assert "optimizer" not in state.components_loaded
        assert warn_mock.called
        events = [
            (c.args[0] if c.args else c.kwargs.get("event"))
            for c in warn_mock.call_args_list
        ]
        assert any("optimizer" in str(e) for e in events)
