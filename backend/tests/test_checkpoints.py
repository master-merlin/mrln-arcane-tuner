"""
Tests for Phase 7: Checkpoint Resume & Staged Training.

Covers:
- CheckpointManager save/load roundtrip
- PEFT adapter saving/loading
- Config override validation (safe/warning/blocked)
- Compatibility checking (rank/alpha/model)
- Checkpoint inspector filesystem scanning
"""

import json
import os
from types import SimpleNamespace
from typing import Any

import peft
import pytest
import torch

from app.engine.components.checkpoints import (
    CheckpointManager,
    CheckpointState,
    apply_overrides,
    validate_compatibility,
)
from app.engine.utils.checkpoint_inspector import inspect_checkpoint


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_linear_module() -> torch.nn.Module:
    """Create a small torch module for testing state_dict save/load."""
    return torch.nn.Linear(4, 2, bias=False)


def _make_optimizer(module: torch.nn.Module) -> torch.optim.Optimizer:
    """Create a simple optimizer for testing."""
    return torch.optim.SGD(module.parameters(), lr=0.01)


def _make_scheduler(optimizer: torch.optim.Optimizer) -> Any:
    """Create a simple LR scheduler for testing."""
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)


class FakeEMAHandler:
    """Minimal EMA handler mock for save/load testing."""

    def __init__(self):
        self._shadow = {"weight": torch.randn(2, 4)}

    def state_dict(self) -> dict:
        return self._shadow

    def load_state_dict(self, sd: dict) -> None:
        self._shadow = sd

    def store_and_swap(self) -> None:
        pass

    def restore(self) -> None:
        pass


class FakePeftModel:
    """Minimal PEFT model mock for adapter save/load testing."""

    def __init__(self, name: str = "test"):
        self._name = name
        self._loaded_from: str | None = None

    def save_pretrained(self, path: str) -> None:
        """Save a fake adapter_config.json and adapter_model.bin."""
        os.makedirs(path, exist_ok=True)
        config = {"peft_type": "LORA", "r": 16, "lora_alpha": 8}
        with open(os.path.join(path, "adapter_config.json"), "w") as f:
            json.dump(config, f)
        # Write a small binary to simulate adapter_model
        torch.save({"dummy": torch.zeros(1)}, os.path.join(path, "adapter_model.bin"))

    def load_adapter(self, path: str, adapter_name: str = "default", is_trainable: bool = True) -> None:
        """Record that an adapter was loaded from this path."""
        self._loaded_from = path

    def parameters(self):
        return iter([])


# ── CheckpointManager Save ──────────────────────────────────────────────


class TestCheckpointManagerSave:
    """Tests for CheckpointManager.save_checkpoint()."""

    def test_save_creates_training_state(self, tmp_path):
        """training_state.json should contain step, config, timestamp."""
        mgr = CheckpointManager(str(tmp_path))
        config = {"lora_name": "test", "job_id": "j1"}

        mgr.save_checkpoint(step=100, components={}, config=config)

        state_path = tmp_path / "checkpoint-000100" / "training_state.json"
        assert state_path.exists()
        with open(state_path) as f:
            state = json.load(f)
        assert state["global_step"] == 100
        assert state["config"]["job_id"] == "j1"
        assert "timestamp" in state

    def test_save_creates_manifest(self, tmp_path):
        """checkpoint_manifest.json should list all saved files with sizes."""
        mgr = CheckpointManager(str(tmp_path))
        module = _make_linear_module()

        mgr.save_checkpoint(
            step=50,
            components={"linear": module},
            config={"lora_name": "test"},
        )

        manifest_path = tmp_path / "checkpoint-000050" / "checkpoint_manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert "linear.pt" in manifest
        assert "training_state.json" in manifest
        assert all(isinstance(v, int) and v > 0 for v in manifest.values())

    def test_save_optimizer_scheduler_scaler(self, tmp_path):
        """Optimizer, scheduler, and scaler .pt files should exist."""
        mgr = CheckpointManager(str(tmp_path))
        module = _make_linear_module()
        optimizer = _make_optimizer(module)
        scheduler = _make_scheduler(optimizer)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        mgr.save_checkpoint(
            step=10,
            components={},
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config={"lora_name": "test"},
        )

        ckpt_dir = tmp_path / "checkpoint-000010"
        assert (ckpt_dir / "optimizer.pt").exists()
        assert (ckpt_dir / "scheduler.pt").exists()
        assert (ckpt_dir / "scaler.pt").exists()

    def test_save_ema_shadow(self, tmp_path):
        """EMA shadow state should be saved when handler is present."""
        mgr = CheckpointManager(str(tmp_path))
        ema = FakeEMAHandler()

        mgr.save_checkpoint(
            step=20, components={}, ema_handler=ema, config={"lora_name": "test"},
        )

        assert (tmp_path / "checkpoint-000020" / "ema_shadow.pt").exists()

    def test_save_peft_adapters(self, tmp_path):
        """PEFT components should produce adapter_config.json in subdirs."""
        mgr = CheckpointManager(str(tmp_path))
        peft = FakePeftModel("unet")

        mgr.save_checkpoint(
            step=30, components={"unet": peft}, config={"lora_name": "test"},
        )

        adapter_config = tmp_path / "checkpoint-000030" / "unet" / "adapter_config.json"
        assert adapter_config.exists()

    def test_save_final_naming(self, tmp_path):
        """Final save should use 'final' folder and '_final' suffix."""
        mgr = CheckpointManager(str(tmp_path))

        path = mgr.save_checkpoint(
            step=1000, components={}, config={"lora_name": "my_lora"}, is_final=True,
        )

        assert "final" in path
        assert (tmp_path / "final" / "training_state.json").exists()

    def test_save_returns_path(self, tmp_path):
        """save_checkpoint should return the checkpoint directory path."""
        mgr = CheckpointManager(str(tmp_path))

        path = mgr.save_checkpoint(step=5, components={}, config={"lora_name": "t"})

        assert os.path.isdir(path)
        assert "checkpoint-000005" in path

    def test_periodic_lora_save_failure_does_not_raise(self, tmp_path):
        """A distribution-LoRA save failure on a PERIODIC checkpoint must be
        logged loudly but must NOT crash the training loop — training
        continues past a bad mid-run checkpoint."""

        class FailingSaver:
            def save(self, components, path, metadata=None):
                raise OSError("disk full")

        mgr = CheckpointManager(str(tmp_path), saver_impl=FailingSaver())

        # Should not raise — periodic saves are best-effort for the
        # distribution LoRA; resume state still gets written.
        path = mgr.save_checkpoint(
            step=10, components={}, config={"lora_name": "t"}, is_final=False,
        )
        assert os.path.isdir(path)

    def test_final_lora_save_failure_raises(self, tmp_path):
        """A distribution-LoRA save failure on the FINAL checkpoint must
        propagate — the job must fail loudly rather than report success
        with no LoRA file written."""

        class FailingSaver:
            def save(self, components, path, metadata=None):
                raise OSError("disk full")

        mgr = CheckpointManager(str(tmp_path), saver_impl=FailingSaver())

        with pytest.raises(OSError, match="disk full"):
            mgr.save_checkpoint(
                step=1000, components={}, config={"lora_name": "t"}, is_final=True,
            )


# ── W5.T3(c): weight_shapes written on first + final save only ──────────


class TestTrainingLogWeightShapesOnceThenFinal:
    """``weight_shapes`` is a static per-tensor snapshot — PEFT adapter
    shapes never change mid-run — so building + serializing it into
    ``training_log.json`` on EVERY periodic save was thousands of
    identical JSON entries for a high-rank 14B LoRA. It must now appear
    only on the FIRST save and again on the FINAL save, never on the
    periodic saves in between.
    """

    @staticmethod
    def _peft_component() -> SimpleNamespace:
        """``hasattr(comp, "peft_config")`` gates both the peft_info AND
        weight_shapes collection blocks in ``_write_training_log``."""
        return SimpleNamespace(
            peft_config={
                "default": SimpleNamespace(
                    peft_type="LORA",
                    r=8,
                    lora_alpha=8,
                    lora_dropout=0.0,
                    bias="none",
                    target_modules={"q_proj"},
                    modules_to_save=None,
                )
            }
        )

    def _write_and_read(
        self, mgr: CheckpointManager, monkeypatch, step: int, is_final: bool = False,
    ) -> dict:
        monkeypatch.setattr(
            peft,
            "get_peft_model_state_dict",
            lambda comp: {"lora_A.weight": torch.zeros(2, 2)},
        )
        mgr._write_training_log(
            step=step,
            components={"unet": self._peft_component()},
            optimizer=None,
            config={"lora_name": "t"},
            is_final=is_final,
            elapsed_time=1.0,
            lora_filename="t.safetensors",
        )
        log_path = os.path.join(mgr.output_dir, "training_log.json")
        with open(log_path) as f:
            return json.load(f)

    def test_weight_shapes_present_on_first_periodic_save_only(
        self, tmp_path, monkeypatch
    ):
        mgr = CheckpointManager(str(tmp_path))

        present = [
            "weight_shapes" in self._write_and_read(mgr, monkeypatch, step=s)
            for s in (10, 20, 30)
        ]

        assert present == [True, False, False]

    def test_weight_shapes_present_on_final_save_even_after_periodic_saves(
        self, tmp_path, monkeypatch
    ):
        mgr = CheckpointManager(str(tmp_path))
        self._write_and_read(mgr, monkeypatch, step=10)
        self._write_and_read(mgr, monkeypatch, step=20)

        final_log = self._write_and_read(mgr, monkeypatch, step=30, is_final=True)

        assert "weight_shapes" in final_log

    def test_weight_shapes_content_matches_peft_state_dict_when_written(
        self, tmp_path, monkeypatch
    ):
        """The FIRST-save write must still carry the real shape/dtype/numel
        payload — the gating must not silently degrade what gets recorded
        on the save that IS written."""
        mgr = CheckpointManager(str(tmp_path))

        log = self._write_and_read(mgr, monkeypatch, step=1)

        assert log["weight_shapes"]["unet"]["lora_A.weight"] == {
            "shape": [2, 2],
            "dtype": "torch.float32",
            "numel": 4,
        }


# ── CheckpointManager Load ──────────────────────────────────────────────


class TestCheckpointManagerLoad:
    """Tests for CheckpointManager.load_checkpoint()."""

    def _create_checkpoint(self, tmp_path, step: int = 42, config: dict | None = None) -> str:
        """Helper to create a valid checkpoint directory."""
        mgr = CheckpointManager(str(tmp_path))
        return mgr.save_checkpoint(
            step=step,
            components={},
            config=config or {"lora_name": "test", "network_rank": 16, "network_alpha": 8.0},
        )

    def test_load_restores_step(self, tmp_path):
        """Should restore global_step from training_state.json."""
        ckpt_path = self._create_checkpoint(tmp_path, step=42)
        mgr = CheckpointManager(str(tmp_path))

        state = mgr.load_checkpoint(ckpt_path)

        assert isinstance(state, CheckpointState)
        assert state.global_step == 42

    def test_load_restores_optimizer(self, tmp_path):
        """Optimizer state should be restored from optimizer.pt."""
        module = _make_linear_module()
        optimizer = _make_optimizer(module)
        # Do a step to create optimizer state
        loss = module(torch.randn(1, 4)).sum()
        loss.backward()
        optimizer.step()

        mgr = CheckpointManager(str(tmp_path))
        ckpt_path = mgr.save_checkpoint(
            step=10, components={}, optimizer=optimizer, config={"lora_name": "test"},
        )

        # Create fresh optimizer and load
        module2 = _make_linear_module()
        optimizer2 = _make_optimizer(module2)
        state = mgr.load_checkpoint(ckpt_path, optimizer=optimizer2)

        assert "optimizer" in state.components_loaded

    def test_load_peft_adapters(self, tmp_path):
        """PEFT adapters should be auto-detected and loaded."""
        peft = FakePeftModel("unet")
        mgr = CheckpointManager(str(tmp_path))
        ckpt_path = mgr.save_checkpoint(
            step=30, components={"unet": peft}, config={"lora_name": "test"},
        )

        # Create fresh PEFT model and load
        peft2 = FakePeftModel("unet")
        state = mgr.load_checkpoint(ckpt_path, peft_components={"unet": peft2})

        assert "unet" in state.adapters_loaded
        assert peft2._loaded_from is not None

    def test_load_missing_path_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent path."""
        mgr = CheckpointManager(str(tmp_path))

        with pytest.raises(FileNotFoundError):
            mgr.load_checkpoint("/nonexistent/checkpoint-999")

    def test_load_with_config_validation(self, tmp_path):
        """Compatible config should load without error."""
        ckpt_path = self._create_checkpoint(
            tmp_path, config={"lora_name": "test", "network_rank": 16, "network_alpha": 8.0},
        )
        mgr = CheckpointManager(str(tmp_path))

        state = mgr.load_checkpoint(
            ckpt_path,
            current_config={"network_rank": 16, "network_alpha": 8.0, "learning_rate": 2e-4},
        )

        assert state.global_step > 0

    def test_load_skip_scheduler_on_override(self, tmp_path):
        """When lr_scheduler changes, scheduler state loading should be skipped."""
        module = _make_linear_module()
        optimizer = _make_optimizer(module)
        scheduler = _make_scheduler(optimizer)
        mgr = CheckpointManager(str(tmp_path))
        ckpt_path = mgr.save_checkpoint(
            step=10, components={}, optimizer=optimizer, scheduler=scheduler,
            config={"lora_name": "test", "lr_scheduler": "constant"},
        )

        # Resume with different scheduler
        module2 = _make_linear_module()
        optimizer2 = _make_optimizer(module2)
        scheduler2 = _make_scheduler(optimizer2)
        state = mgr.load_checkpoint(
            ckpt_path, optimizer=optimizer2, scheduler=scheduler2,
            current_config={"lr_scheduler": "cosine"},
        )

        # Scheduler should NOT be in loaded components
        assert "scheduler" not in state.components_loaded


# ── Config Overrides ─────────────────────────────────────────────────────


class TestConfigOverrides:
    """Tests for apply_overrides()."""

    def test_safe_override_applied(self):
        """Safe overrides (LR, batch size) should be applied."""
        ckpt = {"learning_rate": 1e-4, "train_batch_size": 1}
        curr = {"learning_rate": 2e-4, "train_batch_size": 2}

        merged = apply_overrides(ckpt, curr)

        assert merged["learning_rate"] == 2e-4
        assert merged["train_batch_size"] == 2

    def test_warning_override_applied(self):
        """Warning overrides should be applied (with logging)."""
        ckpt = {"noise_offset": 0.0}
        curr = {"noise_offset": 0.05}

        merged = apply_overrides(ckpt, curr)

        assert merged["noise_offset"] == 0.05

    def test_blocked_override_raises(self):
        """Blocked overrides (rank, alpha) should raise ValueError."""
        ckpt = {"network_rank": 16}
        curr = {"network_rank": 32}

        with pytest.raises(ValueError, match="Cannot override.*network_rank"):
            apply_overrides(ckpt, curr)

    def test_no_change_passthrough(self):
        """Identical values should pass through unchanged."""
        config = {"learning_rate": 1e-4, "network_rank": 16}

        merged = apply_overrides(config, dict(config))

        assert merged == config

    def test_unknown_keys_applied(self):
        """Unknown keys (family-specific) should be applied silently."""
        ckpt = {"custom_flux_param": 3.0}
        curr = {"custom_flux_param": 5.0}

        merged = apply_overrides(ckpt, curr)

        assert merged["custom_flux_param"] == 5.0


# ── Compatibility Validation ─────────────────────────────────────────────


class TestCompatibilityValidation:
    """Tests for validate_compatibility()."""

    def test_compatible_config(self):
        """Same rank/alpha should return no warnings."""
        ckpt = {"network_rank": 16, "network_alpha": 8.0}
        curr = {"network_rank": 16, "network_alpha": 8.0}

        warnings = validate_compatibility(ckpt, curr)

        assert warnings == []

    def test_rank_mismatch_raises(self):
        """Different rank should raise ValueError."""
        ckpt = {"network_rank": 16}
        curr = {"network_rank": 32}

        with pytest.raises(ValueError, match="network_rank"):
            validate_compatibility(ckpt, curr)

    def test_alpha_mismatch_raises(self):
        """Different alpha should raise ValueError."""
        ckpt = {"network_alpha": 8.0}
        curr = {"network_alpha": 16.0}

        with pytest.raises(ValueError, match="network_alpha"):
            validate_compatibility(ckpt, curr)

    def test_model_definition_change_warns(self):
        """Different model_definition should produce a warning."""
        ckpt = {"model_definition": "klein_9b"}
        curr = {"model_definition": "dev"}

        warnings = validate_compatibility(ckpt, curr)

        assert len(warnings) == 1
        assert "Model definition changed" in warnings[0]

    def test_te_training_change_warns(self):
        """Changed train_text_encoder should produce a warning."""
        ckpt = {"train_text_encoder": False}
        curr = {"train_text_encoder": True}

        warnings = validate_compatibility(ckpt, curr)

        assert len(warnings) == 1
        assert "train_text_encoder" in warnings[0]


# ── Checkpoint Inspector ─────────────────────────────────────────────────


class TestCheckpointInspector:
    """Tests for inspect_checkpoint()."""

    def _create_full_checkpoint(self, tmp_path) -> str:
        """Create a checkpoint with various component files."""
        ckpt_dir = str(tmp_path / "checkpoint-100")
        os.makedirs(ckpt_dir)

        # training_state.json
        with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
            json.dump({"global_step": 100, "config": {"lora_name": "test"}, "timestamp": 1234567890.0}, f)

        # Standard .pt files
        torch.save({"state": "opt"}, os.path.join(ckpt_dir, "optimizer.pt"))
        torch.save({"state": "sch"}, os.path.join(ckpt_dir, "scheduler.pt"))
        torch.save({"state": "ema"}, os.path.join(ckpt_dir, "ema_shadow.pt"))

        # PEFT adapter subdir
        adapter_dir = os.path.join(ckpt_dir, "unet")
        os.makedirs(adapter_dir)
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
            json.dump({"peft_type": "LORA"}, f)
        torch.save({"dummy": torch.zeros(1)}, os.path.join(adapter_dir, "adapter_model.bin"))

        return ckpt_dir

    def test_inspector_valid_checkpoint(self, tmp_path):
        """Should return valid=True with correct metadata."""
        ckpt_dir = self._create_full_checkpoint(tmp_path)

        result = inspect_checkpoint(ckpt_dir)

        assert result["valid"] is True
        assert result["global_step"] == 100
        assert result["config"]["lora_name"] == "test"

    def test_inspector_filesystem_scan(self, tmp_path):
        """Should detect .pt files and PEFT adapter dirs."""
        ckpt_dir = self._create_full_checkpoint(tmp_path)

        result = inspect_checkpoint(ckpt_dir)

        assert result["has_optimizer"] is True
        assert result["has_scheduler"] is True
        assert result["has_ema"] is True
        assert "unet" in result["adapters"]
        assert result["total_size_bytes"] > 0

    def test_inspector_missing_path(self):
        """Non-existent path should return valid=False."""
        result = inspect_checkpoint("/nonexistent/checkpoint-999")

        assert result["valid"] is False
        assert "does not exist" in result["error"]

    def test_inspector_no_training_state(self, tmp_path):
        """Directory without training_state.json should return valid=False."""
        empty_dir = str(tmp_path / "empty")
        os.makedirs(empty_dir)

        result = inspect_checkpoint(empty_dir)

        assert result["valid"] is False
        assert "training_state.json" in result["error"]

    def test_inspector_file_sizes(self, tmp_path):
        """File sizes should be positive integers."""
        ckpt_dir = self._create_full_checkpoint(tmp_path)

        result = inspect_checkpoint(ckpt_dir)

        for name, size in result["files"].items():
            assert isinstance(size, int)
            assert size > 0, f"File {name} has zero size"


# ── BaseTrainer State ────────────────────────────────────────────────────


class TestBaseTrainerState:
    """Tests for BaseTrainer save_state/load_state methods."""

    def test_save_state_creates_file(self, tmp_path, mock_definition, mock_config):
        """save_state creates a state file at the given path."""
        from app.engine.core.interfaces import BaseTrainer

        class TestTrainer(BaseTrainer):
            async def setup(self): pass
            async def load_model(self): pass
            async def prepare_data(self): pass
            async def train(self): pass

        trainer = TestTrainer(mock_definition, mock_config)
        trainer.epoch = 5
        trainer.global_step = 500

        state_path = str(tmp_path / "test_state.pt")
        trainer.save_state(state_path)

        assert os.path.exists(state_path), "State file not created"

    def test_load_state_restores_values(self, tmp_path, mock_definition, mock_config):
        """load_state restores epoch and global_step from saved state."""
        from app.engine.core.interfaces import BaseTrainer

        class TestTrainer(BaseTrainer):
            async def setup(self): pass
            async def load_model(self): pass
            async def prepare_data(self): pass
            async def train(self): pass

        # Save
        trainer = TestTrainer(mock_definition, mock_config)
        trainer.epoch = 3
        trainer.global_step = 300

        state_path = str(tmp_path / "test_state.pt")
        trainer.save_state(state_path)

        # Load into fresh trainer
        trainer2 = TestTrainer(mock_definition, mock_config)
        assert trainer2.global_step == 0
        trainer2.load_state(state_path)

        assert trainer2.epoch == 3
        assert trainer2.global_step == 300
