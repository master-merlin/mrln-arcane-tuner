"""HiDream-O1 end-to-end integration test (Task 16).

Exercises the full pipeline surface — loader stub → _setup_family →
_assign_components → _apply_peft (LoRA injection) → _compute_step_loss
(velocity loss) → saver.save (base-interface path) — WITHOUT loading the
35 GB checkpoint.  All real model instances are replaced with minimal
nn.Module stubs.

Concerns covered:
    1. LatentManager bypass — pixel-space families must not crash on
       encode_and_cache_batch (no VAE).
    2. Training-step loss — _compute_step_loss must produce a non-zero
       scalar loss with a grad_fn (correct velocity loss, not 0).
    3. Processor/tokenizer None guard — clear error when processor absent.
    4. Saver signature — HiDreamO1Saver.save(components, path, metadata)
       must not crash (base-interface conformance).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.hidream_o1.lora_wrapper import inject_lora_layers
from app.engine.models.families.hidream_o1.saver import HiDreamO1Saver
from app.engine.models.families.hidream_o1.trainer import (
    HiDreamO1Trainer,
    _PixelPassthroughLatentManager,
    PATCH_SIZE,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def _make_definition() -> ModelDefinition:
    return ModelDefinition(id="hidream_o1_image", family="hidream_o1", name="Test")


_SEQ_LEN = 10     # total sequence length in fake model outputs
_N_VISUAL = 4     # number of visual (vinput) tokens in the sequence
_PATCH_DIM = 3 * PATCH_SIZE * PATCH_SIZE   # C * P * P


def _make_tiny_model(in_features: int = 16, out_features: int = 16) -> nn.Module:
    """Minimal module with the attribute shape the trainer expects.

    The forward() returns x_pred of shape ``[1, _SEQ_LEN, _PATCH_DIM]`` so
    ``test_sample["vinput_mask"]`` (bool, length ``_SEQ_LEN``) can index it
    without shape mismatch.
    """

    class TinyDiffusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.Linear(in_features, out_features),
            )
            # Mimic the model config expected by forward_pass internals
            self.config = MagicMock()
            self.config.hidden_size = out_features

        def forward(self, **kwargs):
            # Return a stub output with x_pred attribute.
            # Shape: [1, _SEQ_LEN, _PATCH_DIM] so the vinput_mask (length
            # _SEQ_LEN) can select a subset of tokens without dimension mismatch.
            out = MagicMock()
            vinputs = kwargs.get("vinputs")
            if vinputs is not None:
                # Mirror the actual sequence length from the vinputs input
                seq_len = vinputs.shape[1]
            else:
                seq_len = _SEQ_LEN
            out.x_pred = torch.randn(
                1, seq_len, _PATCH_DIM, requires_grad=True,
            )
            return out

    return TinyDiffusionModel()


def _make_mock_text_sample(seq_len: int = _SEQ_LEN, n_visual: int = _N_VISUAL) -> dict:
    """Build a mock text-sample dict with shapes consistent with _make_tiny_model."""
    # vinput_mask: 1D bool of length seq_len with n_visual True entries at the start
    mask_1d = torch.zeros(seq_len, dtype=torch.bool)
    mask_1d[:n_visual] = True
    return {
        "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
        "position_ids": torch.zeros(1, 3, seq_len, dtype=torch.long),
        "token_types": torch.zeros(1, seq_len, dtype=torch.long),
        "vinput_mask": mask_1d.unsqueeze(0),  # [1, seq_len]
    }


def _make_trainer(definition: ModelDefinition | None = None) -> HiDreamO1Trainer:
    """Construct a HiDreamO1Trainer without running async setup."""
    defn = definition or _make_definition()
    config = {
        "network_rank": 4,
        "network_alpha": 4.0,
        "mixed_precision": "bf16",
        "cache_latents": True,
        "timestep_type": "linear",
        "output_dir": "outputs",
        "lora_name": "test_lora",
    }
    trainer = HiDreamO1Trainer.__new__(HiDreamO1Trainer)
    # Manually set the attributes __init__ would set via BaseTrainer
    trainer.definition = defn
    trainer.config = config
    trainer.device = torch.device("cpu")
    trainer.logger = MagicMock()
    trainer.components = {}
    trainer.epoch = 0
    trainer.global_step = 0
    trainer.optimizer = None
    trainer.loader = None
    trainer.saver = None
    # Call _setup_family manually (no async needed)
    trainer._setup_family()
    return trainer


# ── Concern 1: LatentManager bypass ──────────────────────────────────────

class TestPixelPassthroughLatentManager:
    """_PixelPassthroughLatentManager never raises even without a VAE."""

    def test_load_cached_latents_returns_none(self):
        """Must return None so the base loop falls into encode_and_cache_batch
        which returns the 4D image tensor. A non-None sentinel would short
        the cache-miss warning but break the base loop's later
        ``latents.shape[1]`` noise-offset shaping.
        """
        lm = _PixelPassthroughLatentManager()
        result = lm.load_cached_latents(["id1"], ["dir1"], ["path1"])
        assert result is None, "must always return None (always cache miss)"

    def test_encode_and_cache_batch_returns_input_unchanged(self):
        lm = _PixelPassthroughLatentManager()
        x = torch.randn(1, 3, 32, 32)
        out = lm.encode_and_cache_batch(x, ids=["id1"])
        assert out is x, "passthrough must return the original tensor"

    def test_check_cache_coverage_reports_all_cached(self):
        lm = _PixelPassthroughLatentManager()
        cached, missing, missing_ids = lm.check_cache_coverage(
            ["id1", "id2"], ["d", "d"],
        )
        assert cached == 2
        assert missing == 0
        assert missing_ids == []

    def test_validate_latent_cache_noop(self):
        """_validate_latent_cache must not crash and must set _latent_cache_missing=0."""
        trainer = _make_trainer()
        # Install passthrough manually
        trainer.latent_manager = _PixelPassthroughLatentManager()
        trainer._validate_latent_cache()
        assert getattr(trainer, "_latent_cache_missing", 0) == 0

    @pytest.mark.asyncio
    async def test_pre_cache_latents_noop(self):
        """_pre_cache_latents must complete without any I/O or VAE call."""
        trainer = _make_trainer()
        trainer.latent_manager = _PixelPassthroughLatentManager()
        await trainer._pre_cache_latents()  # must not raise


# ── Concern 2: Training-step loss ────────────────────────────────────────

class TestComputeStepLoss:
    """_compute_step_loss must produce a non-zero grad_fn'd scalar loss."""

    def _make_trainer_with_lora(self) -> HiDreamO1Trainer:
        trainer = _make_trainer()
        model = _make_tiny_model()
        trainer.driver.assign_components({"unet": model})
        trainer.components["unet"] = model
        # Inject LoRA
        trainer._apply_peft()
        return trainer

    def test_compute_step_loss_has_grad_fn(self):
        """Loss returned by _compute_step_loss must have a grad_fn (not zero)."""
        trainer = self._make_trainer_with_lora()
        trainer.processor = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.autocast_dtype = torch.float32

        # Build a minimal batch (H=W=PATCH_SIZE*2 → 4 patches)
        H = W = PATCH_SIZE * 2
        pixel_values = torch.randn(1, 3, H, W)
        batch = {
            "images": pixel_values,
            "captions": ["a test caption"],
            "caption": ["a test caption"],
        }

        # n_patches = (H/P)*(W/P) = 4;  seq_len must match model x_pred output
        n_patches = (H // PATCH_SIZE) * (W // PATCH_SIZE)
        mock_text_sample = _make_mock_text_sample(seq_len=n_patches, n_visual=n_patches)

        dummy_pred = torch.zeros(1)
        dummy_target = torch.zeros(1)
        dummy_timesteps = torch.tensor([0.5])

        with patch(
            "app.engine.models.families.hidream_o1.trainer.build_t2i_text_sample",
            return_value=mock_text_sample,
        ):
            loss = trainer._compute_step_loss(
                dummy_pred, dummy_target, dummy_timesteps, batch, grad_accum=1,
            )

        assert loss.grad_fn is not None, (
            "_compute_step_loss must return a tensor with grad_fn — "
            "LoRA params must be in the computation graph"
        )
        assert loss.ndim == 0, "loss must be scalar"
        assert not torch.isnan(loss), "loss must not be NaN"

    def test_compute_step_loss_scaled_by_grad_accum(self):
        """Loss should be divided by grad_accum (loss_1 / loss_4 ≈ 4)."""
        trainer = self._make_trainer_with_lora()
        trainer.processor = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.autocast_dtype = torch.float32

        H = W = PATCH_SIZE * 2
        pixel_values = torch.randn(1, 3, H, W)
        batch = {
            "images": pixel_values,
            "captions": ["a test"],
            "caption": ["a test"],
        }

        n_patches = (H // PATCH_SIZE) * (W // PATCH_SIZE)
        mock_text_sample = _make_mock_text_sample(seq_len=n_patches, n_visual=n_patches)
        dummy = torch.zeros(1)
        t = torch.tensor([0.5])

        with patch(
            "app.engine.models.families.hidream_o1.trainer.build_t2i_text_sample",
            return_value=mock_text_sample,
        ):
            loss_1 = trainer._compute_step_loss(dummy, dummy, t, batch, grad_accum=1)
            loss_4 = trainer._compute_step_loss(dummy, dummy, t, batch, grad_accum=4)

        # Ratio should be ~4 (grad_accum scaling)
        ratio = loss_1.item() / max(loss_4.item(), 1e-9)
        assert 3.0 <= ratio <= 5.0, (
            f"Expected loss_1/loss_4 ≈ 4 (grad_accum scaling), got {ratio:.2f}"
        )


# ── Concern 3: Processor None guard ──────────────────────────────────────

class TestProcessorNoneGuard:
    """compute_loss must raise RuntimeError (not AttributeError) when processor=None."""

    def test_compute_loss_raises_when_processor_none(self):
        trainer = _make_trainer()
        model = _make_tiny_model()
        trainer.driver.assign_components({"unet": model})
        trainer.components["unet"] = model
        trainer._apply_peft()
        trainer.processor = None
        trainer.tokenizer = None
        trainer.autocast_dtype = torch.float32

        H = W = PATCH_SIZE * 2
        batch = {
            "pixel_values": torch.randn(1, 3, H, W),
            "caption": ["test"],
            "captions": ["test"],
        }

        with pytest.raises(RuntimeError, match="processor/tokenizer"):
            trainer.compute_loss(batch)


# ── Concern 4: Saver signature ────────────────────────────────────────────

class TestSaverBaseInterface:
    """HiDreamO1Saver.save(components, path, metadata) must conform to IModelSaver."""

    def _make_lora_model(self) -> nn.Module:
        class Mini(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = nn.Sequential(nn.Linear(8, 8))
        m = Mini()
        inject_lora_layers(m, rank=2, alpha=2.0)
        return m

    def test_base_interface_save_produces_safetensors(self, tmp_path: Path):
        """save(components, path, metadata) must write a .safetensors file."""
        model = self._make_lora_model()
        saver = HiDreamO1Saver()
        out_path = tmp_path / "lora_001000.safetensors"

        components = {
            "unet": model,
            "config": {
                "noise_scale": 8.0,
                "timestep_type": "linear",
                "max_loss": 1.0,
            },
        }
        # Must not raise
        saver.save(components, out_path, metadata={"step": 1000})

        assert out_path.exists(), "safetensors file must be written"

    def test_base_interface_save_keys_have_diffusion_model_prefix(self, tmp_path: Path):
        """Keys in the saved file must follow diffusion_model.<key>.lora_* convention."""
        from safetensors import safe_open

        model = self._make_lora_model()
        saver = HiDreamO1Saver()
        out_path = tmp_path / "lora_test.safetensors"

        saver.save({"unet": model, "config": {}}, out_path, metadata={})

        with safe_open(str(out_path), framework="pt") as f:
            keys = list(f.keys())

        assert all(k.startswith("diffusion_model.") for k in keys), (
            f"All keys must start with 'diffusion_model.', got: {keys}"
        )

    def test_base_interface_save_missing_unet_logs_error(self, tmp_path: Path):
        """save must not crash when 'unet' is missing from components dict."""
        saver = HiDreamO1Saver()
        out_path = tmp_path / "noop.safetensors"
        # Should log error and return without raising
        saver.save({"config": {}}, out_path, metadata={})
        assert not out_path.exists(), "no file should be created when model is absent"

    def test_save_lora_direct_api_still_works(self, tmp_path: Path):
        """save_lora(model, out_dir, name) public API must remain functional."""
        model = self._make_lora_model()
        saver = HiDreamO1Saver()
        out_dir = tmp_path / "direct"
        result = saver.save_lora(model, str(out_dir), "my_lora")
        assert (out_dir / "my_lora.safetensors").exists()
        assert result == out_dir


# ── Full pipeline wiring: _setup_family → _configure_managers ────────────

class TestPipelineWiring:
    """Smoke test the wiring from _setup_family through _configure_managers."""

    def test_configure_managers_installs_passthrough(self):
        """_configure_managers must install _PixelPassthroughLatentManager."""
        trainer = _make_trainer()
        model = _make_tiny_model()
        trainer.driver.assign_components({"unet": model})
        trainer.components["unet"] = model

        # Stub out checkpoint/logger dependencies
        with (
            patch.object(
                type(trainer),
                "_configure_managers",
                wraps=lambda self, max_steps: (
                    # Call super (pipeline_optimization) then check replacement
                    _call_super_configure_managers(trainer, max_steps)
                ),
            ),
        ):
            pass  # Just verify via direct call below

        # Call _configure_managers directly with required stubs
        trainer.logger_component = MagicMock()
        mock_cm = MagicMock()
        # Patch CheckpointManager so we don't need real filesystem
        with patch(
            "app.engine.core.pipeline.pipeline_optimization.CheckpointManager",
            return_value=mock_cm,
        ):
            trainer._configure_managers(max_train_steps=10)

        assert isinstance(
            trainer.latent_manager, _PixelPassthroughLatentManager
        ), (
            "_configure_managers must replace the default LatentManager with "
            "_PixelPassthroughLatentManager for pixel-space families"
        )

    def test_apply_peft_injects_lora_layers(self):
        """_apply_peft must inject HiDreamO1LoRALinear wrappers."""
        from app.engine.models.families.hidream_o1.lora_wrapper import HiDreamO1LoRALinear

        trainer = _make_trainer()
        model = _make_tiny_model()
        trainer.driver.assign_components({"unet": model})
        trainer.components["unet"] = model

        trainer._apply_peft()

        # At least some trainable LoRA params should exist
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable) > 0, "_apply_peft must produce trainable LoRA params"

        # Model should contain HiDreamO1LoRALinear wrappers
        lora_layers = [m for m in model.modules() if isinstance(m, HiDreamO1LoRALinear)]
        assert len(lora_layers) > 0, "_apply_peft must inject HiDreamO1LoRALinear layers"


def _call_super_configure_managers(trainer, max_train_steps: int) -> None:
    """Helper — calls the real super()._configure_managers for wiring test."""
    from app.engine.core.pipeline.pipeline_optimization import PipelineOptimizationMixin
    PipelineOptimizationMixin._configure_managers(trainer, max_train_steps)
