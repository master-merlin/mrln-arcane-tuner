"""
Tests for GenericTrainingPipeline — shared base class.

Covers: default hooks, helper methods, freeze/quantize/PEFT logic,
configuration resolution, batch construction utilities, and trainable
component collection.
"""

import torch
import torch.nn as nn
from unittest.mock import MagicMock
from typing import Any


# ── Minimal Concrete Subclass ────────────────────────────────────────────

# We can't instantiate GenericTrainingPipeline directly because it has
# abstract methods. Create a minimal concrete subclass for testing the
# shared (non-abstract) methods.


class _StubPipeline:
    """Minimal stub that mimics pipeline attributes without importing."""

    def __init__(self):
        self.config: dict[str, Any] = {}
        self.components: dict[str, Any] = {}
        self.device = "cpu"
        self.logger = MagicMock()
        self.autocast_dtype = torch.float32


# ── Default Hook Tests ───────────────────────────────────────────────────


class TestDefaultHooks:
    """Tests for the default (non-abstract) hook implementations."""

    def test_compute_loss_weight_default_none(self):
        """Default compute_loss_weight should return None (uniform weighting)."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        # Access via the class — default method returns None
        result = GenericTrainingPipeline.compute_loss_weight(
            MagicMock(), torch.tensor([500.0])
        )
        assert result is None

    def test_build_batch_extra_default_empty(self):
        """Default build_batch_extra should return empty dict (no driver override)."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        stub = MagicMock()
        stub._driver_hook_override.return_value = None  # driver does not override
        result = GenericTrainingPipeline.build_batch_extra(stub, [])
        assert result == {}

    def test_prepare_latents_default_passthrough(self):
        """Default prepare_latents_for_training delegates to driver.prepare_latents."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        latents = torch.randn(1, 4, 64, 64)
        stub = MagicMock()
        stub.driver = MagicMock()
        stub.driver.prepare_latents.return_value = latents
        result = GenericTrainingPipeline.prepare_latents_for_training(stub, latents)
        assert torch.equal(result, latents)

    def test_get_te_cache_default_none(self):
        """Default get_te_cache should return None when text_cache is empty."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        stub = MagicMock()
        stub.text_cache = {}
        stub._driver_hook_override.return_value = None  # driver does not override
        result = GenericTrainingPipeline.get_te_cache(stub)
        assert result is None

    def test_create_sampler_default_none(self):
        """Default _create_sampler should return None."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        result = GenericTrainingPipeline._create_sampler(MagicMock())
        assert result is None

    def test_init_scheduler_default_none(self):
        """Default init_scheduler should return None (flow-matching)."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        stub = MagicMock()
        stub._driver_hook_override.return_value = None  # driver does not override
        result = GenericTrainingPipeline.init_scheduler(stub)
        assert result is None

    def test_compute_target_default_velocity(self):
        """Default compute_target should return noise - latents (flow-matching)."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        latents = torch.tensor([1.0, 2.0, 3.0])
        noise = torch.tensor([4.0, 5.0, 6.0])
        timesteps = torch.tensor([500.0])

        stub = MagicMock()
        stub._driver_hook_override.return_value = None  # driver does not override
        result = GenericTrainingPipeline.compute_target(stub, latents, noise, timesteps)
        expected = noise - latents
        assert torch.allclose(result, expected)

    def test_add_noise_default_linear(self):
        """Default add_noise should delegate to NoiseInterpolation."""
        from app.engine.core.pipeline import GenericTrainingPipeline
        from app.engine.strategies.noise_interpolation import NoiseInterpolation

        stub = MagicMock()
        stub.noise_interpolation = NoiseInterpolation("linear")
        stub._driver_hook_override.return_value = None  # driver does not override

        latents = torch.randn(2, 4, 8, 8)
        noise = torch.randn(2, 4, 8, 8)
        timesteps = torch.tensor([500.0, 250.0])

        result = GenericTrainingPipeline.add_noise(stub, latents, noise, timesteps)
        assert result.shape == latents.shape

        # Verify linear interpolation: at t=500/1000=0.5, result = 0.5*latents + 0.5*noise
        t_01 = timesteps / 1000.0
        expected_0 = (1.0 - t_01[0]) * latents[0] + t_01[0] * noise[0]
        assert torch.allclose(result[0], expected_0, atol=1e-6)

    def test_on_epoch_end_default_noop(self):
        """Default on_epoch_end should be a no-op (no exception)."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        # Should not raise
        GenericTrainingPipeline.on_epoch_end(MagicMock(), epoch=0)


# ── Offload VAE ──────────────────────────────────────────────────────────


class TestOffloadVae:
    """Tests for _offload_vae."""

    def test_offloads_vae_to_cpu(self):
        """With low_vram=True, VAE should be moved to CPU."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        vae = MagicMock()
        stub = MagicMock()
        stub.config = {"low_vram": True}
        stub.vae = vae
        stub.logger = MagicMock()

        GenericTrainingPipeline._offload_vae(stub)
        vae.to.assert_called_once_with("cpu")

    def test_skips_offload_when_low_vram_false(self):
        """With low_vram=False, VAE should not be moved."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        vae = MagicMock()
        stub = MagicMock()
        stub.config = {"low_vram": False}
        stub.vae = vae

        GenericTrainingPipeline._offload_vae(stub)
        vae.to.assert_not_called()

    def test_handles_no_vae(self):
        """When vae is None, should not crash."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        stub = MagicMock()
        stub.config = {"low_vram": True}
        stub.vae = None
        stub.logger = MagicMock()

        # Should not raise
        GenericTrainingPipeline._offload_vae(stub)


# ── Loading Dtype Resolution ────────────────────────────────────────


class TestResolveDtype:
    """Tests for _resolve_loading_dtype."""

    def test_bf16_returns_bfloat16(self):
        """mixed_precision='bf16' should return torch.bfloat16."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        stub = MagicMock()
        stub.config = {"mixed_precision": "bf16"}
        result = GenericTrainingPipeline._resolve_loading_dtype(stub)
        assert result == torch.bfloat16

    def test_fp16_returns_float32(self):
        """mixed_precision='fp16' should return torch.float32 (AMP needs fp32 params)."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        stub = MagicMock()
        stub.config = {"mixed_precision": "fp16"}
        result = GenericTrainingPipeline._resolve_loading_dtype(stub)
        assert result == torch.float32

    def test_default_returns_float32(self):
        """Missing mixed_precision should default to fp16 → float32."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        stub = MagicMock()
        stub.config = {}
        result = GenericTrainingPipeline._resolve_loading_dtype(stub)
        assert result == torch.float32


# ── Freeze All ───────────────────────────────────────────────────────────


class TestFreezeAll:
    """Tests for _freeze_all."""

    def test_freeze_all_freezes_model_and_vae(self):
        """_freeze_all should call requires_grad_(False) on all components."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        model = nn.Linear(4, 4)
        vae = nn.Linear(4, 4)
        te = nn.Linear(4, 4)

        stub = MagicMock()
        stub.components = {"unet": model, "vae": vae, "text_encoder": te}
        stub.logger = MagicMock()

        # Bind the unbound methods
        GenericTrainingPipeline._get_primary_model(stub)  # triggers mock
        stub._get_primary_model = lambda: model
        stub._get_text_encoders = lambda: {"text_encoder": te}

        GenericTrainingPipeline._freeze_all(stub)

        # Verify all params frozen
        for p in model.parameters():
            assert not p.requires_grad
        for p in vae.parameters():
            assert not p.requires_grad
        for p in te.parameters():
            assert not p.requires_grad


# ── Get Text Encoders ────────────────────────────────────────────────────


class TestGetTextEncoders:
    """Tests for _get_text_encoders (delegates to driver)."""

    def test_single_text_encoder(self):
        """Single text_encoder via driver should be returned."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        te = nn.Linear(4, 4)
        stub = MagicMock()
        stub.driver = MagicMock()
        stub.driver.get_text_encoders.return_value = {"text_encoder": te}

        result = GenericTrainingPipeline._get_text_encoders(stub)
        assert result == {"text_encoder": te}

    def test_dual_text_encoders(self):
        """SDXL-style dual TEs via driver should both be returned."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        te1 = nn.Linear(4, 4)
        te2 = nn.Linear(4, 4)
        stub = MagicMock()
        stub.driver = MagicMock()
        stub.driver.get_text_encoders.return_value = {
            "text_encoder_1": te1,
            "text_encoder_2": te2,
        }

        result = GenericTrainingPipeline._get_text_encoders(stub)
        assert result == {"text_encoder_1": te1, "text_encoder_2": te2}

    def test_no_text_encoder_returns_empty(self):
        """Missing TEs via driver should return empty dict."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        stub = MagicMock()
        stub.driver = MagicMock()
        stub.driver.get_text_encoders.return_value = {}

        result = GenericTrainingPipeline._get_text_encoders(stub)
        assert result == {}


# ── Update Primary Model ─────────────────────────────────────────────────


class TestUpdatePrimaryModel:
    """Tests for _update_primary_model reference updating."""

    def test_updates_components_dict(self):
        """_update_primary_model should update components['unet']."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        old_model = nn.Linear(4, 4)
        new_model = nn.Linear(4, 4)
        stub = MagicMock()
        stub.components = {"unet": old_model}

        GenericTrainingPipeline._update_primary_model(stub, new_model)
        assert stub.components["unet"] is new_model

    def test_updates_self_model_if_exists(self):
        """If self.model exists, it should be updated too."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        old = nn.Linear(4, 4)
        new = nn.Linear(4, 4)

        class Stub:
            components = {"unet": old}
            model = old
            unet = old

        stub = Stub()
        GenericTrainingPipeline._update_primary_model(stub, new)
        assert stub.model is new
        assert stub.unet is new


# ── Build Trainable Components ───────────────────────────────────────────


class TestBuildTrainableComponents:
    """Tests for _build_trainable_components."""

    def test_returns_unet_only_by_default(self):
        """Without TE training, should return only unet."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        model = nn.Linear(4, 4)
        stub = MagicMock()
        stub.config = {}
        stub._get_primary_model = MagicMock(return_value=model)
        stub._get_text_encoders = MagicMock(return_value={})

        result = GenericTrainingPipeline._build_trainable_components(stub)
        assert "unet" in result
        assert result["unet"] is model

    def test_includes_text_encoders_when_training(self):
        """With train_text_encoder=True, should include TEs."""
        from app.engine.core.pipeline import GenericTrainingPipeline

        model = nn.Linear(4, 4)
        te1 = nn.Linear(4, 4)
        stub = MagicMock()
        stub.config = {"train_text_encoder": True}
        stub._get_primary_model = MagicMock(return_value=model)
        stub._get_text_encoders = MagicMock(return_value={"text_encoder": te1})

        result = GenericTrainingPipeline._build_trainable_components(stub)
        assert "unet" in result
        assert "text_encoder" in result
