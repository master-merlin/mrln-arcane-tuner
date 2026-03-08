"""
Tests for definition-driven model parameters:
 - ConfigHarvester: reads HF component config.json, extracts ALL params
   with dot-namespace notation (e.g. transformer.hidden_size, vae.latent_channels)

Note: Flux2Params tests are skipped — the BFL Flux2Params class was replaced
by diffusers Flux2Transformer2DModel in the clean FLUX.2 refactor.
"""

import os
import json
from typing import Any

from app.engine.utils.config_harvester import harvest


# ── ConfigHarvester ──────────────────────────────────────────────────────


class TestConfigHarvester:
    """Tests for config_harvester.harvest()."""

    def _write_config(self, dirpath: str, data: dict[str, Any]):
        """Helper to write a config.json file."""
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, "config.json"), "w") as f:
            json.dump(data, f)

    def _write_scheduler_config(self, dirpath: str, data: dict[str, Any]):
        """Helper to write a scheduler_config.json file."""
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, "scheduler_config.json"), "w") as f:
            json.dump(data, f)

    def test_harvest_empty_dir(self, tmp_path):
        """Empty dir should return empty dict."""
        result = harvest(str(tmp_path))
        assert result == {}

    def test_harvest_nonexistent_dir(self):
        """Non-existent dir should return empty dict."""
        result = harvest("/nonexistent/path/xyz")
        assert result == {}

    def test_harvest_transformer_config(self, tmp_path):
        """Should extract all transformer params with transformer.* namespace."""
        self._write_config(str(tmp_path / "transformer"), {
            "num_attention_heads": 48,
            "attention_head_dim": 128,
            "joint_attention_dim": 15360,
            "num_layers": 8,
            "num_single_layers": 48,
            "in_channels": 128,
            "mlp_ratio": 3.0,
            "guidance_embeds": True,
            "axes_dims_rope": [32, 32, 32, 32],
            "rope_theta": 2000,
        })

        result = harvest(str(tmp_path))
        assert result["transformer.num_attention_heads"] == 48
        assert result["transformer.attention_head_dim"] == 128
        assert result["transformer.joint_attention_dim"] == 15360
        assert result["transformer.num_layers"] == 8
        assert result["transformer.num_single_layers"] == 48
        assert result["transformer.in_channels"] == 128
        assert result["transformer.mlp_ratio"] == 3.0
        assert result["transformer.guidance_embeds"] is True
        assert result["transformer.axes_dims_rope"] == [32, 32, 32, 32]
        assert result["transformer.rope_theta"] == 2000

    def test_harvest_hidden_size_derived(self, tmp_path):
        """hidden_size should be derived from num_heads * head_dim when not explicit."""
        self._write_config(str(tmp_path / "transformer"), {
            "num_attention_heads": 32,
            "attention_head_dim": 128,
        })

        result = harvest(str(tmp_path))
        assert result["transformer.hidden_size"] == 32 * 128

    def test_harvest_text_encoder_config(self, tmp_path):
        """Should extract all TE params with te.* namespace, including nested text_config."""
        self._write_config(str(tmp_path / "text_encoder"), {
            "text_config": {
                "hidden_size": 5120,
                "num_hidden_layers": 40,
                "model_type": "mistral",
                "max_position_embeddings": 131072,
            }
        })

        result = harvest(str(tmp_path))
        # Both the nested dict and flattened keys should be stored
        assert result["te.text_config.hidden_size"] == 5120
        assert result["te.text_config.num_hidden_layers"] == 40
        assert result["te.text_config.model_type"] == "mistral"
        assert result["te.text_config.max_position_embeddings"] == 131072

    def test_harvest_te_concat_layers_derived(self, tmp_path):
        """te.concat_layers should be derived as joint_attention_dim // te_hidden_size."""
        self._write_config(str(tmp_path / "transformer"), {
            "joint_attention_dim": 12288,
        })
        self._write_config(str(tmp_path / "text_encoder"), {
            "hidden_size": 4096,
        })

        result = harvest(str(tmp_path))
        assert result["te.concat_layers"] == 3

    def test_harvest_te_concat_layers_flux_dev(self, tmp_path):
        """te.concat_layers for Flux Dev: 15360 / 5120 = 3 (nested text_config)."""
        self._write_config(str(tmp_path / "transformer"), {
            "joint_attention_dim": 15360,
        })
        self._write_config(str(tmp_path / "text_encoder"), {
            "text_config": {"hidden_size": 5120},
        })

        result = harvest(str(tmp_path))
        assert result["te.concat_layers"] == 3

    def test_harvest_scheduler_config(self, tmp_path):
        """Should read scheduler_config.json from scheduler dir."""
        self._write_scheduler_config(str(tmp_path / "scheduler"), {
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "use_dynamic_shifting": True,
            "shift": 3.0,
            "base_shift": 0.5,
            "max_shift": 1.15,
        })

        result = harvest(str(tmp_path))
        assert result["scheduler._class_name"] == "FlowMatchEulerDiscreteScheduler"
        assert result["scheduler.use_dynamic_shifting"] is True
        assert result["scheduler.shift"] == 3.0
        assert result["scheduler.base_shift"] == 0.5

    def test_harvest_vae_config(self, tmp_path):
        """Should extract VAE params with vae.* namespace."""
        self._write_config(str(tmp_path / "vae"), {
            "in_channels": 3,
            "latent_channels": 16,
            "_class_name": "AutoencoderKL",
            "scaling_factor": 0.18215,
        })

        result = harvest(str(tmp_path))
        assert result["vae.in_channels"] == 3
        assert result["vae.latent_channels"] == 16
        assert result["vae._class_name"] == "AutoencoderKL"
        assert result["vae.scaling_factor"] == 0.18215

    def test_harvest_full_klein_repo(self, tmp_path):
        """Integration: full Klein 9B mock repo should give all expected params."""
        self._write_config(str(tmp_path / "transformer"), {
            "num_attention_heads": 32,
            "attention_head_dim": 128,
            "joint_attention_dim": 12288,
            "num_layers": 8,
            "num_single_layers": 24,
            "in_channels": 128,
            "mlp_ratio": 3.0,
            "guidance_embeds": False,
        })
        self._write_config(str(tmp_path / "text_encoder"), {
            "hidden_size": 4096,
            "num_hidden_layers": 36,
            "model_type": "qwen3",
        })
        self._write_scheduler_config(str(tmp_path / "scheduler"), {
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "shift": 3.0,
        })

        result = harvest(str(tmp_path))

        # Transformer
        assert result["transformer.hidden_size"] == 4096  # derived: 32 * 128
        assert result["transformer.num_attention_heads"] == 32
        assert result["transformer.joint_attention_dim"] == 12288
        assert result["transformer.num_layers"] == 8
        assert result["transformer.num_single_layers"] == 24
        assert result["transformer.guidance_embeds"] is False
        # TE
        assert result["te.hidden_size"] == 4096
        assert result["te.num_hidden_layers"] == 36
        assert result["te.model_type"] == "qwen3"
        # Derived: concat_layers
        assert result["te.concat_layers"] == 3
        # Scheduler
        assert result["scheduler._class_name"] == "FlowMatchEulerDiscreteScheduler"

    def test_harvest_unet_dir_alias(self, tmp_path):
        """Should check 'unet' dir as fallback for transformer namespace."""
        self._write_config(str(tmp_path / "unet"), {
            "num_attention_heads": 16,
            "in_channels": 4,
        })

        result = harvest(str(tmp_path))
        assert result["transformer.num_attention_heads"] == 16
        assert result["transformer.in_channels"] == 4

    def test_harvest_corrupt_json_skips(self, tmp_path):
        """Corrupt config.json should be skipped, not crash."""
        dirpath = str(tmp_path / "transformer")
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, "config.json"), "w") as f:
            f.write("{invalid json")

        result = harvest(str(tmp_path))
        assert result == {}

    def test_harvest_text_encoder_2(self, tmp_path):
        """Should extract TE2 params with te2.* namespace for SDXL dual-CLIP."""
        self._write_config(str(tmp_path / "text_encoder_2"), {
            "hidden_size": 1280,
            "num_hidden_layers": 32,
            "model_type": "clip_text_model",
            "max_position_embeddings": 77,
            "architectures": ["CLIPTextModelWithProjection"],
            "projection_dim": 1280,
        })

        result = harvest(str(tmp_path))
        assert result["te2.hidden_size"] == 1280
        assert result["te2.num_hidden_layers"] == 32
        assert result["te2.architectures"] == ["CLIPTextModelWithProjection"]
        assert result["te2.projection_dim"] == 1280

    def test_harvest_flux2_vae_generic(self, tmp_path):
        """Flux 2 VAE should be harvested generically like all components.

        The generic harvester does NOT inject Flux 1 scaling/shift defaults.
        If they're not in config.json, they won't appear in results.
        """
        self._write_config(str(tmp_path / "vae"), {
            "_class_name": "AutoencoderKLFlux2",
            "in_channels": 3,
            "latent_channels": 32,
            "batch_norm_eps": 0.0001,
            "batch_norm_momentum": 0.1,
        })

        result = harvest(str(tmp_path))
        assert result["vae._class_name"] == "AutoencoderKLFlux2"
        assert "vae.scaling_factor" not in result  # Not in config.json = not in result
        assert "vae.shift_factor" not in result    # Not in config.json = not in result
        assert result["vae.batch_norm_eps"] == 0.0001
        assert result["vae.batch_norm_momentum"] == 0.1

    def test_harvest_flux2_vae_explicit_values_preserved(self, tmp_path):
        """When config explicitly has scaling values, they should be harvested."""
        self._write_config(str(tmp_path / "vae"), {
            "_class_name": "AutoencoderKLFlux2",
            "scaling_factor": 0.42,
            "shift_factor": 0.15,
        })

        result = harvest(str(tmp_path))
        assert result["vae.scaling_factor"] == 0.42
        assert result["vae.shift_factor"] == 0.15

    def test_harvest_flux1_vae_scalars(self, tmp_path):
        """Flux 1 VAE (AutoencoderKL) with scaling/shift values should harvest them."""
        self._write_config(str(tmp_path / "vae"), {
            "_class_name": "AutoencoderKL",
            "in_channels": 3,
            "latent_channels": 16,
            "scaling_factor": 0.3611,
            "shift_factor": 0.1159,
        })

        result = harvest(str(tmp_path))
        assert result["vae._class_name"] == "AutoencoderKL"
        assert result["vae.scaling_factor"] == 0.3611
        assert result["vae.shift_factor"] == 0.1159
        # Flux 1 has no BN params in config.json = not in result
        assert "vae.batch_norm_eps" not in result

    def test_harvest_sdxl_unet(self, tmp_path):
        """SDXL UNet params should be under transformer.* namespace."""
        self._write_config(str(tmp_path / "unet"), {
            "_class_name": "UNet2DConditionModel",
            "cross_attention_dim": 2048,
            "sample_size": 128,
            "in_channels": 4,
            "block_out_channels": [320, 640, 1280],
        })

        result = harvest(str(tmp_path))
        assert result["transformer.cross_attention_dim"] == 2048
        assert result["transformer.sample_size"] == 128
        assert result["transformer.block_out_channels"] == [320, 640, 1280]

    def test_harvest_skips_hf_metadata(self, tmp_path):
        """Internal HF metadata keys should be filtered out."""
        self._write_config(str(tmp_path / "transformer"), {
            "_diffusers_version": "0.29.0",
            "_name_or_path": "some/path",
            "transformers_version": "4.40.0",
            "num_attention_heads": 32,
        })

        result = harvest(str(tmp_path))
        assert "transformer._diffusers_version" not in result
        assert "transformer._name_or_path" not in result
        assert "transformer.transformers_version" not in result
        assert result["transformer.num_attention_heads"] == 32

    def test_harvest_full_sdxl_repo(self, tmp_path):
        """Integration: full SDXL repo should yield all expected params."""
        self._write_config(str(tmp_path / "unet"), {
            "_class_name": "UNet2DConditionModel",
            "cross_attention_dim": 2048,
            "in_channels": 4,
            "out_channels": 4,
            "sample_size": 128,
            "block_out_channels": [320, 640, 1280],
            "layers_per_block": 2,
            "addition_embed_type": "text_time",
        })
        self._write_config(str(tmp_path / "text_encoder"), {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "model_type": "clip_text_model",
            "max_position_embeddings": 77,
            "architectures": ["CLIPTextModel"],
        })
        self._write_config(str(tmp_path / "text_encoder_2"), {
            "hidden_size": 1280,
            "num_hidden_layers": 32,
            "model_type": "clip_text_model",
            "max_position_embeddings": 77,
            "architectures": ["CLIPTextModelWithProjection"],
        })
        self._write_scheduler_config(str(tmp_path / "scheduler"), {
            "_class_name": "EulerDiscreteScheduler",
            "num_train_timesteps": 1000,
            "beta_schedule": "scaled_linear",
        })
        self._write_config(str(tmp_path / "vae"), {
            "_class_name": "AutoencoderKL",
            "latent_channels": 4,
            "scaling_factor": 0.13025,
        })

        result = harvest(str(tmp_path))

        # UNet → transformer namespace
        assert result["transformer.cross_attention_dim"] == 2048
        assert result["transformer.sample_size"] == 128
        assert result["transformer.block_out_channels"] == [320, 640, 1280]
        assert result["transformer.addition_embed_type"] == "text_time"
        # TE1
        assert result["te.hidden_size"] == 768
        assert result["te.num_hidden_layers"] == 12
        assert result["te.max_position_embeddings"] == 77
        # TE2
        assert result["te2.hidden_size"] == 1280
        assert result["te2.num_hidden_layers"] == 32
        assert result["te2.architectures"] == ["CLIPTextModelWithProjection"]
        # Scheduler
        assert result["scheduler._class_name"] == "EulerDiscreteScheduler"
        assert result["scheduler.num_train_timesteps"] == 1000
        # VAE
        assert result["vae._class_name"] == "AutoencoderKL"
        assert result["vae.latent_channels"] == 4
        assert result["vae.scaling_factor"] == 0.13025

    def test_harvest_all_keys_preserved(self, tmp_path):
        """Every key from config.json should be preserved — no cherry-picking."""
        self._write_config(str(tmp_path / "transformer"), {
            "num_attention_heads": 32,
            "attention_head_dim": 128,
            "some_future_key": "future_value",
            "another_unknown": [1, 2, 3],
        })

        result = harvest(str(tmp_path))
        assert result["transformer.some_future_key"] == "future_value"
        assert result["transformer.another_unknown"] == [1, 2, 3]

    def test_harvest_ae_dir_alias(self, tmp_path):
        """Should check 'ae' dir as fallback for vae namespace."""
        self._write_config(str(tmp_path / "ae"), {
            "latent_channels": 16,
        })

        result = harvest(str(tmp_path))
        assert result["vae.latent_channels"] == 16


