"""
Tests for LoRA output and ecosystem compatibility.
Phase 5: Validates key naming, metadata, and dtype against working reference LoRA files.
"""

import pytest
import os
import torch
from unittest.mock import MagicMock
from safetensors import safe_open

from app.engine.utils.lora_conversion import convert_peft_to_kohya


# ── Reference File Paths ─────────────────────────────────────────────────

SDXL_REF = os.path.join(os.path.dirname(__file__), "..", "outputs", "sdxl_working.safetensors")
FLUX_REF = os.path.join(os.path.dirname(__file__), "..", "outputs", "flux2-klein-base-9b_working.safetensors")

HAS_SDXL_REF = os.path.exists(SDXL_REF)
HAS_FLUX_REF = os.path.exists(FLUX_REF)


# ── Conversion Tests ─────────────────────────────────────────────────────


class TestConvertPeftToKohya:
    """Tests for the PEFT → Kohya key conversion utility."""

    def test_lora_a_becomes_lora_down(self):
        """lora_A.weight should map to lora_down.weight in Kohya format."""
        sd = {"lora_unet.some.module.lora_A.weight": torch.randn(32, 640)}
        result = convert_peft_to_kohya(sd, model_type="sdxl", alpha=8.0)
        assert "lora_unet_some_module.lora_down.weight" in result

    def test_lora_b_becomes_lora_up(self):
        """lora_B.weight should map to lora_up.weight in Kohya format."""
        sd = {"lora_unet.some.module.lora_B.weight": torch.randn(640, 32)}
        result = convert_peft_to_kohya(sd, model_type="sdxl", alpha=8.0)
        assert "lora_unet_some_module.lora_up.weight" in result

    def test_alpha_injection(self):
        """Every module should get an alpha key when alpha is provided."""
        sd = {
            "lora_unet.mod1.lora_A.weight": torch.randn(32, 640),
            "lora_unet.mod1.lora_B.weight": torch.randn(640, 32),
        }
        result = convert_peft_to_kohya(sd, model_type="sdxl", alpha=16.0)
        assert "lora_unet_mod1.alpha" in result
        assert result["lora_unet_mod1.alpha"].item() == 16.0

    def test_alpha_dtype_is_float32(self):
        """Alpha tensors should always be float32 regardless of weight dtype."""
        sd = {"lora_unet.x.lora_A.weight": torch.randn(4, 8)}
        result = convert_peft_to_kohya(sd, alpha=1.0)
        alpha_keys = [k for k in result if ".alpha" in k]
        for k in alpha_keys:
            assert result[k].dtype == torch.float32

    def test_te1_prefixing(self):
        """Text encoder 1 keys should keep lora_te1 prefix."""
        sd = {"lora_te1.base_model.model.encoder.q_proj.lora_A.weight": torch.randn(4, 768)}
        result = convert_peft_to_kohya(sd, alpha=8.0)
        converted_keys = [k for k in result if k.startswith("lora_te1")]
        assert len(converted_keys) >= 1

    def test_te2_prefixing(self):
        """Text encoder 2 keys should keep lora_te2 prefix."""
        sd = {"lora_te2.base_model.model.encoder.v_proj.lora_B.weight": torch.randn(768, 4)}
        result = convert_peft_to_kohya(sd, alpha=8.0)
        converted_keys = [k for k in result if k.startswith("lora_te2")]
        assert len(converted_keys) >= 1

    def test_dots_are_replaced_with_underscores(self):
        """Module path dots should become underscores in Kohya format."""
        sd = {"lora_unet.down_blocks.0.attentions.0.to_q.lora_A.weight": torch.randn(4, 8)}
        result = convert_peft_to_kohya(sd, alpha=1.0)
        for k in result:
            if ".lora_down.weight" in k or ".lora_up.weight" in k:
                # Key format: lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight
                # The module prefix (before .lora_down) should have no dots
                prefix = k.split(".lora_down.weight")[0] if ".lora_down.weight" in k else k.split(".lora_up.weight")[0]
                assert "." not in prefix, f"Dots found in module prefix: {prefix}"

    def test_base_model_prefix_stripped(self):
        """PEFT 'base_model.model.' prefix should be stripped."""
        sd = {"lora_unet.base_model.model.mid_block.to_q.lora_A.weight": torch.randn(4, 8)}
        result = convert_peft_to_kohya(sd, alpha=1.0)
        assert not any("base_model" in k for k in result)

    def test_empty_input(self):
        """Empty state dict should produce empty output with no alphas."""
        result = convert_peft_to_kohya({}, alpha=8.0)
        assert len(result) == 0


# ── SDXL Saver Tests ─────────────────────────────────────────────────────


class TestSDXLSaver:
    """Tests for the SDXL LoRA saver."""

    def test_save_produces_kohya_format(self, tmp_path):
        """SDXLSaver output should use Kohya-style key naming."""
        from app.engine.models.families.sdxl.saver import SDXLSaver

        saver = SDXLSaver()

        # Create a minimal PEFT-wrapped mock
        unet = MagicMock()
        mock_lora_cfg = MagicMock(r=4, lora_alpha=4)
        unet.peft_config = {"default": mock_lora_cfg}

        # Simulate get_peft_model_state_dict output
        mock_sd = {
            "base_model.model.mid_block.attentions.0.to_q.lora_A.weight": torch.randn(4, 320),
            "base_model.model.mid_block.attentions.0.to_q.lora_B.weight": torch.randn(320, 4),
        }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.engine.models.families.sdxl.saver.get_peft_model_state_dict", lambda _: mock_sd)

            path = tmp_path / "test_sdxl.safetensors"
            saver.save(
                components={"unet": unet, "config": {"network_rank": 4, "network_alpha": 4}},
                path=path,
            )

        assert path.exists()
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            # Should have lora_down, lora_up, and alpha
            down_keys = [k for k in keys if "lora_down" in k]
            up_keys = [k for k in keys if "lora_up" in k]
            alpha_keys = [k for k in keys if ".alpha" in k]
            assert len(down_keys) == 1
            assert len(up_keys) == 1
            assert len(alpha_keys) == 1

    def test_save_metadata_contains_kohya_fields(self, tmp_path):
        """SDXLSaver should write Kohya-compatible metadata."""
        from app.engine.models.families.sdxl.saver import SDXLSaver

        saver = SDXLSaver()
        unet = MagicMock()
        mock_lora_cfg = MagicMock(r=4, lora_alpha=2)
        unet.peft_config = {"default": mock_lora_cfg}
        mock_sd = {
            "base_model.model.x.lora_A.weight": torch.randn(4, 8),
            "base_model.model.x.lora_B.weight": torch.randn(8, 4),
        }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.engine.models.families.sdxl.saver.get_peft_model_state_dict", lambda _: mock_sd)

            path = tmp_path / "meta_test.safetensors"
            saver.save(
                components={"unet": unet, "config": {"network_rank": 4, "network_alpha": 2, "mixed_precision": "fp16"}},
                path=path,
            )

        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata()
            assert "ss_network_module" in meta
            assert "ss_network_dim" in meta
            assert "ss_network_alpha" in meta


# ── Flux2 Saver Tests ────────────────────────────────────────────────────


class TestFlux2Saver:
    """Tests for the Flux2 LoRA saver (ai-toolkit format)."""

    def test_save_uses_peft_format(self, tmp_path):
        """Flux2Saver output should use raw PEFT key naming (lora_A/lora_B, not down/up)."""
        from app.engine.models.families.flux2.saver import Flux2Saver

        saver = Flux2Saver()
        unet = MagicMock()
        mock_lora_cfg = MagicMock(r=32, lora_alpha=32)
        unet.peft_config = {"default": mock_lora_cfg}

        mock_sd = {
            "base_model.model.double_blocks.0.img_attn.qkv.lora_A.weight": torch.randn(32, 4096),
            "base_model.model.double_blocks.0.img_attn.qkv.lora_B.weight": torch.randn(4096, 32),
        }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.engine.models.families.flux2.saver.get_peft_model_state_dict", lambda _: mock_sd)

            path = tmp_path / "test_flux.safetensors"
            saver.save(
                components={"unet": unet, "config": {"network_rank": 32, "save_precision": "bf16"}},
                path=path,
            )

        assert path.exists()
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            # Should use lora_A/lora_B (NOT lora_down/lora_up)
            a_keys = [k for k in keys if "lora_A" in k]
            b_keys = [k for k in keys if "lora_B" in k]
            down_keys = [k for k in keys if "lora_down" in k]
            alpha_keys = [k for k in keys if ".alpha" in k]

            assert len(a_keys) == 1, f"Expected 1 lora_A key, got {a_keys}"
            assert len(b_keys) == 1, f"Expected 1 lora_B key, got {b_keys}"
            assert len(down_keys) == 0, "Should NOT have Kohya-style lora_down keys"
            assert len(alpha_keys) == 0, "Flux format should NOT have alpha keys"

    def test_save_uses_diffusion_model_prefix(self, tmp_path):
        """Flux2Saver keys should start with 'diffusion_model.'."""
        from app.engine.models.families.flux2.saver import Flux2Saver

        saver = Flux2Saver()
        unet = MagicMock()
        mock_lora_cfg = MagicMock(r=32, lora_alpha=32)
        unet.peft_config = {"default": mock_lora_cfg}

        mock_sd = {
            "base_model.model.single_blocks.5.linear1.lora_A.weight": torch.randn(32, 4096),
            "base_model.model.single_blocks.5.linear1.lora_B.weight": torch.randn(4096, 32),
        }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.engine.models.families.flux2.saver.get_peft_model_state_dict", lambda _: mock_sd)

            path = tmp_path / "prefix_test.safetensors"
            saver.save(
                components={"unet": unet, "config": {}},
                path=path,
            )

        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                assert key.startswith("diffusion_model."), f"Key {key} missing diffusion_model prefix"

    def test_save_default_bf16(self, tmp_path):
        """Flux2Saver should default to BF16 save precision."""
        from app.engine.models.families.flux2.saver import Flux2Saver

        saver = Flux2Saver()
        unet = MagicMock()
        mock_lora_cfg = MagicMock(r=4, lora_alpha=4)
        unet.peft_config = {"default": mock_lora_cfg}

        mock_sd = {
            "base_model.model.x.lora_A.weight": torch.randn(4, 8),
            "base_model.model.x.lora_B.weight": torch.randn(8, 4),
        }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.engine.models.families.flux2.saver.get_peft_model_state_dict", lambda _: mock_sd)

            path = tmp_path / "dtype_test.safetensors"
            saver.save(
                components={"unet": unet, "config": {}},
                path=path,
            )

        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                assert t.dtype == torch.bfloat16, f"Key {key} expected bf16, got {t.dtype}"

    def test_non_lora_keys_are_filtered(self, tmp_path):
        """Keys without lora_A/lora_B should be filtered out."""
        from app.engine.models.families.flux2.saver import Flux2Saver

        saver = Flux2Saver()
        unet = MagicMock()
        mock_lora_cfg = MagicMock(r=4, lora_alpha=4)
        unet.peft_config = {"default": mock_lora_cfg}

        mock_sd = {
            "base_model.model.guidance_in.weight": torch.randn(4, 8),  # Not a LoRA key
            "base_model.model.x.lora_A.weight": torch.randn(4, 8),
            "base_model.model.x.lora_B.weight": torch.randn(8, 4),
        }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.engine.models.families.flux2.saver.get_peft_model_state_dict", lambda _: mock_sd)

            path = tmp_path / "filter_test.safetensors"
            saver.save(
                components={"unet": unet, "config": {}},
                path=path,
            )

        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            assert len(keys) == 2  # Only lora_A and lora_B
            assert not any("guidance_in" in k and "lora" not in k for k in keys)


# ── Reference File Validation ────────────────────────────────────────────


@pytest.mark.skipif(not HAS_SDXL_REF, reason="SDXL reference file not found")
class TestSDXLReferenceValidation:
    """Validates that our converter output would match the SDXL reference format."""

    def test_sdxl_ref_uses_kohya_format(self):
        """SDXL reference should use Kohya-style keys (lora_down/up, alphas)."""
        with safe_open(SDXL_REF, framework="pt") as f:
            keys = list(f.keys())
            down = [k for k in keys if "lora_down" in k]
            up = [k for k in keys if "lora_up" in k]
            alpha = [k for k in keys if ".alpha" in k]
            assert len(down) > 0, "SDXL ref should have lora_down keys"
            assert len(up) > 0, "SDXL ref should have lora_up keys"
            assert len(alpha) > 0, "SDXL ref should have alpha keys"
            assert len(down) == len(up), "Each module needs both down and up"
            assert len(alpha) == len(down), "Each module needs an alpha"

    def test_sdxl_ref_all_unet_prefixed(self):
        """All SDXL reference keys should start with lora_unet."""
        with safe_open(SDXL_REF, framework="pt") as f:
            for key in f.keys():
                assert key.startswith("lora_unet") or key.startswith("lora_te"), \
                    f"Unexpected prefix: {key}"

    def test_sdxl_ref_dtype_is_fp16(self):
        """SDXL reference weights should be FP16."""
        with safe_open(SDXL_REF, framework="pt") as f:
            keys = list(f.keys())
            # Check a few weight keys (alphas are always float32)
            weight_keys = [k for k in keys if ".weight" in k]
            for k in weight_keys[:5]:
                assert f.get_tensor(k).dtype == torch.float16

    def test_sdxl_ref_has_metadata(self):
        """SDXL reference should have Kohya-standard metadata."""
        with safe_open(SDXL_REF, framework="pt") as f:
            meta = f.metadata()
            assert meta is not None
            assert "format" in meta


@pytest.mark.skipif(not HAS_FLUX_REF, reason="Flux2 reference file not found")
class TestFlux2ReferenceValidation:
    """Validates that our saver output matches the Flux2 Klein reference format."""

    def test_flux_ref_uses_peft_format(self):
        """Flux reference should use raw PEFT keys (lora_A/lora_B, NOT down/up)."""
        with safe_open(FLUX_REF, framework="pt") as f:
            keys = list(f.keys())
            a_keys = [k for k in keys if "lora_A" in k]
            b_keys = [k for k in keys if "lora_B" in k]
            down_keys = [k for k in keys if "lora_down" in k]
            assert len(a_keys) > 0, "Should have lora_A keys"
            assert len(b_keys) > 0, "Should have lora_B keys"
            assert len(down_keys) == 0, "Should NOT have Kohya-style keys"
            assert len(a_keys) == len(b_keys), "Each module needs A and B"

    def test_flux_ref_no_alpha_keys(self):
        """Flux reference should NOT have alpha keys."""
        with safe_open(FLUX_REF, framework="pt") as f:
            alpha_keys = [k for k in f.keys() if ".alpha" in k]
            assert len(alpha_keys) == 0, f"Unexpected alpha keys: {alpha_keys}"

    def test_flux_ref_uses_diffusion_model_prefix(self):
        """Flux reference keys should start with 'diffusion_model.'."""
        with safe_open(FLUX_REF, framework="pt") as f:
            for key in f.keys():
                assert key.startswith("diffusion_model."), f"Key {key} missing prefix"

    def test_flux_ref_dtype_is_bf16(self):
        """Flux reference weights should be BF16."""
        with safe_open(FLUX_REF, framework="pt") as f:
            for key in list(f.keys())[:10]:
                assert f.get_tensor(key).dtype == torch.bfloat16

    def test_flux_ref_has_expected_modules(self):
        """Flux reference should have double_blocks and single_blocks modules."""
        with safe_open(FLUX_REF, framework="pt") as f:
            keys = list(f.keys())
            has_double = any("double_blocks" in k for k in keys)
            has_single = any("single_blocks" in k for k in keys)
            assert has_double, "Missing double_blocks modules"
            assert has_single, "Missing single_blocks modules"

    def test_flux_ref_dots_preserved(self):
        """Flux reference should use dots (not underscores) in module paths."""
        with safe_open(FLUX_REF, framework="pt") as f:
            keys = list(f.keys())
            # All keys should have dots in the module path
            for k in keys[:10]:
                parts = k.split(".")
                # diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight
                assert len(parts) >= 5, f"Key {k} doesn't have enough dot-separated parts"
