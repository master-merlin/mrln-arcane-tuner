"""
Tests for PEFT → Kohya LoRA state-dict conversion.
Covers: key remapping, dot→underscore, alpha injection, empty dict, pass-through.
"""

import torch

from app.engine.utils.lora_conversion import convert_peft_to_kohya


class TestKeyRemapping:
    def test_lora_A_maps_to_lora_down(self):
        sd = {"lora_unet.base_model.model.down_blocks.0.lora_A.weight": torch.zeros(4, 8)}
        result = convert_peft_to_kohya(sd)
        down_keys = [k for k in result if "lora_down" in k]
        assert len(down_keys) == 1

    def test_lora_B_maps_to_lora_up(self):
        sd = {"lora_unet.base_model.model.up_blocks.0.lora_B.weight": torch.zeros(8, 4)}
        result = convert_peft_to_kohya(sd)
        up_keys = [k for k in result if "lora_up" in k]
        assert len(up_keys) == 1

    def test_dots_replaced_with_underscores(self):
        sd = {
            "lora_unet.base_model.model.mid.attn.q_proj.lora_A.weight": torch.zeros(4, 8),
        }
        result = convert_peft_to_kohya(sd)
        for key in result:
            if key.startswith("lora_unet_"):
                # Between prefix and suffix, dots should be underscores
                module_part = key.split(".")[0]  # e.g. lora_unet_mid_attn_q_proj
                assert "." not in module_part.replace("lora_unet_", "")


class TestTextEncoderPrefixes:
    def test_te1_prefix(self):
        sd = {"lora_te1.base_model.model.layer.0.lora_A.weight": torch.zeros(2, 4)}
        result = convert_peft_to_kohya(sd)
        assert any(k.startswith("lora_te1_") for k in result)

    def test_te2_prefix(self):
        sd = {"lora_te2.base_model.model.layer.0.lora_B.weight": torch.zeros(4, 2)}
        result = convert_peft_to_kohya(sd)
        assert any(k.startswith("lora_te2_") for k in result)


class TestAlphaInjection:
    def test_alpha_injected_for_each_module(self):
        sd = {
            "lora_unet.base_model.model.block.lora_A.weight": torch.zeros(4, 8),
            "lora_unet.base_model.model.block.lora_B.weight": torch.zeros(8, 4),
        }
        result = convert_peft_to_kohya(sd, alpha=16.0)
        alpha_keys = [k for k in result if k.endswith(".alpha")]
        assert len(alpha_keys) >= 1
        assert result[alpha_keys[0]].item() == 16.0

    def test_no_alpha_when_none(self):
        sd = {"lora_unet.base_model.model.block.lora_A.weight": torch.zeros(4, 8)}
        result = convert_peft_to_kohya(sd, alpha=None)
        alpha_keys = [k for k in result if k.endswith(".alpha")]
        assert len(alpha_keys) == 0


class TestEdgeCases:
    def test_empty_state_dict(self):
        result = convert_peft_to_kohya({})
        assert result == {}

    def test_unknown_keys_passed_through(self):
        sd = {"some_random_key": torch.zeros(2)}
        result = convert_peft_to_kohya(sd)
        assert "some_random_key" in result

    def test_non_tensor_values_skipped(self):
        sd = {
            "lora_unet.block.lora_A.weight": torch.zeros(4, 8),
            "metadata_string_key": "not_a_tensor",
        }
        result = convert_peft_to_kohya(sd)
        # String should not appear in output (skipped in loop)
        assert "metadata_string_key" not in result
