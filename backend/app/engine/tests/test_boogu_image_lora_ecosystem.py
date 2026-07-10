"""boogu_image ecosystem LoRA mapping (Task 7, requirement 4 + MUST-FIX).

Exercises ``lora_ecosystem.py`` — the Boogu-adapted counterpart of the
vendored (stock-Lumina2) ``_convert_non_diffusers_lumina2_lora_to_diffusers``
that ``BooguImageLoraLoaderMixin.lora_state_dict()`` unconditionally runs
against any ``diffusion_model.``-prefixed checkpoint
(``.agent/workdir/sdd-boogu/upstream/boogu/pipelines/lora_pipeline.py:165-167``).

Covers:
  1. The portability finding is pinned as a regression guard: exactly 252 of
     the 418 curated modules (504 of 836 keys) have a structural analogue in
     stock Lumina2 and round-trip through the (fixed) converter; the
     remaining 166 modules (332 keys) -- ``ref_image_refiner`` +
     ``double_stream_layers`` -- do not and are boogu_image-native only.
  2. The IMPORT direction (``convert_ecosystem_to_diffusers``): correct GQA
     qkv split + correct Boogu block-attribute names (``single_stream_layers``,
     not stock's ``layers``) + leftover (non-portable) keys pass through
     untouched instead of raising.
  3. The EXPORT direction (``convert_diffusers_to_ecosystem``) and the full
     bidirectional round trip, under the documented shared-``lora_A``
     precondition (see ``lora_ecosystem.py`` module docstring).
  4. The precondition guard: real, independently-PEFT-trained
     to_q/to_k/to_v adapters (as this family's saver actually produces) do
     NOT share a common ``lora_A`` and must raise a clear error rather than
     silently mis-fuse.
"""

from __future__ import annotations

import pathlib
import types
from unittest.mock import MagicMock

import pytest
import torch
import yaml

from app.engine.models.families.boogu_image.lora_ecosystem import (
    NON_PORTABLE_BLOCK_PREFIXES,
    PORTABLE_BLOCK_PREFIXES,
    convert_diffusers_to_ecosystem,
    convert_ecosystem_to_diffusers,
    qkv_split_from_config,
)

TINY_HIDDEN_SIZE = 16
TINY_NUM_ATTENTION_HEADS = 2
TINY_NUM_KV_HEADS = 1
TINY_QKV_SPLIT = (16, 8, 8)

NUM_DOUBLE_STREAM = 8
NUM_SINGLE_STREAM = 32
NUM_REFINER_LAYERS = 2

_DEFINITIONS_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "models" / "families" / "boogu_image" / "definitions"
)


def _load_curated_targets() -> list[str]:
    data = yaml.safe_load((_DEFINITIONS_DIR / "base.yaml").read_text())
    targets = list(data["lora_targetable_modules"])
    assert len(targets) == 418
    return targets


def _build_real_depth_tiny_transformer():
    from app.engine.models.families.boogu_image.vendor.models.transformers.transformer_boogu import (
        BooguImageTransformer2DModel,
    )

    model = BooguImageTransformer2DModel(
        patch_size=2,
        in_channels=4,
        out_channels=None,
        hidden_size=TINY_HIDDEN_SIZE,
        num_layers=NUM_DOUBLE_STREAM + NUM_SINGLE_STREAM,
        num_double_stream_layers=NUM_DOUBLE_STREAM,
        num_refiner_layers=NUM_REFINER_LAYERS,
        num_attention_heads=TINY_NUM_ATTENTION_HEADS,
        num_kv_heads=TINY_NUM_KV_HEADS,
        multiple_of=8,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        axes_dim_rope=(2, 2, 4),
        axes_lens=(64, 64, 64),
        instruction_feature_configs=dict(
            instruction_feat_dim=8, reduce_type="mean", num_instruction_feat_layers=1,
        ),
        prompt_tuning_configs=dict(use_prompt_tuning=False),
        timestep_scale=1000.0,
    )
    model.eval()
    return model


def _build_peft_model(targets: list[str], rank: int = 4):
    from peft import LoraConfig, get_peft_model

    base = _build_real_depth_tiny_transformer()
    lora_cfg = LoraConfig(r=rank, lora_alpha=rank, target_modules=targets)
    return get_peft_model(base, lora_cfg)


def _get_saver():
    from app.engine.models.families.boogu_image.driver import BooguImageDriver

    definition = MagicMock()
    definition.family = "boogu_image"
    definition.id = "boogu-image-test"
    definition.lora_targetable_modules = _load_curated_targets()
    definition.architecture_params = {}
    drv = BooguImageDriver(definition, torch.device("cpu"))
    return drv.get_saver()


class TestPortabilityFinding:
    """Pins the MUST-FIX finding: which curated targets survive the
    ecosystem converter and which don't."""

    def test_portable_and_non_portable_prefixes_partition_all_418_targets(self):
        targets = _load_curated_targets()
        portable = [
            t for t in targets
            if t.split(".", 1)[0] in PORTABLE_BLOCK_PREFIXES
        ]
        non_portable = [
            t for t in targets
            if t.split(".", 1)[0] in NON_PORTABLE_BLOCK_PREFIXES
        ]
        assert len(portable) + len(non_portable) == 418
        assert not (set(portable) & set(non_portable))

    def test_portable_module_count_is_252(self):
        targets = _load_curated_targets()
        portable = [
            t for t in targets
            if t.split(".", 1)[0] in PORTABLE_BLOCK_PREFIXES
        ]
        # noise_refiner(14) + context_refiner(14) + single_stream_layers(224)
        assert len(portable) == 252

    def test_non_portable_module_count_is_166(self):
        targets = _load_curated_targets()
        non_portable = [
            t for t in targets
            if t.split(".", 1)[0] in NON_PORTABLE_BLOCK_PREFIXES
        ]
        # ref_image_refiner(14) + double_stream_layers(152)
        assert len(non_portable) == 166


class TestQkvSplitFromConfig:
    def test_tiny_model_geometry(self):
        config = types.SimpleNamespace(
            hidden_size=TINY_HIDDEN_SIZE,
            num_attention_heads=TINY_NUM_ATTENTION_HEADS,
            num_kv_heads=TINY_NUM_KV_HEADS,
        )
        assert qkv_split_from_config(config) == TINY_QKV_SPLIT

    def test_real_checkpoint_geometry(self):
        # definitions/base.yaml architecture_params.
        config = types.SimpleNamespace(
            hidden_size=3360, num_attention_heads=28, num_kv_heads=7,
        )
        assert qkv_split_from_config(config) == (3360, 840, 840)

    def test_real_checkpoint_geometry_differs_from_stock_lumina2_hardcode(self):
        config = types.SimpleNamespace(
            hidden_size=3360, num_attention_heads=28, num_kv_heads=7,
        )
        assert qkv_split_from_config(config) != (2304, 768, 768)


class TestConvertEcosystemToDiffusers:
    """IMPORT direction: fused-qkv ecosystem state dict -> diffusers-native."""

    def _synthetic_fused_block(self, prefix: str, index: int, rank: int = 4):
        """A fused-qkv LoRA state dict fragment for one block, correctly
        sized for the tiny GQA split (16/8/8)."""
        in_dim = TINY_HIDDEN_SIZE
        qkv_out = sum(TINY_QKV_SPLIT)
        return {
            f"{prefix}.{index}.attention.qkv.lora_A.weight": torch.randn(rank, in_dim),
            f"{prefix}.{index}.attention.qkv.lora_B.weight": torch.randn(qkv_out, rank),
            f"{prefix}.{index}.attention.out.lora_A.weight": torch.randn(rank, in_dim),
            f"{prefix}.{index}.attention.out.lora_B.weight": torch.randn(in_dim, rank),
            f"{prefix}.{index}.feed_forward.w1.lora_A.weight": torch.randn(rank, in_dim),
            f"{prefix}.{index}.feed_forward.w1.lora_B.weight": torch.randn(64, rank),
            f"{prefix}.{index}.feed_forward.w2.lora_A.weight": torch.randn(rank, 64),
            f"{prefix}.{index}.feed_forward.w2.lora_B.weight": torch.randn(in_dim, rank),
            f"{prefix}.{index}.feed_forward.w3.lora_A.weight": torch.randn(rank, in_dim),
            f"{prefix}.{index}.feed_forward.w3.lora_B.weight": torch.randn(64, rank),
        }

    def test_converts_all_three_portable_prefixes_to_diffusers_native_names(self):
        raw = {}
        for prefix in PORTABLE_BLOCK_PREFIXES:
            raw.update(self._synthetic_fused_block(prefix, 0))
        raw = {f"diffusion_model.{k}": v for k, v in raw.items()}

        converted, unconverted = convert_ecosystem_to_diffusers(raw, TINY_QKV_SPLIT)

        assert not unconverted, f"unexpected leftovers: {list(unconverted)[:5]}"
        for prefix in PORTABLE_BLOCK_PREFIXES:
            for attn_key in ("to_q", "to_k", "to_v", "to_out.0"):
                assert f"transformer.{prefix}.0.attn.{attn_key}.lora_A.weight" in converted
                assert f"transformer.{prefix}.0.attn.{attn_key}.lora_B.weight" in converted
            for layer in (1, 2, 3):
                assert (
                    f"transformer.{prefix}.0.feed_forward.linear_{layer}.lora_A.weight"
                    in converted
                )

    def test_gqa_split_widths_correct_on_converted_output(self):
        raw = self._synthetic_fused_block("noise_refiner", 0)
        converted, _ = convert_ecosystem_to_diffusers(raw, TINY_QKV_SPLIT)

        to_q_b = converted["transformer.noise_refiner.0.attn.to_q.lora_B.weight"]
        to_k_b = converted["transformer.noise_refiner.0.attn.to_k.lora_B.weight"]
        to_v_b = converted["transformer.noise_refiner.0.attn.to_v.lora_B.weight"]

        assert to_q_b.shape[0] == 16
        assert to_k_b.shape[0] == 8
        assert to_v_b.shape[0] == 8

        # Same shared lora_A across q/k/v -- mirrors upstream's own
        # (lora_conversion.py:50-56) semantics.
        to_q_a = converted["transformer.noise_refiner.0.attn.to_q.lora_A.weight"]
        to_k_a = converted["transformer.noise_refiner.0.attn.to_k.lora_A.weight"]
        assert torch.equal(to_q_a, to_k_a)

    def test_stock_split_would_produce_wrong_shapes(self):
        """Sanity: using the ORIGINAL vendored (stock Lumina2) hardcoded
        split against Boogu-tiny-sized tensors either crashes (dim mismatch
        in torch.split) or silently produces the wrong shapes -- this is
        the exact class of bug the MUST-FIX guards against."""
        raw = self._synthetic_fused_block("noise_refiner", 0)
        stock_split = (2304, 768, 768)  # vendor/lora_conversion.py:54, verbatim
        with pytest.raises(RuntimeError):
            convert_ecosystem_to_diffusers(raw, stock_split)

    def test_non_portable_prefixed_keys_pass_through_unconverted_no_raise(self):
        """Boogu's ref_image_refiner / double_stream_layers keys have no
        stock-Lumina2 analogue -- must NOT raise (unlike upstream's own
        leftover-key ValueError guard, vendor/lora_conversion.py:97-98)."""
        raw = {
            "diffusion_model.ref_image_refiner.0.attn.to_q.lora_A.weight": torch.randn(4, 16),
            "diffusion_model.double_stream_layers.0.img_instruct_attn.processor.img_to_q.lora_A.weight": torch.randn(4, 16),
        }
        converted, unconverted = convert_ecosystem_to_diffusers(raw, TINY_QKV_SPLIT)
        assert not converted
        assert len(unconverted) == 2
        assert "ref_image_refiner.0.attn.to_q.lora_A.weight" in unconverted
        assert (
            "double_stream_layers.0.img_instruct_attn.processor.img_to_q.lora_A.weight"
            in unconverted
        )


class TestConvertDiffusersToEcosystemAndBackBidirectional:
    """Requirement 4: our export -> (fixed) converter path the upstream
    mixin would run -> lands on diffusers-side names upstream produces ->
    and back. Uses the REAL saver + REAL PEFT-wrapped tiny model, restricted
    to one block per portable family to keep the fixture small.
    """

    _TARGETS = [
        "noise_refiner.0.attn.to_q", "noise_refiner.0.attn.to_k",
        "noise_refiner.0.attn.to_v", "noise_refiner.0.attn.to_out.0",
        "noise_refiner.0.feed_forward.linear_1",
        "noise_refiner.0.feed_forward.linear_2",
        "noise_refiner.0.feed_forward.linear_3",
        "context_refiner.0.attn.to_q", "context_refiner.0.attn.to_k",
        "context_refiner.0.attn.to_v", "context_refiner.0.attn.to_out.0",
        "context_refiner.0.feed_forward.linear_1",
        "context_refiner.0.feed_forward.linear_2",
        "context_refiner.0.feed_forward.linear_3",
        "single_stream_layers.0.attn.to_q", "single_stream_layers.0.attn.to_k",
        "single_stream_layers.0.attn.to_v", "single_stream_layers.0.attn.to_out.0",
        "single_stream_layers.0.feed_forward.linear_1",
        "single_stream_layers.0.feed_forward.linear_2",
        "single_stream_layers.0.feed_forward.linear_3",
    ]

    def _force_shared_qkv_lora_a(self, peft_model, prefix: str, index: int) -> None:
        """Overwrite to_k/to_v's lora_A with to_q's -- simulates the
        ecosystem's shared-A fused-qkv precondition (see lora_ecosystem.py
        module docstring's structural caveat) on top of a REAL PEFT model.
        lora_A shape is [rank, in_dim] for all three (q/k/v share the same
        input width; only lora_B's OUT dim differs via GQA), so this copy
        is shape-valid."""
        params = dict(peft_model.named_parameters())
        q_a = params[f"base_model.model.{prefix}.{index}.attn.to_q.lora_A.default.weight"]
        for attn_key in ("to_k", "to_v"):
            p = params[f"base_model.model.{prefix}.{index}.attn.{attn_key}.lora_A.default.weight"]
            with torch.no_grad():
                p.copy_(q_a)

    def test_real_saver_export_round_trips_through_fixed_converter(self, tmp_path):
        from safetensors.torch import load_file

        model = _build_peft_model(self._TARGETS)
        for prefix in PORTABLE_BLOCK_PREFIXES:
            self._force_shared_qkv_lora_a(model, prefix, 0)

        saver = _get_saver()
        out = tmp_path / "portable_lora.safetensors"
        saver.save(components={"unet": model, "config": {}}, path=out)
        original = load_file(str(out))
        assert len(original) == 2 * len(self._TARGETS)

        # our export -> ecosystem (fused-qkv) format.
        ecosystem, leftover_export = convert_diffusers_to_ecosystem(original)
        assert not leftover_export, f"unexpected leftovers: {list(leftover_export)[:5]}"
        for prefix in PORTABLE_BLOCK_PREFIXES:
            assert f"diffusion_model.{prefix}.0.attention.qkv.lora_A.weight" in ecosystem
            assert f"diffusion_model.{prefix}.0.attention.qkv.lora_B.weight" in ecosystem

        # ecosystem format -> "the converter path the upstream mixin would
        # run" (fixed for Boogu's real GQA widths) -> diffusers-side names.
        back, leftover_import = convert_ecosystem_to_diffusers(ecosystem, TINY_QKV_SPLIT)
        assert not leftover_import

        # Lands back on our own diffusers-native names (transformer.<path>,
        # matching upstream's own output convention) with bit-exact values.
        for prefix in PORTABLE_BLOCK_PREFIXES:
            for attn_key in ("to_q", "to_k", "to_v", "to_out.0"):
                orig = original[f"diffusion_model.{prefix}.0.attn.{attn_key}.lora_A.weight"]
                got = back[f"transformer.{prefix}.0.attn.{attn_key}.lora_A.weight"]
                assert torch.equal(orig, got), f"{prefix}.0.attn.{attn_key} lora_A mismatch"

                orig_b = original[f"diffusion_model.{prefix}.0.attn.{attn_key}.lora_B.weight"]
                got_b = back[f"transformer.{prefix}.0.attn.{attn_key}.lora_B.weight"]
                assert torch.equal(orig_b, got_b), f"{prefix}.0.attn.{attn_key} lora_B mismatch"

            for layer in (1, 2, 3):
                orig = original[
                    f"diffusion_model.{prefix}.0.feed_forward.linear_{layer}.lora_A.weight"
                ]
                got = back[
                    f"transformer.{prefix}.0.feed_forward.linear_{layer}.lora_A.weight"
                ]
                assert torch.equal(orig, got)

    def test_independent_per_target_adapters_raise_not_silently_mis_fuse(self, tmp_path):
        """Without the shared-A doctoring, our saver's REAL output (PEFT
        trains to_q/to_k/to_v as independent adapters -- driver.py
        get_lora_targets) cannot losslessly fuse into the ecosystem's
        shared-A format. Must raise, not silently produce a wrong
        checkpoint."""
        from safetensors.torch import load_file

        model = _build_peft_model(self._TARGETS)
        # deliberately NOT calling _force_shared_qkv_lora_a.
        saver = _get_saver()
        out = tmp_path / "independent_lora.safetensors"
        saver.save(components={"unet": model, "config": {}}, path=out)
        original = load_file(str(out))

        with pytest.raises(ValueError, match="cannot losslessly fuse"):
            convert_diffusers_to_ecosystem(original)

    def test_non_portable_keys_pass_through_diffusers_to_ecosystem_untouched(self):
        """A house-format state dict containing double_stream_layers /
        ref_image_refiner keys must leave them in `unconverted`, not drop
        or raise."""
        raw = {
            "diffusion_model.ref_image_refiner.0.attn.to_q.lora_A.weight": torch.randn(4, 16),
            "diffusion_model.double_stream_layers.0.img_self_attn.to_q.lora_A.weight": torch.randn(4, 16),
        }
        converted, unconverted = convert_diffusers_to_ecosystem(raw)
        assert not converted
        assert len(unconverted) == 2
