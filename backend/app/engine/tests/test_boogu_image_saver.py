"""boogu_image saver: canonical key contract + Base<->Turbo portability (Task 7).

Builds a REAL-DEPTH, tiny-WIDTH ``BooguImageTransformer2DModel`` — the same
tiny per-layer dims as ``test_boogu_image_vendor.py``/
``test_boogu_image_definitions.py`` (``hidden_size=16``,
``num_attention_heads=2``, ``num_kv_heads=1``), but the REAL checkpoint's
block counts (``num_double_stream_layers=8``, ``num_refiner_layers=2``,
``num_layers=40`` -> 32 single-stream layers) — so the shipped
``lora_targetable_modules`` curated list (loaded verbatim from
``definitions/base.yaml``, 418 entries) applies to it exactly, producing the
pinned 836-key (``418 * lora_A/B``) canonical contract on save.

Covers Task 7 requirements 1-3 (saver, canonical key contract, Base<->Turbo
portability). Requirement 4 (ecosystem mapping) lives in
``test_boogu_image_lora_ecosystem.py``.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import torch
import yaml

TINY_HIDDEN_SIZE = 16
TINY_NUM_ATTENTION_HEADS = 2
TINY_NUM_KV_HEADS = 1
# GQA widths for the tiny model: to_q = heads*head_dim = 2*8 = 16;
# to_k/to_v = kv_heads*head_dim = 1*8 = 8 (asymmetric, same ratio shape as
# the real checkpoint's 3360/840/840 -- see definitions/base.yaml).
TINY_TO_Q_WIDTH = 16
TINY_TO_KV_WIDTH = 8

NUM_DOUBLE_STREAM = 8
NUM_SINGLE_STREAM = 32
NUM_REFINER_LAYERS = 2

_DEFINITIONS_DIR = (
    pathlib.Path(__file__).resolve().parents[1]  # .../app/engine
    / "models" / "families" / "boogu_image" / "definitions"
)


def _load_curated_targets() -> list[str]:
    """The shipped, Task-2-pinned 418-entry curated LoRA target list."""
    data = yaml.safe_load((_DEFINITIONS_DIR / "base.yaml").read_text())
    targets = data["lora_targetable_modules"]
    assert len(targets) == 418, f"expected 418 curated targets, got {len(targets)}"
    return list(targets)


def _build_real_depth_tiny_transformer():
    """Tiny per-layer widths, REAL checkpoint depth (8 double + 32 single +
    2 each of noise/ref_image/context refiner) -- matches the shipped
    curated list 1:1 (test_boogu_image_definitions.py precedent)."""
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


def _make_definition(targets: list[str]):
    definition = MagicMock()
    definition.family = "boogu_image"
    definition.id = "boogu-image-test"
    definition.lora_targetable_modules = targets
    definition.architecture_params = {}
    return definition


def _get_saver():
    from app.engine.models.families.boogu_image.driver import BooguImageDriver

    definition = _make_definition(_load_curated_targets())
    drv = BooguImageDriver(definition, torch.device("cpu"))
    return drv.get_saver()


class TestCanonicalKeyContract:
    """Requirement 2: 836 keys (418 modules x lora_A/B), naming schemes,
    GQA width asymmetry survives export."""

    def test_saved_key_count_is_836(self, tmp_path):
        from safetensors.torch import load_file

        targets = _load_curated_targets()
        model = _build_peft_model(targets)
        saver = _get_saver()

        out = tmp_path / "boogu_lora.safetensors"
        saver.save(components={"unet": model, "config": {}}, path=out)
        assert out.exists()

        sd = load_file(str(out))
        assert len(sd) == 836, f"expected 836 canonical keys, got {len(sd)}"

        lora_a = [k for k in sd if k.endswith(".lora_A.weight")]
        lora_b = [k for k in sd if k.endswith(".lora_B.weight")]
        assert len(lora_a) == 418
        assert len(lora_b) == 418

    def test_all_keys_diffusion_model_prefixed(self, tmp_path):
        from safetensors.torch import load_file

        targets = _load_curated_targets()
        model = _build_peft_model(targets)
        saver = _get_saver()

        out = tmp_path / "boogu_lora.safetensors"
        saver.save(components={"unet": model, "config": {}}, path=out)
        sd = load_file(str(out))

        non_dm = [k for k in sd if not k.startswith("diffusion_model.")]
        assert not non_dm, f"keys missing 'diffusion_model.' prefix: {non_dm[:5]}"

    def test_stock_naming_scheme_present(self, tmp_path):
        """Single/refiner blocks: stock ``attn.to_*``/``attn.to_out.0``."""
        from safetensors.torch import load_file

        targets = _load_curated_targets()
        model = _build_peft_model(targets)
        saver = _get_saver()

        out = tmp_path / "boogu_lora.safetensors"
        saver.save(components={"unet": model, "config": {}}, path=out)
        sd = load_file(str(out))

        for key in (
            "diffusion_model.noise_refiner.0.attn.to_q.lora_A.weight",
            "diffusion_model.noise_refiner.0.attn.to_out.0.lora_B.weight",
            "diffusion_model.single_stream_layers.31.feed_forward.linear_1.lora_A.weight",
        ):
            assert key in sd, f"missing stock-scheme key: {key}"

    def test_processor_owned_double_stream_naming_scheme_present(self, tmp_path):
        """Double-stream blocks: processor-owned ``img_instruct_attn.processor.*``."""
        from safetensors.torch import load_file

        targets = _load_curated_targets()
        model = _build_peft_model(targets)
        saver = _get_saver()

        out = tmp_path / "boogu_lora.safetensors"
        saver.save(components={"unet": model, "config": {}}, path=out)
        sd = load_file(str(out))

        for key in (
            "diffusion_model.double_stream_layers.0.img_instruct_attn.processor.img_to_q.lora_A.weight",
            "diffusion_model.double_stream_layers.0.img_instruct_attn.processor.instruct_to_v.lora_B.weight",
            "diffusion_model.double_stream_layers.7.img_instruct_attn.to_out.0.lora_A.weight",
        ):
            assert key in sd, f"missing processor-owned key: {key}"

    def test_gqa_width_asymmetry_survives_export(self, tmp_path):
        """to_k/to_v out width (8) != to_q/to_out width (16) in the saved
        lora_B tensors -- the GQA asymmetry the Task-1 review flagged must
        be visible in the actual exported shapes."""
        from safetensors.torch import load_file

        targets = _load_curated_targets()
        model = _build_peft_model(targets)
        saver = _get_saver()

        out = tmp_path / "boogu_lora.safetensors"
        saver.save(components={"unet": model, "config": {}}, path=out)
        sd = load_file(str(out))

        to_q_b = sd["diffusion_model.noise_refiner.0.attn.to_q.lora_B.weight"]
        to_k_b = sd["diffusion_model.noise_refiner.0.attn.to_k.lora_B.weight"]
        to_v_b = sd["diffusion_model.noise_refiner.0.attn.to_v.lora_B.weight"]
        to_out_b = sd["diffusion_model.noise_refiner.0.attn.to_out.0.lora_B.weight"]

        assert to_q_b.shape[0] == TINY_TO_Q_WIDTH
        assert to_out_b.shape[0] == TINY_TO_Q_WIDTH
        assert to_k_b.shape[0] == TINY_TO_KV_WIDTH
        assert to_v_b.shape[0] == TINY_TO_KV_WIDTH
        assert to_q_b.shape[0] != to_k_b.shape[0]


class TestBaseTurboPortability:
    """Requirement 3: identical geometry -> Base-trained export loads onto
    a Turbo-wrapped tiny model bit-exact, and vice versa."""

    def test_base_export_loads_onto_turbo_bit_exact(self, tmp_path):
        from safetensors.torch import load_file

        targets = _load_curated_targets()
        base_model = _build_peft_model(targets)
        saver = _get_saver()

        out = tmp_path / "boogu_base_lora.safetensors"
        saver.save(components={"unet": base_model, "config": {}}, path=out)
        sd = load_file(str(out))

        turbo_model = _build_peft_model(targets)

        def _remap_to_peft(key: str) -> str:
            module_path = key[len("diffusion_model."):]
            module_path = module_path.replace(".lora_A.weight", ".lora_A.default.weight")
            module_path = module_path.replace(".lora_B.weight", ".lora_B.default.weight")
            return f"base_model.model.{module_path}"

        remapped = {_remap_to_peft(k): v for k, v in sd.items()}
        missing, unexpected = turbo_model.load_state_dict(remapped, strict=False)

        lora_missing = [k for k in missing if "lora" in k.lower()]
        assert not lora_missing, f"missing LoRA keys on turbo: {lora_missing[:5]}"
        assert not unexpected, f"unexpected keys loading onto turbo: {unexpected[:5]}"

        # Bit-exact: every remapped tensor must equal what's now on turbo.
        turbo_sd = turbo_model.state_dict()
        for key, tensor in remapped.items():
            assert torch.equal(turbo_sd[key], tensor), f"value mismatch at {key}"

    def test_turbo_export_loads_onto_base_bit_exact(self, tmp_path):
        """Same proof, reversed direction (geometry is symmetric)."""
        from safetensors.torch import load_file

        targets = _load_curated_targets()
        turbo_model = _build_peft_model(targets)
        saver = _get_saver()

        out = tmp_path / "boogu_turbo_lora.safetensors"
        saver.save(components={"unet": turbo_model, "config": {}}, path=out)
        sd = load_file(str(out))

        base_model = _build_peft_model(targets)

        def _remap_to_peft(key: str) -> str:
            module_path = key[len("diffusion_model."):]
            module_path = module_path.replace(".lora_A.weight", ".lora_A.default.weight")
            module_path = module_path.replace(".lora_B.weight", ".lora_B.default.weight")
            return f"base_model.model.{module_path}"

        remapped = {_remap_to_peft(k): v for k, v in sd.items()}
        missing, unexpected = base_model.load_state_dict(remapped, strict=False)

        lora_missing = [k for k in missing if "lora" in k.lower()]
        assert not lora_missing
        assert not unexpected


class TestSaverMetadataAndRegistration:
    def test_metadata_architecture_is_boogu_image(self, tmp_path):
        from safetensors import safe_open

        targets = _load_curated_targets()
        model = _build_peft_model(targets)
        saver = _get_saver()

        out = tmp_path / "boogu_lora.safetensors"
        saver.save(components={"unet": model, "config": {}}, path=out)

        with safe_open(str(out), framework="pt") as f:
            metadata = f.metadata()
        assert metadata is not None
        assert metadata["modelspec.architecture"] == "boogu_image"

    def test_driver_get_saver_returns_boogu_image_saver(self):
        from app.engine.models.families.boogu_image.driver import BooguImageDriver
        from app.engine.models.families.boogu_image.saver import BooguImageSaver

        definition = _make_definition(_load_curated_targets())
        drv = BooguImageDriver(definition, torch.device("cpu"))
        saver = drv.get_saver()
        assert isinstance(saver, BooguImageSaver)
        assert saver.architecture_name == "boogu_image"
