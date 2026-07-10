"""PRXPixel LoRA portability: canonical keys + pinned key count.

No usable upstream LoRA loader mixin exists for PRX pipelines — OUR saver's
canonical ai-toolkit keys (``diffusion_model.{module}.lora_A/B.weight``) are
the format of record (same shared format as the latent prx sibling).

Pinned key math (checkpoint config: depth=24, FUSED projections):
- per block: img_qkv_proj + txt_kv_proj + to_out.0
             + gate_proj + up_proj + down_proj = 6 modules
- total modules: 24 * 6 = 144 → 144 * 2 (lora_A + lora_B) = **288 keys**
The tiny 1-block PIXEL-variant model used here (bottleneck img_in +
resolution_embeds ON) pins the per-block count (6 → 12 keys) and the
full-model expectation is derived from it.

PORTABILITY NOTE: prx ↔ prx_pixel LoRAs are NOT interchangeable — the
architectures differ (in_channels 16 vs 3, hidden 1792 vs 3584, depth 16
vs 24, plain vs bottleneck img_in), so the shapes can never line up. The
contract asserted here is that the saver METADATA distinguishes the two
families (``modelspec.architecture``: "prx" vs "prx_pixel") so a loader
can reject a cross-family file up front.
"""

from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import MagicMock

import torch


_TINY_CFG = dict(
    in_channels=3,
    patch_size=2,
    context_in_dim=8,
    hidden_size=32,
    num_heads=2,
    depth=1,
    axes_dim=[8, 8],
    bottleneck_size=16,
    resolution_embeds=True,
)

# Checkpoint block count (verified transformer/config.json)
_DEPTH = 24

# Per-block LoRA module count (asserted empirically below)
_BLOCK_MODULES = 6

# THE pinned count for the real checkpoint: 288 keys.
_EXPECTED_FULL_MODEL_KEYS = _DEPTH * _BLOCK_MODULES * 2  # == 288

# Tiny model (1 block): 6 * 2 = 12 keys.
_EXPECTED_TINY_KEYS = _BLOCK_MODULES * 2


def _make_driver():
    from app.engine.models.families.prx_pixel.driver import PRXPixelDriver

    definition = MagicMock()
    definition.family = "prx_pixel"
    definition.id = "prx-pixel-test"
    definition.lora_targetable_modules = []
    definition.architecture_params = {}
    return PRXPixelDriver(definition, torch.device("cpu"))


def _build_peft_model(driver):
    """Tiny pixel-variant PRXTransformer2DModel wrapped with the LoRA spec."""
    from peft import LoraConfig, get_peft_model
    from diffusers.models.transformers.transformer_prx import (
        PRXTransformer2DModel,
    )

    base = PRXTransformer2DModel(**_TINY_CFG)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=driver.get_lora_targets(),
        exclude_modules=driver.get_lora_exclude_modules(),
    )
    return get_peft_model(base, lora_cfg)


def _save_lora(tmp_dir: str):
    from safetensors.torch import load_file

    drv = _make_driver()
    peft_model = _build_peft_model(drv)
    saver = drv.get_saver()
    path = pathlib.Path(tmp_dir) / "prx_pixel_lora.safetensors"
    saver.save(components={"unet": peft_model, "config": {}}, path=path)
    assert path.exists(), "Saver did not produce a safetensors file"
    return path, load_file(str(path))


def test_key_count_pinned_12_tiny_288_full():
    """Tiny 1-block pixel model: exactly 12 keys; the per-block count pins
    the full-checkpoint expectation at 288 keys (144 modules × A/B)."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    assert len(sd) == _EXPECTED_TINY_KEYS, (
        f"expected {_EXPECTED_TINY_KEYS} keys for the 1-block tiny model, "
        f"got {len(sd)}"
    )

    # Per-block module count measured from the actual saved keys
    block_modules = {
        k.rsplit(".lora_", 1)[0]
        for k in sd
        if k.startswith("diffusion_model.blocks.")
    }
    assert len(block_modules) == _BLOCK_MODULES, (
        f"one block must contribute {_BLOCK_MODULES} modules, "
        f"got {len(block_modules)}: {sorted(block_modules)}"
    )

    # Fused-projection module set (no to_q/to_k/to_v exist in PRX)
    suffixes = {m.split("diffusion_model.blocks.0.")[-1] for m in block_modules}
    assert suffixes == {
        "attention.img_qkv_proj",
        "attention.txt_kv_proj",
        "attention.to_out.0",
        "gate_proj",
        "up_proj",
        "down_proj",
    }

    # The full-model pin: 24 * 6 = 144 modules → 288 keys.
    assert _EXPECTED_FULL_MODEL_KEYS == 288


def _real_config_model():
    """Meta-instantiate the REAL checkpoint transformer config (no weights)."""
    from diffusers.models.transformers.transformer_prx import (
        PRXTransformer2DModel,
    )

    with torch.device("meta"):
        return PRXTransformer2DModel(
            in_channels=3,
            patch_size=16,
            context_in_dim=2048,
            hidden_size=3584,
            mlp_ratio=3.5,
            num_heads=28,
            depth=_DEPTH,
            axes_dim=[64, 64],
            theta=10000,
            bottleneck_size=768,
            resolution_embeds=True,
        )


def test_definition_ships_curated_lora_target_list():
    """prx-pixel-t2i MUST ship the curated 144-module list in its YAML.

    dreamlite precedent (2026-07-08 GPU-UAT crash): a YAML with NO
    ``lora_targetable_modules`` gets the field auto-filled at first real model
    load by ``registry.enrich_definition`` with the introspector's EXHAUSTIVE
    Linear catalog (bottleneck img_in.0/img_in.1, txt_in, time_in,
    final_layer, resolution embedders...), and the driver prefers a non-empty
    definition list over the curated prx_shared patterns — silently breaking
    the 288-key pinned surface.

    The shipped list is PRX_BLOCK_LORA_TARGETS fully expanded over the real
    depth-24 pixel config: 24 × 6 = 144 block-scoped module paths.
    """
    from app.engine.models.families.prx_shared import PRX_BLOCK_LORA_TARGETS
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()

    real = _real_config_model()
    expected = {
        n
        for n, m in real.named_modules()
        if isinstance(m, torch.nn.Linear)
        and any(n == p or n.endswith("." + p) for p in PRX_BLOCK_LORA_TARGETS)
    }
    assert len(expected) == _DEPTH * _BLOCK_MODULES  # == 144
    assert all(n.startswith("blocks.") for n in expected)

    defn = ModelRegistry._definitions["prx-pixel-t2i"]
    shipped = set(defn.lora_targetable_modules or [])
    assert shipped, "prx-pixel-t2i: YAML must ship the curated LoRA target list"
    assert shipped == expected, (
        f"prx-pixel-t2i: shipped list diverges from the curated/tested surface "
        f"(+{len(shipped - expected)} extra, -{len(expected - shipped)} missing). "
        f"Extras include e.g. {sorted(shipped - expected)[:3]}"
    )


def test_saver_key_format_is_ai_toolkit():
    """All keys are diffusion_model.{module}.lora_A/B.weight and ONLY
    block-level modules are wrapped — in particular the pixel variant's
    bottleneck img_in.0/img_in.1 must NOT be swept in (the exclusion-free
    contract survives the Sequential img_in)."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    assert sd, "Saved state dict is empty"
    for k in sd:
        assert k.startswith("diffusion_model."), f"bad prefix: {k!r}"
        assert k.endswith(".weight"), f"bad suffix: {k!r}"
        assert ".lora_A." in k or ".lora_B." in k, f"not a LoRA key: {k!r}"
        assert ".default." not in k, f"PEFT adapter name leaked: {k!r}"
        # Exclusion-free contract: every wrapped module lives in a block.
        assert k.startswith("diffusion_model.blocks."), (
            f"non-block module wrapped: {k!r}"
        )

    for forbidden in ("img_in", "txt_in", "time_in", "final_layer",
                      "modulation", "resolution_embedder"):
        assert not any(f".{forbidden}." in k for k in sd), (
            f"{forbidden} must NOT carry LoRA"
        )

    lora_a = [k for k in sd if ".lora_A." in k]
    lora_b = [k for k in sd if ".lora_B." in k]
    assert len(lora_a) == len(lora_b), "lora_A/lora_B counts must match"


def test_saver_fused_projection_shapes():
    """lora_B carries the FUSED output width: 3×hidden for img_qkv_proj,
    2×hidden for txt_kv_proj (rank-4 adapters on the tiny model)."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    hidden = _TINY_CFG["hidden_size"]
    qkv_b = sd["diffusion_model.blocks.0.attention.img_qkv_proj.lora_B.weight"]
    assert tuple(qkv_b.shape) == (3 * hidden, 4)
    kv_b = sd["diffusion_model.blocks.0.attention.txt_kv_proj.lora_B.weight"]
    assert tuple(kv_b.shape) == (2 * hidden, 4)
    out_b = sd["diffusion_model.blocks.0.attention.to_out.0.lora_B.weight"]
    assert tuple(out_b.shape) == (hidden, 4)


def test_saver_architecture_metadata_distinguishes_pixel_from_latent():
    """modelspec.architecture must be 'prx_pixel' — stamped by the family
    subclass, NOT by prx_shared (whose base stays architecture-agnostic).

    prx ↔ prx_pixel portability is NOT expected (different in_channels /
    hidden / depth / img_in) — the metadata stamp is what lets a loader
    reject a cross-family file, so it must differ from the latent
    sibling's."""
    from safetensors import safe_open

    from app.engine.models.families.prx.saver import PRXSaver
    from app.engine.models.families.prx_shared import PRXSharedLoRASaver

    # The shared base must NOT hardcode a family name.
    assert PRXSharedLoRASaver.architecture_name == ""

    with tempfile.TemporaryDirectory() as td:
        path, _ = _save_lora(td)
        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata()

    assert metadata is not None
    assert metadata.get("modelspec.architecture") == "prx_pixel", (
        f"wrong architecture metadata: {metadata.get('modelspec.architecture')!r}"
    )
    # Cross-family distinguishability: the latent sibling stamps "prx".
    assert PRXSaver.architecture_name == "prx"
    assert metadata["modelspec.architecture"] != PRXSaver.architecture_name


def test_lora_round_trips_onto_fresh_model():
    """Saved keys load back onto a fresh identically-wrapped model with zero
    missing LoRA keys and zero unexpected keys (ai-toolkit → PEFT remap)."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    fresh = _build_peft_model(_make_driver())

    def _remap_to_peft(key: str) -> str:
        module_path = key[len("diffusion_model."):]
        module_path = module_path.replace(".lora_A.weight", ".lora_A.default.weight")
        module_path = module_path.replace(".lora_B.weight", ".lora_B.default.weight")
        return f"base_model.model.{module_path}"

    remapped = {_remap_to_peft(k): v for k, v in sd.items()}
    missing, unexpected = fresh.load_state_dict(remapped, strict=False)

    lora_missing = [k for k in missing if "lora" in k.lower()]
    assert not lora_missing, f"LoRA keys missing on reload: {lora_missing[:5]}"
    assert not unexpected, f"Unexpected keys on reload: {unexpected[:5]}"
