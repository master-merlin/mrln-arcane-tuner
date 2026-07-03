"""Krea-2 LoRA portability test: Raw → Turbo round-trip key fidelity.

TDD gate for Task 3a (Phase 3 of the krea2 family).

Verifies that:
1. The saver returned by ``driver.get_saver()`` emits canonical
   ``Krea2Transformer2DModel`` module paths (``transformer_blocks.N.attn/ff``).
2. A LoRA saved from a Raw-config model loads onto an architecturally
   identical Turbo-config model with zero missing LoRA keys and zero
   unexpected keys.
3. The key format is ``diffusion_model.{canonical_module_path}.lora_A/B.weight``
   (ai-toolkit / ComfyUI-compatible).

Key convention used by ``GenericLoRASaver`` (the base of ``QwenImageSaver``):
    ``diffusion_model.{peft_module_path}.lora_A/B.weight``
where ``peft_module_path`` is the PEFT state dict key after stripping the
``base_model.model.`` prefix.  Example:
    ``diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight``

The Turbo load uses ``strict=False`` + strips the ``diffusion_model.`` prefix
so the keys match PEFT's ``base_model.model.{path}`` expectation.
"""

from __future__ import annotations

import torch


# ── Shared tiny config (same as test_krea2_family.py) ─────────────────────────
_TINY_CFG = dict(
    in_channels=64,
    num_layers=2,
    attention_head_dim=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=256,
    timestep_embed_dim=256,
    text_hidden_dim=128,
    num_text_layers=12,
    text_num_attention_heads=4,
    text_num_key_value_heads=4,
    text_intermediate_size=128,
    num_layerwise_text_blocks=1,
    num_refiner_text_blocks=1,
    axes_dims_rope=(32, 48, 48),  # sum=128 == attention_head_dim
    rope_theta=1000.0,
    norm_eps=1e-5,
)

# LoRA targets matching the canonical 8 suffixes from krea2_arch.json
_LORA_TARGETS = [
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_gate",
    "attn.to_out.0",
    "ff.gate", "ff.up", "ff.down",
]


def _build_peft_model():
    """Build a tiny Krea2Transformer2DModel wrapped with PEFT LoRA."""
    from peft import LoraConfig, get_peft_model
    from app.engine.models.families.krea2.vendor.transformer_krea2 import (
        Krea2Transformer2DModel,
    )

    base = Krea2Transformer2DModel.from_config(_TINY_CFG)
    lora_cfg = LoraConfig(r=4, lora_alpha=4, target_modules=_LORA_TARGETS)
    return get_peft_model(base, lora_cfg)


def test_krea2_lora_saves_and_loads_raw_to_turbo(tmp_path):
    """Raw LoRA → save → load onto Turbo with zero missing/unexpected LoRA keys.

    PEFT wraps Raw and Turbo identically (same architecture, same target
    modules).  The saver is whichever ``Krea2Driver.get_saver()`` returns.
    Keys must:
      - Contain canonical ``transformer_blocks.N.attn/ff`` module paths.
      - Round-trip onto a fresh Turbo-config PEFT-wrapped model with no LoRA
        keys missing and no unexpected keys.
    """
    from safetensors.torch import load_file
    from unittest.mock import MagicMock

    from app.engine.models.families.krea2.driver import Krea2Driver

    # ── 1. Build Raw-config PEFT model ────────────────────────────────────────
    raw = _build_peft_model()

    # ── 2. Obtain the saver via the driver (same as training path) ─────────────
    definition = MagicMock()
    definition.family = "krea2"
    definition.id = "krea2-test"
    definition.lora_targetable_modules = _LORA_TARGETS
    definition.architecture_params = {}

    drv = Krea2Driver(definition, torch.device("cpu"))
    saver = drv.get_saver()

    # ── 3. Save the LoRA ───────────────────────────────────────────────────────
    saved_path = tmp_path / "krea2_raw_lora.safetensors"
    saver.save(
        components={"unet": raw, "config": {}},
        path=saved_path,
        metadata=None,
    )
    assert saved_path.exists(), "Saver did not produce a safetensors file"

    # ── 4. Inspect the saved keys ──────────────────────────────────────────────
    sd = load_file(str(saved_path))
    assert sd, "Saved state dict is empty"

    # All keys must start with "diffusion_model."
    non_dm_keys = [k for k in sd if not k.startswith("diffusion_model.")]
    assert not non_dm_keys, (
        f"Keys without 'diffusion_model.' prefix found: {non_dm_keys[:5]}"
    )

    # All keys must reference canonical Krea-2 module paths.
    # Valid containers: transformer_blocks.N.* (main blocks) and
    # text_fusion.layerwise_blocks.N.* / text_fusion.refiner_blocks.N.*
    # (text-fusion blocks — these ALSO match the attn/ff target suffixes).
    # The invariant: all must contain attn or ff sub-modules (the LoRA targets).
    non_canonical = [
        k for k in sd
        if "attn" not in k and "ff" not in k
    ]
    assert not non_canonical, (
        f"Keys without canonical attn/ff module paths: {non_canonical[:5]}"
    )
    # At least some keys must come from the main transformer_blocks container.
    assert any("transformer_blocks" in k for k in sd), (
        "No keys from transformer_blocks found in saved state dict"
    )

    # Must contain both lora_A and lora_B keys
    lora_a_keys = [k for k in sd if "lora_A" in k]
    lora_b_keys = [k for k in sd if "lora_B" in k]
    assert lora_a_keys, "No lora_A keys found in saved state dict"
    assert lora_b_keys, "No lora_B keys found in saved state dict"

    # ── 5. Load onto fresh Turbo-config PEFT model ────────────────────────────
    # Raw and Turbo share the identical Krea2Transformer2DModel architecture.
    #
    # The saver strips the ".default." adapter name from PEFT's internal keys:
    #   PEFT internal:  base_model.model.X.lora_A.default.weight
    #   Saved (ai-toolkit format): diffusion_model.X.lora_A.weight
    #
    # To load back via PEFT's load_state_dict, we must reverse both transformations:
    #   1. Strip "diffusion_model." prefix, add "base_model.model."
    #   2. Re-insert ".default" between "lora_A"/"lora_B" and ".weight"
    turbo = _build_peft_model()

    def _remap_to_peft(key: str) -> str:
        """Reverse ai-toolkit key → PEFT internal key."""
        # Strip "diffusion_model." prefix, add "base_model.model."
        module_path = key[len("diffusion_model."):]
        # Re-insert the adapter name ".default" that the saver stripped
        module_path = module_path.replace(".lora_A.weight", ".lora_A.default.weight")
        module_path = module_path.replace(".lora_B.weight", ".lora_B.default.weight")
        return f"base_model.model.{module_path}"

    remapped = {_remap_to_peft(k): v for k, v in sd.items()}

    missing, unexpected = turbo.load_state_dict(remapped, strict=False)

    # No LoRA keys should be missing on Turbo
    lora_missing = [k for k in missing if "lora" in k.lower()]
    assert not lora_missing, (
        f"LoRA keys missing when loading onto Turbo: {lora_missing[:5]}"
    )

    # No unexpected keys (all remapped keys must have matched)
    assert not unexpected, (
        f"Unexpected keys after loading onto Turbo: {unexpected[:5]}"
    )


def test_krea2_saver_key_format():
    """Saved keys follow the ai-toolkit diffusion_model.{path}.lora_A/B.weight format."""
    from safetensors.torch import load_file
    from unittest.mock import MagicMock

    from app.engine.models.families.krea2.driver import Krea2Driver

    definition = MagicMock()
    definition.family = "krea2"
    definition.id = "krea2-test"
    definition.lora_targetable_modules = _LORA_TARGETS
    definition.architecture_params = {}

    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        saved_path = pathlib.Path(td) / "test.safetensors"

        raw = _build_peft_model()
        drv = Krea2Driver(definition, torch.device("cpu"))
        saver = drv.get_saver()
        saver.save(components={"unet": raw, "config": {}}, path=saved_path)

        sd = load_file(str(saved_path))

    # Sample 3 keys and verify the format
    sample_keys = list(sd.keys())[:3]
    for k in sample_keys:
        assert k.startswith("diffusion_model."), (
            f"Key does not start with 'diffusion_model.': {k!r}"
        )
        assert k.endswith(".weight"), f"Key does not end with '.weight': {k!r}"
        assert "lora_A" in k or "lora_B" in k, (
            f"Key does not contain 'lora_A' or 'lora_B': {k!r}"
        )
        # Key must reference a canonical Krea-2 module path
        # (transformer_blocks.N.* or text_fusion.*_blocks.N.*)
        assert "attn" in k or "ff" in k, (
            f"Key does not reference canonical attn/ff module: {k!r}"
        )
    # At minimum, transformer_blocks must be present in the full saved dict
    assert any("transformer_blocks" in k for k in sd), (
        "No keys from transformer_blocks found in saved state dict"
    )


def test_krea2_saver_is_family_agnostic():
    """Krea2Saver uses family-agnostic key derivation.

    Key format is based purely on the PEFT-wrapped model's module paths,
    NOT on any qwen-specific hardcoded prefix.  This test confirms that
    Krea2Transformer2DModel module paths appear verbatim in the saved keys,
    proving the saver correctly derives canonical keys.
    """
    from safetensors.torch import load_file
    from unittest.mock import MagicMock

    from app.engine.models.families.krea2.driver import Krea2Driver

    definition = MagicMock()
    definition.family = "krea2"
    definition.id = "krea2-test"
    definition.lora_targetable_modules = _LORA_TARGETS
    definition.architecture_params = {}

    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        saved_path = pathlib.Path(td) / "test.safetensors"

        raw = _build_peft_model()

        # Inspect actual PEFT module names before saving
        from peft import get_peft_model_state_dict
        peft_sd = get_peft_model_state_dict(raw)
        peft_keys_sample = [
            k.replace("base_model.model.", "")
            for k in peft_sd.keys()
            if "lora_A" in k
        ][:3]

        drv = Krea2Driver(definition, torch.device("cpu"))
        saver = drv.get_saver()
        saver.save(components={"unet": raw, "config": {}}, path=saved_path)

        sd = load_file(str(saved_path))

    # Each peft key (stripped of base_model.model.) should appear in the
    # saved dict as "diffusion_model.{key}"
    for peft_key in peft_keys_sample:
        expected = f"diffusion_model.{peft_key}"
        assert expected in sd, (
            f"Expected key {expected!r} not found in saved dict.\n"
            f"Available keys (sample): {list(sd.keys())[:5]}"
        )


def test_krea2_saver_architecture_metadata():
    """Krea2Saver writes correct modelspec.architecture metadata.

    The saved safetensors file must contain ``modelspec.architecture: "krea2"``
    (not "qwen_image" or any other value) to ensure proper identification
    and roundtrip onto Krea-2 inference pipelines.
    """
    from safetensors import safe_open
    from unittest.mock import MagicMock

    from app.engine.models.families.krea2.driver import Krea2Driver

    definition = MagicMock()
    definition.family = "krea2"
    definition.id = "krea2-test"
    definition.lora_targetable_modules = _LORA_TARGETS
    definition.architecture_params = {}

    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        saved_path = pathlib.Path(td) / "test.safetensors"

        raw = _build_peft_model()
        drv = Krea2Driver(definition, torch.device("cpu"))
        saver = drv.get_saver()
        saver.save(components={"unet": raw, "config": {}}, path=saved_path)

        # Read metadata using safe_open
        with safe_open(str(saved_path), framework="pt") as f:
            metadata = f.metadata()

        # Assert that modelspec.architecture is set to "krea2"
        assert metadata is not None, "Safetensors metadata is None"
        assert "modelspec.architecture" in metadata, (
            f"modelspec.architecture key not found in metadata. "
            f"Available keys: {list(metadata.keys())}"
        )
        assert metadata["modelspec.architecture"] == "krea2", (
            f"modelspec.architecture is {metadata['modelspec.architecture']!r}, "
            f"expected 'krea2'"
        )
