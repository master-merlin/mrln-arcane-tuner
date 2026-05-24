"""Smoke tests for the HiDream-O1 model family.

Validates wiring (registration, definition loading, class shapes) without
touching the actual 8B checkpoint — safe to run on CI / CPU-only boxes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.engine.core.definitions import ModelDefinition
from app.engine.models.registry import ModelRegistry


# ── Registration ─────────────────────────────────────────────────────────

def test_family_is_discovered():
    """``hidream_o1`` registers itself via the auto-discovery scan."""
    registry = ModelRegistry()
    registry.discover_families()
    family_cls = registry.get_family_class("hidream_o1")
    assert family_cls.family_name == "hidream_o1"


import torch


def test_loader_manifest_is_single_unified_model():
    """The unified architecture exposes one component, mapped to 'unet'.

    HiDream-O1 is a pixel-space Unified Transformer — no separate VAE,
    no separate text encoder. The loader's manifest reflects that with
    a single ``ComponentSpec`` keyed ``"unet"`` so ``GenericTrainingPipeline``
    plumbing recognizes the family has one primary model.
    """
    from app.engine.models.families.hidream_o1.loader import HiDreamO1Loader

    definition = ModelDefinition(id="x", family="hidream_o1", name="X")
    loader = HiDreamO1Loader(device=torch.device("cpu"))
    manifest = loader.get_component_manifest(definition)

    assert len(manifest) == 1
    spec = manifest[0]
    assert spec.key == "unet"


def test_driver_reports_no_vae_no_text_encoder():
    """Pixel-space unified model — VAE and TE are intentionally None.

    HiDream-O1 has no separate VAE or text encoder. The base
    ``GenericTrainingPipeline`` is patched in Task 10 to tolerate this.
    """
    from app.engine.models.families.hidream_o1.driver import HiDreamO1Driver

    definition = ModelDefinition(id="x", family="hidream_o1", name="X")
    driver = HiDreamO1Driver(definition, device=torch.device("cpu"))

    # Before assign_components — all None / not yet wired
    assert driver.get_text_encoders() == {}
    assert driver.get_vae() is None

    # After assigning the unified model
    import torch.nn as nn
    fake_model = nn.Linear(4, 4)
    driver.assign_components({"unet": fake_model})

    assert driver.get_primary_model() is fake_model
    assert driver.get_text_encoders() == {}
    assert driver.get_vae() is None


def test_trainer_recipe_constants_match_spec():
    """Recipe constants must match the ai-toolkit May 2026 documented values
    derived from Saganaki22's trainer (spike_notes.md Task 3a).
    """
    from app.engine.models.families.hidream_o1.trainer import (
        NOISE_SCALE,
        TIMESTEP_TYPE,
        MAX_LOSS,
        LORA_EXCLUDED_SUBSTRINGS,
    )
    assert NOISE_SCALE == 8.0
    assert TIMESTEP_TYPE == "linear"
    assert MAX_LOSS == 1.0
    assert set(LORA_EXCLUDED_SUBSTRINGS) == {"lm_head", "patch_embed", "visual"}


def test_lora_inject_replaces_only_targeted_modules():
    """LoRA injection wraps linear-like modules and skips excluded names."""
    import torch.nn as nn
    from app.engine.models.families.hidream_o1.lora_wrapper import (
        HiDreamO1LoRALinear,
        inject_lora_layers,
    )

    class Mini(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
            self.lm_head = nn.Linear(8, 8)
            self.visual = nn.Sequential(nn.Linear(8, 8))

    m = Mini()
    result = inject_lora_layers(m, rank=4, alpha=4.0)
    # Two linear layers in language_model should be wrapped; lm_head and visual.* skipped.
    assert len(result.layers) == 2
    # The wrapped types are HiDreamO1LoRALinear
    for layer in result.layers:
        assert isinstance(layer, HiDreamO1LoRALinear)
    # Excluded modules remain plain Linear
    assert isinstance(m.lm_head, nn.Linear)
    assert isinstance(m.visual[0], nn.Linear)


def test_lora_wrapper_forward_adds_lora_to_base_output():
    """The wrapper's forward returns base(x) + low-rank residual * scaling."""
    import torch
    import torch.nn as nn
    from app.engine.models.families.hidream_o1.lora_wrapper import (
        HiDreamO1LoRALinear,
    )
    base = nn.Linear(8, 8)
    wrapper = HiDreamO1LoRALinear(
        base, lora_key="test", rank=2, alpha=2.0,
    )
    x = torch.randn(1, 8)
    out = wrapper(x)
    # lora_up is initialized to zeros, so at init time wrapper(x) == base(x)
    assert torch.allclose(out, base(x))
    # After perturbing lora_up, output should differ
    with torch.no_grad():
        wrapper.lora_up.copy_(torch.randn_like(wrapper.lora_up) * 0.01)
    out2 = wrapper(x)
    assert not torch.allclose(out2, base(x))


def test_saver_writes_safetensors_and_sidecar(tmp_path):
    """Saver produces ``<name>.safetensors`` + ``hidream_o1_lora_config.json`` sidecar."""
    import torch
    import torch.nn as nn
    from app.engine.models.families.hidream_o1.lora_wrapper import (
        HiDreamO1LoRALinear,
        inject_lora_layers,
    )
    from app.engine.models.families.hidream_o1.saver import HiDreamO1Saver

    class Mini(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))

    m = Mini()
    inject_lora_layers(m, rank=2, alpha=2.0)

    saver = HiDreamO1Saver()
    out_dir = tmp_path / "my_lora"
    saver.save(
        model=m,
        out_dir=str(out_dir),
        name="my_lora",
        metadata={
            "rank": 2,
            "alpha": 2.0,
            "vendor_revision": "abc123",
            "base_model": "HiDream-ai/HiDream-O1-Image",
            "noise_scale": 8.0,
            "timestep_type": "linear",
            "max_loss": 1.0,
            "excluded_modules": ["lm_head", "patch_embed", "visual"],
            "target_preset": "aitoolkit",
        },
    )

    assert (out_dir / "my_lora.safetensors").exists()
    assert (out_dir / "hidream_o1_lora_config.json").exists()


def test_saver_keys_use_diffusion_model_prefix_kohya_style(tmp_path):
    """LoRA keys MUST use ``diffusion_model.<key>.{lora_down,lora_up,alpha}`` — the
    convention ComfyUI's native HiDream-O1 LoRA loader expects (matches
    Kijai-published reference LoRAs).
    """
    import json
    import torch
    import torch.nn as nn
    from safetensors import safe_open

    from app.engine.models.families.hidream_o1.lora_wrapper import (
        inject_lora_layers,
    )
    from app.engine.models.families.hidream_o1.saver import HiDreamO1Saver

    class Mini(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Sequential(nn.Linear(8, 8))

    m = Mini()
    inject_lora_layers(m, rank=2, alpha=2.0)

    saver = HiDreamO1Saver()
    out_dir = tmp_path / "k"
    saver.save(
        model=m,
        out_dir=str(out_dir),
        name="k",
        metadata={"rank": 2, "alpha": 2.0},
    )

    with safe_open(str(out_dir / "k.safetensors"), framework="pt") as f:
        keys = sorted(f.keys())

    # Every key starts with diffusion_model.
    assert all(k.startswith("diffusion_model.") for k in keys), keys
    # Three keys per LoRA layer (down, up, alpha) — 1 layer × 3 = 3 keys total
    assert len(keys) == 3
    suffixes = sorted({k.rsplit(".", 1)[1] for k in keys})
    assert suffixes == ["alpha", "weight", "weight"][:3] or set(suffixes) == {"alpha", "weight"}
    # Validate the specific suffixes per key
    assert any(k.endswith(".lora_down.weight") for k in keys)
    assert any(k.endswith(".lora_up.weight") for k in keys)
    assert any(k.endswith(".alpha") for k in keys)

    # Sidecar JSON is well-formed
    sidecar = json.loads((out_dir / "hidream_o1_lora_config.json").read_text())
    assert "rank" in sidecar
    assert "alpha" in sidecar


def test_sampler_is_async_callable_with_default_constants():
    """Sampler.sample is async and exposes Full-variant defaults."""
    import asyncio
    from app.engine.models.families.hidream_o1.sampler import (
        HiDreamO1Sampler,
        DEFAULT_STEPS_FULL,
        DEFAULT_GUIDANCE_FULL,
    )
    assert DEFAULT_STEPS_FULL == 50
    assert DEFAULT_GUIDANCE_FULL == 5.0
    assert asyncio.iscoroutinefunction(HiDreamO1Sampler.sample)
