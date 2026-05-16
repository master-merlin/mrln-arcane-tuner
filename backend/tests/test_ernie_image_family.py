"""Smoke tests for the ERNIE-Image model family.

Validates wiring (registration, definition loading, class shapes) without
touching the actual 8B checkpoint -- safe to run on CI / CPU-only boxes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.engine.core.definitions import ModelDefinition
from app.engine.models.registry import ModelRegistry

import torch


# ── Registration ─────────────────────────────────────────────────────────

def test_family_is_discovered():
    """``ernie_image`` registers itself via the auto-discovery scan."""
    registry = ModelRegistry()
    registry.discover_families()
    family_cls = registry.get_family_class("ernie_image")
    assert family_cls.family_name == "ernie_image"


def test_family_returns_trainer_class():
    """``get_trainer_class`` returns the concrete trainer."""
    from app.engine.models.families.ernie_image.family import ErnieImageFamily
    from app.engine.models.families.ernie_image.trainer import ErnieImageTrainer

    definition = ModelDefinition(id="x", family="ernie_image", name="X")
    family = ErnieImageFamily(definition, {})
    assert family.get_trainer_class() is ErnieImageTrainer


# ── Definition YAML ──────────────────────────────────────────────────────

def test_definition_yaml_loads():
    """The base 8B definition parses into a valid ``ModelDefinition``."""
    yaml_path = (
        Path(__file__).parent.parent
        / "app/engine/models/families/ernie_image/definitions/ernie_image_base.yaml"
    )
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if "components" in data:
        for k, v in data["components"].items():
            if isinstance(v, str):
                data["components"][k] = {"path": v}

    definition = ModelDefinition(**data)
    assert definition.id == "ernie-image-base-8b"
    assert definition.family == "ernie_image"
    assert "self_attention.to_q" in definition.lora_targetable_modules
    assert "mlp.linear_fc2" in definition.lora_targetable_modules
    assert definition.architecture_params["transformer.in_channels"] == 128


# ── Component wiring (no real weights) ──────────────────────────────────

def test_loader_manifest_shape():
    """Loader emits four specs in the expected order."""
    from app.engine.models.families.ernie_image.loader import ErnieImageLoader

    definition = ModelDefinition(id="x", family="ernie_image", name="X")
    loader = ErnieImageLoader(device=torch.device("cpu"))
    manifest = loader.get_component_manifest(definition)
    keys = [spec.key for spec in manifest]
    assert keys == ["tokenizer", "text_encoder", "vae", "unet"]

    by_key = {spec.key: spec for spec in manifest}
    assert by_key["unet"].hf_class == "diffusers.ErnieImageTransformer2DModel"
    assert by_key["vae"].hf_class == "diffusers.AutoencoderKLFlux2"
    assert by_key["text_encoder"].hf_class == "transformers.AutoModel"
    # ERNIE checkpoint's tokenizer_config.json declares a Baidu-internal
    # class ("TokenizersBackend") not registered in transformers, so we
    # load via PreTrainedTokenizerFast directly from tokenizer.json.
    assert by_key["tokenizer"].hf_class == "transformers.PreTrainedTokenizerFast"


def test_driver_default_lora_targets():
    """Default LoRA target list covers attn + MLP including the asymmetric ``linear_fc2``."""
    from app.engine.models.families.ernie_image.driver import ErnieImageDriver

    definition = ModelDefinition(id="x", family="ernie_image", name="X")
    driver = ErnieImageDriver(definition, device=torch.device("cpu"))

    targets = driver.get_lora_targets()
    assert "self_attention.to_q" in targets
    assert "self_attention.to_out.0" in targets
    assert "mlp.gate_proj" in targets
    assert "mlp.up_proj" in targets
    assert "mlp.linear_fc2" in targets
    assert "down_proj" not in targets  # not the right name for ERNIE!


def test_driver_definition_overrides_lora_targets():
    """If the definition pre-declares targets, the driver uses them verbatim."""
    from app.engine.models.families.ernie_image.driver import ErnieImageDriver

    definition = ModelDefinition(
        id="x", family="ernie_image", name="X",
        lora_targetable_modules=["self_attention.to_q"],
    )
    driver = ErnieImageDriver(definition, device=torch.device("cpu"))
    assert driver.get_lora_targets() == ["self_attention.to_q"]


def test_saver_architecture_name():
    """ERNIE saver inherits the generic ai-toolkit/ComfyUI format."""
    from app.engine.models.families.ernie_image.saver import ErnieImageSaver
    from app.engine.core.pipeline.saver_base import GenericLoRASaver

    saver = ErnieImageSaver()
    assert isinstance(saver, GenericLoRASaver)
    assert saver.architecture_name == "ernie_image"


# ── Utils ────────────────────────────────────────────────────────────────

def test_patchify_roundtrip():
    """``patchify_latents`` + ``unpatchify_latents`` is identity."""
    from app.engine.models.families.ernie_image.utils import (
        patchify_latents,
        unpatchify_latents,
    )

    x = torch.randn(2, 32, 16, 16)
    packed = patchify_latents(x)
    assert packed.shape == (2, 128, 8, 8)
    restored = unpatchify_latents(packed)
    assert torch.allclose(restored, x)


def test_bn_normalize_denormalize_roundtrip():
    """``bn_normalize`` + ``bn_denormalize`` is approximately identity."""
    from app.engine.models.families.ernie_image.utils import (
        bn_denormalize,
        bn_normalize,
    )

    # Build a minimal VAE-shaped stand-in with the same attributes our utils touch.
    class _StubVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(128)
            self.bn.running_mean.fill_(0.5)
            self.bn.running_var.fill_(2.0)

            class _Cfg:
                batch_norm_eps = 1e-5

            self.config = _Cfg()

    vae = _StubVAE()
    x = torch.randn(2, 128, 4, 4)
    norm = bn_normalize(x, vae)
    back = bn_denormalize(norm, vae)
    assert torch.allclose(back, x, atol=1e-5)


# ── Dispatch (registry round-trip) ──────────────────────────────────────

def test_registry_dispatches_to_trainer():
    """``registry.get_family_class`` resolves and ``get_trainer_class`` returns the trainer."""
    from app.engine.models.families.ernie_image.trainer import ErnieImageTrainer

    registry = ModelRegistry()
    registry.discover_families()
    family_cls = registry.get_family_class("ernie_image")
    definition = ModelDefinition(id="x", family="ernie_image", name="X")
    family = family_cls(definition, {})
    assert family.get_trainer_class() is ErnieImageTrainer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
