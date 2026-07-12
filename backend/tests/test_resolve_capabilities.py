"""Tests for :func:`resolve_capabilities` — the per-definition capability
descriptor the Training UI consumes, plus the consistency safety-net.

Ground-truth note
-----------------
Driver text-encoder LoRA introspection (``get_te_lora_targets``) is
*weight-free*: every driver returns a hardcoded list and its ``__init__``
takes only ``(definition, device)`` without touching weights. So the
safety-net test constructs a fresh driver per family and asserts the
declared ``supports_train_te`` capability matches the driver's actual
TE-LoRA targets — no model weights are loaded.

The remaining ground-truth flags (``has_vae`` / ``has_external_te``)
CANNOT be introspected weight-free: ``get_vae`` / ``get_text_encoders``
read components assigned only after ``assign_components`` (returning
None / {} beforehand), so they reflect emptiness, not the archetype.
Those flags are instead asserted against a hardcoded expectation table
derived from each family's architecture.
"""

from __future__ import annotations

import importlib
import inspect

import pytest
import torch

from app.engine.core.archetypes import resolve_capabilities
from app.engine.core.interfaces import IModelDriver
from app.engine.models.registry import registry


@pytest.fixture(scope="module", autouse=True)
def _loaded_registry():
    registry.discover_families()
    registry.load_definitions("app/engine/models/definitions")
    return registry


# Real definition ids confirmed via registry.list_models().
ALL_DEFINITION_IDS = [
    "sdxl_base_1.0",
    "flux1-dev",
    "flux1-kontext-dev",
    "flux1-schnell",
    "flux2-dev",
    "flux2-klein-base-4b",
    "flux2-klein-base-9b",
    "qwen-image-2512",
    "qwen-image-edit-2509",
    "zimage-base",
    "zimage-de-turbo",
    "ovis-image-base",
    "ernie-image-base-8b",
    "hidream_o1_image",
    "longcat-image-base",
    "prx-sft",
    "dreamlite-base",
    "dreamlite-mobile",
    "prx-pixel-t2i",
]

# Hardcoded ground-truth for flags that are NOT weight-free introspectable:
# family -> (has_vae, has_external_te). The two flags coincided for every
# family until prx_pixel (pixel-space DiT, NO VAE, but a REAL standalone
# Qwen3-VL text encoder), so the table carries them separately.
FAMILY_HAS_VAE_AND_TE = {
    "sdxl": (True, True),
    "flux1": (True, True),
    "flux2": (True, True),
    "qwen_image": (True, True),
    "zimage": (True, True),
    "ernie_image": (True, True),
    "microsoft_lens": (True, True),  # latent_diffusion DiT (VAE) + decoupled external TE
    "ideogram4": (True, True),  # latent_diffusion DiT (custom 32ch VAE) + Qwen3-VL TE
    "wan21": (True, True),  # latent_diffusion DiT (Wan-VAE) + UMT5-XXL external TE
    "wan22": (True, True),  # latent_diffusion dual-expert DiT (Wan-VAE) + UMT5-XXL external TE
    "ltx2": (True, True),  # latent_diffusion DiT (LTX2 VAE) + Gemma3 external TE
    "krea2": (True, True),  # latent_diffusion DiT + Qwen VAE + external stacked Qwen3-VL TE
    "ovis_image": (True, True),  # latent_diffusion MMDiT (Flux VAE) + external Qwen3 TE (no TE training)
    "longcat_image": (True, True),  # latent_diffusion DiT (16ch AutoencoderKL) + Qwen2.5-VL TE
    "prx": (True, True),  # latent_diffusion DiT (Flux 16ch AutoencoderKL) + T5Gemma TE (no TE training)
    "dreamlite": (True, True),  # latent_diffusion U-NET (AutoencoderTiny) + Qwen3-VL TE (no TE training)
    "hunyuan_video15": (True, True),  # latent_diffusion video MMDiT (HV15 16x/4x VAE) + dual external TE (Qwen2.5-VL + ByT5)
    "kandinsky5": (True, True),  # latent_diffusion video DiT (HunyuanVideo VAE) + dual Qwen2.5-VL/CLIP TE
    "hidream_o1": (False, False),  # unified transformer: pixel-space, TE inside the model
    "prx_pixel": (False, True),  # pixel_transformer: NO VAE, external Qwen3-VL text TE
    "boogu_image": (True, True),  # latent_diffusion DiT (FLUX.1-dev VAE) + external Qwen3-VL mllm TE (no TE training)
}


def _driver_class_for_family(family_name: str) -> type[IModelDriver]:
    """Resolve a family's driver class by module convention.

    families.<family>.driver contains exactly one concrete IModelDriver
    subclass.
    """
    module = importlib.import_module(f"app.engine.models.families.{family_name}.driver")
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, IModelDriver)
            and obj is not IModelDriver
            and obj.__module__ == module.__name__
        ):
            return obj
    raise AssertionError(f"No IModelDriver subclass found for {family_name}")


def test_resolve_merges_defaults_and_overrides():
    defn = registry.get_definition("sdxl_base_1.0")
    r = resolve_capabilities(defn)

    assert r["archetype"] == "latent_diffusion"
    # SDXL override flips the archetype default (False) to True.
    assert r["capabilities"]["supports_train_te"] is True
    # YAML default 1e-4 overrides the template's 1e-5.
    assert r["defaults"]["learning_rate"] == 1e-4
    assert r["defaults"]["resolution"] == 1024
    assert r["field_visibility"]["train_text_encoder"]["supported"] is True


def test_hidream_hides_caching():
    defn = registry.get_definition("hidream_o1_image")
    r = resolve_capabilities(defn)

    assert r["archetype"] == "unified_transformer"
    fv = r["field_visibility"]
    assert fv["cache_latents"]["supported"] is False
    assert fv["cache_text_embeddings"]["supported"] is False
    assert fv["train_text_encoder"]["supported"] is False
    assert r["defaults"]["learning_rate"] == 5e-6


def test_declared_train_te_matches_ground_truth():
    """THE SAFETY NET — declared supports_train_te is True iff the family
    actually exposes text-encoder LoRA targets.

    sdxl is the only FAMILY with non-empty get_te_lora_targets() (all its
    definitions inherit it — sdxl_base_1.0, illustrious-xl-v2.0, ...);
    a miscategorized archetype/override would break this.
    """
    for model_id in registry.list_models():
        defn = registry.get_definition(model_id)
        declared = resolve_capabilities(defn)["capabilities"]["supports_train_te"]

        driver_cls = _driver_class_for_family(defn.family)
        driver = driver_cls(defn, torch.device("cpu"))
        has_te_targets = len(driver.get_te_lora_targets()) > 0

        assert declared == has_te_targets, (
            f"{model_id}: declared supports_train_te={declared} but driver "
            f"TE-LoRA targets present={has_te_targets}"
        )
        # Pin the expected truth: only the sdxl family trains the text encoder.
        assert declared is (defn.family == "sdxl")


def test_capability_flags_match_drivers():
    """Stronger ground-truth: has_vae / has_external_te match the family's
    known architecture (hardcoded table — these flags are not weight-free
    introspectable; see module docstring)."""
    for model_id in registry.list_models():
        defn = registry.get_definition(model_id)
        caps = resolve_capabilities(defn)["capabilities"]
        expected_vae, expected_te = FAMILY_HAS_VAE_AND_TE[defn.family]

        assert caps["has_vae"] is expected_vae, (
            f"{model_id}: has_vae={caps['has_vae']} != expected {expected_vae}"
        )
        assert caps["has_external_te"] is expected_te, (
            f"{model_id}: has_external_te={caps['has_external_te']} "
            f"!= expected {expected_te}"
        )
        # Block-swap availability tracks the VAE/latent split (the two
        # pixel-space archetypes both disable it, external TE or not).
        assert caps["supports_block_swap"] is expected_vae


@pytest.mark.parametrize("model_id", ALL_DEFINITION_IDS)
def test_all_definitions_resolve(model_id):
    """Every shipped definition resolves to a well-formed descriptor."""
    defn = registry.get_definition(model_id)
    assert defn is not None, f"missing definition {model_id}"
    r = resolve_capabilities(defn)
    assert set(r) == {"archetype", "capabilities", "field_visibility", "defaults"}
    assert r["archetype"] in (
        "latent_diffusion",
        "unified_transformer",
        "pixel_transformer",
    )
