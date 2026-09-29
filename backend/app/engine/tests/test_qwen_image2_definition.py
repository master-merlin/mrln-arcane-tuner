"""qwen_image2 family + ``qwen-image-2.1`` definition pins (LANE-132 Task 3).

Spec scenarios 1 and 2 (``_harness/lanes/LANE-132/spec.md``):

1. the registry lists ``qwen-image-2.1`` in family ``qwen_image2`` while
   ``qwen-image-2512`` stays in ``qwen_image`` -- a NEW family, not a sibling
   definition (the architecture differs in every component);
2. the architecture params are the weights' own: 32 layers, patch 1, out 64,
   VAE z_dim 64. They were harvested from ``Qwen/Qwen-Image-2.1`` @
   ``790c92633540`` by ``app.engine.utils.config_harvester.harvest`` (script and
   output: ``.agent/workdir/lane-132/harvest_qwen_image_21.*``), never copied
   from ``qwen_image`` -- whose values (60 / 2 / 16 / 16) every assertion below
   would reject.

Also pinned, because a definition that is wrong here fails only at GPU time:

* the loader declares the VENDORED transformer/VAE classes (diffusers 0.40 has
  neither) and ``Qwen3VLForConditionalGeneration`` + ``Qwen3VLProcessor`` for
  the text side -- every dotted path must import;
* ``lora_targetable_modules`` is EXACTLY the per-block Linear suffix set of the
  real vendored transformer (dreamlite 2026-07-08 precedent: a list that
  misses a Linear or names a non-existent one trains the wrong surface
  silently). Measured on a tiny instance of the vendored class, never listed
  from memory;
* ``block_topology`` counts the checkpoint's 32 blocks under the attribute the
  vendored model really has;
* the non-commercial licence lives in the README licence table (RULE-21: the
  ONE place the text lives -- DECISION 2026-09-29 word `a`), same as every
  other restricted family, not on the definition or in saved metadata.
"""

from __future__ import annotations

import pathlib

import pytest
import torch.nn as nn

from app.engine.core.pipeline.loader_base import GenericComponentLoader
from app.engine.models.registry import ModelRegistry

DEF_ID = "qwen-image-2.1"
FAMILY = "qwen_image2"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
README = REPO_ROOT / "README.md"


@pytest.fixture()
def registry():
    def _reset():
        ModelRegistry._discovered = False
        ModelRegistry._families = {}
        ModelRegistry._definitions = {}
        ModelRegistry._paths = {}
        ModelRegistry._definitions_loaded = False

    _reset()
    r = ModelRegistry()
    r.initialize()
    yield r
    _reset()


def _defn(registry):
    defn = registry.get_definition(DEF_ID)
    assert defn is not None, f"{DEF_ID} is not registered"
    return defn


# ── Scenario 1 ──────────────────────────────────────────────────────────────

def test_both_definitions_listed_in_their_own_families(registry):
    listed = set(registry.list_models())
    assert DEF_ID in listed
    assert "qwen-image-2512" in listed
    assert registry.get_definition(DEF_ID).family == FAMILY
    assert registry.get_definition("qwen-image-2512").family == "qwen_image"


def test_family_class_is_registered(registry):
    cls = registry.get_family_class(FAMILY)
    assert cls.family_name == FAMILY
    assert cls.archetype == "latent_diffusion"


# ── Scenario 2 ──────────────────────────────────────────────────────────────

def test_architecture_params_are_the_weights_own(registry):
    arch = _defn(registry).architecture_params
    assert arch["transformer.num_layers"] == 32
    assert arch["transformer.patch_size"] == 1
    assert arch["transformer.out_channels"] == 64
    assert arch["vae.z_dim"] == 64
    # The rest of the checkpoint's identity, so a partial copy from
    # qwen_image cannot pass by matching the four numbers above.
    assert arch["transformer._class_name"] == "QwenImage21Transformer2DModel"
    assert arch["transformer.num_attention_heads"] == 32
    assert arch["transformer.attention_head_dim"] == 128
    assert arch["transformer.in_channels"] == 64
    assert arch["transformer.context_in_dim"] == 4096
    assert arch["vae._class_name"] == "AutoencoderKLQwenImage21"
    assert arch["vae.dim_mult"] == [1, 2, 4, 8, 8]
    assert arch["te.architectures"] == ["Qwen3VLForConditionalGeneration"]
    assert arch["te.text_config.hidden_size"] == 4096


def test_licence_is_in_the_readme_restricted_table():
    """RULE-21: the README licence table, not the definition or saved metadata."""
    text = README.read_text(encoding="utf-8")
    restricted = text.split("**Restricted", 1)[1].split("**Permissive", 1)[0]
    rows = [ln for ln in restricted.splitlines() if ln.startswith("| `qwen_image2`")]
    assert len(rows) == 1, f"expected exactly one qwen_image2 row, found {len(rows)}"
    row = rows[0]
    assert "Qwen/Qwen-Image-2.1" in row
    assert "non-commercial" in row


# ── Loader: every declared class resolves ───────────────────────────────────

def test_loader_declares_vendored_and_qwen3vl_classes(registry):
    from app.engine.models.families.qwen_image2.loader import QwenImage2Loader

    import torch

    manifest = QwenImage2Loader(torch.device("cpu")).get_component_manifest(_defn(registry))
    declared = {spec.key: spec.hf_class for spec in manifest}
    assert declared == {
        "tokenizer": "transformers.Qwen3VLProcessor",
        "text_encoder": "transformers.Qwen3VLForConditionalGeneration",
        "vae": (
            "app.engine.models.families.qwen_image2.vendor."
            "autoencoder_kl_qwenimage21.AutoencoderKLQwenImage21"
        ),
        "unet": (
            "app.engine.models.families.qwen_image2.vendor."
            "transformer_qwenimage21.QwenImage21Transformer2DModel"
        ),
    }
    # The checkpoint's subfolder names (model_index.json: processor, not tokenizer).
    subfolders = {spec.key: spec.subfolder for spec in manifest}
    assert subfolders == {
        "tokenizer": "processor",
        "text_encoder": "text_encoder",
        "vae": "vae",
        "unet": "transformer",
    }
    for path in declared.values():
        assert GenericComponentLoader._import_class(path) is not None, path


# ── LoRA targets measured on the real vendored class ────────────────────────

def _vendored_block_linear_suffixes() -> set[str]:
    from app.engine.models.families.qwen_image2.vendor.transformer_qwenimage21 import (
        QwenImage21Transformer2DModel,
    )

    tiny = QwenImage21Transformer2DModel(
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=8,
        context_in_dim=16,
        in_channels=4,
        out_channels=4,
        axes_dims_rope=(2, 3, 3),
    )
    suffixes = set()
    for name, mod in tiny.named_modules():
        if isinstance(mod, nn.Linear) and name.startswith("transformer_blocks.0."):
            suffixes.add(name[len("transformer_blocks.0."):])
    assert suffixes, "the tiny vendored transformer exposes no block Linear"
    return suffixes


def test_lora_targets_are_exactly_the_vendored_block_linears(registry):
    shipped = set(_defn(registry).lora_targetable_modules)
    assert shipped == _vendored_block_linear_suffixes()


def test_block_topology_counts_the_checkpoint_blocks(registry):
    topo = _defn(registry).block_topology
    assert [(b["attr_path"], b["count"]) for b in topo] == [("transformer_blocks", 32)]


# ── utils: latent arithmetic, cross-checked against the vendored pipeline ───

def test_latent_grid_uses_the_upstream_scale_factor():
    from app.engine.models.families.qwen_image2.utils import VAE_SCALE_FACTOR, latent_grid

    # The value is upstream's constant (pipeline_qwenimage21.py
    # QwenImage21PipelineHelpers.__init__: vae_scale_factor = 16), not a
    # derivation from vae/config.json (spec scope 4).
    assert VAE_SCALE_FACTOR == 16
    assert latent_grid(512, 512) == (32, 32)
    assert latent_grid(1024, 768) == (64, 48)


def test_pack_and_unpack_match_the_vendored_pipeline():
    import torch

    from app.engine.models.families.qwen_image2.utils import pack_latents, unpack_latents
    from app.engine.models.families.qwen_image2.vendor.pipeline_qwenimage21 import (
        QwenImage21PipelineHelpers,
    )

    latents = torch.randn(2, 64, 32, 32, generator=torch.Generator().manual_seed(0))
    packed = pack_latents(latents)
    assert packed.shape == (2, 1024, 64)
    oracle = QwenImage21PipelineHelpers._pack_latents(latents, 2, 64, 32, 32)
    assert torch.equal(packed, oracle)

    unpacked = unpack_latents(packed, 32, 32)
    assert torch.equal(unpacked, latents)
    oracle_unpacked = QwenImage21PipelineHelpers._unpack_latents(packed, 512, 512, 16)
    assert torch.equal(unpacked, oracle_unpacked.squeeze(2))


def test_registry_derives_the_same_topology_when_the_yaml_ships_none():
    # enrich_definition falls back to _derive_block_topology when a definition
    # ships no block_topology; the family must be one it knows.
    from app.engine.models.registry import _derive_block_topology

    topo = _derive_block_topology(FAMILY, {"transformer.num_layers": 32})
    assert [(b["attr_path"], b["count"]) for b in topo] == [("transformer_blocks", 32)]
