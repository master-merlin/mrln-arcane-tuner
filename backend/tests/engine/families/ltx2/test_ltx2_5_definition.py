"""LTX-2.5 definition: pinned to the checkpoint, and to a runtime that can serve it.

Two different jobs, and the second is the one with teeth.

**The values.** ``ltx2_5.yaml`` was generated from LTX-2.5's own component configs
rather than hand-copied from ``ltx2_3.yaml``, because the design spec's risk table
named exactly that failure ("Definition author copies 2.3's VAE params") -- and it
was right to: 24 architecture params differ between the two, eight of them inside
the video VAE, where a copied value produces a model that builds and trains and is
quietly the wrong shape. The literals below are transcribed from those configs, in
the house style of every other ``*_definitions.py`` in this tree.

**The runtime.** A definition declares architecture flags; the transformer class
the loader resolves either accepts them or does not. diffusers 0.39 accepts four of
LTX-2.5's and *silently ignores* the rest -- no error, no warning. The weights load,
the shapes agree, and the model is not the one the checkpoint describes. That is the
whole reason LTX-2.5 could not simply be a YAML file, and it is invisible to every
test that only reads the YAML. So the last test here asks the runtime, not the file.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib

import pytest

from app.engine.models.families.ltx2.loader import Ltx2Loader
from app.engine.models.registry import ModelRegistry

DEFINITION_ID = "ltx2-5-base"


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False
    yield
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False


@pytest.fixture
def definition():
    registry = ModelRegistry()
    registry.initialize()
    defn = registry.get_definition(DEFINITION_ID)
    assert defn is not None, f"{DEFINITION_ID} did not register"
    return defn


def test_ltx2_5_definition_loads(definition):
    assert definition.family == "ltx2"
    assert definition.components["repo"].path == "huggingface:Lightricks/LTX-2.5-Diffusers"


def test_it_is_a_second_definition_not_a_replacement(definition):
    """LTX-2.3 must still be there. 2.5 is an addition, not an upgrade in place."""
    registry = ModelRegistry()
    registry.initialize()
    assert registry.get_definition("ltx2-3-base") is not None, (
        "ltx2-3-base disappeared. 2.5 is a NEW definition inside the same family; "
        "a released id may not be replaced (ARCHITECTURE D2)."
    )


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        # Unchanged from 2.3 -- the transformer geometry really is the same model.
        ("transformer.num_layers", 48),
        ("transformer.num_attention_heads", 32),
        ("transformer.attention_head_dim", 128),
        ("transformer.cross_attention_dim", 4096),
        ("transformer.in_channels", 128),
        ("transformer.out_channels", 128),
        ("transformer.caption_channels", 3840),
        # The four flags diffusers 0.39 ignores. These are why 2.5 is not just a YAML.
        ("transformer.ff_bias", False),
        ("transformer.audio_ff_bias", True),
        ("transformer.use_prompt_adaln_single", True),
        ("transformer.use_keyframes_abs_pos_embedding", True),
        # Scheduler: both changed from 2.3 (which has True / 0.1).
        ("scheduler.use_dynamic_shifting", False),
        ("scheduler.shift_terminal", None),
        # Latent geometry: identical to 2.3, which is what lets the LoRA target
        # list carry over unchanged.
        ("video.vae_spatial", 32),
        ("video.vae_temporal", 8),
        ("video.frame_rule", "8n+1"),
    ],
)
def test_architecture_params_match_the_checkpoint(definition, key, expected):
    assert definition.architecture_params[key] == expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        # The eight video-VAE params that differ from 2.3. Listed explicitly
        # because "copied 2.3's VAE params" is the specific mistake this file
        # exists to make impossible.
        ("vae.block_out_channels", [256, 512, 1024, 1024]),
        ("vae.decoder_block_out_channels", [256, 512, 512, 1024]),
        ("vae.decoder_layers_per_block", [4, 6, 4, 2, 2]),
        ("vae.layers_per_block", [4, 6, 4, 2, 2]),
        ("vae.decoder_spatial_padding_mode", "zeros"),
        ("vae.decoder_spatio_temporal_scaling", [True, True, True, True]),
        ("vae.upsample_factor", [2, 2, 1, 2]),
        ("vae.upsample_residual", [False, False, False, False]),
    ],
)
def test_video_vae_params_are_2_5s_not_2_3s(definition, key, expected):
    assert definition.architecture_params[key] == expected


def test_the_vae_params_actually_differ_from_2_3():
    """Anti-vacuity: if 2.3 ever adopts these values the test above proves nothing."""
    registry = ModelRegistry()
    registry.initialize()
    old = registry.get_definition("ltx2-3-base").architecture_params
    new = registry.get_definition(DEFINITION_ID).architecture_params
    differing = [k for k in new if k.startswith("vae.") and old.get(k) != new[k]]
    assert len(differing) >= 8, (
        f"only {len(differing)} vae.* params differ between 2.3 and 2.5; the "
        "'did someone copy 2.3's VAE?' check above has gone vacuous"
    )


@pytest.mark.parametrize(
    ("definition_id", "te_class", "vocoder_class"),
    [
        ("ltx2-3-base", "transformers.Gemma3ForConditionalGeneration",
         "diffusers.pipelines.ltx2.LTX2Vocoder"),
        (DEFINITION_ID, "transformers.Gemma4UnifiedForConditionalGeneration",
         "diffusers.pipelines.ltx2.LTX2VocoderWithBWE"),
    ],
)
def test_the_loader_resolves_each_definitions_own_classes(
    definition_id, te_class, vocoder_class,
):
    """Both halves, because only one of them was in the original design.

    The spec made the TEXT ENCODER definition-driven and stopped there. The vocoder
    was hardcoded too, and 2.5 changes it as well (16 kHz in, 48 kHz out). That one
    is reached only on an audio run, so a hardcoded class would not fail at load --
    it would surface as a shape error deep inside sampling, on the path that costs
    a GPU hour to reach.
    """
    registry = ModelRegistry()
    registry.initialize()
    defn = registry.get_definition(definition_id)
    loader = Ltx2Loader(device="cpu", train_audio=True)
    manifest = {spec.key: spec.hf_class for spec in loader.get_component_manifest(defn)}
    assert manifest["text_encoder"] == te_class
    assert manifest["vocoder"] == vocoder_class


def test_the_resolved_transformer_accepts_every_flag_the_definition_declares(definition):
    """The one test here that asks the RUNTIME rather than the YAML.

    diffusers accepts unknown keys in a component config without complaint: an
    ignored flag is not an error, it is a different model. On 0.39,
    ``LTX2VideoTransformer3DModel`` ignores all four of the flags LTX-2.5 sets,
    which is why the design called for a vendored fork; diffusers 0.40 accepts
    them natively. Either resolution satisfies this test, and nothing else does --
    which is exactly the property worth pinning, because the failure it guards
    against is silent by construction.

    **What this test does NOT prove**, named because an adversarial review was
    right to name it: acceptance is not implementation. A class can take a flag,
    allocate whatever it implies, and never read it in the forward --
    ``use_keyframes_abs_pos_embedding`` is exactly that on diffusers 0.40, and it
    passes here. That case is pinned separately below rather than left to be
    rediscovered. This test's job is the narrower one it can actually do: catch a
    flag the runtime does not know about at all.
    """
    loader = Ltx2Loader(device="cpu")
    manifest = {spec.key: spec.hf_class for spec in loader.get_component_manifest(definition)}
    module_path, _, class_name = manifest["unet"].rpartition(".")
    cls = getattr(importlib.import_module(module_path), class_name)
    accepted = set(inspect.signature(cls.__init__).parameters)

    declared = {
        key.split(".", 1)[1]
        for key in definition.architecture_params
        if key.startswith("transformer.") and not key.split(".", 1)[1].startswith("_")
    }
    # `hidden_size` is a derived alias this project records for the VRAM estimator,
    # not a constructor argument of any diffusers transformer.
    declared -= {"hidden_size"}
    ignored = sorted(declared - accepted)
    assert not ignored, (
        f"{manifest['unet']} silently ignores {len(ignored)} flag(s) that "
        f"{DEFINITION_ID} declares: {ignored}. Nothing raises -- the checkpoint loads, "
        "the shapes agree, and the model built is not the one the config describes. "
        "Resolve by running a diffusers that accepts them, or by pointing "
        "`transformer._module` at a vendored class that does."
    )


def test_the_prompt_length_is_upstreams_and_a_whole_number_of_registers(definition):
    """256 would load fine and condition the model differently. Twice over.

    The text connectors REPLACE every padded position with a learned register
    (128 of them, indexed by absolute position) and then zero the attention mask,
    so padding here is not something masked away -- it IS conditioning. Changing
    the length changes how many register tokens the transformer cross-attends to.
    Every LTX-2 pipeline in diffusers 0.39 and 0.40 defaults to 1024. LTX-2.3
    shipped 256 and was moved to 1024 on 2026-08-28 on the user's call, the
    compatibility argument being void because it was never broadly released --
    so BOTH definitions are pinned here, and the family no longer disagrees with
    upstream about what a prompt is.

    The divisibility half is a hard runtime constraint, not a preference --
    ``LTX2TextConnector.forward`` raises when ``seq_len % num_learnable_registers``
    is non-zero, and it raises at the first training step, not at load.
    """
    registry = ModelRegistry()
    registry.initialize()
    for def_id in ("ltx2-3-base", DEFINITION_ID):
        max_length = registry.get_definition(def_id).architecture_params["te.max_length"]
        assert max_length == 1024, (
            f"{def_id} te.max_length is {max_length}; upstream's LTX-2 pipelines all "
            "default to 1024. This is not a cost knob -- padded positions become learned "
            "registers, so the length is part of the conditioning."
        )
        assert max_length % 128 == 0, (
            f"{def_id} te.max_length {max_length} is not a multiple of the connectors' 128 "
            "learnable registers; LTX2TextConnector.forward raises on the first step"
        )


def test_the_keyframes_flag_is_declared_for_LOADING_not_for_a_feature(definition):
    """A recorded limitation, so nobody promises I2V keyframes on the strength of it.

    The 2.5 checkpoint carries a ``keyframes_abs_pos_embedding`` weight, so the
    definition sets the flag that allocates it -- otherwise the tensor is an
    unexpected key and the load is no longer clean. But diffusers 0.40 allocates
    it and never reads it: its own docstring says "the regular distilled forward
    path does not consume it until a dedicated keyframes pipeline wires it in".

    So the flag buys load fidelity, not behaviour, and our I2V path is the same
    per-token-timestep conditioning LTX-2.3 uses -- which is also what upstream's
    regular forward does. Asserted rather than commented so that the day diffusers
    wires it up, this test fails and someone decides deliberately whether to use it.
    """
    assert definition.architecture_params["transformer.use_keyframes_abs_pos_embedding"] is True

    import diffusers.models.transformers.transformer_ltx2 as mod

    source = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    # Scope to the MODEL class's own forward. Searching from the file's first
    # `def forward(` would start above the model's __init__ -- where the parameter
    # is allocated -- and the test would report consumption that is really just
    # the allocation it exists to distinguish from consumption.
    class_start = source.find("class LTX2VideoTransformer3DModel")
    assert class_start > 0, "LTX2VideoTransformer3DModel was renamed; re-read this test"
    forward_start = source.find("    def forward(", class_start)
    assert forward_start > 0, "the model class has no forward(); re-read this test"
    assert "keyframes_abs_pos_embedding" not in source[forward_start:], (
        "diffusers now CONSUMES keyframes_abs_pos_embedding in the forward pass. "
        "LTX-2.5's I2V keyframe conditioning has become available -- decide whether "
        "the ltx2 driver should feed it, rather than inheriting this test's assumption."
    )
