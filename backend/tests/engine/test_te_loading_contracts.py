"""Text-encoder LOADING-CONTRACT tests for every family (hardening W2-B).

The B1 "random TE" bug class
============================
A family loader declares a plausible-but-WRONG ``transformers`` class for a
text-encoder checkpoint.  ``from_pretrained`` then loads it anyway with 0/N
tensors matched (a prefix / base-vs-head mismatch), warns only on stderr, and
silently returns a model whose entire ~B-parameter TE is *random* -- so the
family trains and samples on garbage conditioning.  This shipped twice:

* **ideogram4** (historical): declared ``AutoModel`` -> ``Qwen3VLModel`` for a
  ``Qwen3VLForConditionalGeneration`` checkpoint (``model.``-prefixed keys).
* **dreamlite** (2026-07-11, merge 795f49bb): declared ``Qwen3VLModel`` for a
  ``Qwen3VLForConditionalGeneration`` checkpoint -> 0/625 tensors loaded ->
  flat solid-colour previews.

The mechanism that catches it
=============================
For each family's text-encoder component, build a TINY model of the
checkpoint's **TRUE** architecture (what the upstream repo's ``config.json``
``architectures`` field actually says -- pinned per case below from the local
HF cache), ``save_pretrained`` it (real on-disk key layout, no network), then
reload it via the class the family's loader **DECLARES** and assert every
tensor is consumed: ``missing_keys == []`` **and** ``unexpected_keys == []``.
If the declared class is wrong, keys mismatch -> the test goes red.

The declared class is resolved *dynamically* off each family's real
``get_component_manifest()`` (never hardcoded), so the test also pins the
declaration: if anyone edits a loader to a wrong/looser class, this fails.

These tests are fully self-contained: tiny configs (hidden_size 8-16, 1 layer),
no model downloads, no dependency on the local HF cache.  The whole file runs
in a few seconds on CPU.

TRUE-architecture ground truth (source: local HF cache config.json,
D:/AI/huggingface/hub, unless noted):

  family            component        declared class                       checkpoint architectures
  ----------------  ---------------  -----------------------------------  ---------------------------------
  sdxl              text_encoder     CLIPTextModel                        CLIPTextModel
  sdxl              text_encoder_2   CLIPTextModelWithProjection          CLIPTextModelWithProjection
  flux1             text_encoder     CLIPTextModel                        CLIPTextModel
  flux1             text_encoder_2   T5EncoderModel                       T5EncoderModel
  wan21             text_encoder     UMT5EncoderModel                     UMT5EncoderModel
  wan22             text_encoder     UMT5EncoderModel                     UMT5EncoderModel
  ovis_image        text_encoder     Qwen3Model                           Qwen3Model
  flux2 (klein)     text_encoder     Qwen3ForCausalLM (pinned W2-B)       Qwen3ForCausalLM
  zimage            text_encoder     Qwen3ForCausalLM (pinned W2-B)       Qwen3ForCausalLM
  ernie_image       text_encoder     Mistral3Model    (pinned W2-B)       Mistral3Model
  microsoft_lens    text_encoder     GptOssForCausalLM                    GptOssForCausalLM
  ltx2              text_encoder     Gemma3ForConditionalGeneration       Gemma3ForConditionalGeneration
  dreamlite         text_encoder     Qwen3VLForConditionalGeneration      Qwen3VLForConditionalGeneration
  ideogram4         text_encoder     Qwen3VLForConditionalGeneration      Qwen3VLForConditionalGeneration*
  boogu_image       text_encoder     Qwen3VLForConditionalGeneration      Qwen3VLForConditionalGeneration (mllm/)
  krea2             text_encoder     Qwen3VLModel (hand-loaded, see below)Qwen3VLModel
  prx_pixel         text_encoder     Qwen3VLTextModel                     Qwen3VLTextModel
  qwen_image        text_encoder     Qwen2_5_VLForConditionalGeneration   Qwen2_5_VLForConditionalGeneration
  longcat_image     text_encoder     Qwen2_5_VLForConditionalGeneration   Qwen2_5_VLForConditionalGeneration
  kandinsky5        text_encoder     Qwen2_5_VLForConditionalGeneration   Qwen2_5_VLForConditionalGeneration**
  kandinsky5        text_encoder_2   CLIPTextModel                        CLIPTextModel**
  hunyuan_video15   text_encoder     Qwen2_5_VLTextModel                  Qwen2_5_VLTextModel
  hunyuan_video15   text_encoder_2   T5EncoderModel                       T5EncoderModel
  prx               text_encoder     T5GemmaEncoder                       T5GemmaEncoder (flat t5_gemma_module cfg)

  *  ideogram4 loads its TE from the SEPARATE repo ``Qwen/Qwen3-VL-8B-Instruct``
     (``separate_repo=True``), whose config declares
     ``Qwen3VLForConditionalGeneration`` -- NOT the ``ideogram-4-fp8`` fp8 root
     (whose bundled ``text_encoder/`` is a bare ``Qwen3VLModel``).  The loader
     comment documents this; the declared class is correct for the repo it uses.
  ** kandinsky5's Kandinsky-5 repo is not in the local cache; the declared
     classes (Qwen2.5-VL CFG + CLIP-L) are the standard Kandinsky-5 text towers.

Exemptions (see ``test_all_families_covered``):
  * hidream_o1 -- a UNIFIED model (the whole DiT is a vendored
    ``Qwen3VLForConditionalGeneration`` subclass with extra heads); it has NO
    standalone TE component and is loaded ``strict=False`` by design (the
    extra ``x_embedder`` / ``final_layer2`` / ``t_embedder1`` heads legitimately
    create missing keys), so a byte-clean round-trip contract does not apply.
    Its hand-load already surfaces missing/unexpected keys via ``self.warnings``.
  * wan21 ``image_encoder`` (open_clip ``.bin`` CLIP-ViT-H) and
    hunyuan_video15 i2v ``image_encoder`` (SiglipVisionModel) are VISION
    encoders, not text encoders, and load from non-diffusers formats -- outside
    the B1 text-encoder bug class.
"""

from __future__ import annotations

import importlib
import inspect
import tempfile
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from app.engine.core.pipeline.loader_base import GenericComponentLoader


# ---------------------------------------------------------------------------
# Tiny TRUE-architecture builders (keyed by the checkpoint's real
# ``architectures`` value). Each returns a minimal instance whose
# ``save_pretrained`` produces the genuine on-disk key layout.
# ---------------------------------------------------------------------------

def _clip_text():
    from transformers import CLIPTextConfig, CLIPTextModel
    return CLIPTextModel(CLIPTextConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, max_position_embeddings=32, projection_dim=16))


def _clip_text_projection():
    from transformers import CLIPTextConfig, CLIPTextModelWithProjection
    return CLIPTextModelWithProjection(CLIPTextConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, max_position_embeddings=32, projection_dim=16))


def _t5_encoder():
    from transformers import T5Config, T5EncoderModel
    return T5EncoderModel(T5Config(
        vocab_size=64, d_model=16, d_ff=32, d_kv=8, num_layers=1, num_heads=2,
        relative_attention_num_buckets=8))


def _umt5_encoder():
    from transformers import UMT5Config, UMT5EncoderModel
    return UMT5EncoderModel(UMT5Config(
        vocab_size=64, d_model=16, d_ff=32, d_kv=8, num_layers=1, num_heads=2,
        relative_attention_num_buckets=8))


def _qwen3_model():
    from transformers import Qwen3Config, Qwen3Model
    return Qwen3Model(Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=32, tie_word_embeddings=True))


def _qwen3_causal():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=32, tie_word_embeddings=True))


def _mistral3_model():
    from transformers import Mistral3Model
    from transformers.models.mistral.configuration_mistral import MistralConfig
    from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
    from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig
    text = MistralConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=32, tie_word_embeddings=True)
    vision = PixtralVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, head_dim=8, image_size=16, patch_size=4)
    return Mistral3Model(Mistral3Config(
        text_config=text.to_dict(), vision_config=vision.to_dict(),
        vision_feature_layer=-1))


def _gpt_oss_causal():
    from transformers import GptOssForCausalLM
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    return GptOssForCausalLM(GptOssConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        num_local_experts=2, num_experts_per_tok=1, max_position_embeddings=32,
        tie_word_embeddings=False))


def _gemma3_conditional():
    from transformers import Gemma3ForConditionalGeneration
    from transformers.models.gemma3.configuration_gemma3 import (
        Gemma3Config, Gemma3TextConfig,
    )
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
    text = Gemma3TextConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=32, sliding_window=16)
    vision = SiglipVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, image_size=16, patch_size=4)
    return Gemma3ForConditionalGeneration(Gemma3Config(
        text_config=text.to_dict(), vision_config=vision.to_dict()))


def _qwen3vl_configs():
    # NOTE: sub-configs MUST be passed to the top config as DICTs -- the VL
    # config __init__ only materializes text/vision configs for dict/None.
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig,
    )
    text = Qwen3VLTextConfig(
        vocab_size=64, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        max_position_embeddings=32, tie_word_embeddings=True,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
    vision = Qwen3VLVisionConfig(
        depth=1, hidden_size=8, intermediate_size=16, num_heads=2, in_channels=3,
        patch_size=4, spatial_merge_size=1, temporal_patch_size=2,
        out_hidden_size=8, num_position_embeddings=16, deepstack_visual_indexes=[0])
    return text, vision, Qwen3VLConfig


def _qwen3vl_conditional():
    from transformers import Qwen3VLForConditionalGeneration
    text, vision, cfg_cls = _qwen3vl_configs()
    return Qwen3VLForConditionalGeneration(cfg_cls(
        text_config=text.to_dict(), vision_config=vision.to_dict(),
        tie_word_embeddings=True))


def _qwen3vl_model():
    from transformers import Qwen3VLModel
    text, vision, cfg_cls = _qwen3vl_configs()
    return Qwen3VLModel(cfg_cls(
        text_config=text.to_dict(), vision_config=vision.to_dict(),
        tie_word_embeddings=True))


def _qwen3vl_text():
    from transformers import Qwen3VLTextModel
    text, _, _ = _qwen3vl_configs()
    return Qwen3VLTextModel(text)


def _qwen25vl_configs():
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLConfig, Qwen2_5_VLTextConfig, Qwen2_5_VLVisionConfig,
    )
    text = Qwen2_5_VLTextConfig(
        vocab_size=64, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
        tie_word_embeddings=False,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]})
    vision = Qwen2_5_VLVisionConfig(
        depth=1, hidden_size=8, intermediate_size=16, num_heads=2, in_channels=3,
        patch_size=4, spatial_merge_size=1, temporal_patch_size=2, out_hidden_size=8)
    return text, vision, Qwen2_5_VLConfig


def _qwen25vl_conditional():
    from transformers import Qwen2_5_VLForConditionalGeneration
    text, vision, cfg_cls = _qwen25vl_configs()
    return Qwen2_5_VLForConditionalGeneration(cfg_cls(
        text_config=text.to_dict(), vision_config=vision.to_dict(),
        tie_word_embeddings=False))


def _qwen25vl_text():
    from transformers import Qwen2_5_VLTextModel
    text, _, _ = _qwen25vl_configs()
    return Qwen2_5_VLTextModel(text)


def _t5gemma_encoder():
    # The real prx ``text_encoder/config.json`` is a FLAT ``t5_gemma_module``
    # config (T5GemmaModuleConfig, which HAS ``hidden_size`` etc. at top level).
    # ``T5GemmaEncoder.from_pretrained`` reloads it via its own config_class
    # ``T5GemmaConfig`` (retaining the flat fields as kwargs) with a benign
    # model-type warning. Build/save from the module config to mirror that.
    from transformers.models.t5gemma.configuration_t5gemma import T5GemmaModuleConfig
    from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder
    mod = T5GemmaModuleConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=32)
    # These live on the wrapping T5GemmaConfig; the flat checkpoint carries them.
    mod.dropout_rate = 0.0
    mod.attention_dropout = 0.0
    return T5GemmaEncoder(mod)


_BUILDERS = {
    "CLIPTextModel": _clip_text,
    "CLIPTextModelWithProjection": _clip_text_projection,
    "T5EncoderModel": _t5_encoder,
    "UMT5EncoderModel": _umt5_encoder,
    "Qwen3Model": _qwen3_model,
    "Qwen3ForCausalLM": _qwen3_causal,
    "Mistral3Model": _mistral3_model,
    "GptOssForCausalLM": _gpt_oss_causal,
    "Gemma3ForConditionalGeneration": _gemma3_conditional,
    "Qwen3VLForConditionalGeneration": _qwen3vl_conditional,
    "Qwen3VLModel": _qwen3vl_model,
    "Qwen3VLTextModel": _qwen3vl_text,
    "Qwen2_5_VLForConditionalGeneration": _qwen25vl_conditional,
    "Qwen2_5_VLTextModel": _qwen25vl_text,
    "T5GemmaEncoder": _t5gemma_encoder,
}


# ---------------------------------------------------------------------------
# Contract cases
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Case:
    family: str
    loader_module: str
    loader_class: str
    component_key: str
    expected_declared: str          # dotted class path the loader must declare
    true_arch: str                  # key into _BUILDERS
    in_manifest: bool = True        # False => TE is hand-loaded in load(), not the manifest
    ctor_kwargs: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.family}:{self.component_key}"


def _m(fam: str) -> str:
    return f"app.engine.models.families.{fam}.loader"


CASES: list[_Case] = [
    _Case("sdxl", _m("sdxl"), "SDXLLoader", "text_encoder_1",
          "transformers.CLIPTextModel", "CLIPTextModel"),
    _Case("sdxl", _m("sdxl"), "SDXLLoader", "text_encoder_2",
          "transformers.CLIPTextModelWithProjection", "CLIPTextModelWithProjection"),
    _Case("flux1", _m("flux1"), "Flux1Loader", "text_encoder",
          "transformers.CLIPTextModel", "CLIPTextModel"),
    _Case("flux1", _m("flux1"), "Flux1Loader", "text_encoder_2",
          "transformers.T5EncoderModel", "T5EncoderModel"),
    _Case("wan21", _m("wan21"), "Wan21Loader", "text_encoder",
          "transformers.UMT5EncoderModel", "UMT5EncoderModel"),
    _Case("wan22", _m("wan22"), "Wan22Loader", "text_encoder",
          "transformers.UMT5EncoderModel", "UMT5EncoderModel"),
    _Case("ovis_image", _m("ovis_image"), "OvisImageLoader", "text_encoder",
          "transformers.Qwen3Model", "Qwen3Model"),
    _Case("flux2", _m("flux2"), "Flux2Loader", "text_encoder",
          "transformers.Qwen3ForCausalLM", "Qwen3ForCausalLM"),
    _Case("zimage", _m("zimage"), "ZImageLoader", "text_encoder",
          "transformers.Qwen3ForCausalLM", "Qwen3ForCausalLM"),
    _Case("ernie_image", _m("ernie_image"), "ErnieImageLoader", "text_encoder",
          "transformers.Mistral3Model", "Mistral3Model"),
    _Case("microsoft_lens", _m("microsoft_lens"), "MicrosoftLensLoader", "text_encoder",
          "transformers.GptOssForCausalLM", "GptOssForCausalLM"),
    _Case("ltx2", _m("ltx2"), "Ltx2Loader", "text_encoder",
          "transformers.Gemma3ForConditionalGeneration", "Gemma3ForConditionalGeneration"),
    _Case("dreamlite", _m("dreamlite"), "DreamLiteLoader", "text_encoder",
          "transformers.Qwen3VLForConditionalGeneration", "Qwen3VLForConditionalGeneration"),
    _Case("ideogram4", _m("ideogram4"), "IdeogramV4Loader", "text_encoder",
          "transformers.Qwen3VLForConditionalGeneration", "Qwen3VLForConditionalGeneration"),
    _Case("boogu_image", _m("boogu_image"), "BooguImageLoader", "text_encoder",
          "transformers.Qwen3VLForConditionalGeneration", "Qwen3VLForConditionalGeneration"),
    _Case("prx_pixel", _m("prx_pixel"), "PRXPixelLoader", "text_encoder",
          "transformers.Qwen3VLTextModel", "Qwen3VLTextModel"),
    _Case("qwen_image", _m("qwen_image"), "QwenImageLoader", "text_encoder",
          "transformers.Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration"),
    _Case("longcat_image", _m("longcat_image"), "LongCatImageLoader", "text_encoder",
          "transformers.Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration"),
    _Case("kandinsky5", _m("kandinsky5"), "Kandinsky5Loader", "text_encoder",
          "transformers.Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration"),
    _Case("kandinsky5", _m("kandinsky5"), "Kandinsky5Loader", "text_encoder_2",
          "transformers.CLIPTextModel", "CLIPTextModel"),
    _Case("hunyuan_video15", _m("hunyuan_video15"), "Hv15Loader", "text_encoder",
          "transformers.Qwen2_5_VLTextModel", "Qwen2_5_VLTextModel"),
    _Case("hunyuan_video15", _m("hunyuan_video15"), "Hv15Loader", "text_encoder_2",
          "transformers.T5EncoderModel", "T5EncoderModel"),
    _Case("prx", _m("prx"), "PRXLoader", "text_encoder",
          "transformers.models.t5gemma.modeling_t5gemma.T5GemmaEncoder", "T5GemmaEncoder"),
    # krea2's TE is NOT in the manifest -- it is hand-loaded in load() as
    # ``Qwen3VLModel`` with a transformers-5.x rope-config translation. Covered
    # by ``test_krea2_hand_loaded_te_contract`` (round-trip) + a source guard.
    _Case("krea2", _m("krea2"), "Krea2Loader", "text_encoder",
          "transformers.Qwen3VLModel", "Qwen3VLModel", in_manifest=False),
]

# Families that have a loader.py but no standalone-TE loading contract.
_EXEMPT_FAMILIES = {
    "hidream_o1": (
        "Unified vendored Qwen3VLForConditionalGeneration DiT (extra heads); no "
        "standalone TE, loaded strict=False by design -- no byte-clean round-trip."
    ),
}


def _stub_definition() -> SimpleNamespace:
    """Minimal stand-in accepted by every family's get_component_manifest()."""
    return SimpleNamespace(architecture_params={}, components={}, family="__test__")


def _resolve_declared_from_manifest(case: _Case) -> str:
    mod = importlib.import_module(case.loader_module)
    loader_cls = getattr(mod, case.loader_class)
    loader = loader_cls(torch.device("cpu"), **case.ctor_kwargs)
    manifest = loader.get_component_manifest(_stub_definition())
    for spec in manifest:
        if spec.key == case.component_key:
            return spec.hf_class
    raise AssertionError(
        f"{case.family}: component {case.component_key!r} not found in manifest "
        f"(keys: {[s.key for s in manifest]})",
    )


def _roundtrip_missing_unexpected(true_model, declared_cls):
    with tempfile.TemporaryDirectory() as td:
        true_model.save_pretrained(td)
        _loaded, info = declared_cls.from_pretrained(td, output_loading_info=True)
    return info["missing_keys"], info["unexpected_keys"]


# ---------------------------------------------------------------------------
# The contract test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_te_loading_contract(case: _Case):
    """The declared TE class must consume 100% of the checkpoint's tensors.

    Build a tiny model of the TRUE checkpoint architecture, save it (real
    on-disk key layout), reload via the class the family DECLARES, and assert
    ``missing_keys == []`` and ``unexpected_keys == []``. A wrong/looser class
    (the B1 random-TE bug) leaves keys unmatched and fails here.
    """
    # 1. Pin the declaration (also guards manifest-based families against drift).
    if case.in_manifest:
        declared = _resolve_declared_from_manifest(case)
        assert declared == case.expected_declared, (
            f"{case.family}: manifest declares text-encoder class {declared!r}, "
            f"expected {case.expected_declared!r} -- a loader edit changed the "
            f"declared class; verify it still matches the checkpoint's "
            f"architectures before updating this pin."
        )
    else:
        declared = case.expected_declared

    declared_cls = GenericComponentLoader._import_class(declared)

    # 2. Build the TRUE-architecture tiny checkpoint and reload via declared.
    true_model = _BUILDERS[case.true_arch]()
    missing, unexpected = _roundtrip_missing_unexpected(true_model, declared_cls)

    assert missing == [], (
        f"{case.family}/{case.component_key}: declared class {declared!r} failed "
        f"to consume checkpoint tensors (prefix/base-vs-head mismatch -> silent "
        f"random TE): missing={missing[:8]}"
    )
    assert unexpected == [], (
        f"{case.family}/{case.component_key}: declared class {declared!r} left "
        f"checkpoint tensors unmatched: unexpected={unexpected[:8]}"
    )


def test_krea2_hand_loaded_te_contract():
    """krea2's hand-loaded ``Qwen3VLModel`` TE must consume its checkpoint.

    krea2 does NOT list the TE in its manifest: ``Krea2Loader.load`` builds it
    directly via ``Qwen3VLModel.from_pretrained`` (with a transformers-5.x ->
    4.57 rope-config translation). This path previously had ZERO fidelity tests
    -- the exact B1 surface. Guard the class the loader actually uses against the
    checkpoint's real ``Qwen3VLModel`` key layout, and assert the source still
    references that class so a drift is caught.
    """
    from app.engine.models.families.krea2 import loader as krea2_loader

    src = inspect.getsource(krea2_loader.Krea2Loader.load)
    assert "Qwen3VLModel.from_pretrained" in src, (
        "krea2 load() no longer loads the TE via Qwen3VLModel.from_pretrained -- "
        "re-verify the class matches the Krea-2-Raw text_encoder checkpoint "
        "(architectures=['Qwen3VLModel']) and update this guard."
    )

    from transformers import Qwen3VLModel
    true_model = _BUILDERS["Qwen3VLModel"]()
    missing, unexpected = _roundtrip_missing_unexpected(true_model, Qwen3VLModel)
    assert missing == [] and unexpected == [], (
        f"krea2 Qwen3VLModel TE contract broken: missing={missing[:8]} "
        f"unexpected={unexpected[:8]}"
    )


def test_all_families_covered():
    """Every family with a loader.py is either contract-tested or exempted.

    Guards the W2-B rollout: a newly added family that ships a TE must appear in
    ``CASES`` (or be justified in ``_EXEMPT_FAMILIES``) rather than silently
    skipping the loading-contract check.
    """
    from pathlib import Path

    families_dir = Path(
        importlib.import_module("app.engine.models.families").__file__,
    ).parent
    with_loader = {
        p.parent.name
        for p in families_dir.glob("*/loader.py")
    }
    covered = {c.family for c in CASES}
    accounted = covered | set(_EXEMPT_FAMILIES)
    missing = with_loader - accounted
    assert not missing, (
        f"families with a loader.py but no TE loading-contract case or exemption: "
        f"{sorted(missing)} -- add a _Case (build a tiny TRUE-arch model) or an "
        f"_EXEMPT_FAMILIES entry with justification."
    )
