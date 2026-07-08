"""Kandinsky 5.0 loader manifest + VRAM estimator entries.

The manifest must load the dual TE (incl. the processor-as-tokenizer quirk:
the checkpoint's ``tokenizer`` component IS a ``Qwen2VLProcessor``), keep the
HunyuanVideo VAE fp32, and be identical for T2V and I2V (no image encoder).

VRAM: the family fallback carries the meta-measured LITE sizes (2.0B); the
PRO definition ships an authoritative ``model_size_mb`` (36833 MB bf16) that
must drive its estimate instead of the fallback.
"""

from __future__ import annotations

import math

import pytest
import torch

from app.engine.models.families.kandinsky5.loader import Kandinsky5Loader
from app.engine.models.registry import ModelRegistry
from app.engine.utils.vram_estimator import (
    _FAMILY_PARAMS,
    VRAMEstimator,
    _get_primary_params,
    _get_te_params,
    _get_vae_params,
)


@pytest.fixture(scope="module")
def registry():
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False
    r = ModelRegistry()
    r.discover_families()
    r.load_definitions("app/engine/models/definitions")
    yield r
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False


def _manifest(registry, def_id: str):
    loader = Kandinsky5Loader(torch.device("cpu"))
    return loader.get_component_manifest(registry.get_definition(def_id))


def _spec(manifest, key):
    matches = [s for s in manifest if s.key == key]
    assert len(matches) == 1, f"expected exactly one {key!r} spec"
    return matches[0]


def test_manifest_component_set(registry):
    manifest = _manifest(registry, "k5-t2v-lite-sft-5s")
    keys = [s.key for s in manifest]
    assert keys == [
        "tokenizer",
        "text_encoder",
        "tokenizer_2",
        "text_encoder_2",
        "vae",
        "unet",
    ]


def test_tokenizer_is_qwen2vl_processor(registry):
    """The repo's 'tokenizer' component is a PROCESSOR (pipeline type hint)."""
    spec = _spec(_manifest(registry, "k5-t2v-lite-sft-5s"), "tokenizer")
    assert spec.hf_class == "transformers.Qwen2VLProcessor"
    assert spec.is_torch_model is False


def test_dual_text_encoder_classes(registry):
    manifest = _manifest(registry, "k5-t2v-lite-sft-5s")
    te1 = _spec(manifest, "text_encoder")
    te2 = _spec(manifest, "text_encoder_2")
    assert te1.hf_class == "transformers.Qwen2_5_VLForConditionalGeneration"
    assert te2.hf_class == "transformers.CLIPTextModel"
    tok2 = _spec(manifest, "tokenizer_2")
    assert tok2.hf_class == "transformers.CLIPTokenizer"
    assert tok2.is_torch_model is False


def test_vae_is_hunyuan_video_fp32(registry):
    spec = _spec(_manifest(registry, "k5-t2v-lite-sft-5s"), "vae")
    assert spec.hf_class == "diffusers.AutoencoderKLHunyuanVideo"
    assert spec.dtype_override is torch.float32


def test_transformer_mapped_to_unet(registry):
    spec = _spec(_manifest(registry, "k5-t2v-lite-sft-5s"), "unet")
    assert spec.hf_class == "diffusers.Kandinsky5Transformer3DModel"
    assert spec.subfolder == "transformer"


def test_i2v_manifest_identical_no_image_encoder(registry):
    t2v = _manifest(registry, "k5-t2v-lite-sft-5s")
    i2v = _manifest(registry, "k5-i2v-pro-sft-5s")
    assert [s.key for s in t2v] == [s.key for s in i2v]
    assert not any(s.key == "image_encoder" for s in i2v)


# ── VRAM estimator ─────────────────────────────────────────────────────────


def test_estimator_registers_kandinsky5():
    assert "kandinsky5" in _FAMILY_PARAMS
    # Lite fallback sizes (meta-instantiated 2.008B / 8.3B / 0.12B / 0.25B).
    assert _get_primary_params("kandinsky5", {}) == pytest.approx(2.0)
    assert _get_te_params("kandinsky5") == pytest.approx(8.42)  # Qwen + CLIP
    assert _get_vae_params("kandinsky5") == pytest.approx(0.25)


def test_lite_estimate_uses_family_fallback(registry):
    defn = registry.get_definition("k5-t2v-lite-sft-5s")
    report = VRAMEstimator.estimate(
        defn, {"quantization": "none", "num_frames": 17}
    )
    d = report.to_dict()
    # ~2B bf16 ≈ 3.8 GB of primary weights — NOT the generic 2.0B*? default
    # coincidence check: bounded well below an SDXL-sized mistake window.
    assert 3_000 < d["model_weights_mb"] < 6_000, d["model_weights_mb"]
    assert math.isfinite(d["peak_mb"])


def test_pro_estimate_prefers_definition_model_size(registry):
    defn = registry.get_definition("k5-i2v-pro-sft-5s")
    report = VRAMEstimator.estimate(
        defn, {"quantization": "none", "num_frames": 17}
    )
    d = report.to_dict()
    # 36833 MB on disk (19.3B bf16) must drive the estimate, not the 2.0B
    # family fallback (~3.8 GB).
    assert d["model_weights_mb"] > 30_000, d["model_weights_mb"]
    assert math.isfinite(d["peak_mb"])


def test_video_frames_scale_activations(registry):
    defn = registry.get_definition("k5-t2v-lite-sft-5s")
    short = VRAMEstimator.estimate(
        defn, {"quantization": "none", "num_frames": 17}
    ).to_dict()
    long = VRAMEstimator.estimate(
        defn, {"quantization": "none", "num_frames": 121}
    ).to_dict()
    # video.vae_temporal=4 → 5 vs 31 latent frames → more activation memory.
    assert long["activations_mb"] > short["activations_mb"]
