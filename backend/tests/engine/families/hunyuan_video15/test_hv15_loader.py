"""hv15 loader-manifest tests (no weights).

The manifest declares the dual TE + VAE + transformer for BOTH modes; the
Siglip image encoder + feature extractor are gated on ``mode == i2v``; and the
repo's ``guider`` component is EXCLUDED (our sampler implements the
equivalent dual-forward CFG itself).
"""

import torch

from app.engine.models.families.hunyuan_video15.loader import Hv15Loader


class _Defn:
    def __init__(self, mode: str):
        self.architecture_params = {"mode": mode}
        self.components: dict = {}


def _manifest(mode: str):
    loader = Hv15Loader(torch.device("cpu"))
    return loader.get_component_manifest(_Defn(mode))


def test_t2v_manifest_components():
    specs = {s.key: s for s in _manifest("t2v")}
    assert set(specs) == {
        "tokenizer",
        "text_encoder",
        "tokenizer_2",
        "text_encoder_2",
        "vae",
        "unet",
    }
    assert specs["text_encoder"].hf_class == "transformers.Qwen2_5_VLTextModel"
    assert specs["text_encoder_2"].hf_class == "transformers.T5EncoderModel"
    assert specs["vae"].hf_class == "diffusers.AutoencoderKLHunyuanVideo15"
    assert specs["unet"].hf_class == "diffusers.HunyuanVideo15Transformer3DModel"
    assert specs["unet"].subfolder == "transformer"
    # Tokenizers are not torch modules.
    assert specs["tokenizer"].is_torch_model is False
    assert specs["tokenizer_2"].is_torch_model is False
    # VAE loads in the global (bf16) dtype — no fp32 override (1.26B module;
    # the LTX-2 precedent for a >1B video VAE).
    assert specs["vae"].dtype_override is None


def test_guider_is_excluded_from_manifest():
    for mode in ("t2v", "i2v"):
        keys = [s.key for s in _manifest(mode)]
        classes = [s.hf_class for s in _manifest(mode)]
        assert "guider" not in keys
        assert not any("Guidance" in c for c in classes)


def test_i2v_manifest_adds_siglip():
    specs = {s.key: s for s in _manifest("i2v")}
    assert "image_encoder" in specs
    assert "feature_extractor" in specs
    assert specs["image_encoder"].hf_class == "transformers.SiglipVisionModel"
    assert specs["feature_extractor"].hf_class == "transformers.SiglipImageProcessor"
    assert specs["feature_extractor"].is_torch_model is False


def test_t2v_manifest_has_no_siglip():
    keys = [s.key for s in _manifest("t2v")]
    assert "image_encoder" not in keys
    assert "feature_extractor" not in keys
