"""HunyuanVideo 1.5 loader — manifest-driven via GenericComponentLoader.

Components (diffusers-format repos
``hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_{t2v,i2v}``):

- ``tokenizer``       : ``AutoTokenizer``   (Qwen2TokenizerFast, chat template)
- ``text_encoder``    : ``Qwen2_5_VLTextModel`` (7B text tower)
- ``tokenizer_2``     : ``AutoTokenizer``   (ByT5Tokenizer)
- ``text_encoder_2``  : ``T5EncoderModel``  (ByT5 glyph channel, d_model 1472)
- ``vae``             : ``AutoencoderKLHunyuanVideo15`` (1.26B; loaded bf16 —
  the LTX-2 precedent for a >1B video VAE; the scalar ``scaling_factor``
  1.03682 lives in its config)
- ``unet``            : ``HunyuanVideo15Transformer3DModel`` (8.3B MMDiT)

I2V definitions (``mode: i2v``) additionally load:

- ``image_encoder``     : ``SiglipVisionModel``
- ``feature_extractor`` : ``SiglipImageProcessor``

The repo's ``guider`` (``ClassifierFreeGuidance``) is deliberately EXCLUDED
from the manifest: our sampler implements the equivalent classic dual-forward
CFG itself (``pred = uncond + gs * (cond - uncond)``, the guider's
``use_original_formulation=False`` math), so loading the guider module would
be dead weight.
"""

from __future__ import annotations

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader


class Hv15Loader(GenericComponentLoader):
    """Load HunyuanVideo 1.5 components from a diffusers-format repository."""

    def get_component_manifest(
        self, definition: ModelDefinition
    ) -> list[ComponentSpec]:
        arch = getattr(definition, "architecture_params", {}) or {}
        mode = str(arch.get("mode", "t2v")).lower()
        is_i2v = mode == "i2v"

        manifest: list[ComponentSpec] = [
            # -- Tokenizer (Qwen2.5-VL chat template) --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoTokenizer",
                subfolder="tokenizer",
                candidates=["tokenizer"],
                is_torch_model=False,
                fallback_to_root=True,
            ),
            # -- Text encoder 1 (Qwen2.5-VL 7B text tower) --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Qwen2_5_VLTextModel",
                subfolder="text_encoder",
                candidates=["text_encoder"],
                fallback_to_root=True,
            ),
            # -- Tokenizer 2 (ByT5) --
            ComponentSpec(
                key="tokenizer_2",
                hf_class="transformers.AutoTokenizer",
                subfolder="tokenizer_2",
                candidates=["tokenizer_2"],
                is_torch_model=False,
                fallback_to_root=True,
            ),
            # -- Text encoder 2 (ByT5 glyph channel) --
            ComponentSpec(
                key="text_encoder_2",
                hf_class="transformers.T5EncoderModel",
                subfolder="text_encoder_2",
                candidates=["text_encoder_2"],
                fallback_to_root=True,
            ),
            # -- VAE (HunyuanVideo-1.5, 16x spatial / 4x temporal) --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKLHunyuanVideo15",
                subfolder="vae",
                candidates=["vae"],
                fallback_to_root=True,
            ),
            # -- Transformer → mapped to "unet" --
            ComponentSpec(
                key="unet",
                hf_class="diffusers.HunyuanVideo15Transformer3DModel",
                subfolder="transformer",
                candidates=["transformer"],
                fallback_to_root=True,
            ),
        ]

        if is_i2v:
            manifest += [
                # -- Siglip vision encoder (I2V image_embeds) --
                ComponentSpec(
                    key="image_encoder",
                    hf_class="transformers.SiglipVisionModel",
                    subfolder="image_encoder",
                    candidates=["image_encoder"],
                    fallback_to_root=True,
                ),
                ComponentSpec(
                    key="feature_extractor",
                    hf_class="transformers.SiglipImageProcessor",
                    subfolder="feature_extractor",
                    candidates=["feature_extractor"],
                    is_torch_model=False,
                    fallback_to_root=True,
                ),
            ]

        return manifest
