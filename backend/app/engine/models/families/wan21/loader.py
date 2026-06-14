"""WAN 2.1 loader — manifest-driven via GenericComponentLoader.

Components (diffusers-format repo):
- ``tokenizer``     : ``AutoTokenizer`` (UMT5)
- ``text_encoder``  : ``UMT5EncoderModel``
- ``vae``           : ``AutoencoderKLWan`` — kept fp32 (temporal VAE precision)
- ``unet``          : ``WanTransformer3DModel`` (the diffusion transformer)

I2V definitions (``mode: i2v``) additionally load:
- ``image_encoder`` : ``CLIPVisionModel``
- ``image_processor``: ``CLIPImageProcessor``

The image encoder/processor are only added when the definition is I2V, so T2V
loads stay lean and never download CLIP weights.
"""

from __future__ import annotations

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader

import torch


class Wan21Loader(GenericComponentLoader):
    """Load WAN 2.1 components from a diffusers-format repository."""

    def get_component_manifest(
        self, definition: ModelDefinition
    ) -> list[ComponentSpec]:
        arch = getattr(definition, "architecture_params", {}) or {}
        mode = str(arch.get("mode", "t2v")).lower()
        is_i2v = mode == "i2v"

        manifest: list[ComponentSpec] = [
            # -- Tokenizer (UMT5) --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.AutoTokenizer",
                subfolder="tokenizer",
                candidates=["tokenizer"],
                is_torch_model=False,
                fallback_to_root=True,
            ),
            # -- Text Encoder (UMT5-XXL) --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.UMT5EncoderModel",
                subfolder="text_encoder",
                candidates=["text_encoder"],
                fallback_to_root=True,
            ),
            # -- VAE (Wan-VAE) — kept fp32 for temporal-decode precision --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKLWan",
                subfolder="vae",
                candidates=["vae"],
                dtype_override=torch.float32,
                fallback_to_root=True,
            ),
            # -- Transformer → mapped to "unet" --
            ComponentSpec(
                key="unet",
                hf_class="diffusers.WanTransformer3DModel",
                subfolder="transformer",
                candidates=["transformer"],
                fallback_to_root=True,
            ),
        ]

        if is_i2v:
            manifest += [
                # -- CLIP vision image encoder (I2V conditioning) --
                ComponentSpec(
                    key="image_encoder",
                    hf_class="transformers.CLIPVisionModel",
                    subfolder="image_encoder",
                    candidates=["image_encoder"],
                    fallback_to_root=True,
                ),
                ComponentSpec(
                    key="image_processor",
                    hf_class="transformers.CLIPImageProcessor",
                    subfolder="image_processor",
                    candidates=["image_processor"],
                    is_torch_model=False,
                    fallback_to_root=True,
                ),
            ]

        return manifest
