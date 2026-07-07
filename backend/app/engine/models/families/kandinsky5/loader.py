"""Kandinsky 5.0 loader — manifest-driven via GenericComponentLoader.

Components (diffusers-format repo, verified against both checkpoints'
``model_index.json``):

- ``tokenizer``      : **``Qwen2VLProcessor``** — the repo's "tokenizer"
  component is actually a PROCESSOR (the pipeline type-hints it as such);
  loading it as a plain ``AutoTokenizer`` would drop the processor wrapping
  the encode path calls with ``images=None, videos=None``.
- ``text_encoder``   : ``Qwen2_5_VLForConditionalGeneration`` (7B, bf16)
- ``tokenizer_2``    : ``CLIPTokenizer``
- ``text_encoder_2`` : ``CLIPTextModel`` (ViT-L pooled)
- ``vae``            : ``AutoencoderKLHunyuanVideo`` — kept fp32 (temporal
  VAE precision, same policy as WAN)
- ``unet``           : ``Kandinsky5Transformer3DModel``

T2V and I2V share the exact same manifest — I2V conditions through the latent
path (no CLIP-vision image encoder exists for this family).
"""

from __future__ import annotations

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader


class Kandinsky5Loader(GenericComponentLoader):
    """Load Kandinsky 5.0 components from a diffusers-format repository."""

    def get_component_manifest(
        self, definition: ModelDefinition
    ) -> list[ComponentSpec]:
        return [
            # -- Qwen2.5-VL processor (the checkpoint's "tokenizer" component) --
            ComponentSpec(
                key="tokenizer",
                hf_class="transformers.Qwen2VLProcessor",
                subfolder="tokenizer",
                candidates=["tokenizer"],
                is_torch_model=False,
                fallback_to_root=True,
            ),
            # -- Text Encoder 1 (Qwen2.5-VL-7B) --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.Qwen2_5_VLForConditionalGeneration",
                subfolder="text_encoder",
                candidates=["text_encoder"],
                fallback_to_root=True,
            ),
            # -- CLIP tokenizer --
            ComponentSpec(
                key="tokenizer_2",
                hf_class="transformers.CLIPTokenizer",
                subfolder="tokenizer_2",
                candidates=["tokenizer_2"],
                is_torch_model=False,
                fallback_to_root=True,
            ),
            # -- Text Encoder 2 (CLIP ViT-L, pooled projections) --
            ComponentSpec(
                key="text_encoder_2",
                hf_class="transformers.CLIPTextModel",
                subfolder="text_encoder_2",
                candidates=["text_encoder_2"],
                fallback_to_root=True,
            ),
            # -- VAE (HunyuanVideo) — fp32 for temporal-decode precision --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKLHunyuanVideo",
                subfolder="vae",
                candidates=["vae"],
                dtype_override=torch.float32,
                fallback_to_root=True,
            ),
            # -- Transformer → mapped to "unet" --
            ComponentSpec(
                key="unet",
                hf_class="diffusers.Kandinsky5Transformer3DModel",
                subfolder="transformer",
                candidates=["transformer"],
                fallback_to_root=True,
            ),
        ]
