"""WAN 2.1 loader — manifest-driven via GenericComponentLoader.

Components (diffusers-format repo), same for both T2V and I2V definitions:
- ``tokenizer``     : ``AutoTokenizer`` (UMT5)
- ``text_encoder``  : ``UMT5EncoderModel``
- ``vae``           : ``AutoencoderKLWan`` — kept fp32 (temporal VAE precision)
- ``unet``          : ``WanTransformer3DModel`` (the diffusion transformer)

W3.T9: I2V definitions used to additionally load a ``CLIPVisionModel``
``image_encoder`` + ``CLIPImageProcessor`` ``image_processor``, but NOTHING
ever populated ``WanDriverBase.BATCH_IMAGE_EMBED`` from them (only
``wan_shared/driver_base.py``'s ``forward_pass`` READ it, always as ``None`` —
CLIP conditioning was a documented, never-implemented follow-up). Every
wan21-i2v run was paying a multi-GB CLIP download + host-RAM residency + load
time for a component whose output never reached the transformer. Removed.
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

        return manifest
