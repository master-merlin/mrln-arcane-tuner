"""WAN 2.2 loader — dual-transformer MoE, manifest-driven.

Components (diffusers-format repo, ``Wan-AI/Wan2.2-{T2V,I2V}-A14B-Diffusers``):

- ``tokenizer``     : ``AutoTokenizer`` (UMT5)
- ``text_encoder``  : ``UMT5EncoderModel``
- ``vae``           : ``AutoencoderKLWan`` — kept fp32 (temporal VAE precision)
- ``unet``          : ``WanTransformer3DModel`` from ``transformer/`` — the
                      **high-noise** expert (active for ``t >= boundary``)
- ``unet_low``      : ``WanTransformer3DModel`` from ``transformer_2/`` — the
                      **low-noise** expert (active for ``t < boundary``)

Diffusers convention: ``transformer`` = high-noise, ``transformer_2`` =
low-noise (WAN 2.2 dual transformers selected by ``boundary_ratio``).

Unlike WAN 2.1 I2V, **WAN 2.2 I2V has NO CLIP image encoder** — diffusers
asserts ``image_embeds is None`` and conditions on the first-frame latent only.
So even the I2V manifest never loads an image encoder; the 36-channel concat is
built from the first-frame latent with ``encoder_hidden_states_image=None``.
"""

from __future__ import annotations

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader


class Wan22Loader(GenericComponentLoader):
    """Load WAN 2.2 dual-transformer components from a diffusers-format repo."""

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
            # -- High-noise expert → mapped to "unet" (the active-by-default) --
            ComponentSpec(
                key="unet",
                hf_class="diffusers.WanTransformer3DModel",
                subfolder="transformer",
                candidates=["transformer"],
                fallback_to_root=True,
            ),
            # -- Low-noise expert → mapped to "unet_low" --
            ComponentSpec(
                key="unet_low",
                hf_class="diffusers.WanTransformer3DModel",
                subfolder="transformer_2",
                candidates=["transformer_2"],
                fallback_to_root=True,
            ),
        ]
        # NOTE: no image_encoder/image_processor even for I2V — WAN 2.2 I2V is
        # first-frame-latent only (no CLIP-vision conditioning).
        return manifest
