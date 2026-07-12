"""WAN 2.2 TI2V-5B loader — dense single-transformer, manifest-driven.

Components (diffusers-format repo, ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``):

- ``tokenizer``    : ``AutoTokenizer`` (UMT5)
- ``text_encoder`` : ``UMT5EncoderModel``
- ``vae``          : ``AutoencoderKLWan`` (the NEW higher-compression 5B VAE:
                     ``z_dim=48``, ``scale_factor_spatial=16`` — kept fp32)
- ``unet``         : ``WanTransformer3DModel`` (the single dense 5B transformer)

UNLIKE ``wan22`` (A14B MoE): no ``transformer_2/`` (no ``unet_low``), no
expert-mode branching, no deferred second-expert load — this is a plain
single-transformer manifest, structurally identical in shape to ``wan21``'s
loader (minus the CLIP image encoder branch: TI2V-5B has no CLIP-vision tower
at all, in EITHER mode, so the manifest never varies by ``mode``).
"""

from __future__ import annotations

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader


class Wan22Ti2v5bLoader(GenericComponentLoader):
    """Load WAN 2.2 TI2V-5B components from a diffusers-format repository."""

    def get_component_manifest(
        self, definition: ModelDefinition
    ) -> list[ComponentSpec]:
        return [
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
            # -- VAE (new high-compression Wan2.2 VAE) — kept fp32 --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKLWan",
                subfolder="vae",
                candidates=["vae"],
                dtype_override=torch.float32,
                fallback_to_root=True,
            ),
            # -- Transformer (single dense 5B, no second expert) → "unet" --
            ComponentSpec(
                key="unet",
                hf_class="diffusers.WanTransformer3DModel",
                subfolder="transformer",
                candidates=["transformer"],
                fallback_to_root=True,
            ),
        ]
