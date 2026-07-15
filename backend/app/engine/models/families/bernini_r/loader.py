"""Bernini-R loader — component-wise off the repo root.

The HF repos (``ByteDance/Bernini-R-1.3B-Diffusers`` and ``…-Diffusers``) have
**no** ``model_index.json`` — the root ``config.json`` is a transformers-style
``bernini_renderer`` config. So this is NOT a ``DiffusionPipeline.from_pretrained``
layout; components are loaded by subfolder exactly like upstream ``GEN_Wanx22``.

Components (all stock classes present in diffusers 0.39 / transformers 4.57.x):
- ``tokenizer``    : ``AutoTokenizer`` (UMT5 ``T5Tokenizer`` / ``spiece.model``)
- ``text_encoder`` : ``UMT5EncoderModel`` — repo ships fp32; cast to bf16 at load
- ``vae``          : ``AutoencoderKLWan`` (Wan 2.1, z=16) — kept fp32
- ``unet``         : ``WanTransformer3DModel`` from ``transformer/`` — repo ships
                     fp32 shards; cast to bf16 at load

v1 scope: 1.3B single expert (``skip_transformer_2: true`` ⇒ no ``transformer_2``
subfolder, so no second expert is loaded). The 14B dual-expert / MoE boundary
switch is a later extension (mirrors the wan22 expert router).
"""

from __future__ import annotations

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.core.pipeline.loader_base import ComponentSpec, GenericComponentLoader


class BerniniRLoader(GenericComponentLoader):
    """Load Bernini-R components by subfolder off the repo root."""

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
            # -- Text Encoder (UMT5-XXL) — repo fp32; loader casts to bf16 --
            ComponentSpec(
                key="text_encoder",
                hf_class="transformers.UMT5EncoderModel",
                subfolder="text_encoder",
                candidates=["text_encoder"],
                fallback_to_root=True,
            ),
            # -- VAE (Wan 2.1, z=16) — kept fp32 for temporal-decode precision --
            ComponentSpec(
                key="vae",
                hf_class="diffusers.AutoencoderKLWan",
                subfolder="vae",
                candidates=["vae"],
                dtype_override=torch.float32,
                fallback_to_root=True,
            ),
            # -- Transformer → "unet" (fp32 shards; loader casts to bf16) --
            ComponentSpec(
                key="unet",
                hf_class="diffusers.WanTransformer3DModel",
                subfolder="transformer",
                candidates=["transformer"],
                fallback_to_root=True,
            ),
        ]
