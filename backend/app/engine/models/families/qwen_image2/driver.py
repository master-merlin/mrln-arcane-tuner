"""Qwen-Image 2 model driver -- family-specific training behaviour.

Implements ``IModelDriver`` for Qwen-Image 2.1: a single Qwen3-VL text encoder
used text-only (``text_encoding.py``), the vendored single-stream
``QwenImage21Transformer2DModel`` fed unpatched latents (``utils.py``), and
flow matching with the pipeline's default noise/target hooks.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver
from app.engine.core.text_encoding import TextEncoderOutput

from . import text_encoding

logger = structlog.get_logger(__name__)

#: Fallback when a definition declares no ``te.max_length`` (the YAML does).
DEFAULT_TE_MAX_LENGTH = 512


class QwenImage2Driver(IModelDriver):
    """Qwen-Image 2.x family driver."""

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None  # the Qwen3VLProcessor (loader key "tokenizer")
        self._components: dict[str, Any] = {}

        arch = getattr(definition, "architecture_params", None) or {}
        self.max_length: int = int(arch.get("te.max_length") or DEFAULT_TE_MAX_LENGTH)

    # --- Loading & component access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        self._components = components
        self.transformer = components.get("unet")
        self.vae = components.get("vae")
        self.text_encoder = components.get("text_encoder", self.text_encoder)
        self.tokenizer = components.get("tokenizer", self.tokenizer)

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.transformer

    def get_text_encoders(self) -> dict[str, nn.Module]:
        return {"text_encoder": self.text_encoder} if self.text_encoder is not None else {}

    def release_text_encoders(self) -> None:
        self.text_encoder = None

    def get_lora_targets(self) -> list[str]:
        """The definition's list -- exactly the Linears of one vendored block,
        pinned by ``tests/test_qwen_image2_definition.py``."""
        targets = list(getattr(self.definition, "lora_targetable_modules", None) or [])
        if not targets:
            raise ValueError(
                f"{self.definition.id}: qwen_image2 needs lora_targetable_modules in its YAML",
            )
        return targets

    def init_scheduler(self) -> Any:
        """Flow matching -- no external training scheduler."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        return []

    def get_layer_manifest(self) -> Any:
        from app.engine.core.layer_manifest import BlockInfo, ModelLayerManifest

        blocks: list[BlockInfo] = []
        model = self.get_primary_model()
        for i, block in enumerate(getattr(model, "transformer_blocks", None) or []):
            blocks.append(BlockInfo(
                name=f"transformer_blocks.{i}",
                block_type="single",
                param_count=sum(p.numel() for p in block.parameters()),
                depth_index=i,
            ))
        return ModelLayerManifest(
            transformer_blocks=blocks,
            lora_targets=self.get_lora_targets(),
            te_lora_targets=self.get_te_lora_targets(),
        )

    # --- Text encoding ---

    def encode_text(self, captions: list[str], dtype: torch.dtype) -> TextEncoderOutput:
        """Upstream's text-to-image ``encode_prompt`` (see ``text_encoding.py``)."""
        if self.text_encoder is None or self.tokenizer is None:
            raise RuntimeError("qwen_image2: encode_text called without a text encoder and processor")
        embeds, mask = text_encoding.encode_prompts(
            self.text_encoder,
            self.tokenizer,
            list(captions),
            max_length=self.max_length,
            device=self.device,
            dtype=dtype,
        )
        return TextEncoderOutput(embeddings=embeds, attention_mask=mask)

    def te_cache_identity(self) -> str:
        """The TE disk-cache key prefix (encoder identity + template + max length)."""
        return text_encoding.te_cache_identity(self.text_encoder, self.max_length)

    # --- Training loop ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """``QwenImage21Transformer2DModel`` forward on a ``[B, C, H, W]`` latent.

        Evidence: upstream ``QwenImage21Pipeline.__call__`` (``6256aa7666``,
        lines 753-783). The single-stream model returns the WHOLE joint
        sequence, text first; the prediction is its LAST ``H*W`` positions.
        ``img_mask`` marks one vision-language slot per 2x2 group of target
        latent tokens after the text (no condition images in text-to-image).
        ``timesteps`` arrive in ``[0, 1000]`` and are divided by 1000 once: the
        model's own embedder multiplies back (LESSONS: flow-match timestep scale).
        """
        from .utils import pack_latents, unpack_latents

        if isinstance(text_embeddings, tuple):
            enc_hs, enc_mask = text_embeddings
        else:
            enc_hs, enc_mask = text_embeddings, None

        model = self.get_primary_model()
        b, _, h, w = noisy_input.shape
        tokens = h * w
        if tokens % 4:
            raise ValueError(f"qwen_image2: latent grid {h}x{w} is not a whole number of 2x2 slots")
        hidden = pack_latents(noisy_input)
        img_mask = torch.cat(
            [
                torch.zeros(b, enc_hs.shape[1], dtype=torch.bool, device=hidden.device),
                torch.ones(b, tokens // 4, dtype=torch.bool, device=hidden.device),
            ],
            dim=1,
        )
        out = model(
            hidden_states=hidden,
            timestep=timesteps / 1000.0,
            encoder_hidden_states=enc_hs,
            encoder_hidden_states_mask=enc_mask,
            img_shapes=[[(1, h, w)]] * b,
            img_mask=img_mask,
            return_dict=False,
        )[0]
        return unpack_latents(out[:, -tokens:], h, w)

    def get_saver(self):
        from .saver import QwenImage2Saver

        return QwenImage2Saver()

    def get_block_topology(self) -> list[dict[str, Any]]:
        model = self.get_primary_model()
        blocks = getattr(model, "transformer_blocks", None)
        if blocks is None:
            return []
        return [{
            "name": "transformer_blocks",
            "attr_path": "transformer_blocks",
            "count": len(blocks),
            "approx_vram_mb": 400,
        }]
