"""Ideogram 4 model driver -- family-specific training behavior.

Implements :class:`IModelDriver` for the Ideogram 4 diffusion family
(vendored DiT + Qwen3-VL text encoder + FLUX-style VAE).

Key family characteristics:

*  **Multi-layer Qwen3-VL conditioning**: the stock Qwen3-VL language model is
   run with ``output_hidden_states=True`` and the hidden states at
   :data:`~app.engine.models.families.ideogram4.utils.QWEN3VL_SELECTED_LAYERS`
   (13 multi-scale slices: layers ``(0,3,6,9,12,15,18,21,24,27,30,33,35)`` --
   the real upstream ``QWEN3_VL_ACTIVATION_LAYERS``) are concatenated on the
   feature dimension.  The result is ``(B, L, 4096 * 13)`` raw hidden states;
   the DiT projects them internally, so the driver supplies them unprojected.

*  **Fused-QKV / SwiGLU LoRA targets** ``["qkv", "o", "w1", "w2", "w3"]``
   (confirmed against the vendored model in Task 1).

*  Sequence packing (text+image, position_ids/segment_ids/indicator) and the
   denoising ``forward_pass`` land in Task 5; they are stubbed here so the
   driver can instantiate.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver
from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.models.families.ideogram4 import utils

logger = structlog.get_logger(__name__)

# Driver-level cap on tokenised sequence length; governs the driver's padding
# budget, not a shared utils constant.
DEFAULT_TE_MAX_LENGTH = 2048
DEFAULT_SELECTED_LAYERS = utils.QWEN3VL_SELECTED_LAYERS


class IdeogramV4Driver(IModelDriver):
    """Ideogram 4 family driver (multi-layer Qwen3-VL text encoder)."""

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None
        self._components: dict[str, Any] = {}

        arch = getattr(definition, "architecture_params", {}) or {}
        self.te_max_length = int(arch.get("te.max_length", DEFAULT_TE_MAX_LENGTH))
        sel = arch.get("te.selected_layers", DEFAULT_SELECTED_LAYERS)
        self.selected_layers = tuple(int(i) for i in sel)

    # --- Phase 1: component access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        self._components = components
        self.transformer = components["unet"]
        self.vae = components["vae"]
        self.text_encoder = components.get("text_encoder")
        self.tokenizer = components.get("tokenizer")
        self.logger.info(
            "ideogram4_config",
            te_max_length=self.te_max_length,
            selected_layers=self.selected_layers,
        )

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.transformer

    def get_text_encoders(self) -> dict[str, nn.Module]:
        return {"text_encoder": self.text_encoder} if self.text_encoder else {}

    def get_lora_targets(self) -> list[str]:
        defn_targets = getattr(self.definition, "lora_targetable_modules", None)
        if defn_targets:
            return defn_targets
        # Confirmed against the vendored DiT in Task 1: fused qkv + attn out
        # projection + SwiGLU feed-forward (w1/w2/w3).
        return ["qkv", "o", "w1", "w2", "w3"]

    def get_te_lora_targets(self) -> list[str]:
        return []  # Qwen3-VL text encoder is frozen

    def init_scheduler(self) -> Any:
        return None  # flow-matching; no train-time scheduler

    def resolve_loading_dtype(self) -> torch.dtype:
        return torch.bfloat16

    # --- Phase 2: text encoding ---

    @torch.no_grad()
    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Chat-template -> Qwen3-VL -> concat 13 selected hidden states.

        Mirrors upstream ``Ideogram4Pipeline._encode_text``: the selected
        hidden states are stacked and reshaped so each text token's feature
        vector is the per-hidden-unit interleaving of all selected layers
        (``stack -> permute -> reshape``), yielding a final feature width of
        ``4096 * len(QWEN3VL_SELECTED_LAYERS)``.  The DiT projects this raw
        concatenation internally, so no projection/norm is applied here.

        Returns ``TextEncoderOutput`` with ``embeddings`` ``(B, L, H*n_layers)``
        and a bool ``attention_mask`` ``(B, L)``.
        """
        rendered = [utils.render_chat_prompt(c, self.tokenizer) for c in captions]
        encoded = self.tokenizer(
            rendered, padding=True, truncation=True,
            max_length=self.te_max_length, return_tensors="pt",
            add_special_tokens=True,
        )
        # Device-safety: follow the text encoder's parameters, not self.device.
        # Falls back to self.device for parameter-less encoders.
        te_device = self.device
        try:
            te_device = next(self.text_encoder.parameters()).device
        except StopIteration:
            pass
        input_ids = encoded["input_ids"].to(te_device)
        attn = encoded["attention_mask"].to(te_device)

        out = self.text_encoder(
            input_ids=input_ids, attention_mask=attn,
            output_hidden_states=True, use_cache=False,
        )
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "text_encoder returned no hidden_states; ensure the model "
                "supports output_hidden_states=True"
            )

        # Select the activation layers and concat on the feature dim, matching
        # upstream's (num_taps, B, L, H) -> (B, L, H, num_taps) -> (B, L, H*n)
        # interleaved layout.
        layers = [hidden_states[i].to(dtype=dtype) for i in self.selected_layers]
        stacked = torch.stack(layers, dim=0)        # (n, B, L, H)
        stacked = stacked.permute(1, 2, 3, 0)       # (B, L, H, n)
        b, length = input_ids.shape
        embeddings = stacked.reshape(b, length, -1)  # (B, L, H*n)

        return TextEncoderOutput(embeddings=embeddings, attention_mask=attn.bool())

    # --- Phase 5: forward pass + saver (Task 5) ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Ideogram 4 sequence packing + forward_pass land in Task 5."
        )

    def get_saver(self) -> Any:
        raise NotImplementedError(
            "Ideogram 4 saver lands in a later task."
        )
