"""Microsoft Lens model driver -- family-specific training behavior.

Implements :class:`IModelDriver` for the Microsoft Lens diffusion family.

Key family characteristics:

*  **Sequence-space latents** ``[B, S, 128]`` where ``S = latent_h * latent_w``
   (spatial positions after 2x2 patchification) and ``128 = 32 VAE channels *
   2*2 spatial patch``.  The model reuses the FLUX.2 VAE; latents are
   BN-normalized with the VAE's running statistics before being fed to the DiT.

*  **Decoupled GPT-OSS text encoder**: stock ``GptOssForCausalLM`` run with
   ``output_hidden_states=True``.  Four intermediate hidden states are
   extracted at ``DEFAULT_SELECTED_LAYERS`` (Lens post-layer indices
   ``(5, 11, 17, 23)`` == HuggingFace hidden-state indices ``(6, 12, 18, 24)``
   -- i.e. post-layer N is hidden_states[N+1]).  The first ``txt_offset`` = 97
   tokens are dropped to remove the chat-template prefix; the remainder form
   the text conditioning for the DiT.

*  **Flow-matching**, velocity prediction target ``(noise - latents)``; all
   timesteps are in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver
from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.models.families.microsoft_lens import utils

logger = structlog.get_logger(__name__)

# Driver-level cap on tokenised sequence length; not a shared utils constant
# because it governs the *driver's* padding budget, not the tokeniser itself.
DEFAULT_TE_MAX_LENGTH = 512
DEFAULT_SELECTED_LAYERS = utils.DEFAULT_SELECTED_LAYERS
DEFAULT_TXT_OFFSET = utils.DEFAULT_TXT_OFFSET


class MicrosoftLensDriver(IModelDriver):
    """Microsoft Lens family driver (decoupled GPT-OSS text encoder)."""

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
        self.txt_offset = int(arch.get("te.txt_offset", DEFAULT_TXT_OFFSET))
        sel = arch.get("transformer.selected_layer_index", DEFAULT_SELECTED_LAYERS)
        self.selected_layers = tuple(int(i) for i in sel)
        self.hf_layer_indices = utils.lens_layers_to_hf_indices(self.selected_layers)

    # --- Phase 1 ---

    def assign_components(self, components: dict[str, Any]) -> None:
        self._components = components
        self.transformer = components["unet"]
        self.vae = components["vae"]
        self.text_encoder = components.get("text_encoder")
        self.tokenizer = components.get("tokenizer")
        self.logger.info(
            "microsoft_lens_config",
            te_max_length=self.te_max_length,
            txt_offset=self.txt_offset,
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
        return [
            "img_qkv", "txt_qkv", "to_out.0", "to_add_out",
            "w1", "w2", "w3",
        ]

    def get_te_lora_targets(self) -> list[str]:
        return []  # text-encoder LoRA not supported in v1

    def init_scheduler(self) -> Any:
        return None  # flow-matching; no train-time scheduler

    def resolve_loading_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def get_layer_manifest(self) -> Any:
        from app.engine.core.layer_manifest import BlockInfo, ModelLayerManifest

        blocks: list[BlockInfo] = []
        model = self.get_primary_model()
        if model is not None:
            tblocks = getattr(model, "transformer_blocks", None)
            if tblocks is not None:
                for i, block in enumerate(tblocks):
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

    def get_block_topology(self) -> list[dict[str, Any]]:
        model = self.get_primary_model()
        topo: list[dict[str, Any]] = []
        if model is not None:
            tblocks = getattr(model, "transformer_blocks", None)
            if tblocks is not None:
                topo.append({
                    "name": "transformer_blocks",
                    "attr_path": "transformer_blocks",
                    "count": len(tblocks),
                    "approx_vram_mb": 120,
                })
        return topo

    # --- Phase 2: encode_text (Task 5) ---

    @torch.no_grad()
    def encode_text(self, captions: list[str], dtype: torch.dtype) -> TextEncoderOutput:
        """Chat-template -> GPT-OSS -> 4 selected layers -> drop 97-token offset.

        Returns embeddings stacked as ``[B, 4, S, 2880]`` and a bool mask
        ``[B, S]``. ``S`` varies per call (the trainer right-pads at batch
        assembly). The 4-layer list expected by the DiT is reconstructed in
        ``forward_pass`` by splitting dim=1.
        """
        rendered = [utils.render_chat_prompt(c, self.tokenizer) for c in captions]
        encoded = self.tokenizer(
            rendered, padding=True, truncation=True,
            max_length=self.te_max_length, return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attn = encoded["attention_mask"].to(self.device)

        out = self.text_encoder(
            input_ids=input_ids, attention_mask=attn,
            output_hidden_states=True, use_cache=False,
        )
        layers = [out.hidden_states[i] for i in self.hf_layer_indices]
        mask = attn.bool()
        layers, mask = utils.drop_txt_offset(layers, mask, offset=self.txt_offset)

        stacked = torch.stack([f.to(dtype=dtype) for f in layers], dim=1)
        return TextEncoderOutput(embeddings=stacked, attention_mask=mask)

    # --- Phase 5: forward (Task 6) ---

    def prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE latent -> 2x2 patchify -> BN-normalize -> [B, S, 128] (Task 6)."""
        raise NotImplementedError("Implemented in Task 6")

    def prepare_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """Raw noise -> patchify -> [B, S, 128], no BN (Task 6)."""
        raise NotImplementedError("Implemented in Task 6")

    def forward_pass(
        self, noisy_input, timesteps, text_embeddings, batch,
    ) -> torch.Tensor:
        """Run the Lens DiT; flow-matching velocity prediction (Task 6)."""
        raise NotImplementedError("Implemented in Task 6")

    # --- Phase 6: saver ---

    def get_saver(self):
        from app.engine.models.families.microsoft_lens.saver import (
            MicrosoftLensSaver,
        )
        return MicrosoftLensSaver()
