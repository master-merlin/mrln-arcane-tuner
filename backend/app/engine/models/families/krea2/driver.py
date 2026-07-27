"""Krea2Driver — family-specific training behavior for Krea-2.

Implements ``IModelDriver`` for Krea-2 (Raw / Turbo).

Krea-2 specifics:
- Text encoder: Qwen3-VL, 12 tapped layers → 4-D stacked embeddings
  ``(B, text_seq_len, 12, 2560)``
- Transformer: Krea2Transformer2DModel (3-axis RoPE, flow-matching)
- Forward pass: patchify → transformer → unpatchify (4-D output)
- Timestep scale: driver passes ``timesteps / 1000.0``; the transformer
  multiplies by 1000 internally (``Krea2TimestepEmbedding``).
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition
from app.engine.core.interfaces import IModelDriver
from app.engine.core.text_encoding import TextEncoderOutput


logger = structlog.get_logger(__name__)

# Krea-2 uses the same Qwen2 tokenizer context window as qwen_image.
TOKENIZER_MAX_LENGTH = 512

# architecture_params key for the 12 tapped layer indices (from YAML).
_TE_SELECT_LAYERS_KEY = "te.text_encoder_select_layers"

# Default layer tap indices (Krea-2 turbo/raw both use the same 12 layers).
_DEFAULT_SELECT_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]


class Krea2Driver(IModelDriver):
    """Krea-2 family driver.

    Handles:
    - Qwen3-VL 12-layer stacked text encoding (4-D embeddings)
    - 3-axis-RoPE flow-matching forward pass (pack → transformer → unpack)
    - LoRA targets from definition (or sensible defaults)
    - bf16 loading dtype; no external scheduler
    """

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Assigned by assign_components()
        self.model: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None
        self._components: dict[str, Any] = {}

        # Architecture params
        self.max_length: int = TOKENIZER_MAX_LENGTH

        # Layer tap indices — read from definition, fall back to defaults.
        arch = getattr(definition, "architecture_params", {}) or {}
        select_layers = arch.get(_TE_SELECT_LAYERS_KEY, _DEFAULT_SELECT_LAYERS)
        self._select_layers: list[int] = list(select_layers)

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded Krea-2 components into driver state."""
        self._components = components
        self.model = components["unet"]
        self.vae = components["vae"]
        self.text_encoder = components.get(
            "text_encoder", getattr(self, "text_encoder", None),
        )
        self.tokenizer = components.get(
            "tokenizer", getattr(self, "tokenizer", None),
        )

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.model

    def get_text_encoders(self) -> dict[str, nn.Module]:
        result: dict[str, nn.Module] = {}
        if self.text_encoder is not None:
            result["text_encoder"] = self.text_encoder
        return result

    def release_text_encoders(self) -> None:
        """Null the attr get_text_encoders() reads."""
        self.text_encoder = None

    def get_lora_targets(self) -> list[str]:
        """Krea-2 LoRA targets — attention + feed-forward projections."""
        definition_targets = getattr(
            self.definition, "lora_targetable_modules", None,
        )
        if definition_targets and len(definition_targets) > 0:
            self.logger.info(
                "lora_targets_from_definition",
                count=len(definition_targets),
            )
            return definition_targets

        self.logger.info("lora_targets_pattern_defaults")
        return [
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_gate",
            "attn.to_out.0",
            "ff.gate", "ff.up", "ff.down",
        ]

    def init_scheduler(self) -> Any:
        """Krea-2 uses flow matching — no external scheduler needed."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """Krea-2 loads in bf16."""
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported for Krea-2."""
        return []

    def get_layer_manifest(self) -> Any:
        """Krea-2 layer manifest with transformer blocks."""
        from app.engine.core.layer_manifest import (
            BlockInfo,
            ModelLayerManifest,
        )

        blocks: list[BlockInfo] = []
        model = self.get_primary_model()
        if model is not None:
            joint = getattr(model, "transformer_blocks", None)
            if joint is not None:
                for i, block in enumerate(joint):
                    blocks.append(BlockInfo(
                        name=f"transformer_blocks.{i}",
                        block_type="joint",
                        param_count=sum(p.numel() for p in block.parameters()),
                        depth_index=i,
                    ))

        return ModelLayerManifest(
            transformer_blocks=blocks,
            lora_targets=self.get_lora_targets(),
            te_lora_targets=self.get_te_lora_targets(),
        )

    # --- Phase 2: Text Encoding ---

    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Encode captions via Qwen3-VL with 12-layer stacking.

        Delegates to the vendored ``get_text_hidden_states`` helper which:
        1. Wraps each caption in the Krea-2 chat template
        2. Tokenises with cumulative-valid-token position IDs
        3. Forwards through Qwen3VLModel (output_hidden_states=True)
        4. Stacks ``select_layers`` on axis 2 → 4-D ``(B, L, 12, 2560)``
        5. Drops the 34-token system prefix

        Args:
            captions: Batch of caption strings.
            dtype: Target dtype for the returned embeddings.

        Returns:
            ``TextEncoderOutput`` with:
            - ``embeddings``: ``[B, text_seq_len, 12, 2560]``
            - ``attention_mask``: ``[B, text_seq_len]`` bool
        """
        from app.engine.models.families.krea2.vendor.krea2_conditioning import (
            get_text_hidden_states,
        )

        with torch.no_grad():
            hidden_states, attention_mask = get_text_hidden_states(
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                prompt=captions,
                select_layers=self._select_layers,
                max_sequence_length=self.max_length,
                device=self.device,
            )

        return TextEncoderOutput(
            embeddings=hidden_states.to(dtype=dtype),
            attention_mask=attention_mask,
        )

    # --- Phase 5: Training Loop Hooks ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Krea2Transformer2DModel forward pass with patchify / unpatchify.

        Tensor flow:
        1. Unpack ``text_embeddings`` → ``(enc_hs [B,L,12,2560], enc_mask [B,L])``.
        2. Patchify ``noisy_input [B,C,H,W]`` → ``[B, (H/p)*(W/p), C*p*p]``.
        3. Build shared position IDs ``[text_seq+img_seq, 3]``.
        4. Scale timesteps: ``model_ts = timesteps / 1000.0`` (transformer
           multiplies by 1000 internally).
        5. Forward through Krea2Transformer2DModel.
        6. Inline unpatchify ``[B, img_seq, out_C*p*p]`` → ``[B, out_C, H, W]``.

        Args:
            noisy_input: Noisy latents ``[B, C, H, W]`` (C=16 for Krea-2).
            timesteps: Flow-matching timesteps in ``[0, 1000]`` scale.
            text_embeddings: ``(embeddings 4-D, attention_mask)`` tuple or
                tensor (mask assumed None if plain tensor).
            batch: Full batch dict (unused here; kept for interface compat).

        Returns:
            Velocity prediction ``[B, C, H, W]``.
        """
        from app.engine.models.families.krea2.vendor.krea2_conditioning import (
            pack_latents,
            prepare_position_ids,
        )

        # 1. Unpack text embeddings.
        if isinstance(text_embeddings, tuple):
            enc_hs, enc_mask = text_embeddings
        else:
            enc_hs = text_embeddings
            enc_mask = None

        B, C, H, W = noisy_input.shape
        model = self.get_primary_model()
        patch_size = getattr(model.config, "patch_size", 2)
        pH = H // patch_size
        pW = W // patch_size

        # 2. Patchify: [B, C, H, W] → [B, pH*pW, C*p*p]
        packed = pack_latents(noisy_input, patch_size=patch_size)

        # 3. Build position IDs (shared across batch).
        #    text_seq_len = actual sequence length seen by the model.
        text_seq_len = enc_hs.shape[1]
        position_ids = prepare_position_ids(text_seq_len, pH, pW, noisy_input.device)

        # 4. Scale timesteps: [0, 1000] → [0, 1] (transformer multiplies *1000 internally).
        model_timesteps = timesteps / 1000.0

        # 5. Transformer forward.
        # encoder_attention_mask must be bool for SDPA (long tensors fail).
        if enc_mask is not None:
            enc_mask_bool = enc_mask.bool()
        else:
            enc_mask_bool = None
        output = model(
            hidden_states=packed,
            encoder_hidden_states=enc_hs,
            timestep=model_timesteps,
            position_ids=position_ids,
            encoder_attention_mask=enc_mask_bool,
            return_dict=False,
        )

        pred = output[0] if isinstance(output, tuple) else output

        # 6. Inline unpatchify: [B, pH*pW, out_C*p*p] → [B, out_C, H, W].
        # NOTE: We do NOT use unpack_latents() which returns 5-D (B,C,1,H,W).
        out_channels = getattr(model.config, "out_channels", C)
        pred = pred.reshape(B, pH, pW, out_channels, patch_size, patch_size)
        pred = pred.permute(0, 3, 1, 4, 2, 5)
        pred = pred.reshape(B, out_channels, H, W)

        return pred

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self):
        """Return Krea-2 LoRA saver.

        Uses ``Krea2Saver``, which inherits from ``GenericLoRASaver``.  Key
        derivation is family-agnostic — based purely on the PEFT-wrapped model's
        actual module paths with no family-specific prefix hardcoded.  For
        Krea-2 this yields canonical ``diffusion_model.transformer_blocks.N.
        attn/ff.*.lora_A/B.weight`` keys that round-trip identically onto
        Krea-2-Raw and Krea-2-Turbo (same ``Krea2Transformer2DModel``
        architecture).  Only the ``architecture_name`` metadata differs,
        correctly identifying saved LoRAs as "krea2" instead of "qwen_image".
        Validated by ``test_krea2_lora_portability.py``.
        """
        from app.engine.models.families.krea2.saver import Krea2Saver

        return Krea2Saver()

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """Krea-2 block topology: single transformer_blocks container."""
        topology = []
        model = self.get_primary_model()
        if model is not None:
            blocks = getattr(model, "transformer_blocks", None)
            if blocks is not None:
                topology.append({
                    "name": "transformer_blocks",
                    "attr_path": "transformer_blocks",
                    "count": len(blocks),
                    "approx_vram_mb": 500,
                })
        return topology
