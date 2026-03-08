"""FLUX.1 model driver — family-specific training behavior.

Implements ``IModelDriver`` for the FLUX.1 family (Dev and Schnell
variants). Phase 1 scope: loading-related methods only.
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


class Flux1Driver(IModelDriver):
    """FLUX.1 family driver (Dev / Schnell).

    Handles:
    - Dual text encoder assignment (CLIP + T5)
    - FLUX.1-specific LoRA target patterns
    - Flow-matching scheduler (None)
    """

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Assigned by assign_components()
        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.clip_encoder: nn.Module | None = None
        self.clip_tokenizer: Any = None
        self.t5_encoder: nn.Module | None = None
        self.t5_tokenizer: Any = None
        self._components: dict[str, Any] = {}

        # Architecture params
        self.use_guidance_embed: bool = True
        self.te_clip_max_length: int = 77
        self.te_t5_max_length: int = 512

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded FLUX.1 components and cache architecture params."""
        self._components = components
        self.transformer = components["unet"]
        self.vae = components["vae"]
        self.clip_encoder = components.get("text_encoder")
        self.clip_tokenizer = components.get("tokenizer")
        self.t5_encoder = components.get("text_encoder_2")
        self.t5_tokenizer = components.get("tokenizer_2")

        # Cache architecture params
        arch = getattr(self.definition, "architecture_params", {}) or {}
        self.use_guidance_embed = arch.get("transformer.guidance_embeds", True)
        self.te_clip_max_length = arch.get("te.clip_max_length", 77)
        self.te_t5_max_length = arch.get("te.t5_max_length", 512)

        self.logger.info(
            "flux1_config",
            guidance_embed=self.use_guidance_embed,
            clip_max_len=self.te_clip_max_length,
            t5_max_len=self.te_t5_max_length,
        )

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.transformer

    def get_text_encoders(self) -> dict[str, nn.Module]:
        """Return CLIP + T5 text encoders using component-dict keys."""
        result: dict[str, nn.Module] = {}
        if self.clip_encoder is not None:
            result["text_encoder"] = self.clip_encoder
        if self.t5_encoder is not None:
            result["text_encoder_2"] = self.t5_encoder
        return result

    def get_lora_targets(self) -> list[str]:
        """FLUX.1 LoRA targets — diffusers FluxTransformerBlock modules."""
        return [
            # Double stream (FluxTransformerBlock)
            "to_q", "to_k", "to_v", "to_out.0",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
            "ff.net.0.proj", "ff.net.2",
            "ff_context.net.0.proj", "ff_context.net.2",
            # Single stream (FluxSingleTransformerBlock)
            "proj_mlp", "proj_out",
        ]

    def init_scheduler(self) -> Any:
        """FLUX.1 uses flow matching — no external scheduler."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """FLUX.1 loads in bf16."""
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported for FLUX.1."""
        return []

    def get_layer_manifest(self) -> Any:
        """FLUX.1 layer manifest with joint + single transformer blocks."""
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
            single = getattr(model, "single_transformer_blocks", None)
            if single is not None:
                offset = len(blocks)
                for i, block in enumerate(single):
                    blocks.append(BlockInfo(
                        name=f"single_transformer_blocks.{i}",
                        block_type="single",
                        param_count=sum(p.numel() for p in block.parameters()),
                        depth_index=offset + i,
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
        """Encode captions via dual CLIP + T5 encoding."""
        return self.encode_dual_clip_t5(captions, dtype)

    def encode_dual_clip_t5(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Dual CLIP + T5 encoding.

        CLIP provides pooled embeddings (used for conditioning).
        T5 provides the sequence context (primary embeddings).

        Args:
            captions: Batch of caption strings.
            dtype: Target dtype.

        Returns:
            ``TextEncoderOutput`` with T5 sequence embeddings and
            CLIP pooled in ``pooled``.
        """
        # T5 sequence embeddings
        t5_inputs = self.t5_tokenizer(
            captions,
            padding="max_length",
            max_length=self.te_t5_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            t5_out = self.t5_encoder(
                t5_inputs.input_ids.to(self.device),
            )
        t5_emb = t5_out.last_hidden_state.to(dtype=dtype)

        # CLIP pooled embeddings
        clip_inputs = self.clip_tokenizer(
            captions,
            padding="max_length",
            max_length=self.te_clip_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            clip_out = self.clip_encoder(
                clip_inputs.input_ids.to(self.device),
                output_hidden_states=False,
            )
        clip_pooled = clip_out.pooler_output.to(dtype=dtype)

        return TextEncoderOutput(
            embeddings=t5_emb,
            pooled=clip_pooled,
        )

    # --- Phase 5: Training Loop Hooks ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """FluxTransformer2DModel forward: predict velocity.

        Args:
            noisy_input: Packed noisy latents ``[B, L, 64]``.
            timesteps: Scaled timesteps ``[0, 1000]``.
            text_embeddings: T5 context ``[B, L_txt, 4096]``.
            batch: Full batch dict.

        Returns:
            Velocity prediction ``[B, L, 64]``.
        """
        model_timesteps = timesteps / 1000.0

        # txt_ids: zeros [L_txt, 3]
        txt_seq_len = text_embeddings.shape[1]
        txt_ids = torch.zeros(
            txt_seq_len, 3,
            device=self.device, dtype=text_embeddings.dtype,
        )

        # CLIP pooled (cached by encode_text via TextEncoderOutput.pooled)
        pooled = getattr(self, "_clip_pooled", None)
        if pooled is None:
            pooled_dim = self.transformer.config.pooled_projection_dim
            pooled = torch.zeros(
                noisy_input.shape[0], pooled_dim,
                device=self.device, dtype=noisy_input.dtype,
            )

        # Guidance (Dev uses guidance_embed; Schnell does not)
        guidance = None
        if self.use_guidance_embed:
            guidance_scale = 3.5  # Default guidance scale
            guidance = torch.full(
                (noisy_input.shape[0],), guidance_scale,
                device=self.device, dtype=noisy_input.dtype,
            )

        output = self.transformer(
            hidden_states=noisy_input,
            encoder_hidden_states=text_embeddings,
            pooled_projections=pooled,
            timestep=model_timesteps,
            img_ids=self._current_img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )

        return output[0] if isinstance(output, tuple) else output

    def prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Pack image latents ``[B, C, H, W]`` → ``[B, L, C]``.

        Stores ``_current_img_ids`` for the forward pass.
        """
        from app.engine.models.families.flux1.utils import pack_latents

        self._latent_h = latents.shape[2]
        self._latent_w = latents.shape[3]
        packed, img_ids = pack_latents(latents)
        self._current_img_ids = img_ids.to(self.device)
        return packed.to(self.device)

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self):
        """Return FLUX.1 ai-toolkit-format LoRA saver."""
        from app.engine.models.families.flux1.saver import Flux1Saver

        return Flux1Saver()

    # --- Phase 8: Checkpoint Resume ---

    def get_te_cache(self) -> dict[str, dict[str, torch.Tensor]] | None:
        """Return T5 + CLIP pooled caches for checkpoint persistence."""
        text_cache = getattr(self, "text_cache", {})
        if not text_cache:
            return None
        clip_pooled = getattr(self, "_clip_pooled_cache", {})
        return {
            "t5": dict(text_cache),
            "clip_pooled": dict(clip_pooled),
        }

    def set_te_cache(self, caches: dict[str, dict[str, torch.Tensor]]) -> None:
        """Restore T5 + CLIP pooled caches from checkpoint."""
        if "t5" in caches:
            self.text_cache = caches["t5"]
        if "clip_pooled" in caches:
            self._clip_pooled_cache = caches["clip_pooled"]
        self.logger.info(
            "te_cache_restored",
            t5_entries=len(getattr(self, "text_cache", {})),
            clip_entries=len(getattr(self, "_clip_pooled_cache", {})),
        )

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """FLUX.1 block topology: double (joint) + single stream blocks."""
        topology = []
        model = self.get_primary_model()
        if model is not None:
            joint = getattr(model, "transformer_blocks", None)
            if joint is not None:
                topology.append({
                    "name": "double_blocks",
                    "attr_path": "transformer_blocks",
                    "count": len(joint),
                    "approx_vram_mb": 640,
                })
            single = getattr(model, "single_transformer_blocks", None)
            if single is not None:
                topology.append({
                    "name": "single_blocks",
                    "attr_path": "single_transformer_blocks",
                    "count": len(single),
                    "approx_vram_mb": 320,
                })
        return topology




