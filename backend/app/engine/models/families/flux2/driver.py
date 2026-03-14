"""FLUX.2 model driver — family-specific training behavior.

Implements ``IModelDriver`` for the FLUX.2 family (Klein and Dev
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


class Flux2Driver(IModelDriver):
    """FLUX.2 family driver (Klein / Dev).

    Handles:
    - Component assignment with TE type detection (Qwen3 vs Mistral3)
    - Guidance embedder zeroing for guidance-distilled models (Klein)
    - FLUX.2-specific LoRA target patterns (regex for PEFT)
    - Flow-matching scheduler (None — no external scheduler needed)
    """

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Assigned by assign_components()
        self.transformer: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any = None
        self._components: dict[str, Any] = {}

        # Architecture params (populated in assign_components)
        self.use_guidance_embed: bool = False
        self.te_max_length: int = 512
        self.te_concat_layers: int = 3
        self.te_model_type: str = "qwen3"
        self.te_output_layers: list[int] | None = None

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded FLUX.2 components and cache architecture params."""
        self._components = components
        self.transformer = components["unet"]
        self.vae = components["vae"]
        self.text_encoder = components.get("text_encoder")
        self.tokenizer = components.get("tokenizer")

        # Cache architecture params
        arch = getattr(self.definition, "architecture_params", {}) or {}
        self.use_guidance_embed = arch.get("transformer.guidance_embeds", False)
        self.te_max_length = arch.get("te.max_length", 512)
        self.te_concat_layers = int(arch.get("te.concat_layers", 3))
        self.te_model_type = arch.get("te.model_type", "qwen3")
        self.te_output_layers = arch.get("te.output_layers", None)

        # Klein (guidance-distilled): guidance=None skips the embedder entirely
        if not self.use_guidance_embed:
            self.logger.info(
                "guidance_embed_disabled",
                reason="guidance_embeds=False → guidance=None (embedder skipped)",
            )

        self.logger.info(
            "flux2_config",
            guidance_embed=self.use_guidance_embed,
            te_max_length=self.te_max_length,
            te_model_type=self.te_model_type,
            concat_layers=self.te_concat_layers,
            output_layers=self.te_output_layers,
        )

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.transformer

    def get_text_encoders(self) -> dict[str, nn.Module]:
        result: dict[str, nn.Module] = {}
        if self.text_encoder is not None:
            result["text_encoder"] = self.text_encoder
        return result

    def get_lora_targets(self) -> list[str]:
        """FLUX.2 LoRA targets — simple names for PEFT suffix matching.

        PEFT matches these via ``key.endswith(f".{target}")`` so each
        entry must be the trailing portion of a ``named_modules()`` key.

        Double-stream QKV uses **fused** ``to_qkv`` / ``to_added_qkv``
        modules (created by ``fuse_projections()`` before PEFT wrapping).
        This gives PEFT a single ``lora_A`` per fused QKV, avoiding the
        lossy SVD re-decomposition that would be needed to merge separate
        Q/K/V ``lora_A`` matrices at save time.

        The ``to_out`` suffix targets the single-stream output projection
        (``single_transformer_blocks.N.attn.to_out`` — an ``nn.Linear``).
        It also matches the double-stream ``transformer_blocks.N.attn.to_out``
        which is a ``ModuleList``.  :meth:`get_lora_exclude_modules` blocks
        the double-stream match to prevent PEFT from crashing.
        """
        return [
            # Double stream — attention (fused QKV + output projections)
            "to_qkv",
            "to_added_qkv",
            "to_out.0",
            "to_add_out",
            # Double stream — MLP (Flux2FeedForward)
            "ff.linear_in", "ff.linear_out",
            "ff_context.linear_in", "ff_context.linear_out",
            # Single stream — fused QKV+MLP input + output projection
            "to_qkv_mlp_proj",
            "to_out",
        ]

    def get_lora_exclude_modules(self) -> list[str] | None:
        """Exclude double-stream ``attn.to_out`` and unfused Q/K/V modules.

        The ``to_out`` target in :meth:`get_lora_targets` is needed for
        single-stream blocks where it is an ``nn.Linear``.  In double-stream
        blocks the same suffix resolves to a ``ModuleList`` container, which
        PEFT cannot wrap.

        After ``fuse_projections()``, the old unfused ``to_q/k/v`` and
        ``add_q/k/v_proj`` modules still exist on the attention blocks.
        They must be excluded so PEFT only wraps the fused ``to_qkv``
        and ``to_added_qkv`` modules.
        """
        return (
            r"transformer_blocks\.\d+\.attn\.to_out$"
            r"|transformer_blocks\.\d+\.attn\.to_[qkv]$"
            r"|transformer_blocks\.\d+\.attn\.add_[qkv]_proj$"
        )

    def init_scheduler(self) -> Any:
        """FLUX.2 uses flow matching — no external scheduler."""
        return None

    def resolve_loading_dtype(self) -> torch.dtype:
        """FLUX.2 loads in bf16 (always bf16 autocast, no scaler)."""
        return torch.bfloat16

    def get_te_lora_targets(self) -> list[str]:
        """Text encoder LoRA not supported for FLUX.2."""
        return []

    def get_layer_manifest(self) -> Any:
        """FLUX.2 layer manifest with joint + single transformer blocks."""
        from app.engine.core.layer_manifest import (
            BlockInfo,
            ModelLayerManifest,
        )

        blocks: list[BlockInfo] = []
        model = self.get_primary_model()
        if model is not None:
            # Joint (double-stream) blocks
            joint = getattr(model, "transformer_blocks", None)
            if joint is not None:
                for i, block in enumerate(joint):
                    blocks.append(BlockInfo(
                        name=f"transformer_blocks.{i}",
                        block_type="joint",
                        param_count=sum(p.numel() for p in block.parameters()),
                        depth_index=i,
                    ))
            # Single-stream blocks
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
        """Encode captions via layer concatenation (Qwen3 or Mistral3)."""
        return self.encode_layer_concat(captions, dtype)

    def encode_layer_concat(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Hidden-state layer concatenation encoding.

        Supports both Qwen3 (Klein) and Mistral3 (Dev) text encoders.
        Concatenates intermediate hidden states to produce a wide
        embedding (e.g. 3 × 4096 = 12288-dim for Klein).

        Args:
            captions: Batch of caption strings.
            dtype: Target dtype.

        Returns:
            ``TextEncoderOutput`` with embeddings ``[B, L, D*N]``.
        """
        if self.te_model_type == "mistral3":
            return self._encode_mistral3(captions, dtype)
        return self._encode_qwen3(captions, dtype)

    def _encode_qwen3(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Qwen3 encoding: chat template → hidden states → layer concat."""

        all_input_ids = []
        all_attention_masks = []

        for caption in captions:
            messages = [{"role": "user", "content": caption}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.te_max_length,
            )
            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(self.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self.device)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            if self.te_output_layers is not None:
                selected = [outputs.hidden_states[k] for k in self.te_output_layers]
            else:
                selected = list(outputs.hidden_states[-self.te_concat_layers:])
            ctx = torch.cat(selected, dim=-1)

        return TextEncoderOutput(embeddings=ctx.to(dtype=dtype))

    def _encode_mistral3(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Mistral3 encoding: format_input → chat template → layer stack."""
        from diffusers.pipelines.flux2.pipeline_flux2 import SYSTEM_MESSAGE, format_input

        messages_batch = format_input(prompts=captions, system_message=SYSTEM_MESSAGE)

        inputs = self.tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.te_max_length,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        layers = self.te_output_layers or [10, 20, 30]
        out = torch.stack(
            [outputs.hidden_states[k] for k in layers], dim=1,
        )
        out = out.to(dtype=dtype, device=self.device)

        # [B, num_layers, L, D] → [B, L, num_layers * D]
        B, N, L, D = out.shape
        ctx = out.permute(0, 2, 1, 3).reshape(B, L, N * D)

        return TextEncoderOutput(embeddings=ctx)

    # --- Phase 5: Training Loop Hooks ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Flux2Transformer2DModel forward: predict velocity.

        Args:
            noisy_input: Packed noisy latents ``[B, L, 128]``.
            timesteps: Scaled timesteps ``[0, 1000]``.
            text_embeddings: Text context ``[B, L_txt, D_text]``.
            batch: Full batch dict.

        Returns:
            Velocity prediction ``[B, L, 128]``.
        """
        model_timesteps = timesteps / 1000.0

        # txt_ids: 4-col [L_txt, 4] for Flux2's 4-axis RoPE
        txt_seq_len = text_embeddings.shape[1]
        t = torch.arange(1, device=self.device)
        h = torch.arange(1, device=self.device)
        w = torch.arange(1, device=self.device)
        seq_l = torch.arange(txt_seq_len, device=self.device)
        txt_ids = torch.cartesian_prod(t, h, w, seq_l).to(dtype=text_embeddings.dtype)

        # Guidance: Klein passes None (guidance embedder skipped); Dev uses 1.0
        if self.use_guidance_embed:
            guidance = torch.full(
                (noisy_input.shape[0],), 1.0,
                device=self.device, dtype=noisy_input.dtype,
            )
        else:
            guidance = None

        output = self.transformer(
            hidden_states=noisy_input,
            encoder_hidden_states=text_embeddings,
            timestep=model_timesteps,
            img_ids=self._current_img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )

        return output[0] if isinstance(output, tuple) else output

    def prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Patchify + BN normalize + pack latents for Flux2.

        Stores ``_current_img_ids`` for forward pass positioning.
        """
        from app.engine.models.families.flux2.utils import (
            _make_img_ids,
            _pack_spatial,
            bn_normalize,
            patchify_latents,
        )

        patchified = patchify_latents(latents)
        normalized = bn_normalize(patchified, self.vae)
        self._latent_h = normalized.shape[2]
        self._latent_w = normalized.shape[3]
        packed = _pack_spatial(normalized)
        self._current_img_ids = _make_img_ids(
            self._latent_h, self._latent_w, self.device,
        ).to(self.device)
        return packed.to(self.device)

    def prepare_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """Patchify + pack noise WITHOUT BN normalization.

        Flux2 expects raw N(0,1) noise at t=1.0 — only clean signal
        goes through BN.
        """
        from app.engine.models.families.flux2.utils import (
            _pack_spatial,
            patchify_latents,
        )

        patchified = patchify_latents(noise)
        packed = _pack_spatial(patchified)
        return packed.to(self.device)

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self):
        """Return FLUX.2 BFL-format LoRA saver."""
        from app.engine.models.families.flux2.saver import Flux2Saver

        return Flux2Saver()

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """FLUX.2 block topology: double (joint) + single stream blocks."""
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



