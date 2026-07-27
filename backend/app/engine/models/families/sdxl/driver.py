"""SDXL model driver — family-specific training behavior.

Implements ``IModelDriver`` for Stable Diffusion XL.
Phase 1 scope: loading-related methods only.
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


class SDXLDriver(IModelDriver):
    """SDXL family driver.

    Handles:
    - Dual CLIP text encoder assignment
    - DDPM epsilon-prediction scheduler
    - fp32 loading dtype (AMP GradScaler requires fp32)
    - Comprehensive UNet LoRA targets (attention + FF + conv + embeddings)
    """

    def __init__(self, definition: ModelDefinition, device: torch.device):
        self.definition = definition
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Assigned by assign_components()
        self.unet: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.text_encoder_1: nn.Module | None = None
        self.text_encoder_2: nn.Module | None = None
        self._components: dict[str, Any] = {}

        # Architecture params
        self.te_max_length: int = 77
        self.scheduler_beta_start: float = 0.00085
        self.scheduler_beta_end: float = 0.012
        self.scheduler_prediction_type: str = "epsilon"

    # --- Phase 1: Loading & Component Access ---

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded SDXL components and cache architecture params."""
        self._components = components
        self.unet = components["unet"]
        self.vae = components["vae"]
        # TEs may have been offloaded (removed from components dict)
        self.text_encoder_1 = components.get(
            "text_encoder_1", getattr(self, "text_encoder_1", None),
        )
        self.text_encoder_2 = components.get(
            "text_encoder_2", getattr(self, "text_encoder_2", None),
        )

        arch = getattr(self.definition, "architecture_params", {}) or {}
        self.te_max_length = arch.get("te.max_length", 77)
        self.scheduler_beta_start = float(
            arch.get("scheduler.beta_start", 0.00085),
        )
        self.scheduler_beta_end = float(
            arch.get("scheduler.beta_end", 0.012),
        )
        self.scheduler_prediction_type = arch.get(
            "scheduler.prediction_type", "epsilon",
        )

    def get_components(self) -> dict[str, Any]:
        return self._components

    def get_primary_model(self) -> nn.Module:
        return self.unet

    def get_text_encoders(self) -> dict[str, nn.Module]:
        return {
            k: v
            for k, v in {
                "text_encoder_1": self.text_encoder_1,
                "text_encoder_2": self.text_encoder_2,
            }.items()
            if v is not None
        }

    def release_text_encoders(self) -> None:
        """Null both CLIP text encoders — the exact attrs get_text_encoders() reads."""
        self.text_encoder_1 = None
        self.text_encoder_2 = None

    def get_lora_targets(self) -> list[str]:
        """UNet LoRA targets: from definition YAML or comprehensive defaults."""
        definition_targets = getattr(
            self.definition, "lora_targetable_modules", None,
        )
        if definition_targets and len(definition_targets) > 0:
            self.logger.info(
                "lora_targets_from_definition",
                count=len(definition_targets),
            )
            return definition_targets

        self.logger.info("lora_targets_comprehensive_defaults")
        return [
            # Attention (cross + self)
            "to_q", "to_k", "to_v", "to_out.0",
            "add_k_proj", "add_v_proj",
            # Feed-forward networks
            "ff.net.0.proj", "ff.net.2",
            # Transformer projections
            "proj_in", "proj_out",
            # ResNet convolutions
            "conv1", "conv2", "conv_shortcut",
            # Time embedding
            "time_emb_proj",
            # Global convolutions
            "conv_in", "conv_out",
            # Time embedding MLPs
            "time_embedding.linear_1", "time_embedding.linear_2",
            # Addition embedding (SDXL-specific: pooled text embeddings)
            "add_embedding.linear_1", "add_embedding.linear_2",
            # Downsampler / upsampler convolutions
            "downsamplers.0.conv", "upsamplers.0.conv",
        ]

    def init_scheduler(self) -> Any:
        """Create DDPMScheduler for epsilon prediction."""
        from diffusers import DDPMScheduler

        return DDPMScheduler(
            beta_start=self.scheduler_beta_start,
            beta_end=self.scheduler_beta_end,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=self.scheduler_prediction_type,
        )

    def resolve_loading_dtype(self) -> torch.dtype:
        """SDXL loads in fp32 — AMP GradScaler requires fp32 params/grads."""
        return torch.float32

    def get_te_lora_targets(self) -> list[str]:
        """SDXL text encoder LoRA targets (CLIP attention projections)."""
        return ["q_proj", "v_proj"]

    def get_layer_manifest(self) -> Any:
        """SDXL layer manifest with down/mid/up UNet blocks."""
        from app.engine.core.layer_manifest import (
            BlockInfo,
            ModelLayerManifest,
        )

        blocks: list[BlockInfo] = []
        model = self.get_primary_model()
        if model is not None:
            depth = 0
            # Down blocks
            down = getattr(model, "down_blocks", None)
            if down is not None:
                for i, block in enumerate(down):
                    blocks.append(BlockInfo(
                        name=f"down_blocks.{i}",
                        block_type="down",
                        param_count=sum(p.numel() for p in block.parameters()),
                        depth_index=depth,
                    ))
                    depth += 1
            # Mid block
            mid = getattr(model, "mid_block", None)
            if mid is not None:
                blocks.append(BlockInfo(
                    name="mid_block",
                    block_type="mid",
                    param_count=sum(p.numel() for p in mid.parameters()),
                    depth_index=depth,
                ))
                depth += 1
            # Up blocks
            up = getattr(model, "up_blocks", None)
            if up is not None:
                for i, block in enumerate(up):
                    blocks.append(BlockInfo(
                        name=f"up_blocks.{i}",
                        block_type="up",
                        param_count=sum(p.numel() for p in block.parameters()),
                        depth_index=depth,
                    ))
                    depth += 1

        return ModelLayerManifest(
            transformer_blocks=blocks,
            lora_targets=self.get_lora_targets(),
            te_lora_targets=self.get_te_lora_targets(),
        )

    # --- Phase 2: Text Encoding ---

    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Encode captions via dual CLIP encoding."""
        return self.encode_dual_clip(captions, dtype)

    def encode_dual_clip(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Dual CLIP hidden-state concatenation + pooled.

        Uses penultimate hidden states from both CLIP encoders,
        concatenated along the feature dimension.

        Args:
            captions: Batch of caption strings.
            dtype: Target dtype.

        Returns:
            ``TextEncoderOutput`` with concat embeddings ``[B, L, D1+D2]``
            and CLIP-2 pooled in ``pooled``.
        """
        t1 = self._components["tokenizer_1"](
            captions, padding="max_length", max_length=self.te_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)
        t2 = self._components["tokenizer_2"](
            captions, padding="max_length", max_length=self.te_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            e1 = self.text_encoder_1(t1, output_hidden_states=True)
            e2 = self.text_encoder_2(t2, output_hidden_states=True)

        h1 = e1.hidden_states[-2]
        h2 = e2.hidden_states[-2]
        prompt_embeds = torch.cat([h1, h2], dim=-1).to(dtype=dtype)
        pooled_embeds = e2.text_embeds.to(dtype=dtype)

        return TextEncoderOutput(
            embeddings=prompt_embeds,
            pooled=pooled_embeds,
        )

    # --- Phase 5: Training Loop Hooks ---

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """UNet forward with SDXL conditioning (text_embeds + time_ids).

        Expects ``batch`` to contain ``time_ids`` and ``_pooled_embeds``
        to be cached on the driver from ``encode_text()``.

        Returns:
            UNet epsilon prediction ``[B, C, H, W]``.
        """
        pooled = getattr(self, "_pooled_embeds", None)
        if pooled is None:
            raise RuntimeError(
                "SDXLDriver.forward_pass: _pooled_embeds not set. "
                "Call encode_text() first."
            )
        return self.unet(
            noisy_input,
            timesteps,
            encoder_hidden_states=text_embeddings,
            added_cond_kwargs={
                "text_embeds": pooled,
                "time_ids": batch["time_ids"],
            },
        ).sample

    def compute_target(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Epsilon prediction: target = noise."""
        return noise

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        config: dict[str, Any],
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Discrete uniform timesteps ``[0, num_train_timesteps)``."""
        scheduler = getattr(self, "_scheduler", None)
        num_steps = 1000
        if scheduler is not None:
            num_steps = scheduler.config.num_train_timesteps
        return torch.randint(
            0, num_steps, (batch_size,), device=device,
        ).long()

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """DDPM noise addition via scheduler."""
        scheduler = getattr(self, "_scheduler", None)
        if scheduler is not None:
            return scheduler.add_noise(latents, noise, timesteps)
        # Fallback to linear interpolation
        return super().add_noise(latents, noise, timesteps)

    # --- Phase 6: LoRA Output & Saver ---

    def get_saver(self):
        """Return SDXL Kohya-format LoRA saver."""
        from app.engine.models.families.sdxl.saver import SDXLSaver

        return SDXLSaver()

    # --- Phase 7: Data Pipeline ---

    def build_batch_extra(self, items: list[dict]) -> dict[str, Any]:
        """Build SDXL time_ids (6-component conditioning vector).

        Components: (original_h, original_w, crop_top, crop_left, target_h, target_w)
        """
        time_ids_list = []
        for item in items:
            tw, th = item["target_w"], item["target_h"]
            w, h = item.get("orig_w", tw), item.get("orig_h", th)
            scale = max(tw / w, th / h)
            nw, nh = int(w * scale), int(h * scale)
            left = (nw - tw) // 2
            top = (nh - th) // 2
            t_id = [nh, nw, top, left, th, tw]
            time_ids_list.append(torch.tensor(t_id, dtype=torch.float32))
        return {"time_ids": torch.stack(time_ids_list).to(self.device)}

    # --- Phase 8: Checkpoint Resume ---
    #
    # NOTE: get_te_cache/set_te_cache live on ``SDXLTrainer`` (trainer.py),
    # not here. The real checkpoint dispatch path
    # (pipeline_train.py save sites + pipeline_optimization.py resume merge)
    # calls ``self.get_te_cache()``/``set_te_cache()`` on the TRAINER, never
    # on the driver — a driver-level override here would be dead code (and
    # ``self.text_cache``/``self._pooled_cache`` don't even live on the
    # driver instance; they're set on the trainer).

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """SDXL UNet block topology: down + mid + up blocks."""
        topology = []
        model = self.get_primary_model()
        if model is not None:
            down = getattr(model, "down_blocks", None)
            if down is not None:
                topology.append({
                    "name": "down_blocks",
                    "attr_path": "down_blocks",
                    "count": len(down),
                    "approx_vram_mb": 200,
                })
            mid = getattr(model, "mid_block", None)
            if mid is not None:
                topology.append({
                    "name": "mid_block",
                    "attr_path": "mid_block",
                    "count": 1,
                    "approx_vram_mb": 100,
                })
            up = getattr(model, "up_blocks", None)
            if up is not None:
                topology.append({
                    "name": "up_blocks",
                    "attr_path": "up_blocks",
                    "count": len(up),
                    "approx_vram_mb": 200,
                })
        return topology




