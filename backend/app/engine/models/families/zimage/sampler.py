"""Z-Image sampling — matches ``ZImagePipeline`` from diffusers.

Key differences from the previous sampler:

*  **Chat template** — applies Qwen3 ``apply_chat_template`` with
   ``enable_thinking=True`` before tokenizing.
*  **Hidden layer** — uses ``hidden_states[-2]`` (second-to-last).
*  **Variable-length embeddings** — extracts non-padding tokens into a list
   of per-sample tensors, not a batched ``[B, L, D]``.
*  **Velocity negation** — ``noise_pred = -noise_pred`` before scheduler step.
*  **Scheduler** — uses ``FlowMatchEulerDiscreteScheduler`` with dynamic shift.
*  **CFG** — supports classifier-free guidance with negative prompts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline

if TYPE_CHECKING:
    from .trainer import ZImageTrainer

logger = structlog.get_logger(__name__)


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Empirical mu schedule — copied from ZImagePipeline."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class ZImageSampler(GenericSamplingPipeline):
    """Z-Image sampler — matches ``ZImagePipeline`` from diffusers."""

    pipeline: ZImageTrainer

    def __init__(self, pipeline: ZImageTrainer) -> None:
        super().__init__(pipeline)
        self._scheduler = None

    # ── Lazy scheduler ───────────────────────────────────────────────────

    def _get_scheduler(self):
        if self._scheduler is not None:
            return self._scheduler
        from diffusers import FlowMatchEulerDiscreteScheduler

        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        self._scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=int(arch.get("scheduler.num_train_timesteps", 1000)),
            shift=float(arch.get("scheduler.shift", 6.0)),
            use_dynamic_shifting=bool(arch.get("scheduler.use_dynamic_shifting", False)),
        )
        return self._scheduler

    # ── Text encoding (matches ZImagePipeline._encode_prompt) ────────────

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Encode prompt via the trainer's cache-aware ``encode_text()``.

        Delegates all caching and TE management to the trainer.
        Returns dict with ``embeds`` (list of [L, D] tensors).
        """
        embeds = self.pipeline.encode_text(
            [prompt],
            dtype=next(self.pipeline.model.parameters()).dtype,
        )
        return {"embeds": embeds}

    # ── Core sampling methods ────────────────────────────────────────────

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator
    ) -> Tensor:
        """Create noise in standard VAE latent space.

        Shape: [1, C, H/vae_sf, W/vae_sf].
        """
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        in_channels = int(arch.get("in_channels", 16))

        # Match reference: vae_scale_factor = 2^(len(block_out_channels)-1)
        vae = self.pipeline.vae
        if hasattr(vae.config, "block_out_channels"):
            vae_sf = 2 ** (len(vae.config.block_out_channels) - 1)
        else:
            vae_sf = 8
        # Reference: height = 2 * (int(height) // (vae_sf * 2))
        lat_h = 2 * (height // (vae_sf * 2))
        lat_w = 2 * (width // (vae_sf * 2))

        return torch.randn(
            (1, in_channels, lat_h, lat_w),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Tensor:
        """Flow-matching denoising matching ZImagePipeline.__call__.

        Uses FlowMatchEulerDiscreteScheduler with dynamic shift, supports
        CFG, and negates velocity before scheduler step.
        """
        device = self.device
        transformer = self.pipeline.model
        scheduler = self._get_scheduler()
        # Use the loaded model's dtype, not training-time autocast_dtype
        # (which defaults to fp16 while the model may be loaded in bf16).
        dtype = next(transformer.parameters()).dtype

        latents = noise.to(dtype=torch.float32)
        prompt_embeds = prompt_embedding["embeds"]  # list of [L, D]

        # Negative prompt for CFG (empty string)
        do_cfg = guidance_scale > 1.0
        if do_cfg:
            neg_result = self.encode_prompt("")
            negative_embeds = neg_result["embeds"]
        else:
            negative_embeds = []

        # Image sequence length for mu calculation
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)

        # Timestep schedule via scheduler
        mu = _calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        scheduler.sigma_min = 0.0
        scheduler.set_timesteps(num_inference_steps=num_steps, device=device, mu=mu)
        timesteps = scheduler.timesteps

        # Denoising loop
        self._ensure_transformer_on_device(transformer)
        with torch.no_grad():
            total_steps = len(timesteps)
            for step_i, t in enumerate(timesteps, 1):
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Sampling {step_i}/{total_steps}")
                # Timestep transform: (1000 - t) / 1000
                timestep = t.expand(latents.shape[0])
                timestep_model = (1000 - timestep) / 1000

                if do_cfg:
                    # Concat cond + uncond
                    latent_input = latents.to(dtype).repeat(2, 1, 1, 1)
                    emb_input = prompt_embeds + negative_embeds
                    t_input = timestep_model.repeat(2)
                else:
                    latent_input = latents.to(dtype)
                    emb_input = prompt_embeds
                    t_input = timestep_model

                # Add frame dim: [B, C, H, W] → list of [C, 1, H, W]
                x_list = [latent_input[i].unsqueeze(1) for i in range(latent_input.shape[0])]

                model_out = transformer(
                    x=x_list,
                    t=t_input,
                    cap_feats=emb_input,
                    return_dict=False,
                )[0]

                if do_cfg:
                    actual_bs = latents.shape[0]
                    pos_out = model_out[:actual_bs]
                    neg_out = model_out[actual_bs:]

                    noise_pred_list = []
                    for j in range(actual_bs):
                        pos = pos_out[j].float()
                        neg = neg_out[j].float()
                        pred = pos + guidance_scale * (pos - neg)
                        noise_pred_list.append(pred)

                    noise_pred = torch.stack(noise_pred_list, dim=0)
                else:
                    noise_pred = torch.stack([s.float() for s in model_out], dim=0)

                # Squeeze frame dim: [B, C, 1, H, W] → [B, C, H, W]
                noise_pred = noise_pred.squeeze(2)

                # CRITICAL: Negate velocity (reference line 558)
                noise_pred = -noise_pred

                # Scheduler step
                latents = scheduler.step(
                    noise_pred.to(torch.float32), t, latents, return_dict=False
                )[0]

        return latents

    def decode_latents(self, latents: Any) -> Image.Image:
        """Decode latent tensor to PIL image."""
        vae = self.pipeline.vae
        vae_dtype = next(vae.parameters()).dtype

        vae.to(self.device)
        with torch.no_grad():
            # Reference: (latents / scaling_factor) + shift_factor
            scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
            shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
            scaled = latents.to(dtype=vae_dtype) / scaling_factor + shift_factor
            decoded = vae.decode(scaled, return_dict=False)

        if isinstance(decoded, (tuple, list)):
            image_tensor = decoded[0]
        else:
            image_tensor = decoded

        image_tensor = image_tensor.clamp(-1, 1)
        image_tensor = (image_tensor + 1.0) / 2.0
        image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)
        image_np = image_tensor.cpu().float().numpy()
        image_np = (image_np * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(image_np, mode="RGB")
