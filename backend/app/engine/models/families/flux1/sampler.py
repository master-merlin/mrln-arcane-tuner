"""FLUX.1 sampler — Euler denoising loop for sample image generation.

Uses ``FluxTransformer2DModel`` forward with dual CLIP+T5 prompt encoding
and standard ``AutoencoderKL`` VAE decoding.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING, Any

import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.core.sampling import GenericSamplingPipeline
from .utils import pack_latents, unpack_latents

if TYPE_CHECKING:
    from .trainer import Flux1Trainer

logger = structlog.get_logger(__name__)


# ── Resolution-dependent time shift (matches FLUX pre-training) ──────────
# Ref: diffusers FlowMatchEulerDiscreteScheduler, kohya-ss get_schedule(),
#      ai-toolkit calculate_shift().
# Values from scheduler config in dev.yaml:
#   base_image_seq_len=256, max_image_seq_len=4096,
#   base_shift=0.5, max_shift=1.15, time_shift_type=exponential.

def _flux_time_shift(
    mu: float, sigma: float, t: torch.Tensor,
) -> torch.Tensor:
    """Shifted sigmoid schedule: ``exp(mu) / (exp(mu) + (1/t - 1)^sigma)``."""
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def _compute_mu(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Resolution-dependent mu via linear interpolation."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class Flux1Sampler(GenericSamplingPipeline):
    """FLUX.1 sampler — flow-matching Euler denoising loop.

    Generates sample images using ``FluxTransformer2DModel`` and
    ``AutoencoderKL``.  Uses the trainer's CLIP+T5 text encoding.
    """

    pipeline: Flux1Trainer

    # ── Abstract Hook Implementations ────────────────────────────────────

    def encode_prompt(self, prompt: str) -> Any:
        """Encode a prompt using the trainer's CLIP+T5 pipeline.

        Args:
            prompt: The fully-expanded prompt text.

        Returns:
            T5 text embedding ``[1, L, 4096]``.
        """
        return self.pipeline.encode_text(
            [prompt],
            dtype=next(self.pipeline.transformer.parameters()).dtype,
        )

    def _create_initial_noise(
        self, width: int, height: int, generator: torch.Generator
    ) -> Tensor:
        """Create noise in FLUX.1 VAE latent space.

        Shape: ``[1, 16, H/8, W/8]``.
        (16 VAE latent_channels at VAE spatial scale; ``pack_latents``
        in ``denoise`` will 2×2-pack this to ``[1, L, 64]``)

        Args:
            width: Output image width in pixels.
            height: Output image height in pixels.
            generator: Seeded random generator.

        Returns:
            Noise tensor on ``self.device``.
        """
        # VAE latent_channels = 16; pack_latents (called in denoise)
        # applies 2×2 patch packing: 16 * 4 = 64 packed channels.
        # Spatial dims must be VAE scale (÷8), NOT patched (÷16).
        in_channels = 16
        latent_h = height // 8
        latent_w = width // 8
        return torch.randn(
            (1, in_channels, latent_h, latent_w),
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
    ) -> tuple[Tensor, int, int]:
        """Flow-matching Euler denoising loop.

        Matches the diffusers ``FluxPipeline.__call__`` execution context:

        - The transformer forward pass runs under autocast (needed for NF4
          quantized models that dequantize to float32).
        - The Euler accumulation step runs in float32 outside autocast,
          matching ``FlowMatchEulerDiscreteScheduler.step()``.
        - Guidance is created in float32 (diffusers convention).

        Args:
            noise: Initial noise ``[1, 16, H/8, W/8]``.
            prompt_embedding: T5 text context ``[1, L_txt, 4096]``.
            num_steps: Number of denoising steps.
            guidance_scale: CFG scale for Dev variant.
            seed: Random seed (unused here).

        Returns:
            Tuple of (packed_latents ``[1, L, 64]``, latent_h, latent_w).
        """
        latent_h, latent_w = noise.shape[2], noise.shape[3]
        # Use the loaded transformer's dtype, not the training-time
        # autocast_dtype (which defaults to fp16 even when the model is
        # bf16 / NF4-dequant-to-bf16 -- there is no outer autocast at
        # sample time, so a mismatched cast causes per-op repromotion
        # and drift across the denoising loop).
        model_dtype = next(self.pipeline.transformer.parameters()).dtype

        # 1. Pack noise → sequence [1, L, 64] + img_ids [L, 3]
        latents, img_ids = pack_latents(noise)
        latents = latents.to(self.device, dtype=model_dtype)
        img_ids = img_ids.to(self.device)

        # 2. Text embeddings
        txt = prompt_embedding.to(self.device, dtype=model_dtype)
        txt_ids = torch.zeros(
            txt.shape[1], 3,
            device=self.device, dtype=txt.dtype,
        )

        # 3. CLIP pooled
        pooled = getattr(self.pipeline, "_clip_pooled", None)
        if pooled is not None:
            pooled = pooled.to(self.device, dtype=model_dtype)
        else:
            pooled_dim = self.pipeline.transformer.config.pooled_projection_dim
            pooled = torch.zeros(
                1, pooled_dim,
                device=self.device, dtype=model_dtype,
            )

        # 4. Guidance (Dev only) — float32, matching diffusers FluxPipeline
        guidance = None
        if self.pipeline.use_guidance_embed:
            guidance = torch.full(
                (1,), guidance_scale,
                device=self.device, dtype=torch.float32,
            )

        # 5. Timestep schedule: 1.0 → 0.0 with resolution-dependent shift
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        image_seq_len = (latent_h // 2) * (latent_w // 2)
        mu = _compute_mu(image_seq_len)
        timesteps[1:-1] = _flux_time_shift(mu, 1.0, timesteps[1:-1])

        self.logger.debug(
            "denoising_start",
            num_steps=num_steps,
            latent_shape=list(latents.shape),
            text_shape=list(txt.shape),
        )

        # 6. Euler denoising loop
        # Disable outer autocast (from generate_samples) for the loop body.
        # Diffusers FluxPipeline does NOT use autocast; autocast is only
        # needed for the transformer forward to handle NF4 dequantisation.
        autocast_dtype = model_dtype
        use_amp = getattr(self.pipeline, "use_amp", True)

        for i in range(num_steps):
            print(f"[STATUS:Sampling {i + 1}/{num_steps}]", flush=True)
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t

            # Transformer forward — under autocast for NF4 compatibility
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                output = self.pipeline.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=txt,
                    pooled_projections=pooled,
                    timestep=t.unsqueeze(0),
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance,
                    return_dict=False,
                )
            velocity = output[0] if isinstance(output, tuple) else output

            # Euler step in float32 (matches FlowMatchEulerDiscreteScheduler)
            latents = (latents.float() + dt.float() * velocity.float()).to(model_dtype)

            if i == 0 or i == num_steps - 1:
                self.logger.debug(
                    "denoise_step",
                    step=i,
                    t=round(t.item(), 4),
                    dt=round(dt.item(), 4),
                    vel_mean=round(velocity.float().mean().item(), 4),
                    vel_std=round(velocity.float().std().item(), 4),
                    lat_mean=round(latents.float().mean().item(), 4),
                    lat_std=round(latents.float().std().item(), 4),
                )

        return latents, latent_h, latent_w

    def decode_latents(self, latents_bundle: Any) -> Image.Image:
        """Unpack + VAE decode → PIL image.

        Args:
            latents_bundle: Tuple of (packed_latents, latent_h, latent_w).

        Returns:
            PIL Image in RGB mode.
        """
        latents, latent_h, latent_w = latents_bundle

        # Unpack [1, L, 64] → [1, 16, H, W]
        spatial = unpack_latents(latents, latent_h, latent_w)

        # VAE decode
        vae = self.pipeline.vae
        vae_dtype = next(vae.parameters()).dtype
        vae_device = next(vae.parameters()).device

        # Undo the scaling applied during training encode
        # Formula matches diffusers FluxPipeline.__call__:
        #   latents = (latents / scaling_factor) + shift_factor
        scaling_factor = getattr(vae.config, "scaling_factor", 0.3611)
        shift_factor = getattr(vae.config, "shift_factor", 0.1159)
        spatial = spatial / scaling_factor + shift_factor

        if vae_device != self.device:
            vae.to(self.device)

        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            decoded = vae.decode(spatial.to(dtype=vae_dtype))

        if vae_device != self.device:
            vae.to(vae_device)
            torch.cuda.empty_cache()

        # Tensor → PIL
        if hasattr(decoded, "sample"):
            image_tensor = decoded.sample
        elif isinstance(decoded, torch.Tensor):
            image_tensor = decoded
        else:
            self.logger.warning("unexpected_vae_output", type=type(decoded).__name__)
            return Image.new("RGB", (latent_w * 16, latent_h * 16), (128, 128, 128))

        image_tensor = image_tensor.clamp(-1, 1)
        image_tensor = (image_tensor + 1.0) / 2.0
        image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)
        image_np = image_tensor.cpu().float().numpy()
        image_np = (image_np * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(image_np, mode="RGB")
