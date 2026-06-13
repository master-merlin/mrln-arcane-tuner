"""FLUX.1 Kontext sampler — image-conditioned Euler denoising.

Subclasses :class:`Flux1Sampler`; reuses its prompt encoding, noise
construction, decode, and precision regime (float32 Euler accumulation
outside autocast — the autocast-collapse + timestep-scale gotchas are
pinned by test_kontext_sampler_precision.py). The only delta is that a
CLEAN control latent is concatenated onto the sequence at every step and
the velocity used for the Euler update is sliced back to the target tokens.

The control image comes from the sample prompt's ``control_images``
(set by the base ``_sample_single`` on ``self._active_prompt_cfg``); when
none is given the sampler falls back to plain text-to-image behaviour so a
preview never crashes mid-run.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

from .sampler import Flux1Sampler
from .utils import pack_latents

logger = structlog.get_logger(__name__)


class Flux1KontextSampler(Flux1Sampler):
    """FLUX.1 Kontext sampler — clean control tokens concatenated each step."""

    def _resolve_control_paths(self) -> list[str]:
        cfg = getattr(self, "_active_prompt_cfg", None) or {}
        paths = cfg.get("control_images") or []
        return [p for p in paths if p]

    def _encode_control_latent(self, path: str) -> Tensor:
        """VAE-encode one control image to a CLEAN latent ``[1, 16, H/8, W/8]``."""
        vae = self.pipeline.vae
        vae_dtype = next(vae.parameters()).dtype
        img = Image.open(path).convert("RGB")
        # Quantize to the VAE's /8 grid (Kontext keeps the control's own size).
        w, h = img.width - img.width % 16, img.height - img.height % 16
        img = img.resize((max(16, w), max(16, h)), Image.Resampling.LANCZOS)
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        arr = to_tensor(img).unsqueeze(0)
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            posterior = vae.encode(arr.to(self.device, dtype=vae_dtype))
        latent = posterior.latent_dist.mode() if hasattr(posterior, "latent_dist") else posterior.sample()
        scaling = getattr(vae.config, "scaling_factor", 0.3611)
        shift = getattr(vae.config, "shift_factor", 0.1159)
        return (latent - shift) * scaling

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> tuple[Tensor, int, int]:
        """Image-conditioned flow-matching Euler loop.

        Identical precision regime to :meth:`Flux1Sampler.denoise` — trajectory
        ``linspace(1.0→0.0)`` (never ×1000), Euler accumulation in float32
        OUTSIDE autocast. The control tokens are concatenated into every
        forward and the velocity is sliced to the target tokens before the step.
        """
        control_paths = self._resolve_control_paths()
        if not control_paths:
            self.logger.warning("kontext_sample_no_control_image")
            return super().denoise(
                noise, prompt_embedding, num_steps, guidance_scale, seed
            )

        latent_h, latent_w = noise.shape[2], noise.shape[3]
        model_dtype = next(self.pipeline.transformer.parameters()).dtype

        # Target tokens.
        latents, img_ids = pack_latents(noise)
        latents = latents.to(self.device, dtype=model_dtype)
        img_ids = img_ids.to(self.device)
        target_seq_len = latents.shape[1]

        # Clean control tokens (packed once, reused every step) + offset ids.
        ctrl_packed: list[Tensor] = []
        ctrl_ids: list[Tensor] = []
        vae_moved = self._ensure_on_gpu(["vae"])
        for slot_idx, path in enumerate(control_paths):
            ctrl_latent = self._encode_control_latent(path)
            packed, cids = pack_latents(ctrl_latent)
            cids = cids.clone()
            cids[:, 0] = slot_idx + 1
            ctrl_packed.append(packed.to(self.device, dtype=model_dtype))
            ctrl_ids.append(cids.to(self.device))
        self._offload_to_cpu(vae_moved)
        combined_ids = torch.cat([img_ids, *ctrl_ids], dim=0)

        # Text + pooled + guidance (same as base).
        txt = prompt_embedding.to(self.device, dtype=model_dtype)
        txt_ids = torch.zeros(txt.shape[1], 3, device=self.device, dtype=txt.dtype)
        pooled = getattr(self.pipeline, "_clip_pooled", None)
        if pooled is not None:
            pooled = pooled.to(self.device, dtype=model_dtype)
        else:
            pooled_dim = self.pipeline.transformer.config.pooled_projection_dim
            pooled = torch.zeros(1, pooled_dim, device=self.device, dtype=model_dtype)
        guidance = None
        if self.pipeline.use_guidance_embed:
            guidance = torch.full(
                (1,), guidance_scale, device=self.device, dtype=torch.float32,
            )

        # Timestep schedule: identical to base (no ×1000).
        from .sampler import _compute_mu, _flux_time_shift
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        image_seq_len = (latent_h // 2) * (latent_w // 2)
        mu = _compute_mu(image_seq_len)
        timesteps[1:-1] = _flux_time_shift(mu, 1.0, timesteps[1:-1])

        use_amp = getattr(self.pipeline, "use_amp", True)
        for i in range(num_steps):
            if getattr(self, "_log_writer", None):
                self._log_writer.status(f"Sampling {i + 1}/{num_steps}")
            t = timesteps[i]
            dt = timesteps[i + 1] - t

            hidden = torch.cat([latents, *ctrl_packed], dim=1)
            with torch.autocast("cuda", dtype=model_dtype, enabled=use_amp):
                output = self.pipeline.transformer(
                    hidden_states=hidden,
                    encoder_hidden_states=txt,
                    pooled_projections=pooled,
                    timestep=t.unsqueeze(0),
                    img_ids=combined_ids,
                    txt_ids=txt_ids,
                    guidance=guidance,
                    return_dict=False,
                )
            velocity = output[0] if isinstance(output, tuple) else output
            # Step the TARGET tokens only (drop the context tail).
            velocity = velocity[:, :target_seq_len]
            latents = (latents.float() + dt.float() * velocity.float()).to(model_dtype)

        return latents, latent_h, latent_w
