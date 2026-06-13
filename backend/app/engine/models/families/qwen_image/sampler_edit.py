"""Qwen-Image-Edit sampler — image-conditioned flow-matching Euler denoising.

Subclasses :class:`QwenImageSampler`; reuses its packing, scheduler, prompt
encoding, and VAE decode. The only delta is that a CLEAN control latent is
packed once and concatenated onto the packed target tokens at every step,
with ``img_shapes`` carrying both image shapes, and the prediction sliced
back to the target tokens before the scheduler step (matches diffusers
``QwenImageEditPipeline``: ``cat([latents, image_latents])`` →
``noise_pred[:, :latents.size(1)]``).

The control image path comes from the sample prompt's ``control_images``
(set on ``self._active_prompt_cfg`` by the base ``_sample_single``). When no
control image is given the sampler falls back to plain text-to-image so a
preview never crashes mid-run. Timesteps stay in flow time (``t/1000``, never
an extra ×1000); the trajectory is accumulated by the scheduler step.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
import torch
from PIL import Image
from torch import Tensor

from .sampler import QwenImageSampler, _calculate_shift

logger = structlog.get_logger(__name__)


class QwenImageEditSampler(QwenImageSampler):
    """Qwen-Image-Edit sampler — clean control tokens concatenated each step."""

    def _resolve_control_paths(self) -> list[str]:
        cfg = getattr(self, "_active_prompt_cfg", None) or {}
        paths = cfg.get("control_images") if isinstance(cfg, dict) else getattr(
            cfg, "control_images", None,
        )
        return [p for p in (paths or []) if p]

    def _encode_control_latent(self, path: str, lat_h: int, lat_w: int) -> Tensor:
        """VAE-encode one control image to a CLEAN packed latent ``[1, Lc, C*4]``.

        The control is resized to the target's pixel dims so its latent grid
        matches the target (control follows the target bucket) and the packed
        token counts line up.
        """
        vae = self.pipeline.vae
        vae_dtype = next(vae.parameters()).dtype
        num_channels = self.pipeline.transformer.config.in_channels // 4

        img = Image.open(path).convert("RGB")
        img = img.resize((self._sample_width, self._sample_height), Image.LANCZOS)
        arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)
        arr = arr.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]

        with torch.no_grad():
            posterior = vae.encode(arr.to(self.device, dtype=vae_dtype))
        latent = (
            posterior.latent_dist.mode()
            if hasattr(posterior, "latent_dist") else posterior.sample()
        )
        if latent.ndim == 5:  # [B, C, T, H, W] → drop the (single) frame dim
            latent = latent[:, :, 0]

        # Transformer latent space is (x - mean) / std (inverse of decode).
        mean = torch.tensor(vae.config.latents_mean, device=latent.device,
                            dtype=latent.dtype).view(1, -1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=latent.device,
                           dtype=latent.dtype).view(1, -1, 1, 1)
        latent = (latent - mean) / std

        return self._pack_latents(latent, 1, num_channels, lat_h, lat_w)

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Any:
        """Image-conditioned flow-matching Euler loop (control tokens concat)."""
        control_paths = self._resolve_control_paths()
        if not control_paths:
            self.logger.warning("qwen_edit_sample_no_control_image")
            return super().denoise(
                noise, prompt_embedding, num_steps, guidance_scale, seed,
            )

        device = self.device
        transformer = self.pipeline.transformer
        scheduler = self._get_scheduler()
        dtype = next(transformer.parameters()).dtype

        height, width = self._sample_height, self._sample_width
        lat_h, lat_w, vae_sf = self._lat_h, self._lat_w, self._vae_sf

        prompt_embeds = prompt_embedding["embeds"]
        prompt_mask = prompt_embedding["mask"]
        latents = noise.to(device=device, dtype=dtype)
        target_tokens = latents.shape[1]

        # Clean control tokens (packed once, reused every step).
        control_packed: list[Tensor] = []
        inner_shapes = [(1, lat_h // 2, lat_w // 2)]
        for path in control_paths:
            packed = self._encode_control_latent(path, lat_h, lat_w)
            control_packed.append(packed.to(device=device, dtype=dtype))
            inner_shapes.append((1, lat_h // 2, lat_w // 2))
        img_shapes = [inner_shapes]

        txt_seq_lens = prompt_mask.sum(dim=1).tolist()

        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        mu = _calculate_shift(
            latents.shape[1],
            int(arch.get("scheduler.base_image_seq_len", 256)),
            int(arch.get("scheduler.max_image_seq_len", 4096)),
            float(arch.get("scheduler.base_shift", 0.5)),
            float(arch.get("scheduler.max_shift", 1.15)),
        )
        scheduler.set_timesteps(
            num_inference_steps=num_steps, device=device, sigmas=sigmas, mu=mu,
        )
        timesteps = scheduler.timesteps

        self._ensure_transformer_on_device(transformer)
        scheduler.set_begin_index(0)
        with torch.no_grad():
            total = len(timesteps)
            for step_i, t in enumerate(timesteps, 1):
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Sampling {step_i}/{total}")
                ts = t.expand(latents.shape[0]).to(dtype)
                hidden = torch.cat([latents, *control_packed], dim=1)

                noise_pred = transformer(
                    hidden_states=hidden,
                    timestep=ts / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]
                # Step the TARGET tokens only (drop the control tail).
                noise_pred = noise_pred[:, :target_tokens]
                latents = scheduler.step(
                    noise_pred, t, latents, return_dict=False,
                )[0]

        latents = self._unpack_latents(latents, height, width, vae_sf)
        return {"latents": latents, "height": height, "width": width}
