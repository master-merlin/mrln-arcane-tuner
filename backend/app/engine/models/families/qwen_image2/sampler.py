"""Qwen-Image 2.1 preview sampler -- upstream ``QwenImage21Pipeline.__call__``, text-to-image.

Evidence: diffusers ``6256aa7666`` ``pipeline_qwenimage21.py`` 700-849: noise
``[B, 1, C, h, w]`` packed to tokens, ``sigmas = linspace(1, 1/N, N)`` with a
dynamic shift ``mu`` from the scheduler config, one conditional forward per
step (``true_cfg_scale`` defaults to 1.0), Euler steps, then
``z * std + mean`` and the VAE's frame 0.

The forward goes through ``QwenImage2Driver.forward_pass`` -- the one producer
of the calling convention (image slots, ``/1000``, last-token slice) the
trainer uses too.

PRECISION CONTRACT (LESSONS: autocast sampler collapse): the trajectory, the
scheduler step and the timesteps stay float32, and the DiT forward runs with
autocast explicitly OFF even when a caller wraps sampling in one. Only the
model's inputs are cast to its weight dtype.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch
from PIL import Image
from torch import Tensor

from app.engine.components.latents import LatentManager
from app.engine.core.sampling import GenericSamplingPipeline

from . import utils

if TYPE_CHECKING:
    from .trainer import QwenImage2Trainer

logger = structlog.get_logger(__name__)


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Upstream ``calculate_shift`` (``pipeline_qwenimage21.py`` 61-72)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class QwenImage2Sampler(GenericSamplingPipeline):
    """Preview sampler for Qwen-Image 2.1."""

    pipeline: QwenImage2Trainer

    def __init__(self, pipeline: QwenImage2Trainer) -> None:
        super().__init__(pipeline)
        self._scheduler = None
        self._cfg_ignored_warned = False

    def _scheduler_config(self) -> dict[str, Any]:
        """The checkpoint's ``scheduler_config.json``, as harvested into the YAML."""
        arch = getattr(self.pipeline.definition, "architecture_params", None) or {}
        return {
            k.removeprefix("scheduler."): v
            for k, v in arch.items()
            if k.startswith("scheduler.") and k != "scheduler._class_name"
        }

    def _get_scheduler(self):
        if self._scheduler is None:
            from diffusers import FlowMatchEulerDiscreteScheduler

            self._scheduler = FlowMatchEulerDiscreteScheduler.from_config(self._scheduler_config())
        return self._scheduler

    def encode_prompt(self, prompt: str) -> dict[str, Any]:
        """Through the trainer's cache-aware ``encode_text`` (pre-warmed prompts)."""
        embeds, mask = self.pipeline.encode_text([prompt], dtype=torch.float32)
        return {"embeds": embeds, "mask": mask}

    def _create_initial_noise(self, width: int, height: int, generator: torch.Generator) -> Tensor:
        """Float32 noise ``[1, h*w, C]`` on upstream's grid (``latent_grid``)."""
        transformer = self.pipeline.transformer
        channels = int(transformer.config.in_channels)
        lat_h, lat_w = utils.latent_grid(height, width)
        noise = torch.randn(
            (1, 1, channels, lat_h, lat_w), generator=generator,
            device=generator.device, dtype=torch.float32,
        ).to(self.device)
        self._lat_h, self._lat_w = lat_h, lat_w
        return utils.pack_latents(noise[:, 0])

    def _warn_cfg_ignored_once(self, guidance_scale: float) -> None:
        if not self._cfg_ignored_warned:
            self._cfg_ignored_warned = True
            logger.warning(
                "qwen_image2 preview runs one conditional forward per step "
                "(upstream true_cfg_scale default 1.0); guidance_scale is ignored",
                guidance_scale=guidance_scale,
            )

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> dict[str, Any]:
        transformer = self.pipeline.transformer
        driver = self.pipeline.driver
        scheduler = self._get_scheduler()
        cfg = scheduler.config
        model_dtype = next(transformer.parameters()).dtype
        if float(guidance_scale) > 1.0:
            self._warn_cfg_ignored_once(guidance_scale)

        latents = noise.to(self.device, dtype=torch.float32)
        lat_h, lat_w = self._lat_h, self._lat_w
        embeds = prompt_embedding["embeds"].to(self.device, dtype=model_dtype)
        mask = prompt_embedding["mask"].to(self.device)

        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        mu = calculate_shift(
            latents.shape[1],
            cfg.get("base_image_seq_len", 256),
            cfg.get("max_image_seq_len", 4096),
            cfg.get("base_shift", 0.5),
            cfg.get("max_shift", 1.15),
        )
        scheduler.set_timesteps(num_steps, device=self.device, sigmas=sigmas, mu=mu)
        scheduler.set_begin_index(0)

        self._ensure_transformer_on_device(transformer)
        total = len(scheduler.timesteps)
        with torch.no_grad():
            for i, t in enumerate(scheduler.timesteps, 1):
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Sampling {i}/{total}")
                spatial = utils.unpack_latents(latents, lat_h, lat_w).to(model_dtype)
                ts = t.expand(latents.shape[0]).to(torch.float32)
                # Autocast OFF around the DiT forward (precision contract above).
                with torch.autocast(self.device.type, enabled=False):
                    pred = driver.forward_pass(spatial, ts, (embeds, mask), {})
                pred = utils.pack_latents(pred.float())
                latents = scheduler.step(pred, t, latents, return_dict=False)[0].float()

        return {"latents": utils.unpack_latents(latents, lat_h, lat_w)}

    def decode_latents(self, latents_bundle: Any) -> Image.Image:
        """``z * std + mean``, VAE frame 0, then an RGB preview.

        Assumption: the preview drops the decoded alpha channel (upstream
        returns an RGBA image); the sample gallery shows RGB.
        """
        vae = self.pipeline.vae
        latents = latents_bundle["latents"].to(vae.dtype).unsqueeze(2)  # [B, C, 1, h, w]
        latents = LatentManager.denormalize_latents(latents, vae)
        with torch.no_grad():
            decoded = vae.decode(latents, return_dict=False)[0][:, :, 0]
        rgb = decoded[0, :3].float().clamp(-1, 1)
        arr = ((rgb + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr, mode="RGB")
