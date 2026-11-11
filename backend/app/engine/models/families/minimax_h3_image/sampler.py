"""MiniMax-H3-Image preview sampler: the model's fixed 6-step grid, one head per step.

``MiniMaxH3ImageTrainer._create_sampler()`` returns this when
``sample_every_n_steps > 0``; the shared ``GenericSamplingPipeline`` drives it
(prompt -> noise -> denoise -> decode -> PNG). It follows the reference pipeline,
diffusers PR #15230 at merge commit 875373ac, ``modular_pipelines/minimax_h3_image``:

- **The schedule** (``before_denoise.py`` ``MiniMaxH3ImageSetTimestepsStep``): the
  definition's head grid, each sigma rounded through its float32 timestep
  ``1 - sigma`` and back, then a terminal 0. It is never shifted. Step ``i`` runs at
  ``t = 1 - sigma_i`` through head ``i`` (``denoise.py``: ``pdd_step_index=i``).
- **The update** is the scheduler's ``euler_update_form="velocity"``:
  ``x_next = x + (sigma - sigma_next) * v`` in float32 (the heads were distilled on
  that form). The trajectory stays float32; the DiT runs with autocast OFF.
- **No CFG and no negative prompt**: the checkpoint is guidance-distilled, so there is
  one forward per step. A ``sample_negative_prompt``, a ``guidance_scale`` above 1 and
  a step count other than the grid's are ignored, each with one warning.
- **The canvas**: an image under ``MIN_GENERATION_AREA`` pixels is generated on a
  larger canvas of the same aspect and resized back after decoding
  (``get_image_canvas_size``, ``decoders.py``). The decoded image is quantised to
  8 bits by truncation and resized with Lanczos, as the reference does.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from torch import Tensor

from app.engine.components.latents import LatentManager
from app.engine.core.sampling import GenericSamplingPipeline
from app.engine.models.families.minimax_h3_shared.pixel_adapter import IMAGENET_MEAN, IMAGENET_STD

if TYPE_CHECKING:
    from .trainer import MiniMaxH3ImageTrainer

# before_denoise.py MiniMaxH3ImagePrepareCanvasStep: `min_generation_area` default=2048 * 2048.
MIN_GENERATION_AREA = 2048 * 2048
# The released VAE compresses 16x; the reference's fallback when no VAE is loaded.
_VAE_SPATIAL = 16
_LATENT_CHANNELS = 24
_DEFAULT_RESOLUTION = 1024  # the training area (about 1 MP); the canvas rule lifts it to 4 MP


def image_canvas_size(height: int, width: int, min_generation_area: int, multiple: int) -> tuple[int, int]:
    """The ``(height, width)`` canvas a ``height x width`` image is generated on.

    The reference rule (``before_denoise.py`` ``get_image_canvas_size``), verbatim: an
    image below ``min_generation_area`` pixels has both sides scaled by
    ``sqrt(min_generation_area / (height * width))``; either way both sides round to the
    nearest ``multiple``."""
    scale = max(1.0, math.sqrt(min_generation_area / (width * height)))
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


class MiniMaxH3ImageSampler(GenericSamplingPipeline):
    """Six velocity-Euler steps on the head grid, one head per step, no CFG."""

    pipeline: MiniMaxH3ImageTrainer
    needs_live_te = False  # prompts are served from the trainer's text cache

    def __init__(self, pipeline: MiniMaxH3ImageTrainer) -> None:
        super().__init__(pipeline)
        self._output_size: tuple[int, int] | None = None  # (height, width) of the request
        self._warned: set[str] = set()

    def _warn_once(self, event: str, **kw: Any) -> None:
        if event not in self._warned:
            self._warned.add(event)
            self.logger.warning(event, **kw)

    # ── The model's own facts ───────────────────────────────────────────

    def sampling_sigmas(self) -> Tensor:
        """The head grid through its float32 timestep and back, then the terminal 0
        (``before_denoise.py``: ``timesteps = tensor([1.0 - sigma ...], float32)``,
        ``sigmas = cat([1.0 - timesteps, zeros(1)])``)."""
        timesteps = torch.tensor([1.0 - s for s in self.pipeline.driver.head_sigmas()], dtype=torch.float32)
        return torch.cat([1.0 - timesteps, torch.zeros(1)])

    def _sample_prompt_fallbacks(self) -> dict[str, Any]:
        """The grid's step count and no guidance; the definition's resolution, else about 1 MP."""
        defaults = getattr(self.pipeline.definition, "defaults", {}) or {}
        resolution = int(defaults.get("resolution", _DEFAULT_RESOLUTION) or _DEFAULT_RESOLUTION)
        return {
            "width": resolution,
            "height": resolution,
            "num_inference_steps": len(self.pipeline.driver.head_sigmas()),
            "guidance_scale": 1.0,
        }

    def _vae(self) -> Any:
        return self._resolve_component("vae")

    def _vae_spatial(self) -> int:
        vae = self._vae()
        return int(getattr(vae, "spatial_compression_ratio", None) or _VAE_SPATIAL) if vae is not None else _VAE_SPATIAL

    def _latent_channels(self) -> int:
        config = getattr(self._vae(), "config", None)
        return int(getattr(config, "latent_channels", None) or _LATENT_CHANNELS)

    def _model_dtype(self) -> torch.dtype:
        try:
            return next(self.pipeline.transformer.parameters()).dtype
        except (AttributeError, StopIteration, TypeError):
            return torch.float32

    # ── Abstract hooks ──────────────────────────────────────────────────

    def encode_prompt(self, prompt: str) -> Any:
        """The trainer's cached text path (the encoder was released after pre-caching).
        Only the prompt is encoded: the model has no unconditional branch."""
        if str(self.config.get("sample_negative_prompt", "") or "").strip():
            self._warn_once(
                "minimax_h3_image_negative_prompt_ignored",
                message="MiniMax-H3-Image is guidance-distilled: it samples without CFG, so the negative prompt is not used",
            )
        return self.pipeline.encode_text([prompt], self._model_dtype())

    def _create_initial_noise(self, width: int, height: int, generator: torch.Generator) -> Tensor:
        """fp32 noise on the canvas: the reference draws ``(1, C, 1, H, W)``, one frame."""
        multiple = self._vae_spatial() * int(self.pipeline.driver.patch_size()[2])
        canvas_h, canvas_w = image_canvas_size(int(height), int(width), MIN_GENERATION_AREA, multiple)
        self._output_size = (int(height), int(width))
        spatial = self._vae_spatial()
        shape = (1, self._latent_channels(), 1, canvas_h // spatial, canvas_w // spatial)
        noise = torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)
        return noise[:, :, 0]

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Tensor:
        """Step ``i`` runs head ``i`` at ``t = 1 - sigma_i``; ``x += (sigma_i - sigma_{i+1}) * v`` in fp32.
        ``num_steps`` and ``guidance_scale`` are the base's signature; the grid fixes both."""
        driver = self.pipeline.driver
        sigmas = self.sampling_sigmas().to(self.device)
        heads = int(sigmas.numel()) - 1
        if int(num_steps) != heads:
            self._warn_once(
                "minimax_h3_image_steps_fixed",
                requested=int(num_steps),
                steps=heads,
                message=f"MiniMax-H3-Image samples on its fixed {heads}-step grid, one output head per step",
            )
        if float(guidance_scale) > 1.0:
            self._warn_once(
                "minimax_h3_image_guidance_ignored",
                guidance_scale=float(guidance_scale),
                message="MiniMax-H3-Image is guidance-distilled: it samples without CFG",
            )
        timesteps = 1.0 - sigmas[:-1]
        emb, mask = driver._text_rows(prompt_embedding)
        n_text = int(mask[0].sum()) if mask is not None else int(emb.shape[1])
        text_rows = emb[:1, :n_text].to(self.device)

        transformer = self.pipeline.transformer
        if transformer is not None:
            self._ensure_transformer_on_device(transformer)
        x = noise.to(device=self.device, dtype=torch.float32)
        for i in range(heads):
            if getattr(self.pipeline, "_log_writer", None):
                self.pipeline._log_writer.status(f"Sampling {i + 1}/{heads}")
            # The DiT sees autocast OFF on every device type, whatever context the caller entered.
            with (
                torch.no_grad(),
                torch.autocast("cuda", enabled=False),
                torch.autocast("cpu", enabled=False),
            ):
                v = driver.predict_velocity(x, text_rows, timesteps[i].reshape(1), i)
            x = x + (sigmas[i] - sigmas[i + 1]) * v.to(torch.float32)
        return x

    def decode_latents(self, latents: Any) -> Image.Image:
        """Denormalise, decode, revert the VAE's ImageNet space in fp32, quantise to 8 bits
        by truncation, and resize back to the request (Lanczos) when the canvas was larger."""
        vae = self._vae()
        if vae is None:
            raise RuntimeError("minimax_h3_image sampler: no VAE to decode with")
        inner = getattr(vae, "inner", vae)  # the shared pixel adapter's wrapped VAE
        with torch.no_grad():
            raw = LatentManager.denormalize_latents(latents.to(torch.float32), vae)
            out = inner.decode(raw, return_dict=False)
            decoded = (out.sample if hasattr(out, "sample") else out[0]).float()
        mean = torch.tensor(IMAGENET_MEAN, device=decoded.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=decoded.device).view(1, 3, 1, 1)
        pixels = (decoded * std + mean).clamp(0.0, 1.0)
        pixels = (pixels * 255).clamp(0, 255).to(torch.uint8)
        image = Image.fromarray(pixels[0].permute(1, 2, 0).cpu().numpy())
        height, width = self._output_size or (image.height, image.width)
        if (image.width, image.height) != (width, height):
            image = image.resize((width, height), resample=Image.LANCZOS)
        return image
