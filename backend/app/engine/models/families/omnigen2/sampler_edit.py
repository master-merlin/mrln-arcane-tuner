"""OmniGen2EditSampler — image-conditioned in-training previews.

Subclasses :class:`OmniGen2Sampler`; reuses its whole denoise loop
(including the 2-pass/3-pass guidance combines — the base loop already
implements the pipeline's full L672-723 branch structure and keys the
3-pass on a control being present). The delta here (boogu sampler_edit
pattern):

- The sample prompt's ``control_images`` (``self._active_prompt_cfg``,
  stashed by the base ``_sample_single``) are VAE-encoded ONCE to clean
  latents and fed into the CONDITIONAL and ref-CFG forwards via the
  :meth:`_forward_batch` hook — the driver's
  ``_build_ref_image_hidden_states`` routes them through the transformer's
  native ref-image pathway, exactly as in training. The UNCONDITIONAL
  forward keeps ``batch={}`` (pipeline L702/L721: uncond drops the ref).
- :meth:`_resolve_image_guidance` surfaces ``image_guidance_scale``:
  per-prompt ``image_guidance_scale`` key -> training config
  ``sample_image_guidance_scale`` -> definition ``defaults.
  image_guidance_scale`` -> 2.0 (upstream ``example_edit.sh``:
  ``--text_guidance_scale 5.0 --image_guidance_scale 2.0``). With
  ``image_guidance_scale > 1`` AND ``guidance_scale > 1`` the base loop
  runs the pipeline's 3-pass combine; setting it to 1.0 collapses to the
  cheaper 2-pass, mirroring the pipeline's own ``image_guidance_scale=1``
  default semantics.

When no control image is configured the preview falls back to plain T2I
(base behavior — a preview never crashes mid-run; qwen/boogu precedent).

VAE encode order matches upstream ``encode_vae`` (pipeline L217-233):
``(sample - shift_factor) * scaling_factor`` — the exact inverse of the
decode tail, and the same normalization the house control-latent cache
applies in training. Deterministic ``latent_dist.mode()`` (documented
deviation from upstream's stochastic ``.sample()``, for reproducible
previews — qwen/boogu edit-sampler precedent).

Control sizing: the control is resized to the target's pixel dims (control
follows the target bucket — the house ``control_resolution: 0``
convention; documented deviation from the pipeline's ``align_res``, which
instead snaps the OUTPUT resolution to the reference image's).
"""

from __future__ import annotations

import numpy as np
import structlog
import torch
from PIL import Image
from torch import Tensor

from .sampler import OmniGen2Sampler

logger = structlog.get_logger(__name__)

# Upstream example_edit.sh: --image_guidance_scale 2.0 (with text 5.0).
_DEFAULT_IMAGE_GUIDANCE = 2.0


class OmniGen2EditSampler(OmniGen2Sampler):
    """OmniGen2 Edit sampler — clean control latents fed each step via the
    ``_forward_batch`` hook + the pipeline's image-guidance CFG."""

    def __init__(self, pipeline) -> None:
        super().__init__(pipeline)
        # Set for the duration of one denoise() call (list of per-slot
        # [1, C, h, w] clean control latents), None otherwise.
        self._active_control_latents: list[Tensor] | None = None

    # ── Control resolution ────────────────────────────────────────────────

    def _resolve_control_paths(self) -> list[str]:
        """Control image path(s) from the active sample prompt config."""
        cfg = getattr(self, "_active_prompt_cfg", None) or {}
        paths = cfg.get("control_images") if isinstance(cfg, dict) else getattr(
            cfg, "control_images", None,
        )
        return [p for p in (paths or []) if p]

    def _forward_batch(self) -> dict:
        """Feed the active control latents into the cond/ref forwards.

        Same ``{"control_latents": List[Tensor[B, C, h, w]]}`` per-slot
        layout the training loop produces — previews exercise the identical
        driver seam.
        """
        if self._active_control_latents:
            return {"control_latents": self._active_control_latents}
        return {}

    def _resolve_image_guidance(self) -> float:
        """``image_guidance_scale`` for the 3-pass combine (module
        docstring: prompt key -> config -> definition default -> 2.0)."""
        cfg = getattr(self, "_active_prompt_cfg", None) or {}
        if isinstance(cfg, dict) and cfg.get("image_guidance_scale"):
            return float(cfg["image_guidance_scale"])
        from_config = self.config.get("sample_image_guidance_scale")
        if from_config:
            return float(from_config)
        defaults = getattr(self.pipeline.definition, "defaults", {}) or {}
        return float(defaults.get("image_guidance_scale", _DEFAULT_IMAGE_GUIDANCE))

    # ── Control encode ────────────────────────────────────────────────────

    def _encode_control_latent(self, path: str, width: int, height: int) -> Tensor:
        """VAE-encode one control image to a CLEAN ``[1, C, H/8, W/8]``
        latent in upstream ``encode_vae``'s order:
        ``(z - shift_factor) * scaling_factor`` (pipeline L227-233)."""
        vae = self.pipeline.vae
        vae_dtype = next(vae.parameters()).dtype

        img = Image.open(path).convert("RGB")
        img = img.resize((width, height), Image.LANCZOS)
        arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)
        arr = arr.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        with torch.no_grad():
            posterior = vae.encode(arr.to(self.device, dtype=vae_dtype))
        latent = (
            posterior.latent_dist.mode()
            if hasattr(posterior, "latent_dist")
            else posterior
        )

        scaling_factor = getattr(vae.config, "scaling_factor", 1.0) or 1.0
        shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
        latent = (latent - shift_factor) * scaling_factor
        return latent.to(torch.float32)

    # ── Denoise (control stash around the inherited loop) ────────────────

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Tensor:
        control_paths = self._resolve_control_paths()
        if not control_paths:
            self.logger.warning("omnigen2_edit_sample_no_control_image")
            return super().denoise(
                noise, prompt_embedding, num_steps, guidance_scale, seed,
            )

        # Pixel dims from the initial-noise latent grid (control follows
        # the target bucket — module docstring sizing note).
        arch = getattr(self.pipeline.definition, "architecture_params", {}) or {}
        vae_sf = int(arch.get("vae.vae_scale_factor", 8))
        height = noise.shape[-2] * vae_sf
        width = noise.shape[-1] * vae_sf

        # VAE on GPU only for the control encode (house phased-VRAM
        # bracket — Phase 2 normally has the VAE offloaded).
        vae_moved = self._ensure_on_gpu(["vae"])
        try:
            self._active_control_latents = [
                self._encode_control_latent(p, width, height)
                for p in control_paths
            ]
        finally:
            self._offload_to_cpu(vae_moved)

        try:
            return super().denoise(
                noise, prompt_embedding, num_steps, guidance_scale, seed,
            )
        finally:
            self._active_control_latents = None
