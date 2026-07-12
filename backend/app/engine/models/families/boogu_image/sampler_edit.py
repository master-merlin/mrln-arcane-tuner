"""BooguImageEditSampler — image-conditioned in-training previews (task A4
fix wave, finding 2).

Subclasses :class:`BooguImageSampler`; reuses BOTH of its denoise loops
(Base CFG + Turbo DMD), its prompt encoding, native-default fill and VAE
decode unchanged. The only delta: the sample prompt's ``control_images``
(``self._active_prompt_cfg``, stashed by the base ``_sample_single`` before
``denoise()`` runs) are VAE-encoded ONCE to clean latents and fed into every
``driver.forward_pass`` call via the :meth:`_forward_batch` hook — the
driver's ``_build_ref_image_hidden_states`` then routes them through the
transformer's native ref-image pathway, exactly as in training. No loop
duplication (contrast qwen's ``sampler_edit.py``, which must re-implement
its whole denoise loop because its control conditioning is a trainer-level
sequence concat; Boogu's is a model-native input).

When no control image is configured the sampler falls back to plain T2I so
a preview never crashes mid-run (qwen precedent). CFG note: the clean
control latents are fed to BOTH the conditional and the unconditional
forward — the negative is therefore upstream's "drop text, KEEP image" term
(the empty-caption encode already lands on the DROP/TI2I system prompt, and
upstream's TI2I image-guidance branch keeps ``ref_latents`` in its
drop-text forward), not the T2I drop-all term.

VAE encode order matches upstream ``encode_vae`` (pipeline_boogu.py:876-892):
``(sample - shift_factor) * scaling_factor`` — the exact inverse of this
family's decode tail (``latents / scaling_factor + shift_factor``).
Deterministic ``latent_dist`` mode (not ``.sample()``) — documented
deviation from upstream's stochastic encode, matching the qwen edit
sampler's choice for reproducible previews.
"""

from __future__ import annotations

import numpy as np
import structlog
import torch
from PIL import Image
from torch import Tensor

from .sampler import BooguImageSampler

logger = structlog.get_logger(__name__)


class BooguImageEditSampler(BooguImageSampler):
    """Boogu-Image Edit sampler — clean control latents fed each step via
    the ``_forward_batch`` hook."""

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
        """Feed the active control latents into driver.forward_pass.

        The driver's ``_build_ref_image_hidden_states`` consumes the same
        ``{"control_latents": List[Tensor[B, C, h, w]]}`` per-slot layout the
        training loop produces — previews exercise the identical seam.
        """
        if self._active_control_latents:
            return {"control_latents": self._active_control_latents}
        return {}

    # ── Control encode ────────────────────────────────────────────────────

    def _encode_control_latent(self, path: str, width: int, height: int) -> Tensor:
        """VAE-encode one control image to a CLEAN ``[1, C, H/8, W/8]`` latent.

        The control is resized to the target's pixel dims (control follows
        the target bucket — this family's ``control_resolution: 0``
        convention), normalized to ``[-1, 1]``, encoded, then mapped to the
        transformer latent space in upstream ``encode_vae``'s order:
        ``(z - shift_factor) * scaling_factor``.
        """
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

    # ── Denoise (control stash around the inherited loops) ───────────────

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
            self.logger.warning("boogu_edit_sample_no_control_image")
            return super().denoise(
                noise, prompt_embedding, num_steps, guidance_scale, seed,
            )

        # Pixel dims from the initial-noise latent grid (control follows the
        # target bucket) — same arch-param source as _create_initial_noise.
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
