"""FLUX.1 Kontext LoRA trainer — image-conditioned ("edit") variant.

Subclasses :class:`Flux1Trainer`; the only delta is the forward pass.
Kontext conditions on a CLEAN control image: its VAE latent is packed and
its tokens are sequence-concatenated AFTER the noisy target tokens, with
the context tokens' first position-id coordinate offset by +1 (per slot)
so RoPE separates context from target. Loss is computed on the target
tokens only, so the output is sliced back to the target sequence length —
``compute_target`` (velocity on the target latents) then needs no change.

Text encoding, packing, timestep sampling, and caching are inherited
unchanged from :class:`Flux1Trainer`.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch

from .trainer import Flux1Trainer
from .utils import pack_latents

logger = structlog.get_logger(__name__)


class Flux1KontextTrainer(Flux1Trainer):
    """FLUX.1 Kontext (image-edit) trainer — clean control tokens concat."""

    def _create_sampler(self):
        """Use the image-conditioned sampler when sampling is configured."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler_kontext import Flux1KontextSampler
            return Flux1KontextSampler(self)
        return None

    def _pack_control(
        self, control_latents: torch.Tensor, slot_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack one clean control latent and build its offset position ids.

        Returns ``(packed [B, Lc, 64], img_ids [Lc, 3])`` with the context
        offset applied on coordinate 0 (``slot_index + 1``) so the context
        block never shares RoPE positions with the target grid.
        """
        packed, ids = pack_latents(control_latents)
        ids = ids.clone()
        ids[:, 0] = slot_index + 1
        return (
            packed.to(self.device, dtype=self.autocast_dtype),
            ids.to(self.device, dtype=packed.dtype),
        )

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """FluxTransformer2DModel forward with concatenated clean control tokens.

        Args:
            noisy_input: Packed noisy TARGET latents ``[B, L_target, 64]``.
            timesteps: Scaled timesteps ``[0, 1000]`` (÷1000 for the model).
            text_embeddings: T5 context ``[B, L_txt, 4096]``.
            batch: Full batch dict; ``control_latents`` is a list of clean
                spatial control latents (one per slot).

        Returns:
            Velocity prediction for the TARGET tokens only ``[B, L_target, 64]``.
        """
        control_latents = batch.get("control_latents") or []
        if not control_latents:
            # No controls (e.g. a partial batch slipped through) — behave
            # exactly like the base Flux1 trainer.
            return super().forward_pass(
                noisy_input, timesteps, text_embeddings, batch
            )

        target_seq_len = noisy_input.shape[1]

        # Pack + offset every control slot, then concat onto the target tokens.
        packed_controls: list[torch.Tensor] = []
        control_ids: list[torch.Tensor] = []
        for slot_idx, ctrl in enumerate(control_latents):
            packed, ids = self._pack_control(ctrl, slot_idx)
            packed_controls.append(packed)
            control_ids.append(ids)

        hidden_states = torch.cat([noisy_input, *packed_controls], dim=1)
        combined_img_ids = torch.cat(
            [self._current_img_ids, *control_ids], dim=0
        )

        # Diffusers model multiplies timestep by 1000 internally.
        model_timesteps = timesteps / 1000.0

        txt_seq_len = text_embeddings.shape[1]
        txt_ids = torch.zeros(
            txt_seq_len, 3, device=self.device, dtype=text_embeddings.dtype,
        )

        pooled = getattr(self, "_clip_pooled", None)
        if pooled is None:
            pooled_dim = self.transformer.config.pooled_projection_dim
            pooled = torch.zeros(
                hidden_states.shape[0], pooled_dim,
                device=self.device, dtype=self.autocast_dtype,
            )

        guidance = None
        if self.use_guidance_embed:
            guidance_scale = float(self.config.get("guidance_scale", 1.0))
            guidance = torch.full(
                (hidden_states.shape[0],), guidance_scale,
                device=self.device, dtype=self.autocast_dtype,
            )

        output = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=text_embeddings,
            pooled_projections=pooled,
            timestep=model_timesteps,
            img_ids=combined_img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )
        pred = output[0] if isinstance(output, tuple) else output

        # Loss is on the target tokens only — drop the context tail so the
        # shape matches compute_target(prepared_latents).
        return pred[:, :target_seq_len]
