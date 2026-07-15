"""Bernini-R driver — subclasses the shared :class:`WanDriverBase`.

Bernini-R reuses all of the shared Wan flow-match training behaviour (raw
``[0,1000]`` timestep ``add_noise``, UMT5 encoding, 5D ``prepare_latents``, the
wan-canonical LoRA target set). The ONE family specific is the forward path: it
runs the vendored **packed** forward (``vendor/transformer_forward.py``) that
token-concatenates the clean condition-video latents with the noisy target and
reads the velocity back for the target tokens only, rather than the stock
channel-wise Wan forward.

v1 scope: v2v only — one condition stream at ``source_id=1``, target at
``source_id=0``. Additional ordered condition streams (rv2v) map to
``source_id = slot_index + 1`` and are already handled generically here.
"""

from __future__ import annotations

from typing import Any

import torch

from app.engine.models.families.bernini_r.vendor.transformer_forward import (
    bernini_packed_forward,
)
from app.engine.models.families.wan_shared.driver_base import WanDriverBase


class BerniniRDriver(WanDriverBase):
    """Bernini-R family driver (renderer-only video edit, 1.3B v1)."""

    # Clean control-video latents, one 5D tensor per ordered condition slot,
    # attached by the training data pipeline (``pipeline_data._load_control_latents``).
    BATCH_CONTROL_LATENTS = "control_latents"

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Vendored packed forward — velocity over the target's 16 channels.

        Builds ``[cond..., target]`` token streams (condition latents from
        ``batch['control_latents']``, ``source_id = slot + 1``; target at
        ``source_id=0``), runs the full-bidirectional packed forward, and returns
        the velocity for the TARGET tokens only, shaped ``[B, 16, F, H, W]``.

        With no condition latents this degenerates to a stock Wan t2v forward
        (``source_id=0``, single stream), so mixed edit/plain batches are safe.

        Args:
            noisy_input: noised target latent ``[B, 16, F, H, W]`` (or 4D still,
                lifted to 5D).
            timesteps: raw ``[0,1000]`` timestep(s) — shared by ALL tokens,
                including the clean condition tokens.
            text_embeddings: ``TextEncoderOutput`` / tuple / raw ``[B, L, D]``.
            batch: full batch dict; condition latents live under
                ``BATCH_CONTROL_LATENTS``.

        Returns:
            Velocity prediction ``[B, 16, F, H, W]``.
        """
        enc_hs = self._as_text_tensor(text_embeddings)
        target = self.prepare_latents(noisy_input)  # ensure 5D [B,C,F,H,W]

        cond_latents: list[torch.Tensor] = []
        cond_source_ids: list[float] = []
        for slot_idx, control in enumerate(batch.get(self.BATCH_CONTROL_LATENTS) or []):
            if control is None:
                continue
            cond = control if control.ndim == 5 else control.unsqueeze(2)
            cond_latents.append(cond.to(device=target.device, dtype=target.dtype))
            # source_id 1..N — ordered condition streams (v2v uses the first).
            cond_source_ids.append(float(slot_idx + 1))

        # RAW [0, 1000] timestep — the diffusers Wan time embedder consumes the
        # FlowMatchEuler value directly (the /1000 lives only in add_noise's lerp).
        output = bernini_packed_forward(
            self.transformer,
            cond_latents=cond_latents,
            cond_source_ids=cond_source_ids,
            target_latent=target,
            timestep=timesteps,
            encoder_hidden_states=enc_hs,
            return_dict=False,
        )
        return output[0] if isinstance(output, tuple) else output

    def get_saver(self) -> Any:
        """Reuse the Wan 2.1 ComfyUI-format saver — Bernini-R weights are
        stock-key Wan, so the ``diffusion_model.*`` mapping is identical.
        """
        from app.engine.models.families.wan21.saver import Wan21Saver

        return Wan21Saver(mode="t2v")

    def get_block_topology(self) -> list[dict[str, Any]]:
        """Single stack of ``blocks`` (mirrors wan21)."""
        topology: list[dict[str, Any]] = []
        model = self.get_primary_model()
        if model is not None:
            blocks = getattr(model, "blocks", None)
            if blocks is not None:
                topology.append(
                    {
                        "name": "blocks",
                        "attr_path": "blocks",
                        "count": len(blocks),
                        "approx_vram_mb": 320,
                    }
                )
        return topology
