"""WAN 2.2 TI2V-5B driver — dense single-transformer, ``expand_timesteps`` I2V.

Recon (diffusers 0.39, ``pipelines/wan/pipeline_wan_i2v.py`` — the REAL pipeline
the diffusers team uses for THIS checkpoint's i2v generation, gated by
``self.config.expand_timesteps``, verified against the installed lib in this
repo's venv):

- The 5B transformer's ``in_channels == out_channels == z_dim == 48`` — there
  is NO 36-channel ``[noisy(16), mask(4), cond(16)]`` concat like
  ``wan21``/``wan22`` A14B I2V (those models have ``image_dim: null`` too, but
  their in-channels are hard-doubled+padded; TI2V-5B's patch_embedding takes
  the raw 48-channel latent unchanged in BOTH modes).
- Inference conditions on a starting image by (a) VAE-encoding it, (b) at every
  denoise step substituting that CLEAN encoded latent into the first temporal
  slot of the model's input (``latent_model_input = (1-mask)*condition +
  mask*latents`` with ``mask[...,0]=0``), and (c) feeding the transformer a
  PER-TOKEN timestep (``timestep.ndim == 2``, ``WanTransformer3DModel.forward``
  branches on this — see ``transformer_wan.py:673-676``) that is ZERO on every
  token belonging to frame 0 and the scalar ``t`` elsewhere
  (``pipeline_wan_i2v.py:760-767``).

Training-time equivalent (this driver): our "condition" is the SAME clip's own
clean frame-0 latent (self-reconstruction — the same convention ``wan21``/
``wan22`` already use for their 36-channel ``cond``), so forcing the flow-match
noise SCALE to zero on frame 0 in :meth:`add_noise` reproduces the pipeline's
substitution exactly (``noisy = 0*noise + 1*latents == latents`` at frame 0) —
no separate stashed conditioning tensor is needed (unlike ``wan21``/``wan22``'s
``BATCH_FIRST_FRAME_LATENT``). :meth:`forward_pass` mirrors the per-token
timestep construction for the (downsampled by the transformer's spatial patch
size) token grid.

Because frame 0's input is never actually noised, the flow-match target
``noise - latents`` is NOT predictable there (the model was shown no noise to
undo) — an i2v ANSWER LEAK / degenerate-loss risk identical in kind to the
``ltx2``/``kandinsky5`` first-frame token exclusion. :class:`Wan22Ti2v5bTrainer`
masks those tokens out of the loss (mirrors ``ltx2``'s ``_compute_step_loss``).

F=1 STILL GUARD: a single still on an i2v-active run has no frame to predict
beyond the conditioning frame — :meth:`_conditioning_engaged` gates on F>1
(ltx2/kandinsky5/wan21/wan22 parity), so a still silently takes the plain T2V
scalar-timestep path with zero special-casing (no channel padding needed here,
unlike the 36-channel families, since the channel count never changes).
"""

from __future__ import annotations

from typing import Any

import torch

from app.engine.models.families.wan_shared.driver_base import WanDriverBase


class Wan22Ti2v5bDriver(WanDriverBase):
    """WAN 2.2 TI2V-5B driver (dense, ``mode: both`` — T2V + I2V in one checkpoint)."""

    def __init__(self, definition: Any, device: torch.device) -> None:
        super().__init__(definition, device)
        # The base class derives ``is_i2v`` from a FIXED ``mode`` architecture
        # param (t2v XOR i2v per definition) — TI2V-5B's single ``mode: both``
        # definition instead has its video_mode chosen PER TRAINING STEP by the
        # trainer (ltx2 precedent). ``is_i2v`` is intentionally left at the base
        # default (False from ``mode != "i2v"``, since arch declares "both") so
        # the base's 36-channel ``get_lora_targets`` i2v branch never engages —
        # TI2V-5B has no image cross-attention projections at all (no
        # ``added_kv_proj_dim`` in the real checkpoint config). The per-step
        # flag lives here instead:
        self._i2v_active: bool = False

        arch = getattr(definition, "architecture_params", {}) or {}
        patch = arch.get("transformer.patch_size", [1, 2, 2])
        # Spatial patch factors (H, W) — the transformer's Conv3d patch_embedding
        # downsamples the latent grid by these before the sequence is flattened;
        # the per-token i2v timestep must match that grid exactly.
        self._patch_hw: tuple[int, int] = (
            (int(patch[1]), int(patch[2])) if len(patch) == 3 else (2, 2)
        )

    # ── i2v engagement gate (per-step flag ANDed with the F>1 still guard) ──

    def _conditioning_engaged(self, latents: torch.Tensor) -> bool:
        if not self._i2v_active:
            return False
        lat = latents if latents.ndim == 5 else latents.unsqueeze(2)
        return lat.shape[2] > 1

    # ── Flow-match noising: frame-0 scale pinned to 0 (== clean) when engaged ──

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Flow-match lerp; frame 0 gets scale 0 (stays clean) when i2v-engaged.

        Non-engaged (t2v, or an F=1 still on an i2v-active step) is
        byte-identical to the base scalar lerp.
        """
        if not self._conditioning_engaged(latents):
            return super().add_noise(latents, noise, timesteps)

        b = latents.shape[0]
        f = latents.shape[2]
        frac = (timesteps / 1000.0).reshape(b, 1, 1, 1, 1).to(latents.dtype)
        frac = frac.expand(b, 1, f, 1, 1).clone()
        frac[:, :, 0] = 0.0  # frame 0: noisy == latents (the "condition")
        return frac * noise + (1.0 - frac) * latents

    # ── Forward: 48ch input unchanged; per-token timestep only when engaged ──

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """WAN transformer forward over the RAW 48-channel latent (no concat).

        Engaged i2v steps swap the scalar ``[B]`` timestep for a per-token
        ``[B, seq_len]`` tensor (zero on frame-0 tokens, ``t`` elsewhere) —
        ``WanTransformer3DModel.forward`` branches on ``timestep.ndim`` (the
        diffusers ``expand_timesteps`` contract). ``encoder_hidden_states_image``
        stays ``None`` unconditionally — TI2V-5B has no CLIP image encoder
        (``image_dim: null``).
        """
        enc_hs = self._as_text_tensor(text_embeddings)

        if self._conditioning_engaged(noisy_input):
            ts_arg = self._per_token_timestep(timesteps, noisy_input)
        else:
            ts_arg = timesteps

        output = self.transformer(
            hidden_states=noisy_input,
            timestep=ts_arg,
            encoder_hidden_states=enc_hs,
            encoder_hidden_states_image=None,
            return_dict=False,
        )
        return output[0] if isinstance(output, tuple) else output

    def _per_token_timestep(
        self, timesteps: torch.Tensor, latents: torch.Tensor
    ) -> torch.Tensor:
        """``[B]`` scalar → ``[B, F * gh * gw]`` with frame-0 tokens forced to 0.

        ``(gh, gw)`` is the patch-downsampled spatial grid — matches
        ``diffusers``' ``mask[:, :, ::2, ::2]`` subsample (the mask is spatially
        uniform per frame, so subsampling never changes its values, only the
        token count) — and the temporal patch is 1 (frame count unchanged).
        """
        b, _, f, h, w = latents.shape
        ph, pw = self._patch_hw
        gh, gw = h // ph, w // pw
        grid = timesteps.reshape(b, 1, 1, 1).to(latents.dtype).expand(b, f, gh, gw).clone()
        grid[:, 0] = 0.0
        return grid.flatten(1)

    # ── Saver / topology ─────────────────────────────────────────────────

    def get_saver(self) -> Any:
        from app.engine.models.families.wan22_ti2v_5b.saver import Wan22Ti2v5bSaver

        return Wan22Ti2v5bSaver()

    def get_block_topology(self) -> list[dict[str, Any]]:
        """WAN 2.2 TI2V-5B block topology: a single stack of ``blocks``."""
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
                        "approx_vram_mb": 160,
                    }
                )
        return topology
