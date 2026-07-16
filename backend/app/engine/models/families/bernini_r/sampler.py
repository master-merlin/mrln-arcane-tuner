"""Bernini-R v2v sampler — packed-token conditioning + UniPC flow denoise.

Subclasses :class:`WanVideoSamplerBase` for the Wan-VAE decode, UMT5 cached
prompt encoding, 5D noise creation, and the video ``SampleArtifact`` output; the
ONE delta is the denoise loop, which:

- VAE-encodes the control video(s) to CLEAN condition latents ONCE, before the
  loop, and reuses those exact tensors (by reference) every step — the condition
  tokens re-enter frozen while only the noisy target latent is stepped (upstream
  ``GEN_Wanx22.sample()``: condition tokens re-enter frozen every step, only
  ``noisy_vae_latent`` is stepped). The packed sequence is assembled through the
  vendored :func:`bernini_packed_forward`, which patch-embeds each stream with
  its ``source_id`` rope; a fixed condition latent → deterministic condition
  tokens, so they are bit-identical across steps.
- steps the target latent with the native diffusers
  ``UniPCMultistepScheduler(prediction_type="flow_prediction",
  use_flow_sigmas=True, flow_shift=…)`` — upstream ships ``use_unipc: true`` in
  every config; the model output is the flow-match velocity (``noise - x0``),
  which is exactly the ``flow_prediction`` model output UniPC expects.

House precision contract: the trajectory is fp32 (the scheduler steps the fp32
target latent, and the velocity is upcast to fp32 before the step). The DiT
forward runs in the SAME autocast regime as training — the Wan transformer is a
mixed-dtype module (bf16 weights + fp32 ``scale_shift_table``/norms), so a single
input cast cannot satisfy every layer; per-op autocast handles it while the
trajectory accumulation stays fp32 OUTSIDE the autocast (the
autocast-sampler-collapse guard). Mirrors the wan21/wan22 sampler dtype regime.

──────────────────────────────────────────────────────────────────────────────
CFG VARIANT #5 (v2v) — DO NOT COPY the sibling ``_combine_cfg`` implementations.
──────────────────────────────────────────────────────────────────────────────
Pinned to upstream ``bernini/models/wan_diffusion.py`` ``GEN_Wanx22.sample()``
``guidance_mode == "v2v"`` (recon §5): the **condition video stays in the
UNCOND branch** — only the TEXT is swapped, and the negative text is the EMPTY
UMT5 embedding (``""``), NOT the long Chinese Wan negative prompt::

    eps_uncond = fwd(cond_video, EMPTY_text)      # cond video present
    eps_cond   = fwd(cond_video, real_text)       # cond video present
    pred       = eps_uncond + omega_txt * (eps_cond - eps_uncond)   # omega_txt=4.0

FOUR sibling families each pin a DIFFERENT CFG regime to THEIR own upstream
(chroma / lumina2 / nucleus / ace ``_combine_cfg``). They coexist deliberately;
copying any of them here would break the v2v contract. This one is written
against Bernini's ``GEN_Wanx22.sample()`` and pinned by
``test_bernini_r_sampler.py``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from app.engine.models.families.bernini_r.vendor.transformer_forward import (
    bernini_packed_forward,
)
from app.engine.models.families.wan_shared.sampler_base import WanVideoSamplerBase
from app.engine.models.families.wan_shared.vae_utils import (
    WAN_VAE_SPATIAL,
    WAN_VAE_TEMPORAL,
    normalize_wan_latents,
)

# UniPC ``flow_shift`` fallback — v2v ships 5.0 (matches the v2v training shift);
# the definition pins it via ``scheduler.flow_shift`` + ``enrich_pinned_keys``.
BERNINI_DEFAULT_FLOW_SHIFT: float = 5.0


class BerniniRSampler(WanVideoSamplerBase):
    """Bernini-R (renderer-only video edit, 1.3B v1) in-training v2v sampler."""

    # ── Control-video resolution ─────────────────────────────────────────

    def _resolve_control_paths(self) -> list[str]:
        """Control VIDEO path(s) for this preview.

        Read off ``SamplePromptConfig.control_images`` (stashed on
        ``self._active_prompt_cfg`` by the base ``_sample_single``). For this
        family the field holds video file path(s). No control ⇒ ``[]`` ⇒ the
        degenerate t2v path (see :meth:`denoise`).
        """
        cfg = getattr(self, "_active_prompt_cfg", None) or {}
        paths = (
            cfg.get("control_images")
            if isinstance(cfg, dict)
            else getattr(cfg, "control_images", None)
        )
        return [p for p in (paths or []) if p]

    def _encode_control_video(self, path: str, target: Tensor) -> Tensor:
        """VAE-encode one control video to a CLEAN normalized latent stream.

        The control clip is decoded to the TARGET's pixel dims and frame count
        (derived from the target latent's shape, same Wan-VAE 8×/4× compression)
        so its latent grid lines up with the target's. The VAE encode is
        bracketed with the phased-GPU management (``_ensure_on_gpu(["vae"])`` /
        ``_offload_to_cpu``) — step-0 baseline previews run right after
        pre-caching offloads the VAE to CPU, so an unmanaged encode would feed
        CUDA pixels to CPU weights (qwen step-0 fix precedent, GPU UAT
        2026-07-14).
        """
        from app.engine.components.video import VideoFrameLoader

        _, _, latent_f, lat_h, lat_w = target.shape
        num_frames = (int(latent_f) - 1) * WAN_VAE_TEMPORAL + 1
        width = int(lat_w) * WAN_VAE_SPATIAL
        height = int(lat_h) * WAN_VAE_SPATIAL

        clip = VideoFrameLoader().load_clip(
            path,
            target_frames=num_frames,
            target_fps=float(self.output_fps),
            trim_start_s=0.0,
            trim_end_s=None,
            target_w=width,
            target_h=height,
            h_flip=False,
        )  # [3, F, H, W] in [-1, 1]
        pixels = clip.unsqueeze(0)  # [1, 3, F, H, W]

        vae = self.pipeline.driver.vae
        vae_moved = self._ensure_on_gpu(["vae"])
        vae_dtype = next(vae.parameters()).dtype
        with torch.no_grad():
            posterior = vae.encode(pixels.to(self.device, dtype=vae_dtype))
        latent = (
            posterior.latent_dist.mode()
            if hasattr(posterior, "latent_dist")
            else posterior.sample()
        )
        # Transformer latent space is (z - mean) / std (per-channel Wan-VAE stats).
        latent = normalize_wan_latents(latent, vae)
        self._offload_to_cpu(vae_moved)
        return latent

    def _build_condition_streams(
        self, target: Tensor
    ) -> tuple[list[Tensor], list[float]]:
        """Encode the control video(s) ONCE → (cond_latents, cond_source_ids).

        Ordered streams map to ``source_id = slot + 1`` (target implicitly 0),
        matching the driver's training-side assignment. The returned latents are
        the exact tensors reused (by reference) every denoise step — this is what
        keeps the condition tokens frozen across the trajectory.
        """
        cond_latents: list[Tensor] = []
        cond_source_ids: list[float] = []
        for slot_idx, path in enumerate(self._resolve_control_paths()):
            lat = self._encode_control_video(path, target)
            cond_latents.append(lat.to(device=target.device, dtype=target.dtype))
            cond_source_ids.append(float(slot_idx + 1))
        return cond_latents, cond_source_ids

    # ── Scheduler ────────────────────────────────────────────────────────

    def _flow_shift(self) -> float:
        """UniPC ``flow_shift`` from the definition (``scheduler.flow_shift``).

        v2v ships 5.0 (matches the v2v training shift); the HF ``scheduler_config``
        ships 3.0 (the t2v value), so the definition pins 5.0 via
        ``enrich_pinned_keys``. Falls back to the v2v default if unset.
        """
        defn = getattr(self.pipeline, "definition", None)
        arch = getattr(defn, "architecture_params", {}) or {}
        return float(arch.get("scheduler.flow_shift", BERNINI_DEFAULT_FLOW_SHIFT))

    def _build_scheduler(self):
        """A fresh native ``UniPCMultistepScheduler`` (flow-prediction).

        Built fresh per denoise so the multistep solver's state (step index,
        cached model outputs) never leaks between sampling rounds. Kwargs
        verified against the local diffusers 0.39 source: ``prediction_type``,
        ``use_flow_sigmas``, ``flow_shift`` are all ``__init__`` args, and
        ``use_flow_sigmas=True`` is what makes ``flow_prediction`` /
        ``x0 = sample - sigma * v`` the model-output conversion.
        """
        from diffusers import UniPCMultistepScheduler

        return UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=self._flow_shift(),
        )

    # ── Expert selection (14B MoE boundary switch) ───────────────────────

    def _select_expert(self, timestep: Any) -> Any:
        """The transformer that serves this step.

        Dual-expert (14B): route the RAW ``[0,1000]`` timestep to its expert
        (``t >= boundary·1000`` → high ``transformer``; ``t < boundary`` → low
        ``transformer_2``), so a descending UniPC schedule switches experts at the
        boundary crossing (recon §3). Single-expert (1.3B) / any driver without a
        low expert: the single primary model (byte-identical to v1).
        """
        driver = self.pipeline.driver
        if (
            getattr(driver, "is_dual", False)
            and getattr(driver, "transformer_low", None) is not None
        ):
            return driver.transformer_for_timestep(timestep)
        return driver.get_primary_model()

    # ── Forward + CFG ────────────────────────────────────────────────────

    def _packed_forward(
        self,
        transformer: Any,
        cond_latents: list[Tensor],
        cond_source_ids: list[float],
        target: Tensor,
        timestep: Tensor,
        text: Tensor,
        autocast_dtype: torch.dtype,
        device_type: str,
    ) -> Tensor:
        """One vendored packed forward under the training autocast regime.

        The DiT forward runs inside ``autocast`` (mixed-dtype Wan module); the
        result is returned in whatever dtype the forward produced and upcast to
        fp32 by :meth:`_cfg_velocity` before it touches the fp32 trajectory.
        """
        with (
            torch.no_grad(),
            torch.autocast(device_type=device_type, dtype=autocast_dtype),
        ):
            out = bernini_packed_forward(
                transformer,
                cond_latents=cond_latents,
                cond_source_ids=cond_source_ids,
                target_latent=target,
                timestep=timestep,
                encoder_hidden_states=text,
                return_dict=False,
            )
        return out[0] if isinstance(out, tuple) else out

    def _cfg_velocity(
        self,
        transformer: Any,
        cond_latents: list[Tensor],
        cond_source_ids: list[float],
        target: Tensor,
        timestep: Tensor,
        text_cond: Tensor,
        text_uncond: Tensor | None,
        guidance_scale: float,
        autocast_dtype: torch.dtype,
        device_type: str,
    ) -> Tensor:
        """CFG VARIANT #5 (v2v): condition video in BOTH branches; text swaps.

        ``pred = eps_uncond + omega_txt * (eps_cond - eps_uncond)`` with
        ``omega_txt = guidance_scale`` — the SAME ``cond_latents`` (source video)
        ride in the uncond forward, only the text embedding is swapped for the
        empty UMT5 embedding. Combined in fp32 (the fp32-accumulation contract).
        ``text_uncond is None`` ⇒ CFG off ⇒ the single conditional velocity.

        DO NOT replace this with a copied chroma/lumina2/nucleus/ace combine —
        each of those pins a different upstream CFG regime (see module docstring).
        """
        v_cond = self._packed_forward(
            transformer,
            cond_latents,
            cond_source_ids,
            target,
            timestep,
            text_cond,
            autocast_dtype,
            device_type,
        ).to(torch.float32)
        if text_uncond is None:
            return v_cond
        v_uncond = self._packed_forward(
            transformer,
            cond_latents,
            cond_source_ids,
            target,
            timestep,
            text_uncond,
            autocast_dtype,
            device_type,
        ).to(torch.float32)
        return v_uncond + float(guidance_scale) * (v_cond - v_uncond)

    # ── Denoise loop ─────────────────────────────────────────────────────

    def denoise(
        self,
        noise: Tensor,
        prompt_embedding: Any,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Any:
        """v2v UniPC denoise with frozen condition tokens + CFG variant #5.

        The condition latents are VAE-encoded ONCE before the loop and reused by
        reference every step (frozen); the UniPC scheduler steps ONLY the target
        latent. With no control video the streams are empty ⇒ the vendored packed
        forward runs its degenerate stock-t2v path (source_id=0, single stream),
        so a preview never crashes for want of a control input.
        """
        driver = self.pipeline.driver
        self._ensure_transformer_on_device(driver.get_primary_model())
        # Dual-expert (14B): both experts must be resident for the boundary
        # switch (a descending schedule crosses the boundary once). Idempotent.
        if getattr(driver, "is_dual", False):
            for m in (
                getattr(driver, "transformer_high", None),
                getattr(driver, "transformer_low", None),
            ):
                if m is not None:
                    self._ensure_transformer_on_device(m)

        autocast_dtype = (
            getattr(self.pipeline, "autocast_dtype", None) or torch.bfloat16
        )
        device_type = torch.device(self.device).type

        # fp32 target trajectory (house precision contract).
        target = noise.to(torch.float32)
        text_cond = prompt_embedding

        # CFG variant #5: the negative branch is the EMPTY UMT5 embedding (""),
        # never the wan Chinese negative. guidance_scale <= 1 keeps a single
        # conditional forward (byte-identical to CFG-off).
        gs = float(guidance_scale) if guidance_scale is not None else None
        cfg_on = gs is not None and gs > 1.0
        text_uncond = self.encode_prompt("") if cfg_on else None

        # Condition streams — VAE-encoded ONCE, reused (by reference) every step.
        cond_latents, cond_source_ids = self._build_condition_streams(target)

        scheduler = self._build_scheduler()
        scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = scheduler.timesteps

        total = len(timesteps)
        for i, t in enumerate(timesteps):
            if getattr(self, "_log_writer", None):
                self._log_writer.status(f"Sampling {i + 1}/{total}")
            # RAW [0, 1000] timestep (the scheduler's own value) shared by every
            # token, including the clean condition tokens (upstream t.expand(1)).
            ts = t.reshape(1).expand(target.shape[0]).to(torch.float32)
            # Boundary switch (dual) — the raw scheduler timestep ``t`` picks the
            # active expert; single-expert returns the one primary model.
            transformer = self._select_expert(t)
            velocity = self._cfg_velocity(
                transformer,
                cond_latents,
                cond_source_ids,
                target,
                ts,
                text_cond,
                text_uncond,
                gs if cfg_on else 1.0,
                autocast_dtype,
                device_type,
            )
            # UniPC steps ONLY the target latent (fp32 trajectory).
            target = scheduler.step(velocity, t, target, return_dict=False)[0]

        return target
