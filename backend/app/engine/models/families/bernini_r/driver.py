"""Bernini-R driver — subclasses the shared :class:`WanDriverBase`.

Bernini-R reuses all of the shared Wan flow-match training behaviour (raw
``[0,1000]`` timestep ``add_noise``, UMT5 encoding, 5D ``prepare_latents``, the
wan-canonical LoRA target set). The ONE family specific is the forward path: it
runs the vendored **packed** forward (``vendor/transformer_forward.py``) that
token-concatenates the clean condition-video latents with the noisy target and
reads the velocity back for the target tokens only, rather than the stock
channel-wise Wan forward.

Single expert (1.3B) — byte-identical v1 path
---------------------------------------------
v2v only: one condition stream at ``source_id=1``, target at ``source_id=0``.
``assign_components`` wires ``self.transformer`` exactly as :class:`WanDriverBase`
does; nothing dual-expert engages (``is_dual`` is False).

Dual expert (14B MoE)
---------------------
Mirrors :class:`Wan22Driver`: the high-noise expert (``transformer`` →
``transformer_high``) serves ``t >= boundary·1000`` and the low-noise expert
(``transformer_2`` → ``transformer_low``) serves ``t < boundary`` (recon §3;
boundary = ``switch_dit_boundary`` = 0.875). The active expert is chosen per
optimizer step by a reused :class:`wan22.ExpertRouter` (see the trainer);
``get_primary_model`` returns the active expert (via ``self.transformer``), so
the generic loop + the packed ``forward_pass`` always run the right transformer.
Timesteps for the step are BAND-sampled to the active expert's serving range by
the trainer (the deliberate range-split divergence — see ``trainer.py``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.engine.models.families.bernini_r.vendor.transformer_forward import (
    bernini_packed_forward,
)
from app.engine.models.families.wan22.expert_router import HIGH, LOW, ExpertRouter
from app.engine.models.families.wan_shared.driver_base import WanDriverBase

# Default MoE boundary (timestep fraction in [0, 1]). Bernini-R 14B = 0.875
# (upstream ``switch_dit_boundary``), identical to wan2.2 T2V-A14B.
DEFAULT_BOUNDARY = 0.875


class BerniniRDriver(WanDriverBase):
    """Bernini-R family driver (renderer-only video edit; 1.3B single / 14B MoE)."""

    # Clean control-video latents, one 5D tensor per ordered condition slot,
    # attached by the training data pipeline (``pipeline_data._load_control_latents``).
    BATCH_CONTROL_LATENTS = "control_latents"

    def __init__(self, definition: Any, device: torch.device) -> None:
        super().__init__(definition, device)
        arch = getattr(definition, "architecture_params", {}) or {}
        # Dual-expert (14B) vs single-expert (1.3B). The 1.3B path never touches
        # any of the expert bookkeeping below, staying byte-identical to v1.
        self.is_dual: bool = bool(arch.get("dual_expert", False))
        # Boundary as a [0,1] fraction (× num_train_timesteps for the raw split).
        self.boundary: float = float(
            arch.get(
                "switch_dit_boundary",
                arch.get("moe.boundary_ratio", DEFAULT_BOUNDARY),
            )
        )
        self.boundary_timestep: float = self.boundary * float(
            arch.get("scheduler.num_train_timesteps", 1000)
        )
        self.transformer_high: nn.Module | None = None
        self.transformer_low: nn.Module | None = None
        self._active_expert: str = HIGH
        self.router: ExpertRouter | None = None
        # "both" (dual) | "high" | "low" — set by the trainer before loading.
        self.expert_mode: str = "both"
        self.swap_mode: str = "resident"
        # Name (HIGH/LOW) of the expert whose deep blocks are under ACTIVE
        # ``BlockSwappingManager`` management, or ``None`` (mirrors
        # :class:`Wan22Driver`). Set by the trainer once it knows (see
        # ``BerniniRTrainer._configure_optimization``) — the driver has no
        # visibility into ``self._block_swap_managers`` (lives on the
        # pipeline/trainer). Read by :meth:`place_experts_for_start` and
        # :meth:`_set_active`.
        self.block_swap_active_expert: str | None = None

    # ── Component wiring ──────────────────────────────────────────────────

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded components.

        Single-expert (1.3B): defer to :class:`WanDriverBase` verbatim (sets
        ``self.transformer`` from ``unet``). Dual-expert (14B): additionally wire
        the high/low expert slots per ``expert_mode`` (mirrors :class:`Wan22Driver`).
        """
        super().assign_components(components)
        if not self.is_dual:
            return
        if self.expert_mode == "low":
            self.transformer_low = components.get("unet")
            self.transformer_high = None
            self._set_active(LOW)
        elif self.expert_mode == "high":
            self.transformer_high = components.get("unet")
            self.transformer_low = None
            self._set_active(HIGH)
        else:  # both
            self.transformer_high = components.get("unet")
            self.transformer_low = components.get("unet_low")
            self._set_active(HIGH)
        self.logger.info(
            "bernini_r_experts_assigned",
            expert_mode=self.expert_mode,
            has_high=self.transformer_high is not None,
            has_low=self.transformer_low is not None,
            active=self._active_expert,
            boundary=self.boundary,
        )

    def configure_expert_mode(self, mode: str) -> None:
        """Set ``expert_mode`` (``both``/``high``/``low``) before loading."""
        self.expert_mode = str(mode or "both").lower()

    def configure_swap_mode(self, mode: str) -> None:
        """Set ``expert_swap_mode`` (``auto``/``swap``/``resident``).

        ``auto`` was documented as a resident-vs-swap VRAM probe, but no such
        probe exists anywhere in this driver — it has always silently resolved
        to ``resident`` with no signal that the "auto" choice was never
        actually evaluated. Rather than keep lying, resolve it explicitly and
        say so once (mirrors :class:`Wan22Driver`).
        """
        resolved = str(mode or "resident").lower()
        if resolved == "auto":
            self.logger.warning(
                "expert_swap_auto_unimplemented",
                message=(
                    "expert_swap_mode='auto' has no resident-vs-swap VRAM "
                    "probe implemented; resolving to 'resident' (both experts "
                    "on GPU)."
                ),
            )
            resolved = "resident"
        self.swap_mode = resolved

    def _expert_model(self, expert: str) -> nn.Module | None:
        return self.transformer_high if expert == HIGH else self.transformer_low

    def _set_active(self, expert: str) -> None:
        """Point ``self.transformer`` (the primary model) at ``expert``.

        Placement safety net (mirrors :class:`Wan22Driver._set_active`): in
        ``resident`` mode both experts should already be on ``self.device``
        via :meth:`place_experts_for_start`; this catches anything that
        reaches here without that having run first, before it turns into a
        device-mismatch ``RuntimeError`` mid-forward.

        Skipped when ``expert`` is under active block-swap management
        (``self.block_swap_active_expert``) — its blocks are cycled between
        pinned CPU shadow buffers and GPU by ``BlockSwappingManager``'s
        forward hooks, so a bulk ``model.to(device)`` here would defeat the
        swap (mirrors :class:`Wan22Driver._set_active`).
        """
        self._active_expert = expert
        model = self._expert_model(expert)
        if model is not None and self.swap_mode == "resident":
            if expert == self.block_swap_active_expert:
                self.logger.info(
                    "expert_block_swap_placement_skipped",
                    expert=expert,
                    call_site="_set_active",
                )
            else:
                p = next(model.parameters(), None)
                if p is not None and p.device != self.device:
                    model.to(self.device)
        self.transformer = model

    @property
    def active_expert(self) -> str:
        return self._active_expert

    def expert_for_timestep(self, timestep: Any) -> str:
        """Route a RAW ``[0,1000]`` timestep to its serving expert.

        ``t >= boundary·1000`` → ``HIGH`` (``transformer``, noisy steps);
        ``t < boundary·1000`` → ``LOW`` (``transformer_2``). Pure function — the
        mutation test drives a timestep on each side of the boundary through it
        (recon §3: ``model_id = "transformer_1" if t >= boundary else
        "transformer_2"``).
        """
        t = timestep
        if isinstance(t, torch.Tensor):
            t = float(t.reshape(-1)[0].item()) if t.numel() else 0.0
        return HIGH if float(t) >= self.boundary_timestep else LOW

    def transformer_for_timestep(self, timestep: Any) -> nn.Module | None:
        """The expert MODULE that serves ``timestep`` (dual only)."""
        return self._expert_model(self.expert_for_timestep(timestep))

    # ── Router wiring (reuses wan22's ExpertRouter) ───────────────────────

    def set_router(self, router: ExpertRouter) -> None:
        """Attach the :class:`ExpertRouter` and sync the active expert."""
        self.router = router
        self._set_active(router.choose_expert(0))

    def on_optimizer_step(self, optimizer_step: int) -> None:
        """Advance the router and set the active expert for the NEXT step.

        Called by the generic loop right after ``optimizer.step()`` for the
        just-completed ``optimizer_step``. In ``swap`` mode the experts migrate
        across the CPU/GPU boundary on a change; otherwise the pointer flips.
        """
        if self.router is None:
            return
        next_expert = self.router.choose_expert(optimizer_step + 1)
        if next_expert == self._active_expert:
            return
        if self.swap_mode == "swap":
            self._swap_to(next_expert)
        else:
            self._set_active(next_expert)

    def _swap_to(self, expert: str) -> None:
        """Move ``expert`` onto ``self.device`` and the other onto CPU."""
        target = self._expert_model(expert)
        current = self._expert_model(self._active_expert)
        if target is None:
            self._set_active(expert)
            return
        if current is not None and current is not target:
            current.to("cpu")
        if torch.cuda.is_available():  # pragma: no cover - GPU only
            torch.cuda.empty_cache()
        target.to(self.device)
        self._set_active(expert)
        self.logger.info("bernini_r_expert_swapped", active=expert, mode=self.swap_mode)

    def place_experts_for_start(self) -> None:
        """Place experts on devices per ``swap_mode`` before the loop starts.

        Mirrors :class:`Wan22Driver.place_experts_for_start`: in
        ``resident``/``auto`` mode, an expert under active block-swap
        management (``self.block_swap_active_expert``) is left alone —
        ``BlockSwappingManager``'s forward hooks own its placement, and a
        bulk ``.to(device)`` would defeat the swap. The other expert is
        still placed, preserving the original fix (first router flip must
        not land on a CPU-resident model).
        """
        if not self.is_dual:
            return
        high, low = self.transformer_high, self.transformer_low
        if self.swap_mode == "swap":
            active = self._expert_model(self._active_expert)
            inactive = low if self._active_expert == HIGH else high
            if active is not None:
                active.to(self.device)
            if inactive is not None:
                inactive.to("cpu")
            return
        for name, m in ((HIGH, high), (LOW, low)):
            if m is None:
                continue
            if name == self.block_swap_active_expert:
                self.logger.info(
                    "expert_block_swap_placement_skipped",
                    expert=name,
                    call_site="place_experts_for_start",
                )
                continue
            m.to(self.device)

    def get_primary_model(self) -> nn.Module:
        """Return the ACTIVE model. For 1.3B this is the single transformer; for
        14B it is whichever expert ``_set_active`` currently points at."""
        return self.transformer

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

        Batches are bucket-uniform and this packed forward applies ONE
        condition-slot list to the whole batch, so every item in a batch must
        carry the same control-slot structure — mixed edit/plain items in a
        SINGLE batch are not supported. A batch with no condition latents at
        all degenerates to a stock Wan t2v forward (``source_id=0``, single
        stream). On 14B, ``self.transformer`` is the router-selected active
        expert (the batch's timesteps are band-sampled to that expert's range).

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
        """Bernini-R saver.

        Dual-expert (14B) → :class:`BerniniRDualSaver` (two files, wan22 high/low
        naming, ``bernini-r`` provenance). Single-expert (1.3B) → the single-file
        :class:`BerniniRSaver` (wan-canonical ``diffusion_model.*`` keys,
        byte-equal to wan21's export).
        """
        if self.is_dual:
            from app.engine.models.families.bernini_r.saver import BerniniRDualSaver

            return BerniniRDualSaver(mode="t2v")
        from app.engine.models.families.bernini_r.saver import BerniniRSaver

        return BerniniRSaver(mode="t2v")

    def get_block_topology(self) -> list[dict[str, Any]]:
        """Single stack of ``blocks`` on the ACTIVE model (mirrors wan21/wan22)."""
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
