"""WAN 2.2 driver — dual-expert MoE on top of the shared :class:`WanDriverBase`.

WAN 2.2 A14B holds TWO transformers (experts):

- ``transformer_high`` — the **high-noise** expert (diffusers ``transformer``),
  active when the sampled timestep ``t >= boundary``.
- ``transformer_low``  — the **low-noise** expert (diffusers ``transformer_2``),
  active when ``t < boundary``.

In a single run the active expert is switched **per optimizer step** by an
:class:`ExpertRouter` (routed by the configured timestep distribution). The
generic training loop is expert-agnostic: it always operates on
``get_primary_model()``, which this driver makes return the *active* expert — so
``.train()``, grad-clip, gradient-checkpointing, optimizer step, and the
forward/loss all hit the right transformer. Timesteps for the step come from the
router, truncated to the active expert's range (keeping the marginal timestep
distribution unbiased — see :mod:`expert_router`).

VRAM modes (``expert_swap_mode``):

- ``resident`` — both experts on GPU (fastest; needs ~2× transformer VRAM).
- ``swap``     — only the active expert on GPU; the inactive one sits on CPU
  (pinned) and is swapped in on switch (``active.to('cpu'); empty_cache();
  next.to(device)``). Hysteresis (``switch_interval``) amortizes the PCIe cost.
- ``auto``     — ``resident`` if both experts + headroom fit, else ``swap``.

The actual VRAM behavior is GPU-only; the swap helper + state transitions are
unit-tested with fakes (a recorder ``.to()``). WAN 2.2 I2V uses the wan_shared
36-channel first-frame conditioning WITHOUT a CLIP image embed
(``encoder_hidden_states_image=None``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.engine.models.families.wan_shared.driver_base import WanDriverBase
from app.engine.models.families.wan22.expert_router import HIGH, LOW, ExpertRouter

# Default expert boundaries (timestep fraction in [0, 1]). T2V≈0.875, I2V≈0.9.
DEFAULT_BOUNDARY_T2V = 0.875
DEFAULT_BOUNDARY_I2V = 0.9


class Wan22Driver(WanDriverBase):
    """WAN 2.2 (T2V-A14B / I2V-A14B) dual-expert driver."""

    def __init__(self, definition: Any, device: torch.device) -> None:
        super().__init__(definition, device)
        self.transformer_high: nn.Module | None = None
        self.transformer_low: nn.Module | None = None
        self._active_expert: str = HIGH
        self.router: ExpertRouter | None = None
        # "both" (dual) | "high" | "low" — set by the trainer before loading.
        self.expert_mode: str = "both"

        arch = getattr(definition, "architecture_params", {}) or {}
        default_boundary = DEFAULT_BOUNDARY_I2V if self.is_i2v else DEFAULT_BOUNDARY_T2V
        self.boundary: float = float(arch.get("moe.boundary_ratio", default_boundary))
        self.swap_mode: str = "resident"

    # ── Component wiring ──────────────────────────────────────────────────

    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire the expert(s) per ``expert_mode``.

        - ``both``: ``unet`` = high expert, ``unet_low`` = low expert (active high).
        - ``high``: ``unet`` = high expert, low = ``None`` (active high).
        - ``low``: the loader put ``transformer_2/`` under ``unet`` → that IS the
          low expert; high = ``None`` (active low). The single loaded transformer
          is the only one resident, halving VRAM.
        """
        super().assign_components(components)
        # super() set self.transformer = components["unet"] (the loaded primary).
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
            "wan22_experts_assigned",
            expert_mode=self.expert_mode,
            has_high=self.transformer_high is not None,
            has_low=self.transformer_low is not None,
            active=self._active_expert,
            boundary=self.boundary,
        )

    def configure_expert_mode(self, mode: str) -> None:
        """Set ``expert_mode`` (``both``/``high``/``low``) before loading."""
        self.expert_mode = str(mode or "both").lower()

    def _expert_model(self, expert: str) -> nn.Module | None:
        return self.transformer_high if expert == HIGH else self.transformer_low

    def _set_active(self, expert: str) -> None:
        """Point ``self.transformer`` (the primary model) at ``expert``.

        Placement safety net: in ``resident`` mode BOTH experts are supposed to
        already be on ``self.device`` (via :meth:`place_experts_for_start`), but
        anything that reaches this method without that having run first (a
        future call site, a test, a code path we haven't audited) would
        otherwise hand back a CPU-resident expert as the new primary model —
        the first forward/backward on it raises a device-mismatch
        ``RuntimeError`` deep in the loop instead of failing where the mistake
        was made. One cheap parameter-device check closes that gap.
        """
        self._active_expert = expert
        model = self._expert_model(expert)
        if model is not None and self.swap_mode == "resident":
            p = next(model.parameters(), None)
            if p is not None and p.device != self.device:
                model.to(self.device)
        self.transformer = model

    @property
    def active_expert(self) -> str:
        return self._active_expert

    def get_primary_model(self) -> nn.Module:
        """Return the ACTIVE expert so the generic loop targets the right model."""
        return self.transformer

    # ── Router wiring ─────────────────────────────────────────────────────

    def set_router(self, router: ExpertRouter) -> None:
        """Attach the :class:`ExpertRouter` and sync the active expert."""
        self.router = router
        self._set_active(router.choose_expert(0))

    def configure_swap_mode(self, mode: str) -> None:
        """Set ``expert_swap_mode`` (``auto``/``swap``/``resident``).

        ``auto`` was documented as a resident-vs-swap VRAM probe, but no such
        probe exists anywhere in this driver — it has always silently resolved
        to ``resident`` with no signal that the "auto" choice was never
        actually evaluated. Rather than keep lying, resolve it explicitly and
        say so once.
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

    # ── Timestep sampling (delegates to the router for the active expert) ─

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        config: dict[str, Any],
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample timesteps for the ACTIVE expert (truncated to its range).

        Falls back to the base flow-match sampler if no router is attached
        (e.g. before training setup).
        """
        if self.router is None:
            return super().sample_timesteps(batch_size, device, config, latents)
        return self.router.sample_timesteps_for(
            self._active_expert, batch_size, device, config, latents
        )

    # ── Optimizer-step hook (advance the router, swap for the NEXT step) ──

    def on_optimizer_step(self, optimizer_step: int) -> None:
        """Advance the router and set the active expert for the NEXT step.

        Called by the training loop right after ``optimizer.step()`` /
        ``zero_grad`` for the just-completed step ``optimizer_step``. We choose
        the expert for ``optimizer_step + 1`` and, in ``swap`` mode, migrate the
        experts across the CPU/GPU boundary if the expert changed.
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

    # ── VRAM swap helper (state transitions unit-tested with fakes) ───────

    def _swap_to(self, expert: str) -> None:
        """Move ``expert`` onto ``self.device`` and the other onto CPU.

        Order: offload the currently-active expert to CPU, free the cache, then
        bring the target expert onto the device. ``set active`` last so the
        primary-model pointer only flips once the target is resident.
        """
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
        self.logger.info("wan22_expert_swapped", active=expert, mode=self.swap_mode)

    def place_experts_for_start(self) -> None:
        """Place experts on devices per ``swap_mode`` before the loop starts.

        - ``resident``/``auto`` → both experts on ``self.device``.
        - ``swap`` → active expert on device, inactive on CPU.

        The ``auto`` resident-vs-swap decision is a GPU-memory probe that only
        makes sense on CUDA; off-GPU (tests) it behaves as ``resident``.
        """
        high, low = self.transformer_high, self.transformer_low
        if self.swap_mode == "swap":
            active = self._expert_model(self._active_expert)
            inactive = (
                self.transformer_low
                if self._active_expert == HIGH
                else self.transformer_high
            )
            if active is not None:
                active.to(self.device)
            if inactive is not None:
                inactive.to("cpu")
            return
        # resident / auto → both resident.
        for m in (high, low):
            if m is not None:
                m.to(self.device)

    # ── Saver ─────────────────────────────────────────────────────────────

    def get_saver(self) -> Any:
        from app.engine.models.families.wan22.saver import Wan22Saver

        return Wan22Saver(mode=self.mode)

    def get_block_topology(self) -> list[dict[str, Any]]:
        """Block topology of the ACTIVE expert (single ``blocks`` stack)."""
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
