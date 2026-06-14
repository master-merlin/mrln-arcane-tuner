"""WAN 2.2 dual-expert router — single-run auto expert-switching.

WAN 2.2 A14B is a Mixture-of-Experts: a **high-noise expert** active for
``t >= boundary`` and a **low-noise expert** active for ``t < boundary`` (in the
``[0, 1000]`` timestep space; boundary is given as a fraction, e.g. T2V≈0.875 →
``875``). To train BOTH experts in a single run, each *optimizer* step picks ONE
expert and uses it for all of that step's gradient-accumulation micro-batches,
so gradients never mix experts within a step.

The router has two jobs:

1. **Choose the active expert per optimizer step** so that, over the run, the
   fraction of high-expert steps matches ``p_high = P(t >= boundary)`` under the
   *configured* timestep distribution. ``p_high`` is estimated ONCE at setup by
   Monte-Carlo sampling the real :class:`TimestepSampler` and cached. The draw
   is **quantized into runs of ``switch_interval`` optimizer steps** (hysteresis
   — useful when swapping experts across PCIe is slow): we pre-draw one
   Bernoulli outcome per ``switch_interval``-step block, so the long-run
   frequency still ≈ ``p_high`` while consecutive steps reuse the same expert.

2. **Sample timesteps for the active expert** by drawing from the SAME
   configured distribution but **truncated** to the expert's range via rejection
   sampling (resample until ``t >= boundary`` for high / ``t < boundary`` for
   low). Because step selection is Bernoulli(``p_high``) and each expert's
   timesteps are the distribution conditioned on its range, the *marginal*
   timestep distribution over the whole run is identical to single-model
   training — i.e. the routing is **unbiased**. (Law of total probability:
   ``p_high * f(t | t>=b) + (1-p_high) * f(t | t<b) = f(t)``.)

Everything here is weight-free and unit-tests with a seeded RNG + tiny tensors.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch

from app.engine.strategies.timestep_sampling import TimestepSampler

logger = structlog.get_logger(__name__)

HIGH = "high"
LOW = "low"

# Hard cap on rejection-sampling retries so a pathological distribution (almost
# no mass on one side of the boundary) can't spin forever. On exhaustion we
# clamp the last draw into the expert's range — a negligible bias vs. hanging.
_MAX_REJECTION_ROUNDS = 64


class ExpertRouter:
    """Routes optimizer steps + timesteps between the high/low WAN 2.2 experts.

    Args:
        boundary: Expert boundary as a FRACTION in ``[0, 1]`` (e.g. ``0.875``).
            Internally compared against timesteps in ``[0, 1000]`` (``boundary *
            1000``). ``t >= boundary`` → high expert; ``t < boundary`` → low.
        switch_interval: Number of consecutive optimizer steps that reuse one
            expert decision (hysteresis run length). ``1`` = re-draw every step.
        timestep_cfg: The training config dict (``timestep_sampling`` mode +
            mode params). Used both to estimate ``p_high`` and to sample the
            per-expert truncated timesteps from the SAME distribution.
        seed: RNG seed for deterministic step-selection draws (tests pin this).
        mc_samples: Monte-Carlo sample count for the ``p_high`` estimate.
        scale: Timestep scale (``1000`` for FlowMatchEuler).
    """

    def __init__(
        self,
        boundary: float,
        switch_interval: int = 1,
        timestep_cfg: dict[str, Any] | None = None,
        *,
        seed: int = 0,
        mc_samples: int = 100_000,
        scale: float = 1000.0,
    ) -> None:
        self.boundary_frac = float(boundary)
        self.scale = float(scale)
        self.boundary_scaled = self.boundary_frac * self.scale
        self.switch_interval = max(1, int(switch_interval))
        self.config: dict[str, Any] = dict(timestep_cfg or {})
        self.mode: str = str(self.config.get("timestep_sampling", "logit_normal"))
        self.mc_samples = int(mc_samples)
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Deterministic CPU generator for the Bernoulli step-selection draws.
        # Kept separate from the (config-driven) timestep sampling so resuming
        # only needs to persist/restore THIS generator's state.
        self._seed = int(seed)
        self._step_rng = torch.Generator(device="cpu").manual_seed(self._seed)

        # Estimate + cache p_high once.
        self.p_high: float = self._estimate_p_high()

        # Per-interval-block decision cache: block_index -> "high"|"low".
        self._block_decisions: dict[int, str] = {}
        # Active expert for the CURRENT optimizer step (advanced by the driver
        # hook). Initialized to the decision for block 0.
        self._active_expert: str = self._decide_block(0)

        self.logger.info(
            "expert_router_init",
            boundary_frac=self.boundary_frac,
            switch_interval=self.switch_interval,
            mode=self.mode,
            p_high=round(self.p_high, 5),
        )

    # ── p_high estimation (Monte-Carlo, cached once) ──────────────────────

    def _estimate_p_high(self) -> float:
        """Estimate ``P(t >= boundary)`` under the configured distribution.

        Draws ``mc_samples`` timesteps from the REAL :class:`TimestepSampler`
        (same mode + params the trainer uses) and returns the empirical fraction
        ``>= boundary``. Computed on CPU with a fixed generator so the estimate
        is reproducible.
        """
        gen = torch.Generator(device="cpu").manual_seed(self._seed + 1)
        # TimestepSampler reads torch's default RNG; seed it for reproducibility
        # of the estimate (does not touch the step-selection generator).
        prev_state = torch.random.get_rng_state()
        try:
            torch.random.manual_seed(self._seed + 1)
            samples = TimestepSampler.sample_scaled(
                self.mode,
                self.mc_samples,
                torch.device("cpu"),
                self.config,
                scale=self.scale,
            )
        finally:
            torch.random.set_rng_state(prev_state)
        del gen
        p = float((samples >= self.boundary_scaled).float().mean().item())
        # Guard against degenerate 0/1 (would make one expert never train).
        return min(max(p, 1e-6), 1.0 - 1e-6)

    # ── Step selection (quantized Bernoulli, deterministic) ───────────────

    def _decide_block(self, block_index: int) -> str:
        """Draw (and cache) the expert decision for an interval block.

        Each ``switch_interval``-step block gets ONE Bernoulli(``p_high``) draw;
        all steps in the block reuse it. Drawing per-block (not per-step) keeps
        the long-run high/low frequency ≈ ``p_high`` while giving hysteresis.
        """
        if block_index in self._block_decisions:
            return self._block_decisions[block_index]
        u = torch.rand((), generator=self._step_rng).item()
        decision = HIGH if u < self.p_high else LOW
        self._block_decisions[block_index] = decision
        return decision

    def choose_expert(self, optimizer_step: int) -> str:
        """Return the expert (``"high"``/``"low"``) for ``optimizer_step``.

        Deterministic given the seed: step ``s`` belongs to interval block
        ``s // switch_interval``, and every step in a block shares that block's
        single Bernoulli draw.
        """
        block_index = int(optimizer_step) // self.switch_interval
        decision = self._decide_block(block_index)
        self._active_expert = decision
        return decision

    @property
    def active_expert(self) -> str:
        """The expert chosen for the current optimizer step."""
        return self._active_expert

    # ── Per-expert truncated timestep sampling (unbiased marginal) ────────

    def sample_timesteps_for(
        self,
        expert: str,
        batch_size: int,
        device: torch.device,
        config: dict[str, Any],
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample ``[batch_size]`` timesteps in ``[0, scale]`` for ``expert``.

        Draws from the SAME configured distribution but truncated to the
        expert's range by **rejection sampling**:

        - ``high`` → ``t >= boundary``
        - ``low``  → ``t < boundary``

        Conditioning each expert's timesteps on its range while selecting steps
        with probability ``p_high`` keeps the run's MARGINAL timestep
        distribution identical to single-model training (the routing is
        unbiased — see the module docstring).
        """
        expert = expert.lower()
        if expert not in (HIGH, LOW):
            raise ValueError(f"unknown expert {expert!r} (expected 'high'/'low')")

        out = torch.empty(batch_size, device=device)
        filled = torch.zeros(batch_size, dtype=torch.bool, device=device)

        def _in_range(t: torch.Tensor) -> torch.Tensor:
            if expert == HIGH:
                return t >= self.boundary_scaled
            return t < self.boundary_scaled

        rounds = 0
        while not bool(filled.all()) and rounds < _MAX_REJECTION_ROUNDS:
            need = int((~filled).sum().item())
            cand = TimestepSampler.sample_scaled(
                self.mode,
                need,
                device,
                config,
                scale=self.scale,
                latents=latents,
            )
            ok = _in_range(cand)
            if bool(ok.any()):
                # Place accepted candidates into the still-unfilled slots.
                idx_unfilled = torch.nonzero(~filled, as_tuple=False).flatten()
                accept_pos = idx_unfilled[ok]
                out[accept_pos] = cand[ok]
                filled[accept_pos] = True
            rounds += 1

        if not bool(filled.all()):
            # Pathological distribution: clamp the remaining slots strictly into
            # the expert's range rather than hang. Logged so it's visible.
            self.logger.warning(
                "expert_timestep_rejection_exhausted",
                expert=expert,
                rounds=rounds,
                unfilled=int((~filled).sum().item()),
            )
            eps = 1e-3 * self.scale
            if expert == HIGH:
                out[~filled] = self.boundary_scaled
            else:
                out[~filled] = max(self.boundary_scaled - eps, 0.0)
        return out

    # ── Resume state (router RNG + cached decisions) ──────────────────────

    def state_dict(self) -> dict[str, Any]:
        """Serialize router state for checkpoint resume (RNG + decisions)."""
        return {
            "seed": self._seed,
            "p_high": self.p_high,
            "boundary_frac": self.boundary_frac,
            "switch_interval": self.switch_interval,
            "active_expert": self._active_expert,
            "rng_state": self._step_rng.get_state(),
            "block_decisions": dict(self._block_decisions),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore router state saved by :meth:`state_dict`."""
        self._seed = int(state.get("seed", self._seed))
        self.p_high = float(state.get("p_high", self.p_high))
        self._active_expert = str(state.get("active_expert", self._active_expert))
        self._block_decisions = {
            int(k): str(v) for k, v in state.get("block_decisions", {}).items()
        }
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._step_rng.set_state(rng_state)
