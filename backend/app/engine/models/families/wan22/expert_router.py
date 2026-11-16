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

import contextlib
import inspect
from collections.abc import Sequence
from typing import Any, Callable

import structlog
import torch

from app.engine.strategies.timestep_sampling import TimestepSampler

logger = structlog.get_logger(__name__)

HIGH = "high"
LOW = "low"


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
        pinned_expert: If ``"high"``/``"low"``, this is a SINGLE-expert run —
            every optimizer step routes to that one expert and its timesteps are
            truncated to that expert's boundary range. Bernoulli step-selection
            and the ``p_high`` Monte-Carlo estimate are skipped entirely (there
            is nothing to switch). ``None`` (default) = dual-expert routing.
        timestep_draw: Optional override for the ``p_high`` Monte-Carlo draw —
            a callable ``(n) -> Tensor`` returning ``n`` RAW ``[0, scale]``
            timesteps from the family's ACTUAL per-step distribution. When
            ``None`` (default, wan22's usage), the estimate draws from the
            generic :class:`TimestepSampler` per ``timestep_cfg`` — correct
            whenever the per-step timesteps ARE that same generic draw. A
            family whose per-step timesteps come from its OWN transform
            (e.g. bernini_r's mode+shift-warp band sampling) must pass its own
            sampler here, or the step-selection frequency and the actual
            per-band timestep mass silently stop composing to the intended
            marginal (see :meth:`BerniniRTrainer._build_router`). The callable
            may also declare ``progress`` and/or ``latents`` keyword parameters;
            it then receives the training progress and the reference latent
            (see below) for the estimate it serves.
        reference_latent_shape: Optional ``[B, C, (F,) H, W]`` shape of ONE
            reference latent the p_high estimate is drawn at, for a
            resolution-aware mode (``flux_shift``, ``model_shift``, ``radc``
            with ``radc_res_influence > 0``). ``None`` = no latent (wan22, in
            every mode but ``radc``). It serves only the step-boundary
            prediction until the first batch is seen; each step's expert is
            then decided for its own batch's latent shape
            (:meth:`expert_for_batch`), and the prediction follows the last
            batch's shape.
        progress_provider: Optional callable returning the current training
            progress in ``[0, 1]``. When given, ``p_high`` is re-estimated at
            every NEW switch-interval block from it (a progress-dependent mode
            such as ``radc``); a resumed job estimates from its restored step
            with no extra state. ``None`` (wan22, in every mode but ``radc``) =
            the estimate is made once at progress ``0.0``.
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
        pinned_expert: str | None = None,
        timestep_draw: Callable[..., torch.Tensor] | None = None,
        reference_latent_shape: Sequence[int] | None = None,
        progress_provider: Callable[[], float] | None = None,
    ) -> None:
        self.boundary_frac = float(boundary)
        self.scale = float(scale)
        self.boundary_scaled = self.boundary_frac * self.scale
        self.switch_interval = max(1, int(switch_interval))
        self.config: dict[str, Any] = dict(timestep_cfg or {})
        # The trainer hands its config after the shared setup() filled the definition's mode; a
        # config without one never passed through it, so routing on a guessed mode is refused.
        mode = self.config.get("timestep_sampling")
        if not mode:
            raise ValueError(
                "ExpertRouter needs the job's timestep_sampling (the shared setup() fills the "
                "definition's default before the router is built)"
            )
        self.mode: str = str(mode)
        self.mc_samples = int(mc_samples)
        self._timestep_draw = timestep_draw
        self._reference_latent_shape = (
            tuple(int(d) for d in reference_latent_shape) if reference_latent_shape else None
        )
        self._progress_provider = progress_provider
        self.logger = structlog.get_logger(self.__class__.__name__)

        pinned = pinned_expert.lower() if pinned_expert else None
        if pinned is not None and pinned not in (HIGH, LOW):
            raise ValueError(
                f"pinned_expert must be 'high'/'low'/None, got {pinned_expert!r}"
            )
        self.pinned_expert: str | None = pinned

        # Deterministic CPU generator for the Bernoulli step-selection draws.
        # Kept separate from the (config-driven) timestep sampling so resuming
        # only needs to persist/restore THIS generator's state.
        self._seed = int(seed)
        self._step_rng = torch.Generator(device="cpu").manual_seed(self._seed)

        # Per-interval-block decision cache: block_index -> "high"|"low", and the block's
        # uniform draw the decision compares with p_high (kept: a step re-decides it for its
        # own batch's latent shape, see :meth:`expert_for_batch`).
        self._block_decisions: dict[int, str] = {}
        self._block_u: dict[int, float] = {}
        self._current_block = 0
        # The expert the current optimizer step trains, fixed at its first batch.
        self._step_expert: str | None = None
        # The latent shape of the last batch drawn for: the step-boundary prediction's shape.
        self._last_shape: tuple[int, ...] | None = None
        self._p_cache: dict[tuple[float, tuple[int, ...] | None], float] = {}
        # True while the loop runs a throwaway warm-up batch: it decides, locks and records
        # nothing, and draws for the expert already active.
        self.paused = False

        if self.pinned_expert is not None:
            # Single-expert: no switching, no MC estimate. p_high is degenerate
            # (1.0 for high-only, 0.0 for low-only) purely for logging/state.
            self.p_high = 1.0 if self.pinned_expert == HIGH else 0.0
            self._active_expert: str = self.pinned_expert
        else:
            # Estimate + cache p_high once, then seed the active expert.
            self.p_high = self._estimate_p_high()
            # Active expert for the CURRENT optimizer step (advanced by the
            # driver hook). Initialized to the decision for block 0.
            self._active_expert = self._decide_block(0)

        self.logger.info(
            "expert_router_init",
            boundary_frac=self.boundary_frac,
            switch_interval=self.switch_interval,
            mode=self.mode,
            p_high=round(self.p_high, 5),
            pinned_expert=self.pinned_expert,
        )

    # ── p_high estimation (Monte-Carlo, cached once) ──────────────────────

    def _estimate_p_high(self) -> float:
        """Estimate ``P(t >= boundary)`` under the REAL per-step distribution.

        Draws ``mc_samples`` timesteps and returns the empirical fraction
        ``>= boundary``, computed on CPU with the default CPU generator seeded
        inside a CPU-only fork so the estimate is reproducible (does not touch
        the step-selection generator, the caller's CPU state or any CUDA RNG). By default draws from the generic :class:`TimestepSampler`
        (same mode + params the trainer uses) — correct whenever the per-step
        timesteps ARE that same generic draw. When ``timestep_draw`` was
        supplied at construction, that callable draws instead — for a family
        whose ACTUAL per-step timesteps come from its own transform (see the
        constructor docstring).
        """
        return self._p_high_at(self._last_shape)

    def _p_high_at(self, shape: tuple[int, ...] | None) -> float:
        """:meth:`estimate_p_high` at the current progress and ``shape`` (the reference when
        ``None``), cached: the estimate is a pure function of the two."""
        progress = 0.0 if self._progress_provider is None else float(self._progress_provider())
        key = (progress, shape)
        if key not in self._p_cache:
            if len(self._p_cache) >= 64:
                self._p_cache.clear()
            self._p_cache[key] = self.estimate_p_high(progress=progress, latent_shape=shape)
        return self._p_cache[key]

    def estimate_p_high(
        self, progress: float = 0.0, latent_shape: Sequence[int] | None = None
    ) -> float:
        """``P(t >= boundary)`` at ``progress`` and a reference latent shape.

        ``latent_shape`` defaults to the constructor's ``reference_latent_shape``.
        Pure with respect to the router state and the training RNG: the CPU generator
        is restored on exit and no CUDA generator is touched.
        """
        shape = tuple(latent_shape) if latent_shape else self._reference_latent_shape
        # Only the SHAPE reaches the resolution-aware modes (token count); no data is read.
        latents = torch.empty(*shape) if shape else None
        # The shared TimestepSampler draws from the default CPU generator and takes no
        # generator argument, so the estimate seeds THAT generator only, inside a CPU-only
        # fork that restores it on exit. Never ``torch.manual_seed``: it also reseeds every
        # CUDA generator, and each interval would replay the training draws (LANE-168 1.01).
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(self._seed + 1)
            if self._timestep_draw is not None:
                accepted = inspect.signature(self._timestep_draw).parameters
                kwargs: dict[str, Any] = {}
                if "progress" in accepted:
                    kwargs["progress"] = progress
                if "latents" in accepted:
                    kwargs["latents"] = latents
                samples = self._timestep_draw(self.mc_samples, **kwargs)
            else:
                samples = TimestepSampler.sample_scaled(
                    self.mode,
                    self.mc_samples,
                    torch.device("cpu"),
                    self.config,
                    scale=self.scale,
                    latents=latents,
                    progress=progress,
                )
        p =float((samples >= self.boundary_scaled).float().mean().item())
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
        if self.pinned_expert is None and block_index > 0:  # block 0 was just estimated by __init__
            # At the current progress and the last batch's shape (cached: a mode that reads
            # neither estimates once per shape), so the prediction is what that batch would decide.
            self.p_high = self._estimate_p_high()
        u = self._block_u.get(block_index)
        if u is None:
            u = torch.rand((), generator=self._step_rng).item()
            self._block_u[block_index] = u
        decision = HIGH if u < self.p_high else LOW
        self._block_decisions[block_index] = decision
        return decision

    def choose_expert(self, optimizer_step: int) -> str:
        """Return the expert (``"high"``/``"low"``) for ``optimizer_step``.

        Pinned (single-expert) runs always return the pinned expert. Otherwise
        deterministic given the seed: step ``s`` belongs to interval block
        ``s // switch_interval``, and every step in a block shares that block's
        single Bernoulli draw.
        """
        if self.pinned_expert is not None:
            self._active_expert = self.pinned_expert
            return self.pinned_expert
        block_index = int(optimizer_step) // self.switch_interval
        self._current_block = block_index
        self._step_expert = None
        decision = self._decide_block(block_index)
        self._active_expert = decision
        return decision

    def expert_for_batch(self, latent_shape: Sequence[int] | None) -> str:
        """The expert the current optimizer step trains, for the batch it draws for.

        :meth:`choose_expert` predicts it at the step boundary, before the batch exists. The
        step's first batch decides it: the block's uniform draw against ``p_high`` for that
        batch's latent shape, so each batch's high share is ``p_high`` at its own resolution
        and the conditional draws compose to the configured distribution at that resolution.
        Fixed for the rest of the step: a routing family's batches come one bucket per
        accumulation window, so every batch of a step has its shape. A batch of another shape
        means the step it was decided for never reached the optimizer (a window whose every
        micro-step was skipped, which calls no :meth:`choose_expert`): it starts a new
        decision rather than inherit the other resolution's odds.
        """
        if self.pinned_expert is not None:
            return self.pinned_expert
        if self.paused:
            return self._active_expert
        shape = None if latent_shape is None else (1, *(int(d) for d in tuple(latent_shape)[1:]))
        if self._step_expert is not None and (shape is None or shape == self._last_shape):
            return self._step_expert
        if self._step_expert is not None:
            self.logger.warning("expert_step_shape_changed", decided_for=self._last_shape, batch=shape,
                                reason="the step it was decided for never reached the optimizer")
        u = self._block_u.get(self._current_block)
        if shape is not None and u is not None:
            self._last_shape = shape
            self._active_expert = HIGH if u < self._p_high_at(self._last_shape) else LOW
            # The block's later boundaries predict what this batch decided, never the older guess.
            self._block_decisions[self._current_block] = self._active_expert
        self._step_expert = self._active_expert
        return self._active_expert

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
        progress: float = 0.0,
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

        The redrawing is :meth:`TimestepSampler.draw_in_band`'s, so a thin range
        keeps the distribution's shape. Only a range with next to no mass, which
        its candidate limit doesn't reach, leaves slots on the range's edge, with
        a warning on each such draw.
        """
        expert = expert.lower()
        if expert not in (HIGH, LOW):
            raise ValueError(f"unknown expert {expert!r} (expected 'high'/'low')")

        def _in_range(t: torch.Tensor) -> torch.Tensor:
            if expert == HIGH:
                return t >= self.boundary_scaled
            return t < self.boundary_scaled

        def _draw(k: int) -> torch.Tensor:
            return TimestepSampler.sample_scaled(
                self.mode,
                k,
                device,
                config,
                scale=self.scale,
                latents=latents,
                progress=progress,
            )

        out, filled = TimestepSampler.draw_in_band(_draw, batch_size, device, _in_range)

        if not bool(filled.all()):
            # Next to no mass in this range: clamp the remaining slots strictly into
            # it rather than hang. Logged so it's visible.
            self.logger.warning(
                "expert_timestep_rejection_exhausted",
                expert=expert,
                candidates=TimestepSampler.BAND_MAX_CANDIDATES,
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
            "pinned_expert": self.pinned_expert,
            "rng_state": self._step_rng.get_state(),
            "block_decisions": dict(self._block_decisions),
            "block_u": dict(self._block_u),
            "last_shape": self._last_shape,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore router state saved by :meth:`state_dict`."""
        self._seed = int(state.get("seed", self._seed))
        self.p_high = float(state.get("p_high", self.p_high))
        if "pinned_expert" in state:
            pinned = state["pinned_expert"]
            self.pinned_expert = str(pinned).lower() if pinned else None
        self._active_expert = str(state.get("active_expert", self._active_expert))
        self._block_decisions = {
            int(k): str(v) for k, v in state.get("block_decisions", {}).items()
        }
        # A state saved before the per-batch decision carries no uniforms: its decided blocks
        # keep their decision, the next undecided block draws one.
        self._block_u = {int(k): float(v) for k, v in state.get("block_u", {}).items()}
        last = state.get("last_shape")
        self._last_shape = tuple(int(d) for d in last) if last else None
        # The next step belongs to the last block decided (the one ``active_expert`` names).
        self._current_block = max(self._block_decisions, default=0)
        self._step_expert = None
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._step_rng.set_state(rng_state)


class ExpertRoutingHooks:
    """The training loop's hooks into a dual-expert trainer's router (wan22, bernini_r)."""

    def _routing_driver(self):
        driver = getattr(self, "driver", None)
        if getattr(driver, "router", None) is None or not getattr(driver, "is_dual", True):
            return None
        return driver

    @contextlib.contextmanager
    def routing_paused(self):
        """While the loop runs a throwaway warm-up batch, the router decides nothing for it."""
        driver = self._routing_driver()
        if driver is None:
            yield
            return
        driver.router.paused = True
        try:
            yield
        finally:
            driver.router.paused = False

    def route_training_batch(self, latents: torch.Tensor) -> None:
        """Activate the expert the step trains, decided from this training batch, before any
        warm-up runs: the warm-up then reserves the peak on that expert."""
        driver = self._routing_driver()
        if driver is not None:
            driver.route_for_batch(latents)

    def take_rewarm_request(self) -> bool:
        """Whether an expert swap emptied the CUDA cache since the last call: the loop then runs a
        warm-up on the expert now active before the next training forward."""
        driver = getattr(self, "driver", None)
        due = bool(getattr(driver, "rewarm_due", False))
        if due:
            driver.rewarm_due = False
        return due
