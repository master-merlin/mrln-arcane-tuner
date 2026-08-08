"""In-flight adaptive LoRA layer targeting (spec §5).

Measures per-module recent learning at analysis events (windowed ‖ΔW‖² of the
effective LoRA delta, EMA-smoothed across windows) and freezes modules that
stopped contributing. Freeze-only mode is monotonic; opt-in reactivation adds
probe windows that reopen the universe for a bounded number of steps and may
re-admit a module an earlier event froze. The rebuild action is layered on in a
later task.

Zero per-step overhead between events: the only per-step work is an integer
compare. Event cost is rank-space Gram products over the module registry.

**Critical invariant (shared with ``targeted_training``):** base model
parameters are NEVER touched. This module only flips ``requires_grad`` on
``lora_A`` / ``lora_B`` adapter weights.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

import structlog
import torch

from app.engine.core.optimization.adaptive_heat import delta_frobenius_sq, select_active
from app.engine.core.optimization.targeted_training import normalize_module_name
from app.engine.models.adaptive import AdaptiveTargetingConfig

logger = structlog.get_logger(__name__)

# An analysis event may never kill a multi-hour run, but a metric that keeps
# failing is also not steering anything — after this many consecutive failures
# the feature switches itself off (loudly) instead of burning event budget.
_MAX_CONSECUTIVE_FAILURES = 3
# Hard cap on checkpoint+restart cycles per run (spec §5). Each rebuild costs a
# full checkpoint, a process teardown and a model reload; past a handful the
# restart overhead outweighs the optimizer-VRAM it reclaims. Hitting the cap is
# never fatal — the run continues with in-place freezing.
_MAX_REBUILDS = 5
HISTORY_FILENAME = "adaptive_targeting.json"

# First numeric path segment of a module name — ``blocks.7.attn.to_q`` → 7.
_BLOCK_INDEX_RE = re.compile(r"(?:^|\.)(\d+)(?:\.|$)")


def earliest_active_block(module_names: list[str]) -> int | None:
    """Smallest numeric path segment among ``module_names``, or ``None``.

    Reported so the user can see how much of the backward pass freezing can
    actually skip: autograd only shortens the graph when a *contiguous prefix*
    of early blocks goes fully cold, so a low earliest-active block means the
    freeze is saving parameters but almost no backward time.
    """
    indices = [
        int(match.group(1))
        for match in (_BLOCK_INDEX_RE.search(name) for name in module_names)
        if match
    ]
    return min(indices) if indices else None


def _json_heat(value: float) -> float | None:
    """Round a heat value for the history file, or ``None`` if non-finite.

    Six SIGNIFICANT figures, not six decimal places: a per-window ‖ΔW‖² at real
    learning rates sits far below 1e-6, and decimal rounding would flatten the
    whole map to zeros — the diagnostic would read "nothing learned" on exactly
    the runs it exists to explain. A non-finite value (diverged run) becomes
    null rather than 0.0, which would read as a layer that never learned.
    """
    if not math.isfinite(value):
        return None
    return float(f"{value:.6g}")


def _unmanaged_adapter_names(module) -> list[str]:
    """Adapter entries on ``module`` that this controller cannot read.

    A peft ``Embedding`` stores its adapter in ``lora_embedding_A``/``_B`` and
    leaves ``lora_A``/``lora_B`` as EMPTY dicts; a model loaded under a
    non-``"default"`` adapter name has populated dicts under a different key.
    Both are invisible to the ``(lora_A, lora_B)["default"]`` lookup — but
    ``TargetedLayerManager`` DOES manage ``lora_embedding_A``/``_B``, so a
    module missing from our registry is one that ``keep_patterns()`` would
    silently freeze on a rebuild restart. Never drop one without saying so.
    """
    found: list[str] = []
    for attr in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
        container = getattr(module, attr, None)
        if container is None or not hasattr(container, "keys"):
            continue
        found.extend(f"{attr}.{key}" for key in container.keys())
    return found


def _discover_lora_modules(
    model,
) -> tuple[dict[str, tuple[torch.nn.Parameter, torch.nn.Parameter]], list[str]]:
    """Map normalized module path → its ``(lora_A, lora_B)`` weights.

    Names go through :func:`normalize_module_name`, the SAME normalization
    ``TargetedLayerManager`` applies: these names become the anchored patterns a
    rebuild restart re-applies in a different process, whose PEFT/compile
    wrapping must not change which modules they select.

    Returns ``(modules, unmanaged)``; ``unmanaged`` names adapter-bearing
    modules whose layout this controller cannot read (see
    :func:`_unmanaged_adapter_names`). The caller must surface them.
    """
    modules: dict[str, tuple[torch.nn.Parameter, torch.nn.Parameter]] = {}
    unmanaged: list[str] = []
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        adapter: tuple[torch.nn.Parameter, torch.nn.Parameter] | None = None
        if lora_a is not None and lora_b is not None:
            try:
                adapter = (lora_a["default"].weight, lora_b["default"].weight)
            except (KeyError, TypeError, AttributeError):
                adapter = None
        if adapter is not None:
            modules[normalize_module_name(name)] = adapter
        elif _unmanaged_adapter_names(module):
            # Ordinary (non-adapter) modules land here too — only the ones
            # actually carrying an adapter are worth reporting.
            unmanaged.append(normalize_module_name(name))
    return modules, unmanaged


class AdaptiveTargetingController:
    """Freezes LoRA modules that stopped learning, at fixed step intervals.

    What this buys, so nobody re-derives it from a disappointing benchmark:
    NOT step time. Freezing flips ``requires_grad`` on a module's A/B pair and
    nothing else. The forward is unchanged by contract (a frozen module's delta
    is part of the model — see ``keep_patterns``), and autograd still walks
    every block down to ``earliest_active_block``, which stays near zero
    whenever the surviving modules are spread across the depth. The LoRA
    weight-gradients that do get skipped are a low single-digit share of a
    step's work, inside run-to-run noise.

    What it does buy is capacity control late in training, and — in
    ``action="rebuild"`` only — optimizer state for the dropped parameters.
    """

    def __init__(
        self,
        model,
        config: AdaptiveTargetingConfig,
        total_steps: int,
        log_writer,
        output_dir: str,
    ) -> None:
        self.config = config
        self.total_steps = max(int(total_steps), 1)
        self.log_writer = log_writer
        self.output_dir = output_dir
        self.enabled = True
        self.event_index = 0
        self.rebuild_count = 0
        self._consecutive_failures = 0
        self._events: list[dict[str, Any]] = []
        # Module names earlier segments of this run managed. A rebuild restart
        # discovers a NARROWED universe, so without this the history file's
        # module list would shrink below the set its own events refer to.
        self._prior_modules: list[str] = []
        self._heat: dict[str, float] = {}
        self._hot: list[str] = []
        self._universe: list[str] = []
        self._active: list[str] = []
        self._snapshot: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        # Non-None only while a probe window is open (reactivation mode).
        self._probe_open_step: int | None = None
        self._pre_probe_active: list[str] = []
        # Rebuild bookkeeping (rebuild action only). The baseline is the
        # active-param count the CURRENT optimizer was built over; the pending
        # step carries a requested-but-not-yet-emitted rebuild.
        self._params_at_last_rebuild = 0
        self._pending_rebuild_step: int | None = None
        self._rebuild_cap_logged = False
        self._warmup_end = int(self.config.warmup_pct * self.total_steps)
        self._next_event = self._warmup_end + self.config.interval_steps

        self._modules, unmanaged = _discover_lora_modules(model)
        if unmanaged:
            # ONE line, not one per module: a big model can carry hundreds.
            self.log_writer.warning(
                f"adaptive_targeting: {len(unmanaged)} adapter-bearing module(s) "
                f"use a layout this feature cannot measure (e.g. '{unmanaged[0]}') "
                "— they keep training, but a rebuild restart would freeze them"
            )
            logger.warning(
                "adaptive_targeting_unmanaged_adapters",
                count=len(unmanaged),
                example=unmanaged[0],
            )
        # Universe = modules still trainable AFTER the manual targeted_layers
        # freeze. Adaptive targeting operates strictly WITHIN the user's
        # selection and never re-enables something the user turned off.
        self._universe = [
            name
            for name, (a_p, b_p) in self._modules.items()
            if a_p.requires_grad and b_p.requires_grad
        ]
        # Size of the universe the RUN started from. On a fresh controller that
        # is this process's universe; a resumed/rebuilt one overwrites it from
        # the persisted state (see restore_state and _min_active_floor).
        self._original_total = len(self._universe)
        if not self._universe:
            self._disable(
                "no LoRA modules found on the model"
                if not self._modules
                else "every LoRA module is already frozen by targeted_layers"
            )
            return

        self._active = list(self._universe)
        self._params_at_last_rebuild = self._active_param_count()
        self._snapshot = self._take_snapshot()

    # ── public counters (per-step metrics, Task 5) ────────────────────────
    @property
    def total_count(self) -> int:
        return len(self._universe)

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def hot_count(self) -> int:
        return len(self._hot)

    def keep_patterns(self) -> list[str]:
        """Anchored regexes of the CURRENT active set, for ``TargetedLayerManager``."""
        return [f"^{re.escape(name)}$" for name in self._active]

    def _active_param_count(self) -> int:
        """LoRA params the optimizer is currently stepping — the rebuild metric.

        Counted in PARAMS, not modules: a rebuild is worth its restart only for
        the optimizer state it stops allocating, and modules differ in size by
        orders of magnitude.
        """
        return sum(
            self._modules[name][0].numel() + self._modules[name][1].numel()
            for name in self._active
        )

    # ── step hook ─────────────────────────────────────────────────────────
    def on_optimizer_step(self, step: int) -> str | None:
        if not self.enabled:
            return None
        probe_open_step = self._probe_open_step
        if probe_open_step is not None:
            # The interval clock PAUSES while a probe window is open. A regular
            # narrowing event firing mid-probe would truncate the probe's own
            # measurement window AND rank a keep-set over a universe that is
            # only temporarily wide open.
            if step < probe_open_step + self.config.probe_steps:
                return None
        elif step < self._next_event:
            return None
        try:
            if probe_open_step is not None:
                self._close_probe(step)
                result = None
            elif self._probe_is_due():
                self._open_probe(step)
                result = None
            else:
                result = self._run_event(step)
            # Reset only on a step that actually ran an event: the budget is
            # three CONSECUTIVE failed EVENTS, and resetting on the ordinary
            # in-between steps would make it unreachable.
            self._consecutive_failures = 0
            return result
        except Exception as exc:  # noqa: BLE001 — must never kill a run (spec §7)
            self._consecutive_failures += 1
            logger.warning("adaptive_event_failed", step=step, error=str(exc))
            self.log_writer.warning(
                f"adaptive_targeting event failed at step {step}: {exc}"
            )
            if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                self._disable(
                    f"{_MAX_CONSECUTIVE_FAILURES} consecutive analysis failures"
                )
            self._next_event = step + self.config.interval_steps
            return None

    def _disable(self, reason: str) -> None:
        """Switch the feature off — after undoing a half-applied probe.

        Order is the contract: a failure inside ``_open_probe`` past the
        un-freeze trips this latch while the window is open, and from then on
        ``on_optimizer_step`` returns early forever, so ``_close_probe`` never
        runs. Without the rollback every module the run had frozen stays
        trainable for the rest of the job — the exact compute this feature
        exists to save, lost permanently and invisibly.
        """
        if self._probe_open_step is not None:
            try:
                if self._pre_probe_active:
                    self._apply_active_set(self._pre_probe_active)
            except Exception as exc:  # noqa: BLE001 — never mask the disable
                logger.warning("adaptive_probe_rollback_failed", error=str(exc))
                self.log_writer.warning(
                    "adaptive_targeting: could not restore the pre-probe active "
                    f"set while disabling ({exc}) — the universe stays trainable"
                )
            finally:
                self._probe_open_step = None
                self._pre_probe_active = []
        self.enabled = False
        self.log_writer.warning(f"adaptive_targeting_disabled: {reason}")
        logger.error("adaptive_targeting_disabled", reason=reason)

    # ── internals ─────────────────────────────────────────────────────────
    def _take_snapshot(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Clone the universe's LoRA weights as the next window's baseline.

        Universe-only: modules the user froze can never move, so a delta for
        them would always be zero and the clone would be pure overhead.
        """
        with torch.no_grad():
            return {
                name: (
                    self._modules[name][0].detach().float().cpu().clone(),
                    self._modules[name][1].detach().float().cpu().clone(),
                )
                for name in self._universe
            }

    def _compute_window_heat(self) -> dict[str, float]:
        heat: dict[str, float] = {}
        ema = self.config.heat_ema
        with torch.no_grad():
            for name in self._universe:
                a_p, b_p = self._modules[name]
                a_prev, b_prev = self._snapshot[name]
                delta = delta_frobenius_sq(
                    b_p.detach().float().cpu(),
                    a_p.detach().float().cpu(),
                    b_prev,
                    a_prev,
                )
                heat[name] = ema * self._heat.get(name, 0.0) + (1.0 - ema) * delta
        return heat

    def _apply_active_set(self, keep: list[str]) -> None:
        """Flip ``requires_grad`` so exactly ``keep`` stays trainable.

        Iterates the universe only — base-model params and user-frozen adapters
        are never touched.
        """
        keep_set = set(keep)
        for name in self._universe:
            a_p, b_p = self._modules[name]
            trainable = name in keep_set
            a_p.requires_grad_(trainable)
            b_p.requires_grad_(trainable)
        self._active = [name for name in self._universe if name in keep_set]

    def _min_active_floor(self) -> int:
        """Smallest keep-set this controller may produce.

        ``min_active_pct`` is a share of the universe the RUN started from, not
        of whatever the current process inherited: a rebuild restart re-applies
        the previous keep-set as its manual ``targeted_layers``, so this
        process's universe IS that keep-set. Recomputing the share against it
        makes every segment's floor a fraction of an already-narrowed set, and
        across the capped rebuild cycles the guarantee compounds down toward a
        single module — while the UI promises a share of the LoRA's modules.
        Capped at the current universe: a floor this process cannot satisfy
        would be met by keeping everything, which is the same thing said
        without the arithmetic.
        """
        anchor = max(self._original_total, len(self._universe))
        floor = math.ceil(self.config.min_active_pct * anchor)
        return max(1, min(floor, len(self._universe)))

    def _pad_to_floor(self, keep: list[str]) -> list[str]:
        """Top the keep-set back up to the min-active floor.

        ``select_active`` already applies the floor, but intersecting its result
        with the current active set (monotonicity) can drop below it. Padding
        pulls the *hottest* still-active modules that were dropped — never a
        positional prefix, which could evict a hot module in favour of a cold
        neighbour that merely sorts earlier.
        """
        floor = self._min_active_floor()
        if len(keep) >= floor:
            return keep
        keep_set = set(keep)
        candidates = sorted(
            (name for name in self._active if name not in keep_set),
            key=lambda name: self._heat.get(name, 0.0),
            reverse=True,
        )
        return keep + candidates[: floor - len(keep)]

    def _run_event(self, step: int) -> str | None:
        # Computed into a local and committed only once the snapshot it was
        # measured against has been replaced. Advancing the EMA first would let
        # a failure part-way through leave the heat one window ahead of its own
        # baseline, so the next window double-counts this interval.
        heat = self._compute_window_heat()
        selection = select_active(
            heat,
            self._universe,
            self.config.energy_threshold,
            self.config.min_active_pct,
            min_active_count=self._min_active_floor(),
        )
        self._snapshot = self._take_snapshot()
        self._heat = heat
        self._next_event = step + self.config.interval_steps

        if selection.total_heat <= 0.0:
            # Nothing moved in this window (e.g. accumulation-only steps): the
            # ranking would be arbitrary, so freezing on it would be a coin flip.
            logger.info("adaptive_event_skipped_zero_heat", step=step)
            self.log_writer.log(
                f"adaptive_targeting: step {step} window had no learning signal "
                "— skipped"
            )
            return None

        # Freeze mode is monotonic: intersect with the CURRENT active set so a
        # module that was frozen can never come back (its optimizer state and
        # its heat are both stale — reactivation is probe-gated, Task 4).
        active_set = set(self._active)
        keep = self._pad_to_floor([n for n in selection.keep if n in active_set])
        keep_set = set(keep)
        self._hot = [name for name in selection.hot if name in keep_set]
        frozen_now = len(self._active) - len(keep)
        self._apply_active_set(keep)
        self.event_index += 1
        self._emit_event(
            step, kind="narrow", frozen_this_event=frozen_now, reactivated_this_event=0
        )
        if self.config.action != "rebuild":
            return None
        return self._maybe_request_rebuild(step)

    # ── rebuild action (spec §5) ──────────────────────────────────────────
    def _maybe_request_rebuild(self, step: int) -> str | None:
        """Ask the train loop for a checkpoint+restart, if the shrink earned it.

        The narrowing above has ALREADY been applied in place, so returning
        ``None`` here is simply freeze-mode behaviour — nothing is lost when the
        threshold is not met or the cap is spent.

        The ``adapt`` event is deliberately NOT emitted here: it carries the
        checkpoint directory the backend relaunches from, and that checkpoint
        does not exist until the loop has written it (see
        :meth:`notify_rebuild_checkpoint`).
        """
        active_params = self._active_param_count()
        # Measured against the last rebuild, never the run start: each restart
        # has to earn its own shrink, or every event past the first threshold
        # would request one.
        baseline = max(self._params_at_last_rebuild, 1)
        shrink = 1.0 - (active_params / baseline)
        if shrink < self.config.rebuild_min_shrink_pct / 100.0:
            return None
        if self.rebuild_count >= _MAX_REBUILDS:
            if not self._rebuild_cap_logged:
                # Once per run: the condition holds at every later event too,
                # and a repeated line would read as repeated restarts.
                self._rebuild_cap_logged = True
                self.log_writer.log(
                    f"adaptive_targeting: rebuild cap of {_MAX_REBUILDS} reached "
                    "— training continues with in-place freezing only"
                )
                logger.info("adaptive_rebuild_cap_reached", step=step)
            return None
        # Both counters advance at the DECISION, not once the checkpoint the
        # caller is about to write exists. A checkpoint that then fails
        # therefore burns a rebuild slot and moves the shrink baseline, so the
        # retry must earn a further shrink against the narrower set. That
        # conservatism is deliberate: rolling them back would let a run whose
        # checkpoint save keeps failing request a restart at every subsequent
        # event, and a restart nobody can act on is pure teardown cost.
        self.rebuild_count += 1
        self._params_at_last_rebuild = active_params
        self._pending_rebuild_step = step
        return "rebuild_request"

    def notify_rebuild_checkpoint(self, checkpoint_dir: str) -> None:
        """Emit the deferred ``rebuild_request`` now that the checkpoint exists.

        ``checkpoint_dir`` is the directory NAME (not a path): the backend
        resolves it against the run's output dir, which only it knows in its own
        process.

        Ignores a call with nothing pending — the backend relaunches the job on
        this event, so an unrequested one would restart a healthy run.

        Raises only if the event never reached the log writer. Once it HAS, the
        backend is already relaunching this job, so a failure past that point
        must not be raised: the caller would keep training a job that a second
        process is about to pick up.
        """
        step = self._pending_rebuild_step
        if step is None:
            logger.warning(
                "adaptive_rebuild_notify_without_request",
                checkpoint_dir=checkpoint_dir,
            )
            return
        self._pending_rebuild_step = None
        # Its own event index: this is a second `adapt` payload, and a consumer
        # keying on event_index must not see the narrow event's index twice.
        self.event_index += 1
        self._emit_event(
            step,
            kind="rebuild_request",
            frozen_this_event=0,
            reactivated_this_event=0,
            extra={
                "checkpoint_dir": checkpoint_dir,
                "keep_patterns": self.keep_patterns(),
                "rebuild_count": self.rebuild_count,
            },
            is_handoff=True,
        )

    # ── probe windows (reactivation mode, spec §5) ────────────────────────
    def _probe_is_due(self) -> bool:
        """Is the event now due a probe window rather than a regular narrowing?

        ``event_index > 0`` deliberately skips the very first event: a probe
        measures which modules deserve re-admission, and before any narrowing
        has happened there is nothing to re-admit — it would burn a whole
        window re-measuring an already fully-trainable universe.
        """
        if not self.config.reactivation:
            return False
        return (
            self.event_index > 0
            and (self.event_index + 1) % self.config.probe_every == 0
        )

    def _open_probe(self, step: int) -> None:
        """Temporarily unfreeze the whole universe and start a fresh window.

        The pre-probe set is captured BEFORE the unfreeze because
        ``_apply_active_set`` overwrites ``self._active`` with the universe: a
        close that finds no signal would otherwise "restore" the wide-open
        universe and silently discard every narrowing decision of the run.
        """
        pre_probe = list(self._active)
        # The measurement window is the probe and nothing else. Heat carried in
        # from the preceding interval was accumulated while most of the universe
        # was frozen and physically could not move, so it cannot rank the
        # modules the probe exists to re-rank.
        #
        # Snapshotting FIRST is deliberate: it only READS weight values, so the
        # unfreeze cannot affect it, and every fallible step therefore runs
        # before the one irreversible one. Unfreezing first would let a snapshot
        # failure leave the model genuinely trainable across the whole universe
        # while ``_probe_open_step`` is still None — the controller would not
        # know a probe was open, so nothing would ever close it.
        snapshot = self._take_snapshot()
        self._pre_probe_active = pre_probe
        self._snapshot = snapshot
        self._probe_open_step = step
        self._apply_active_set(list(self._universe))
        self.event_index += 1
        # Both counters are 0 by contract: the unfreeze is temporary and the
        # real accounting belongs to the matching "probe_apply". The widened
        # ``active_count`` on this event is what makes the probe visible.
        self._emit_event(
            step, kind="probe_open", frozen_this_event=0, reactivated_this_event=0
        )

    def _close_probe(self, step: int) -> None:
        """Measure the probe window and apply a possibly-re-admitting keep-set."""
        before = list(self._pre_probe_active)
        try:
            # Same ordering as ``_run_event``: computed into a LOCAL so a failure
            # part-way through cannot leave the EMA one window ahead of the
            # snapshot it was measured against.
            heat = self._compute_window_heat()
            selection = select_active(
                heat,
                self._universe,
                self.config.energy_threshold,
                self.config.min_active_pct,
                min_active_count=self._min_active_floor(),
            )
        except Exception:
            # Roll the MODEL back, not just the bookkeeping. ``_open_probe``
            # flipped real ``requires_grad`` flags across the universe; leaving
            # them set would train every module the run had already frozen —
            # burning exactly the compute this feature exists to save — and
            # would turn the next event's monotonic intersect against
            # ``self._active`` into a no-op, silently bypassing the invariant.
            # Re-raised so on_optimizer_step still logs, surfaces and counts it.
            self._apply_active_set(before)
            raise
        finally:
            # Released even when the measurement raised. A probe left open
            # pauses the interval clock forever, so every later event of the run
            # is lost and the run finishes wide open and unmeasured.
            self._probe_open_step = None
            self._pre_probe_active = []
            self._next_event = step + self.config.interval_steps
        self._snapshot = self._take_snapshot()
        self._heat = heat

        if selection.total_heat <= 0.0:
            # Nothing moved even with the whole universe trainable, so the
            # ranking would be arbitrary and re-admitting on it a coin flip.
            # Restore the PRE-probe set, never ``self._active`` — that is the
            # universe this probe opened.
            self._apply_active_set(before)
            logger.info("adaptive_probe_zero_heat", step=step)
            self.log_writer.log(
                f"adaptive_targeting: probe window ending at step {step} had no "
                "learning signal — the pre-probe active set is restored unchanged"
            )
            return

        keep = selection.keep
        keep_set = set(keep)
        before_set = set(before)
        self._hot = [name for name in selection.hot if name in keep_set]
        reactivated = len([name for name in keep if name not in before_set])
        frozen_now = len([name for name in before if name not in keep_set])
        # Deliberately NOT intersected with the pre-probe set: re-admitting a
        # module that went cold earlier and learned again during the probe is
        # the entire point of this mode. Both remaining bounds still hold —
        # ``select_active`` only ever ranks names from the universe (so a
        # user-frozen module can never come back) and applies the min-active
        # floor over that same universe (so no padding is needed here).
        self._apply_active_set(keep)
        self._emit_event(
            step,
            kind="probe_apply",
            frozen_this_event=frozen_now,
            reactivated_this_event=reactivated,
        )

    def _emit_event(
        self,
        step: int,
        kind: str,
        frozen_this_event: int,
        reactivated_this_event: int,
        extra: dict | None = None,
        is_handoff: bool = False,
    ) -> None:
        """Append, broadcast and persist one analysis event.

        ``is_handoff`` marks the one event the BACKEND acts on irreversibly —
        the rebuild request. It inverts two orderings, both for the same
        reason: for a handoff the emit IS the decision, everywhere else the
        decision was already applied to the model before we got here.

        * Emit vs. record. A handoff emits first, so a failed emit records
          nothing — ``_write_history`` rewrites the file from ``self._events``
          and would otherwise publish, on the next successful write, a restart
          the pipeline never performed. Every other kind records first, so a
          broken log channel cannot erase a narrowing the run really made from
          the only durable record of it.
        * History-write failure. Once a handoff payload is on the log-writer
          channel the backend has it, so failing to rewrite the file must not
          be reported back as "the event never happened" — it is surfaced, not
          raised. Every other caller wants the failure: an event that cannot
          be persisted is a failed event and must count against the budget.
        """
        active_params = sum(
            self._modules[n][0].numel() + self._modules[n][1].numel()
            for n in self._active
        )
        total_params = (
            sum(
                self._modules[n][0].numel() + self._modules[n][1].numel()
                for n in self._universe
            )
            or 1
        )
        event = {
            "step": step,
            "event_index": self.event_index,
            "kind": kind,
            "active_count": self.active_count,
            "total_count": self.total_count,
            "hot_count": self.hot_count,
            "frozen_this_event": frozen_this_event,
            "reactivated_this_event": reactivated_this_event,
            "active_param_pct": round(100.0 * active_params / total_params, 1),
            "earliest_active_block": earliest_active_block(self._active),
            "top_modules": self._hot[:10],
            **(extra or {}),
        }
        summary = (
            f"adaptive_targeting[{kind}] step {step}: "
            f"{self.active_count}/{self.total_count} layers active"
        )
        if is_handoff:
            self.log_writer.emit("adapt", event)
            self._events.append(event)
            self.log_writer.log(summary)
            try:
                self._write_history()
            except Exception as exc:  # noqa: BLE001 — the event already shipped
                logger.warning(
                    "adaptive_history_write_failed", kind=kind, error=str(exc)
                )
                self.log_writer.warning(
                    f"adaptive_targeting: could not update {HISTORY_FILENAME} after "
                    f"the {kind} event ({exc}) — the event itself was delivered"
                )
            return

        self._events.append(event)
        try:
            self.log_writer.emit("adapt", event)
        except Exception:
            # The freeze this event describes is already applied. Persist it
            # before re-raising, so the durable record matches the model even
            # though nobody is listening on the live channel. A history write
            # that also fails must not mask the emit failure the caller's
            # budget is counting.
            try:
                self._write_history()
            except Exception as hist_exc:  # noqa: BLE001 — emit error wins
                logger.warning(
                    "adaptive_history_write_failed", kind=kind, error=str(hist_exc)
                )
            raise
        self.log_writer.log(summary)
        self._write_history()

    def _seed_events_from_history(self, current_step: int | None) -> None:
        """Adopt the run's earlier events before this segment appends to them.

        ``_write_history`` REWRITES the file from ``self._events``, and that
        list starts empty in every process — so a resume, and by construction
        every rebuild restart, would otherwise truncate the durable history to
        the segment that happens to be running. This file is the only durable
        record: ``job.logs`` is cleared by the relaunch.

        Events past the resume point are dropped. Resuming an earlier
        checkpoint discards the steps after it, and republishing their events
        would advertise narrowing decisions this run no longer made. Callers
        that cannot name a resume step keep the file's events verbatim.

        An unreadable or malformed file degrades to "history restarts here"
        with the reason surfaced — never a failed resume, and never a silent
        one.
        """
        path = os.path.join(self.output_dir, HISTORY_FILENAME)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                document = json.load(fh)
        except FileNotFoundError:
            return  # first segment of the run — nothing to inherit
        except (OSError, ValueError) as exc:
            self._warn_history_unusable(str(exc))
            return
        # A file that parses but is not an object (``[...]``, ``null``, ``3``)
        # must degrade like any other unusable history. Reaching for ``.get``
        # on it raises AttributeError, which no caller's except clause names —
        # it would surface as "adaptive targeting failed to configure" and
        # disable the feature for the whole segment over a cosmetic file.
        if not isinstance(document, dict):
            self._warn_history_unusable("its top level is not a JSON object")
            return
        prior = document.get("events")
        if not isinstance(prior, list):
            self._warn_history_unusable("its event list is not an array")
            return
        self._prior_modules = [
            name for name in document.get("modules", []) if isinstance(name, str)
        ]

        kept: list[dict[str, Any]] = []
        malformed = 0
        for entry in prior:
            if not isinstance(entry, dict):
                malformed += 1
                continue
            try:
                entry_step = int(entry.get("step", 0))
            except (TypeError, ValueError):
                malformed += 1
                continue
            if current_step is not None and entry_step > int(current_step):
                continue
            kept.append(entry)
        self._events = kept
        if malformed:
            self.log_writer.warning(
                f"adaptive_targeting: dropped {malformed} unreadable event(s) while "
                f"re-adopting {HISTORY_FILENAME} — the rest of the history is intact"
            )
            logger.warning("adaptive_history_entries_dropped", count=malformed)

    def _warn_history_unusable(self, reason: str) -> None:
        """Surface a history file this run cannot build on. Never silent."""
        self.log_writer.warning(
            f"adaptive_targeting: could not re-adopt {HISTORY_FILENAME} ({reason}) "
            "— this run's event history restarts from here"
        )
        logger.warning("adaptive_history_read_failed", reason=reason)

    def _write_history(self) -> None:
        """Rewrite the run-dir history file atomically.

        A reader (UI, post-run report) must never observe a half-written file:
        a partial JSON document is indistinguishable from a corrupt run.

        Schema — the two spans differ, and a consumer must not conflate them:

        ``events``   every analysis event of the RUN, across rebuild segments
                     (seeded from the prior file, pruned at the resume step).
        ``modules``  every module the RUN managed, same span as ``events``. A
                     rebuild restart's own universe is already narrowed, so
                     this is the union with what earlier segments recorded —
                     otherwise it would name fewer modules than its own events
                     reference.
        ``heat``     live heat of the modules THIS segment still measures, and
                     only those. Narrower than ``modules`` by construction: a
                     module frozen in an earlier segment is no longer measured,
                     so any value carried for it would be a stale reading
                     presented as a current one.
        """
        path = os.path.join(self.output_dir, HISTORY_FILENAME)
        seen = set(self._universe)
        payload = {
            "events": self._events,
            "modules": self._universe
            + [n for n in self._prior_modules if n not in seen],
            "heat": {k: _json_heat(v) for k, v in self._heat.items()},
        }
        tmp = f"{path}.tmp"
        os.makedirs(self.output_dir, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            # allow_nan=False is the backstop for _json_heat: a non-finite value
            # reaching the file would produce `NaN`, which is not JSON and which
            # the browser's JSON.parse rejects outright.
            json.dump(payload, fh, allow_nan=False)
        os.replace(tmp, path)

    # ── persistence (wired in Task 5) ─────────────────────────────────────
    def get_state(self) -> dict[str, Any]:
        return {
            "active_modules": list(self._active),
            "heat": {k: float(v) for k, v in self._heat.items()},
            "hot_modules": list(self._hot),
            "event_index": self.event_index,
            "next_event": self._next_event,
            "rebuild_count": self.rebuild_count,
            # Universe size the min-active floor is a share of. A rebuild
            # restart's own universe is already narrowed, so the floor must
            # travel with the state or it erodes segment by segment.
            "original_total": self._original_total,
            # Active-param baseline the next shrink is measured against — the
            # restart's optimizer is built over exactly this many params.
            "params_at_last_rebuild": self._params_at_last_rebuild,
            # Diagnostics only — a resume never reopens it (see restore_state).
            "probe_open_step": self._probe_open_step,
        }

    def restore_state(
        self, state: dict[str, Any], current_step: int | None = None
    ) -> None:
        """Re-adopt a persisted state; ``current_step`` is the resume point.

        ``current_step`` is optional only for callers that genuinely cannot name
        one (tests, direct state inspection). A resume that CAN name it must, so
        the schedule below can be clamped against it.
        """
        if not self.enabled or not state:
            return
        # The floor stays a share of the universe the RUN started from. Falls
        # back to this process's universe for state written before the key
        # existed, which is exactly the fresh-controller value.
        self._original_total = max(
            int(state.get("original_total") or 0), len(self._universe)
        )
        universe = set(self._universe)
        persisted = list(state.get("active_modules", []))
        restored = [n for n in persisted if n in universe]
        if persisted and len(restored) != len(persisted):
            # Names that do not resolve mean the state was written against a
            # different module graph (rebuilt model, different family, renamed
            # modules). Staying wide open by simply skipping the restore would
            # discard the run's narrowing decision without a trace.
            outcome = (
                f"restoring the {len(restored)} that matched"
                if restored
                else "keeping every module active for this run"
            )
            self.log_writer.warning(
                f"adaptive_targeting: resumed state lists {len(persisted)} active "
                f"module(s) but only {len(restored)} match this model's "
                f"{len(self._universe)}-module universe — {outcome}"
            )
            logger.warning(
                "adaptive_state_module_mismatch",
                persisted=len(persisted),
                restored=len(restored),
                universe=len(self._universe),
            )
        if restored:
            self._apply_active_set(restored)
        # All three filtered on the UNIVERSE, not the raw module registry: a
        # user-frozen module's persisted heat would otherwise survive and be
        # reported in hot_count / top_modules despite never being managed.
        self._heat = {
            k: float(v) for k, v in state.get("heat", {}).items() if k in universe
        }
        self._hot = [n for n in state.get("hot_modules", []) if n in universe]
        self.event_index = int(state.get("event_index", 0))
        self.rebuild_count = int(state.get("rebuild_count", 0))
        # Falls back to the set just restored, never to the full universe: a
        # baseline wider than the optimizer's real param set would read as a
        # shrink this run never made and fire a rebuild on the first event.
        self._params_at_last_rebuild = int(
            state.get("params_at_last_rebuild") or self._active_param_count()
        )
        self._next_event = int(state.get("next_event", self._next_event))
        if current_step is not None:
            # A persisted next_event that predates the resume point would fire
            # on the very next step, over a window one step long — ranking
            # modules on that is the freeze-on-noise the zero-heat guard exists
            # to prevent. The first post-resume window is always a full interval,
            # measured from the resume point (the baseline snapshot below is
            # taken there too).
            self._next_event = max(
                self._next_event, int(current_step) + self.config.interval_steps
            )
        if state.get("probe_open_step") is not None:
            # A probe cannot be resumed: the baseline snapshot it was being
            # measured against died with the process, so the surviving half of
            # the window would be compared against the resume point and read as
            # near-zero heat everywhere — a re-admission decision on noise.
            self.log_writer.log(
                "adaptive_targeting: a probe window was open when this run "
                "stopped — it is abandoned; the next event is a fresh normal "
                "window over the restored active set"
            )
        # Unconditional: the restored active set is the one the probe had
        # widened, and only a fresh normal event may narrow it again.
        self._probe_open_step = None
        self._pre_probe_active = []
        self._snapshot = self._take_snapshot()  # window restarts at resume point
        # LAST, deliberately. Re-adopting the run's earlier events is the least
        # critical part of a restore and the only part that reads a file this
        # process did not write. Going first, any failure in it would discard
        # the whole restore above — active set, heat, schedule clamp — and the
        # run would silently resume with none of its narrowing decisions.
        self._seed_events_from_history(current_step)
