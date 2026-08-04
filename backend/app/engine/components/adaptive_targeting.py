"""In-flight adaptive LoRA layer targeting (spec §5).

Measures per-module recent learning at analysis events (windowed ‖ΔW‖² of the
effective LoRA delta, EMA-smoothed across windows) and freezes modules that
stopped contributing. Freeze-only mode is monotonic; reactivation (probe
windows) and the rebuild action are layered on in later tasks.

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
from app.engine.models.adaptive import AdaptiveTargetingConfig

logger = structlog.get_logger(__name__)

# PEFT prefix injected by ``get_peft_model()``.
_PEFT_PREFIX = "base_model.model."
# An analysis event may never kill a multi-hour run, but a metric that keeps
# failing is also not steering anything — after this many consecutive failures
# the feature switches itself off (loudly) instead of burning event budget.
_MAX_CONSECUTIVE_FAILURES = 3
HISTORY_FILENAME = "adaptive_targeting.json"

# First numeric path segment of a module name — ``blocks.7.attn.to_q`` → 7.
_BLOCK_INDEX_RE = re.compile(r"(?:^|\.)(\d+)(?:\.|$)")


def _strip_prefix(name: str) -> str:
    """Remove the ``base_model.model.`` prefix added by PEFT."""
    return name[len(_PEFT_PREFIX) :] if name.startswith(_PEFT_PREFIX) else name


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
    """Map PEFT-prefix-stripped module path → its ``(lora_A, lora_B)`` weights.

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
            modules[_strip_prefix(name)] = adapter
        elif _unmanaged_adapter_names(module):
            # Ordinary (non-adapter) modules land here too — only the ones
            # actually carrying an adapter are worth reporting.
            unmanaged.append(_strip_prefix(name))
    return modules, unmanaged


class AdaptiveTargetingController:
    """Freezes LoRA modules that stopped learning, at fixed step intervals."""

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
        self._heat: dict[str, float] = {}
        self._hot: list[str] = []
        self._universe: list[str] = []
        self._active: list[str] = []
        self._snapshot: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
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
        if not self._universe:
            reason = (
                "no LoRA modules found on the model"
                if not self._modules
                else "every LoRA module is already frozen by targeted_layers"
            )
            self.enabled = False
            self.log_writer.warning(f"adaptive_targeting_disabled: {reason}")
            logger.error("adaptive_targeting_disabled", reason=reason)
            return

        self._active = list(self._universe)
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

    # ── step hook ─────────────────────────────────────────────────────────
    def on_optimizer_step(self, step: int) -> str | None:
        if not self.enabled or step < self._next_event:
            return None
        try:
            result = self._run_event(step)
            self._consecutive_failures = 0
            return result
        except Exception as exc:  # noqa: BLE001 — must never kill a run (spec §7)
            self._consecutive_failures += 1
            logger.warning("adaptive_event_failed", step=step, error=str(exc))
            self.log_writer.warning(
                f"adaptive_targeting event failed at step {step}: {exc}"
            )
            if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                self.enabled = False
                self.log_writer.warning(
                    "adaptive_targeting_disabled: "
                    f"{_MAX_CONSECUTIVE_FAILURES} consecutive analysis failures"
                )
                logger.error(
                    "adaptive_targeting_disabled",
                    reason="consecutive analysis failures",
                )
            self._next_event = step + self.config.interval_steps
            return None

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

    def _pad_to_floor(self, keep: list[str]) -> list[str]:
        """Top the keep-set back up to ``min_active_pct`` of the universe.

        ``select_active`` already applies the floor, but intersecting its result
        with the current active set (monotonicity) can drop below it. Padding
        pulls the *hottest* still-active modules that were dropped — never a
        positional prefix, which could evict a hot module in favour of a cold
        neighbour that merely sorts earlier.
        """
        floor = max(1, math.ceil(self.config.min_active_pct * len(self._universe)))
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
        return None

    def _emit_event(
        self,
        step: int,
        kind: str,
        frozen_this_event: int,
        reactivated_this_event: int,
        extra: dict | None = None,
    ) -> None:
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
        self._events.append(event)
        self.log_writer.emit("adapt", event)
        self.log_writer.log(
            f"adaptive_targeting[{kind}] step {step}: "
            f"{self.active_count}/{self.total_count} layers active"
        )
        self._write_history()

    def _write_history(self) -> None:
        """Rewrite the run-dir history file atomically.

        A reader (UI, post-run report) must never observe a half-written file:
        a partial JSON document is indistinguishable from a corrupt run.
        """
        path = os.path.join(self.output_dir, HISTORY_FILENAME)
        payload = {
            "events": self._events,
            "modules": self._universe,
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
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if not self.enabled or not state:
            return
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
        self._next_event = int(state.get("next_event", self._next_event))
        self._snapshot = self._take_snapshot()  # window restarts at resume point
