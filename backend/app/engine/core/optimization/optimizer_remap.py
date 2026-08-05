"""Remap a saved optimizer state onto a narrowed trainable-param set (spec §5).

A rebuild restart recreates the optimizer over ONLY the params that are still
trainable — that is where the VRAM reclaim comes from — while the checkpoint's
``optimizer.pt`` still covers the wider pre-rebuild set. Optimizer state is
keyed by a param's POSITION in the flat param group, so the two are not
comparable directly: the remap goes through the ordered param-NAME list saved
beside ``optimizer.pt``.

A param whose moments cannot be carried over starts fresh. The caller names it
in a WARNING — a silently zeroed momentum reads as a converged param and drags
the next few hundred steps.
"""

from __future__ import annotations

import copy
from typing import Any


def remap_optimizer_state(
    saved_state: dict[str, Any],
    saved_names: list[str],
    current_names: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Re-key ``saved_state`` from ``saved_names`` order onto ``current_names``.

    Returns ``(state_dict, unmapped)`` where ``state_dict`` is ready for
    ``optimizer.load_state_dict`` and ``unmapped`` names every current param
    that starts from fresh moments (absent from the saved list, or present but
    never stepped — optimizer state is allocated lazily).

    Emits a SINGLE param group, which is what the pipeline always builds (every
    optimizer strategy receives one flat param list). Per-param state indices in
    a torch ``state_dict`` are global across groups, so collapsing a
    multi-group saved state loses nothing but the extra groups' hyperparameters.

    ``saved_state`` is never mutated: the caller may still fall back to it.
    """
    old_index = {name: position for position, name in enumerate(saved_names)}
    saved_per_param = saved_state.get("state") or {}

    new_state: dict[int, Any] = {}
    unmapped: list[str] = []
    for new_position, name in enumerate(current_names):
        old_position = old_index.get(name)
        if old_position is None or old_position not in saved_per_param:
            unmapped.append(name)
            continue
        new_state[new_position] = saved_per_param[old_position]

    saved_groups = saved_state.get("param_groups") or [{}]
    # Group-level state — Prodigy's learned ``d``, the LR, betas — belongs to
    # the run, not to any one param, and must survive the narrowing.
    group = copy.deepcopy(saved_groups[0])
    group["params"] = list(range(len(current_names)))
    return {"state": new_state, "param_groups": [group]}, unmapped
