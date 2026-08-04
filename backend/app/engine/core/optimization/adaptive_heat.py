"""Heat math for adaptive LoRA layer targeting (spec §5).

``delta_frobenius_sq`` computes ‖B_n@A_n − B_p@A_p‖²_F entirely in rank space:
ΔW = [B_n | −B_p] @ [[A_n],[A_p]], and ‖BA‖²_F = tr((BᵀB)(AAᵀ)) needs only two
(2r×2r) Grams — the out×in delta matrix is never materialized, so an analysis
event over hundreds of modules costs milliseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_HOT_TIER_ENERGY = 0.90  # spec §5: hot (essential) tier = 90% cumulative energy


def delta_frobenius_sq(
    b_now: torch.Tensor,
    a_now: torch.Tensor,
    b_prev: torch.Tensor,
    a_prev: torch.Tensor,
) -> float:
    # fp32 promotion: live LoRA weights are bf16, whose accumulation error
    # would swamp the small deltas this metric exists to detect.
    b = torch.cat([b_now.float(), -b_prev.float()], dim=1)  # (out, 2r)
    a = torch.cat([a_now.float(), a_prev.float()], dim=0)  # (2r, in)
    g_b = b.T @ b  # (2r, 2r)
    g_a = a @ a.T  # (2r, 2r)
    return max(float((g_b * g_a).sum().item()), 0.0)


@dataclass
class Selection:
    keep: list[str]
    hot: list[str]
    total_heat: float


def select_active(
    heat: dict[str, float],
    universe: list[str],
    energy_threshold: float,
    min_active_pct: float,
) -> Selection:
    total = sum(heat.get(m, 0.0) for m in universe)
    if total <= 0.0:
        # No signal in this window — caller skips the event (spec §7).
        return Selection(keep=list(universe), hot=[], total_heat=0.0)

    ranked = sorted(universe, key=lambda m: heat.get(m, 0.0), reverse=True)
    floor = max(1, math.ceil(min_active_pct * len(universe)))

    keep: list[str] = []
    hot: list[str] = []
    cum = 0.0
    for module in ranked:
        threshold_met = cum >= energy_threshold * total
        if threshold_met and len(keep) >= floor:
            break
        keep.append(module)
        if cum < _HOT_TIER_ENERGY * total:
            hot.append(module)
        cum += heat.get(module, 0.0)
    return Selection(keep=keep, hot=hot, total_heat=total)
