"""Heat math for adaptive LoRA layer targeting (spec §5).

``delta_frobenius_sq`` computes ‖B_n@A_n − B_p@A_p‖²_F entirely in rank space:
ΔW = [B_n | −B_p] @ [[A_n],[A_p]], and ‖BA‖²_F = tr((BᵀB)(AAᵀ)) needs only two
(2r×2r) Grams — the out×in delta matrix is never materialized, so an analysis
event over hundreds of modules costs milliseconds.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

import torch

_HOT_TIER_ENERGY = 0.90  # spec §5: hot (essential) tier = 90% cumulative energy

# Numeric path segments are the per-block index; everything else is the
# projection's identity. Only segments FOLLOWED by a dot are stripped, so a
# trailing index that is part of the name survives (``attn.to_out.0``).
_BLOCK_SEGMENTS_RE = re.compile(r"(?:\.\d+)+(?=\.)")

#: Key of the single fallback group. No module name can produce it, so its
#: presence is the unambiguous signal that per-projection grouping failed.
GLOBAL_GROUP = "*"


def projection_group(name: str) -> str:
    """The module's projection identity, with its block index removed.

    ``transformer_blocks.27.attn.to_v`` -> ``transformer_blocks.attn.to_v``,
    so every block's ``to_v`` lands in one group.
    """
    return _BLOCK_SEGMENTS_RE.sub("", name)


def group_universe(universe: list[str]) -> dict[str, list[str]]:
    """Partition ``universe`` by projection, or one global group if it cannot be.

    Selection runs per group because raw ‖ΔW‖²_F is NOT comparable across
    projections: under grouped-query attention a ``to_v`` delta has an order of
    magnitude fewer elements than an ``ff.gate`` delta, so one global ranking
    retires every K and V in the model while the widest matrices monopolise the
    keep-set — silently deleting the text-conditioning pathway. Ranking within a
    group only ever compares like with like.

    A universe whose names carry no block index degenerates to one group per
    module, which would keep everything and turn narrowing into a no-op. That
    falls back to a single global group; the caller reports it (see
    ``Selection.groups_used``), because a family that lands here gets the old
    biased ranking and its user should know.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for name in universe:
        groups[projection_group(name)].append(name)
    if len(groups) >= len(universe) > 1:
        return {GLOBAL_GROUP: list(universe)}
    return dict(groups)


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
    # How many projection groups the ranking ran over. 1 means grouping was
    # impossible for this module naming and the old global ranking applied.
    groups_used: int = field(default=1)


def _select_within(
    members: list[str],
    heat: dict[str, float],
    energy_threshold: float,
    floor: int,
) -> tuple[list[str], list[str]]:
    """Hottest prefix of one group, above that group's own floor."""
    total = sum(heat.get(m, 0.0) for m in members)
    ranked = sorted(members, key=lambda m: (-heat.get(m, 0.0), m))
    if total <= 0.0:
        # Nothing in this group moved. Ranking would be a coin flip, so keep
        # exactly the floor — never zero, or a group that goes quiet for one
        # window is retired for the rest of the run. Name order makes the
        # arbitrary pick at least reproducible across processes.
        return ranked[:floor], []

    keep: list[str] = []
    hot: list[str] = []
    cum = 0.0
    for module in ranked:
        if cum >= energy_threshold * total and len(keep) >= floor:
            break
        keep.append(module)
        if cum < _HOT_TIER_ENERGY * total:
            hot.append(module)
        cum += heat.get(module, 0.0)
    return keep, hot


def select_active(
    heat: dict[str, float],
    universe: list[str],
    energy_threshold: float,
    min_active_pct: float,
    min_active_count: int | None = None,
) -> Selection:
    """Keep the hottest modules of EACH projection group, above its own floor.

    Selection is per projection (see ``group_universe``) so that no pathway can
    be eliminated wholesale by a metric that is not comparable across matrix
    shapes. Within a group the rule is unchanged: rank by heat, take the prefix
    covering ``energy_threshold`` of that group's energy, never fewer than the
    group's share of the floor.

    ``min_active_count`` overrides the ``min_active_pct``-derived floor. The
    caller needs it because the percentage is a promise about the universe the
    RUN started with, while this function only ever sees the universe of the
    current process — which a rebuild restart has already narrowed. Passing the
    count keeps the floor anchored on the original size (spec §5); omitting it
    keeps the standalone percentage behaviour.

    ``keep`` and ``hot`` come back in global heat-descending order, not group
    order — callers treat them as rankings.
    """
    total = sum(heat.get(m, 0.0) for m in universe)
    if total <= 0.0:
        # No signal anywhere in this window — caller skips the event (spec §7).
        return Selection(keep=list(universe), hot=[], total_heat=0.0)

    groups = group_universe(universe)
    size = len(universe)
    global_floor = (
        math.ceil(min_active_pct * size)
        if min_active_count is None
        else min_active_count
    )
    global_floor = max(1, min(global_floor, size))

    keep: list[str] = []
    hot: list[str] = []
    for members in groups.values():
        # The floor is shared out in proportion to group size, and rounded UP
        # per group, so the total kept is never below the global floor the
        # caller asked for — the guarantee only ever gets stronger.
        floor = max(1, math.ceil(global_floor * len(members) / size))
        floor = min(floor, len(members))
        g_keep, g_hot = _select_within(members, heat, energy_threshold, floor)
        keep.extend(g_keep)
        hot.extend(g_hot)

    by_heat = lambda m: (-heat.get(m, 0.0), m)  # noqa: E731 — local sort key
    return Selection(
        keep=sorted(keep, key=by_heat),
        hot=sorted(hot, key=by_heat),
        total_heat=total,
        groups_used=len(groups),
    )
