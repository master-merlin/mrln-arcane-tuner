"""Heat math for adaptive targeting: Gram-trick ΔW norm + keep-set selection."""

import pytest
import torch

from app.engine.core.optimization.adaptive_heat import (
    GLOBAL_GROUP,
    Selection,
    delta_frobenius_sq,
    group_universe,
    projection_group,
    select_active,
)


@pytest.mark.parametrize("out_f,in_f,r", [(8, 8, 2), (32, 16, 1), (16, 48, 4)])
def test_gram_trick_equals_dense_computation(out_f, in_f, r):
    torch.manual_seed(0)
    b_now, a_now = torch.randn(out_f, r), torch.randn(r, in_f)
    b_prev, a_prev = torch.randn(out_f, r), torch.randn(r, in_f)
    dense = torch.linalg.matrix_norm(b_now @ a_now - b_prev @ a_prev, ord="fro") ** 2
    assert delta_frobenius_sq(b_now, a_now, b_prev, a_prev) == pytest.approx(
        float(dense), rel=1e-4
    )


def test_gram_trick_zero_when_unchanged():
    b, a = torch.randn(8, 2), torch.randn(2, 8)
    assert delta_frobenius_sq(b, a, b.clone(), a.clone()) == pytest.approx(
        0.0, abs=1e-6
    )


def test_gram_trick_handles_bf16_inputs():
    """Live LoRA weights are bf16; math must be done in fp32 without erroring."""
    b, a = (
        torch.randn(8, 2, dtype=torch.bfloat16),
        torch.randn(2, 8, dtype=torch.bfloat16),
    )
    z = torch.zeros_like(b)
    val = delta_frobenius_sq(b, a, z, a)
    assert val > 0.0


def test_selection_keeps_top_energy_until_threshold():
    heat = {"m1": 80.0, "m2": 15.0, "m3": 4.0, "m4": 1.0}
    sel = select_active(
        heat, ["m1", "m2", "m3", "m4"], energy_threshold=0.93, min_active_pct=0.01
    )
    # cumulative: m1=0.80 < 0.93 → continue; m1+m2=0.95 ≥ 0.93 → stop after m2
    assert sel.keep == ["m1", "m2"]


def test_selection_respects_min_active_floor():
    heat = {"m1": 99.0, "m2": 0.5, "m3": 0.3, "m4": 0.2}
    sel = select_active(heat, list(heat), energy_threshold=0.5, min_active_pct=0.75)
    assert len(sel.keep) == 3  # ceil(0.75 * 4) == 3

    # Non-integer product pins the authorized rounding direction: ceil(2.4) == 3,
    # which floor(2.4) == 2 and round(2.4) == 2 would both get wrong.
    sel_ceil = select_active(heat, list(heat), energy_threshold=0.5, min_active_pct=0.6)
    assert len(sel_ceil.keep) == 3  # ceil(0.6 * 4) == 3, not floor/round's 2


def test_hot_tier_is_90pct_energy_prefix():
    heat = {"m1": 80.0, "m2": 15.0, "m3": 4.0, "m4": 1.0}
    sel = select_active(heat, list(heat), energy_threshold=1.0, min_active_pct=0.01)
    # m1=0.80 < 0.90 → m1+m2=0.95 ≥ 0.90 → hot = [m1, m2]
    assert sel.hot == ["m1", "m2"]
    assert sel.keep == ["m1", "m2", "m3", "m4"]  # threshold 1.0 keeps all

    # Tighter fixture straddling the 0.90 constant on both sides of m3, so a
    # drift of the hot-tier cutoff to 0.85 or 0.95 would change membership and
    # fail this assertion (the fixture above alone tolerates any cutoff in
    # roughly (0.80, 0.95]).
    heat_tight = {"m1": 79.0, "m2": 8.0, "m3": 7.0, "m4": 6.0}
    sel_tight = select_active(
        heat_tight, list(heat_tight), energy_threshold=1.0, min_active_pct=0.01
    )
    # cum before m3 = 87: ≥ 0.85*100 (excluded at 0.85) but < 0.90*100 (included at 0.90)
    # cum before m4 = 94: ≥ 0.90*100 (excluded at 0.90) but < 0.95*100 (included at 0.95)
    assert sel_tight.hot == ["m1", "m2", "m3"]


def _blocks(n, *projections):
    return [f"transformer_blocks.{i}.{p}" for p in projections for i in range(n)]


def test_projection_group_strips_the_block_index_not_the_name():
    assert projection_group("transformer_blocks.27.attn.to_v") == (
        "transformer_blocks.attn.to_v"
    )
    # A TRAILING index is part of the projection's name, not a block index.
    assert projection_group("transformer_blocks.3.attn.to_out.0") == (
        "transformer_blocks.attn.to_out.0"
    )
    # Nested indices collapse together, so UNet-style paths still group.
    assert projection_group("down_blocks.0.attentions.1.tb.0.attn1.to_q") == (
        "down_blocks.attentions.tb.attn1.to_q"
    )
    # Two blocks of the same projection share one group; two projections do not.
    assert projection_group("transformer_blocks.0.ff.gate") == projection_group(
        "transformer_blocks.9.ff.gate"
    )
    assert projection_group("transformer_blocks.0.ff.gate") != projection_group(
        "transformer_blocks.0.ff.up"
    )


def test_a_systematically_smaller_projection_is_never_wiped_out():
    """The defect this grouping exists to prevent.

    Raw ||dW||_F^2 is not comparable across projections: under grouped-query
    attention a to_v delta has ~10x fewer elements than an ff.gate delta, so a
    single global ranking retires EVERY to_v in the model — deleting the
    text-conditioning pathway — while the widest matrices take the whole
    keep-set. The fixture reproduces that shape ratio.
    """
    universe = _blocks(28, "ff.gate", "attn.to_v")
    heat = {
        m: (100.0 if "ff.gate" in m else 1.0) + i * 0.01
        for i, m in enumerate(universe)
    }

    sel = select_active(heat, universe, energy_threshold=0.90, min_active_pct=0.25)

    kept_v = [m for m in sel.keep if "attn.to_v" in m]
    kept_gate = [m for m in sel.keep if "ff.gate" in m]
    assert kept_v, "the entire to_v pathway was frozen — this is the bug"
    assert len(kept_v) >= 7  # ceil(0.25 * 56 * 28/56) == 7, its own floor
    assert kept_gate  # …and the hot projection is still narrowed on its merits
    assert len(kept_gate) < 28
    assert sel.groups_used == 2


def test_each_group_carries_its_own_share_of_the_floor():
    universe = _blocks(10, "ff.gate", "attn.to_v")
    heat = {m: (1000.0 if "ff.gate" in m else 0.001) for m in universe}

    sel = select_active(heat, universe, energy_threshold=0.10, min_active_pct=0.30)

    # ceil(0.30 * 20) == 6 globally -> ceil(6 * 10/20) == 3 per group. Rounding
    # up per group can only make the total kept exceed the global floor.
    assert len([m for m in sel.keep if "attn.to_v" in m]) >= 3
    assert len([m for m in sel.keep if "ff.gate" in m]) >= 3
    assert len(sel.keep) >= 6


def test_a_group_that_goes_quiet_keeps_its_floor_instead_of_retiring():
    """One silent window must not retire a projection for the rest of the run."""
    universe = _blocks(8, "ff.gate", "attn.to_v")
    heat = {m: (5.0 if "ff.gate" in m else 0.0) for m in universe}

    sel = select_active(heat, universe, energy_threshold=0.90, min_active_pct=0.25)

    kept_v = [m for m in sel.keep if "attn.to_v" in m]
    assert len(kept_v) == 2  # ceil(ceil(0.25*16) * 8/16) == 2
    assert sel.total_heat > 0.0  # the window as a whole did move


def test_names_without_a_block_index_fall_back_to_one_global_group():
    """Grouping every module into its own group would keep everything and turn
    narrowing into a silent no-op, so an ungroupable universe uses the old
    single ranking — and says so through ``groups_used``."""
    universe = ["alpha", "beta", "gamma", "delta"]
    assert group_universe(universe) == {GLOBAL_GROUP: universe}

    heat = {"alpha": 80.0, "beta": 15.0, "gamma": 4.0, "delta": 1.0}
    sel = select_active(heat, universe, energy_threshold=0.93, min_active_pct=0.01)
    assert sel.keep == ["alpha", "beta"]  # unchanged global behaviour
    assert sel.groups_used == 1


def test_zero_total_heat_returns_full_universe():
    """Nothing learned in the window → caller must skip freezing (never freeze on noise)."""
    sel = select_active({}, ["m1", "m2"], energy_threshold=0.9, min_active_pct=0.1)
    assert isinstance(
        sel, Selection
    )  # pins the return type, not just its attribute shape
    assert sel.total_heat == 0.0
    assert sel.keep == ["m1", "m2"]
