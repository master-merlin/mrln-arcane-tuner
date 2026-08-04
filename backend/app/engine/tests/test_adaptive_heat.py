"""Heat math for adaptive targeting: Gram-trick ΔW norm + keep-set selection."""

import pytest
import torch

from app.engine.core.optimization.adaptive_heat import (
    Selection,  # noqa: F401 - imported to pin the interface name Task 3 consumes
    delta_frobenius_sq,
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
    assert len(sel.keep) == 3  # ceil(0.75 * 4)


def test_hot_tier_is_90pct_energy_prefix():
    heat = {"m1": 80.0, "m2": 15.0, "m3": 4.0, "m4": 1.0}
    sel = select_active(heat, list(heat), energy_threshold=1.0, min_active_pct=0.01)
    # m1=0.80 < 0.90 → m1+m2=0.95 ≥ 0.90 → hot = [m1, m2]
    assert sel.hot == ["m1", "m2"]
    assert sel.keep == ["m1", "m2", "m3", "m4"]  # threshold 1.0 keeps all


def test_zero_total_heat_returns_full_universe():
    """Nothing learned in the window → caller must skip freezing (never freeze on noise)."""
    sel = select_active({}, ["m1", "m2"], energy_threshold=0.9, min_active_pct=0.1)
    assert sel.total_heat == 0.0
    assert sel.keep == ["m1", "m2"]
