"""LANE-158: the whole-model fp32 contract of the vendor-parity tests, exercised THROUGH ``_check``.

The contract is an empirical one: a residual bounded by ``1e-5 + 2**-16 * max|fixture|`` per output
array (128 fp32 ulps of the output's largest magnitude plus an atol for the near-zero elements),
validated on the capture host and two GitHub-hosted runner images. These tests pin both edges of it
and the negative (every parameter scaled by 1.001), by driving each parity module's own ``_check``;
they compute only an OFFSET from the contract, never a bound to compare against.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_TESTS = Path(__file__).resolve().parent
FAMILIES = ["boogu_image", "omnigen2"]
ATOL = 1e-5
RTOL = 2.0**-16
MSG = "LANE-158 host-independent fp32 bound"


def _load(family: str):
    path = _TESTS / f"test_{family}_vendor_parity.py"
    spec = importlib.util.spec_from_file_location(f"_lane158_{family}_parity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fp32_row(mod):
    rows = [r for r in mod.DTYPES if r[2] == "fp32"]
    assert len(rows) == 1, f"{MSG}: expected one fp32 row in DTYPES, got {rows}"
    return rows[0]


def _offsets(ref: list, k: float) -> list:
    return [k * (ATOL + RTOL * float(np.abs(r).max())) for r in ref]


@pytest.mark.parametrize("family", FAMILIES, ids=FAMILIES)
def test_check_accepts_a_residual_inside_the_fp32_contract(family):
    mod = _load(family)
    row = _fp32_row(mod)
    ref = mod._fixture("transformer.fp32")
    offs = _offsets(ref, 0.5)
    outs = [torch.from_numpy(np.ascontiguousarray(r + o)) for r, o in zip(ref, offs)]
    try:
        mod._check("transformer", row, outs)
    except AssertionError as exc:
        pytest.fail(f"{MSG}: _check rejects a residual of {max(offs):.3e} inside the contract ({exc})")


@pytest.mark.parametrize("family", FAMILIES, ids=FAMILIES)
def test_check_rejects_a_residual_outside_the_fp32_contract(family):
    mod = _load(family)
    row = _fp32_row(mod)
    ref = mod._fixture("transformer.fp32")
    offs = _offsets(ref, 1.5)
    outs = [torch.from_numpy(np.ascontiguousarray(r + o)) for r, o in zip(ref, offs)]
    with pytest.raises(AssertionError, match="LANE-150 parity"):
        mod._check("transformer", row, outs)


@pytest.mark.parametrize("family", FAMILIES, ids=FAMILIES)
def test_check_rejects_a_weight_perturbation(family, monkeypatch):
    mod = _load(family)
    row = _fp32_row(mod)
    ref = mod._fixture("transformer.fp32")
    original = mod._randomize

    def perturbed(module, g):
        module = original(module, g)
        with torch.no_grad():
            for p in module.parameters():
                p.mul_(1.001)
        return module

    monkeypatch.setattr(mod, "_randomize", perturbed)
    outs = mod._tiny_transformer_outputs(torch.float32)
    bounds = [ATOL + RTOL * float(np.abs(r).max()) for r in ref]
    ratios = [float(np.abs(mod._np(o) - r).max()) / b for o, r, b in zip(outs, ref, bounds)]
    print(f"{family}: residual/bound per output = {[f'{x:.1f}x' for x in ratios]}")
    assert min(ratios) > 1.0, f"{MSG}: the all-parameter 1e-3 mutant is within the contract (residual/bound {ratios})"
    with pytest.raises(AssertionError, match="LANE-150 parity"):
        mod._check("transformer", row, outs)
