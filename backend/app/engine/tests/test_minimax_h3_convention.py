"""minimax_h3 convention-contract suite (concept §5.2; plan rows 1.1–1.5, 1.7).

The failure this suite guards is silent at training time and visible only at
inference: a sign, scale or schedule error produces a LoRA of pure noise, and
this project has shipped that defect once (memory `flowmatch-timestep-scale-
gotcha`). Each test names the wrong answer it rejects.

Row 1.1 — tests 6 and 7 (the dual-shift schedule):

* 6 ``test_dual_shift_comes_from_the_definition_not_a_literal`` — the drawn
  σ_v CHANGES when the definition's ``video.sigma_shift`` changes (the exact
  PR0 Task-7 defect: a literal nothing reads), matches the closed form at
  ``12.0`` to 1e-9, and a definition without the key RAISES.
* 7 ``test_one_u_drives_both_curves`` — for ONE ``u``, ``σ_v == shift(u, 12)``
  AND ``σ_a == shift(u, 3)`` to 1e-9, and ``σ_a`` is a deterministic function
  of ``σ_v`` (the reference identity ``remap_sigma(σ_v, 12→3) == σ_a``,
  research §3.2). Rejects two independent draws, which a naive ``σ_a != σ_v``
  assertion would pass.

Closed form: ``s·u / (1 + (s−1)·u)`` — diffusers ``scheduling_minimax_h3.py:157``
(0.40.0); ai-toolkit ``src/packing.py`` ``shift_sigma`` / ``remap_sigma``
(method re-derived, no line copied).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.minimax_h3.schedule import H3SigmaSchedule
from app.engine.models.families.minimax_h3.settings import resolve_h3_settings

_ARCH: dict[str, Any] = {
    "audio.loss_weight": 0.1,
    "video.sigma_shift": 12.0,
    "audio.sigma_shift": 3.0,
}


def _definition(arch: dict[str, Any]) -> ModelDefinition:
    return ModelDefinition(
        id="minimax-h3-t2va",
        family="minimax_h3",
        name="fixture",
        defaults={"train_audio": True},
        architecture_params=arch,
    )


def _schedule(arch: dict[str, Any]) -> H3SigmaSchedule:
    return H3SigmaSchedule.from_settings(
        resolve_h3_settings(_definition(arch), {"train_audio": True})
    )


def _closed_form(u: torch.Tensor, s: float) -> torch.Tensor:
    # Written independently of schedule.py so the test is an oracle, not an echo.
    return (s * u) / (1.0 + (s - 1.0) * u)


_U = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)


# ── 6. The shifts come from the definition, never a literal ─────────────


def test_dual_shift_comes_from_the_definition_not_a_literal():
    at_12 = _schedule(dict(_ARCH))
    at_7 = _schedule(dict(_ARCH, **{"video.sigma_shift": 7.0}))

    sigma_v_12, sigma_a_12 = at_12.draw(_U)
    sigma_v_7, _ = at_7.draw(_U)

    assert not torch.allclose(sigma_v_12[1:-1], sigma_v_7[1:-1]), (
        "σ_v unchanged under video.sigma_shift: 7.0 — the schedule reads a "
        "literal, not the definition"
    )
    assert torch.allclose(sigma_v_12, _closed_form(_U, 12.0), atol=1e-9, rtol=0)
    assert torch.allclose(sigma_a_12, _closed_form(_U, 3.0), atol=1e-9, rtol=0)
    assert torch.allclose(sigma_v_7, _closed_form(_U, 7.0), atol=1e-9, rtol=0)

    arch_without = dict(_ARCH)
    del arch_without["video.sigma_shift"]
    with pytest.raises(ValueError, match="sigma_shift_video"):
        _schedule(arch_without)


# ── 7. ONE u drives BOTH curves ──────────────────────────────────────────


def test_one_u_drives_both_curves():
    schedule = _schedule(dict(_ARCH))
    sigma_v, sigma_a = schedule.draw(_U)

    assert torch.allclose(sigma_v, _closed_form(_U, 12.0), atol=1e-9, rtol=0)
    assert torch.allclose(sigma_a, _closed_form(_U, 3.0), atol=1e-9, rtol=0), (
        "σ_a not a function of σ_v — audio drew its own u"
    )

    # Reference identity (research §3.2): un-shift σ_v by 12, re-shift by 3.
    # Inverse of s·u/(1+(s−1)·u):  u = σ / (s − (s−1)·σ).
    u_recovered = sigma_v / (12.0 - 11.0 * sigma_v)
    remapped = _closed_form(u_recovered, 3.0)
    assert torch.allclose(remapped, sigma_a, atol=1e-9, rtol=0), (
        "σ_a not a function of σ_v — remap_sigma(σ_v, 12→3) != σ_a"
    )

    # Endpoints are fixed points of the shift: u=0 → 0, u=1 → 1 on both curves.
    assert sigma_v[0] == 0.0 and sigma_a[0] == 0.0
    assert sigma_v[-1] == 1.0 and sigma_a[-1] == 1.0
