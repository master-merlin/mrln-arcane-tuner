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

import re
from pathlib import Path
from typing import Any

import pytest
import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.minimax_h3.schedule import H3SigmaSchedule
from app.engine.models.families.minimax_h3.settings import resolve_h3_settings

_TESTS_DIR = Path(__file__).resolve().parent

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


# ── Row 1.2: the training convention on the driver ───────────────────────
#
# Oracle (concept §5.1, closed, no free parameters), against the INSTALLED
# diffusers 0.40.0 `MiniMaxH3Scheduler` (`scheduling_minimax_h3.py:170-171`
# t = 1 − σ; `:225` scale_noise x_t = t·x₀ + (1−t)·noise; `:273` step
# x̂₀ = x_t + σ·v):  x₀ = x_t + σ·v  ⇒  v = x₀ − noise, unique, no scale.
# Research §3.1: ai-toolkit `t_v = 1.0 − sigma_v`, `return -noise_pred`;
# diffusion-pipe `t_v = 1.0 − sigma_v`, `-video_out` — the same contract.

_CONTROLS = _TESTS_DIR / "fixtures" / "h3_convention_controls.py"


def _driver(arch: dict[str, Any] | None = None):
    from app.engine.models.families.minimax_h3.driver import MiniMaxH3Driver

    return MiniMaxH3Driver(_definition(dict(_ARCH) if arch is None else arch), torch.device("cpu"))


def _scheduler(shift: float = 12.0):
    from diffusers import MiniMaxH3Scheduler

    return MiniMaxH3Scheduler(shift=shift)


def _family_sources() -> list[Path]:
    family_dir = _TESTS_DIR.parents[0] / "models" / "families" / "minimax_h3"
    return sorted(
        p for p in family_dir.rglob("*.py") if "vendor" not in p.relative_to(family_dir).parts
    )


# ── 1. The target is the unique velocity the scheduler inverts ───────────


def test_target_is_the_unique_velocity_the_scheduler_inverts():
    drv = _driver()
    sched = _scheduler()
    torch.manual_seed(1)
    x0 = torch.randn(2, 24, 3, 4, 4)
    noise = torch.randn(2, 24, 3, 4, 4)
    for sigma in (0.999, 0.9, 0.5, 0.1, 0.01):
        t = torch.full((2,), 1.0 - sigma)
        x_t = drv.add_noise(x0, noise, t)
        # The driver's forward process IS the reference's forward process.
        assert torch.allclose(x_t, sched.scale_noise(x0, t, noise), atol=1e-6), (
            f"add_noise != MiniMaxH3Scheduler.scale_noise at sigma={sigma}"
        )
        v = drv.compute_target(x0, noise, t)
        assert torch.allclose(x_t + sigma * v, x0, atol=1e-5), (
            f"x_t + sigma*v != x0 at sigma={sigma}: the target is not x0 - noise"
        )


# ── 2. A perfect-velocity walk recovers x₀ ───────────────────────────────


def test_perfect_velocity_round_trip_recovers_x0():
    drv = _driver()
    sched = _scheduler(shift=12.0)
    sched.set_timesteps(20, device="cpu")
    torch.manual_seed(2)
    x0 = torch.randn(1, 24, 2, 4, 4)
    noise = torch.randn(1, 24, 2, 4, 4)
    v = drv.compute_target(x0, noise, torch.tensor([0.0]))
    sample = noise.clone()
    for t in sched.timesteps:
        sample = sched.step(v, t, sample, return_dict=False)[0]
    assert torch.allclose(sample, x0, atol=1e-4), (
        "the oracle velocity walked through the reference scheduler does not "
        "land on x0 — a direction/sign error, not a precision one"
    )


# ── 3. add_noise runs on H3's clock: t=1 clean, t=0 noise ────────────────


def test_add_noise_endpoints_are_h3_clockwise():
    drv = _driver()
    torch.manual_seed(3)
    x0 = torch.randn(2, 24, 2, 4, 4)
    noise = torch.randn(2, 24, 2, 4, 4)
    assert torch.equal(drv.add_noise(x0, noise, torch.tensor([1.0, 1.0])), x0), (
        "t=1 is not x0 — the standard clock (t=0 clean) is the wrong one here"
    )
    assert torch.equal(drv.add_noise(x0, noise, torch.tensor([0.0, 0.0])), noise), (
        "t=0 is not pure noise"
    )
    # Strictly monotone: the distance to x0 falls as t rises.
    dist = [
        (drv.add_noise(x0, noise, torch.full((2,), t)) - x0).norm().item()
        for t in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]
    assert all(b < a for a, b in zip(dist, dist[1:])), f"not monotone: {dist}"


# ── sample_timesteps: the drawn u, through the VIDEO shift, as t ─────────


def test_sample_timesteps_are_video_clock_t_of_the_drawn_u():
    drv = _driver()
    config = {"train_audio": True, "timestep_sampling": "uniform"}
    torch.manual_seed(4)
    t = drv.sample_timesteps(8, torch.device("cpu"), config)
    torch.manual_seed(4)
    u = torch.rand((8,))
    expected = 1.0 - _closed_form(u, 12.0)
    assert t.shape == (8,) and t.dtype.is_floating_point
    assert torch.all((t >= 0) & (t <= 1))
    assert torch.allclose(t, expected.to(t.dtype), atol=1e-6), (
        "sample_timesteps is not 1 - shift(u, video.sigma_shift)"
    )
    # The shift is the definition's, not a literal.
    torch.manual_seed(4)
    t7 = _driver(dict(_ARCH, **{"video.sigma_shift": 7.0})).sample_timesteps(
        8, torch.device("cpu"), config
    )
    assert not torch.allclose(t, t7)


# ── 5. No ×1000 anywhere in the family ───────────────────────────────────

TRIPLE = '"' * 3

_THOUSAND_RE = re.compile(r"[*/]\s*1000(?:\.0)?\b|num_train_timesteps")


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """``(lineno, code)`` with ``#`` comments and triple-quoted docstrings
    stripped — the guards scan what EXECUTES; a docstring that names the
    contract (``t = 1 - sigma``) is documentation, not a conversion site."""
    out: list[tuple[int, str]] = []
    in_doc = False
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        code = raw.split("#", 1)[0]
        quotes = code.count(TRIPLE)
        if quotes % 2 == 1:
            in_doc = not in_doc
            continue
        if in_doc or quotes:
            continue
        out.append((n, code))
    return out


def _thousand_offenders(paths: list[Path]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for p in paths:
        hits = [n for n, code in _code_lines(p) if _THOUSAND_RE.search(code)]
        if hits:
            out[p.name] = hits
    return out


def test_no_thousand_scaling_in_family_source():
    assert _thousand_offenders([_CONTROLS]), f"control not flagged: {_CONTROLS}"
    offenders = _thousand_offenders(_family_sources())
    assert not offenders, f"timestep rescale in family source: {offenders}"


# ── 8. ONE σ↔t conversion module ─────────────────────────────────────────

_CONVERSION_RE = re.compile(r"\b1(?:\.0)?\s*-\s*(?:sigma|timesteps?|t)\b")


def _conversion_sites(paths: list[Path]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for p in paths:
        hits = [n for n, code in _code_lines(p) if _CONVERSION_RE.search(code)]
        if hits:
            out[p.name] = hits
    return out


def test_trainer_and_sampler_share_one_conversion_module():
    assert _conversion_sites([_CONTROLS]), f"control not flagged: {_CONTROLS}"
    sites = _conversion_sites(_family_sources())
    assert set(sites) == {"schedule.py"}, (
        "the sigma<->t conversion must live in schedule.py and nowhere else in "
        f"the family (training and sampling would diverge): {sites}"
    )
