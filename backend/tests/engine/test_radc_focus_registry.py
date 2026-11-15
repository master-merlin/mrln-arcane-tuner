"""RADC trains at the noise focus its settings name, in every family that offers it.

``radc_start`` / ``radc_end`` / ``radc_width`` / ``radc_res_influence`` describe a curve over the
model's own noise fraction (0 clean, 1 pure noise). A family that shifts, warps or reverses the
draw afterwards, or weights the loss by the same curve again, trains somewhere else. This file
draws through every registered definition's own ``sample_timesteps`` (the real trainer class, no
weights), maps each draw to its noise fraction through the family's own ``add_noise``
(``noise_bands.noise_fraction_of``), weights it by the family's ``compute_loss_weight`` and
compares the weighted quartiles with the quartiles of the set curve.

A definition either passes that check or declares that it offers no flow-matching timestep
sampling, and a declaration counts only once it is shown to be true: the timestep fields are
hidden, the allowlist drops them, and a config forced to ``radc`` draws what it draws without.
A definition that can't be built or drawn fails by name. No family is named in a check: a new
family joins with no edit here.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
import torch

from app.engine.core.archetypes import FLOW_TIMESTEP_FIELDS, resolve_capabilities
from app.engine.core.config_allowlist import apply_capability_allowlist
from app.engine.core.pipeline.noise_bands import noise_fraction_of
from app.engine.core.pipeline.pipeline_base import PipelineBaseMixin
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.registry import registry
from app.engine.strategies import timestep_sampling
from app.engine.strategies.noise_interpolation import NoiseInterpolation
from app.engine.strategies.timestep_sampling import TimestepSampler, _make_radc_pdf
from tests.test_engine_fields_reach_the_trainer import _PROFILES, _Profile, _trainer

# One worker holds the memoised results the coverage test and the controls read.
pytestmark = pytest.mark.xdist_group("radc_focus_registry")

CALLS, BATCH = 2048, 4  # 8192 draws per distribution
STEPS = 100
START, END, WIDTH = 0.9, 0.3, 0.3
PROGRESS = (0.0, 0.5, 1.0)
QUANTILES = (0.25, 0.5, 0.75)
TOLERANCE = 0.012
CENTRE_TOLERANCE = 0.001
INFLUENCE = 0.2
# On the resolution term's two clamps for every patchify factor from 1 to 4: at most 256
# tokens (16x16 cells) and at least 4096 (256x256 cells), whatever the family's own factor.
SMALL_GRID, LARGE_GRID = 16, 256
A3_CALLS = 64


@dataclass
class Result:
    kind: str  # "checked" (A1 + A2), "declared" (A3) or "undrawable"
    problems: list[tuple[str, str]] = field(default_factory=list)  # (line, text)

    def lines(self, *names: str) -> list[str]:
        return [f"{line}: {text}" for line, text in self.problems if line in names]


# ── The registry ─────────────────────────────────────────────────────────────


def _definitions():
    registry.initialize()
    return sorted(registry._definitions.values(), key=lambda d: (d.family, d.id))


def _name(defn) -> str:
    return f"{defn.family}/{defn.id}"


def _caps(defn) -> dict:
    return resolve_capabilities(defn)["capabilities"]


def _profile(defn) -> _Profile:
    return _PROFILES.get(defn.family, _Profile())


def _latent_shape(defn, grid: int, batch: int = BATCH) -> tuple[int, ...]:
    """The latent shape the family's trainer receives, from its declared capabilities.

    An audio family's latents are ``[B, C, frames]`` and have no grid; a video family's are
    ``[B, C, F, H, W]`` (one frame here); every other family's ``[B, C, H, W]``.
    """
    caps = _caps(defn)
    profile = _profile(defn).latents
    channels = profile[1]
    if caps.get("is_audio_family"):
        return (batch, channels, profile[-1])
    if caps.get("is_video"):
        return (batch, channels, 1, grid, grid)
    return (batch, channels, grid, grid)


def _reference(defn) -> torch.Tensor:
    """A one-item latent of the family's rank for the noise oracle (three frames for video)."""
    caps = _caps(defn)
    channels = _profile(defn).latents[1]
    if caps.get("is_audio_family"):
        return torch.zeros(1, channels, 4)
    if caps.get("is_video"):
        return torch.zeros(1, channels, 3, 2, 2)
    return torch.zeros(1, channels, 2, 2)


def _radc(influence: float, share: float = 0.0) -> dict:
    return dict(timestep_sampling="radc", radc_start=START, radc_end=END, radc_width=WIDTH,
                radc_res_influence=influence, radc_uniform_prob=share)


def _build(defn, **config):
    """The real trainer around its real driver, with the noise oracle ``setup()`` builds."""
    trainer = _trainer(defn, **{**_profile(defn).config, **config})
    trainer.noise_interpolation = NoiseInterpolation(
        mode=trainer.config.get("noise_interpolation", "linear"), scheduler=trainer.scheduler)
    trainer.max_train_steps = STEPS
    return trainer


def _set_centre(progress: float) -> float:
    return START + (END - START) * progress


# ── Drawing and measuring ────────────────────────────────────────────────────


def _draw(trainer, latents, progress: float, calls: int = CALLS) -> torch.Tensor:
    trainer.global_step = round(progress * STEPS)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        draws = [trainer.sample_timesteps(BATCH, latents).detach().float().reshape(-1) for _ in range(calls)]
    return torch.cat(draws)


def _router(trainer, defn):
    if not _caps(defn).get("dual_expert"):
        return None
    router = getattr(trainer.driver, "router", None)
    if router is None:
        raise AssertionError("declares dual_expert but its driver has no router")
    return router


def _mix(trainer, defn, latents, progress: float, router_shape=None) -> list[tuple[torch.Tensor, float]]:
    """``(draws, share)`` per expert: one group for a single model, both experts for a dual one,
    mixed by the router's ``p_high`` for the SAME latents the draw gets."""
    router = _router(trainer, defn)
    if router is None:
        return [(_draw(trainer, latents, progress), 1.0)]
    shape = tuple(latents.shape) if router_shape is None else router_shape
    p_high = router.estimate_p_high(progress=progress, latent_shape=shape)
    groups = []
    for expert, share in (("high", p_high), ("low", 1.0 - p_high)):
        trainer.driver._set_active(expert)
        groups.append((_draw(trainer, latents, progress), share))
    return groups


def _trained(trainer, defn, latents, progress: float, router_shape=None) -> tuple[torch.Tensor, torch.Tensor]:
    """The trained distribution: noise fractions and their weights (draw share x loss weight)."""
    reference = _reference(defn)
    fractions, weights = [], []
    for draws, share in _mix(trainer, defn, latents, progress, router_shape):
        loss_weight = trainer.compute_loss_weight(draws)
        w = torch.ones_like(draws) if loss_weight is None else loss_weight.detach().float().reshape(-1)
        fractions.append(noise_fraction_of(trainer, draws, reference))
        weights.append(w * share / draws.numel())
    return torch.cat(fractions), torch.cat(weights)


def _weighted_quartiles(values: torch.Tensor, weights: torch.Tensor) -> list[float]:
    order = torch.argsort(values)
    ordered = values[order].double()
    cdf = torch.cumsum(weights[order].double(), dim=0)
    cdf = cdf / cdf[-1]
    return [float(ordered[min(int(torch.searchsorted(cdf, torch.tensor(q, dtype=torch.float64))), len(ordered) - 1)])
            for q in QUANTILES]


def _curve_quartiles(centre: float, width: float, share: float = 0.0) -> list[float]:
    """The quartiles of the set curve over noise fractions ``i / 1000``, mixed with U(0, 1) by ``share``."""
    pdf = _make_radc_pdf(centre, width).double()
    pdf = (1.0 - share) * pdf / pdf.sum() + share / pdf.numel()
    cdf = torch.cumsum(pdf, dim=0)
    cdf = cdf / cdf[-1]
    grid = torch.arange(1, 1001, dtype=torch.float64) / 1000.0
    return [float(grid[int(torch.searchsorted(cdf, torch.tensor(q, dtype=torch.float64)))]) for q in QUANTILES]


def _off_by(got: list[float], want: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(got, want, strict=True))


def _fmt(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in values) + "]"


@contextmanager
def _asked_centres():
    """Every centre the draw asks ``_radc_cdf`` for."""
    asked: list[float] = []
    real = timestep_sampling._radc_cdf

    def spy(center_q, width_q):
        asked.append(center_q / 1000.0)
        return real(center_q, width_q)

    with patch.object(timestep_sampling, "_radc_cdf", spy):
        yield asked


@contextmanager
def _sampler_calls():
    """Every mode the shared ``TimestepSampler.sample`` is called with (``sample_scaled`` included)."""
    modes: list[str] = []
    real = TimestepSampler.sample

    def spy(mode, *args, **kwargs):
        modes.append(mode)
        return real(mode, *args, **kwargs)

    with patch.object(TimestepSampler, "sample", staticmethod(spy)):
        yield modes


# ── The checks ───────────────────────────────────────────────────────────────


def _focus_problems(defn) -> list[tuple[str, str]]:
    """A1 (the focus) and A2 (the resolution term) for a definition that offers RADC."""
    problems: list[tuple[str, str]] = []
    trainer = _build(defn, **_radc(0.0))
    small = torch.zeros(_latent_shape(defn, SMALL_GRID))
    for progress in PROGRESS:
        got = _weighted_quartiles(*_trained(trainer, defn, small, progress))
        want = _curve_quartiles(_set_centre(progress), WIDTH)
        if _off_by(got, want) > TOLERANCE:
            problems.append(("A1", f"at progress {progress} the trained quartiles are {_fmt(got)}, "
                                   f"the set curve's {_fmt(want)}"))
    problems += _resolution_problems(defn)
    return problems


def _resolution_problems(defn, router_shape=None, progresses=(0.5, 1.0)) -> list[tuple[str, str]]:
    problems: list[tuple[str, str]] = []
    declared = bool(_caps(defn).get("supports_radc_resolution", True))
    trainer = _build(defn, **_radc(INFLUENCE))
    for grid, sign in ((SMALL_GRID, 1.0), (LARGE_GRID, -1.0)):
        latents = torch.zeros(_latent_shape(defn, grid))
        for progress in progresses:
            expected = _set_centre(progress) + (sign * INFLUENCE * progress if declared else 0.0)
            where = f"{grid}x{grid} cells at progress {progress}"
            with _asked_centres() as asked:
                fractions, weights = _trained(trainer, defn, latents, progress, router_shape)
            if not asked:
                problems.append(("A2", f"{where}: the draw never asked for a radc curve"))
            elif any(abs(c - expected) > CENTRE_TOLERANCE for c in asked):
                problems.append(("A2", f"{where}: the draw asked for centres {sorted(set(asked))}, "
                                       f"the settings give {expected:.3f}"))
            got = _weighted_quartiles(fractions, weights)
            want = _curve_quartiles(expected, WIDTH)
            if _off_by(got, want) > TOLERANCE:
                problems.append(("A2", f"{where}: the trained quartiles are {_fmt(got)}, "
                                       f"the curve at centre {expected:.3f} has {_fmt(want)}"))
    if not declared:
        rule = resolve_capabilities(defn)["field_visibility"].get("radc_res_influence") or {}
        if rule.get("supported", True) or not rule.get("reason"):
            problems.append(("A2", f"declares no resolution term but the form shows radc_res_influence ({rule})"))
    return problems


def _no_flow_problems(defn) -> list[tuple[str, str]]:
    """A3: a definition that declares no flow-matching timestep sampling is shown to apply none."""
    problems: list[tuple[str, str]] = []
    visibility = resolve_capabilities(defn)["field_visibility"]
    for name in FLOW_TIMESTEP_FIELDS:
        rule = visibility.get(name) or {}
        if rule.get("supported", True) or not rule.get("reason"):
            problems.append(("A3", f"declares no flow timestep sampling but shows {name} ({rule})"))
    config = {name: BaseTrainingConfig.model_fields[name].default for name in FLOW_TIMESTEP_FIELDS}
    apply_capability_allowlist(config, defn)
    if set(config) & set(FLOW_TIMESTEP_FIELDS):
        problems.append(("A3", f"the allowlist keeps {sorted(set(config) & set(FLOW_TIMESTEP_FIELDS))}"))
    latents = torch.zeros(_latent_shape(defn, SMALL_GRID))
    plain, forced = _build(defn), _build(defn, **_radc(INFLUENCE))
    with _sampler_calls() as modes:
        for progress in (0.0, 1.0):
            without = _draw(plain, latents, progress, A3_CALLS)
            with_radc = _draw(forced, latents, progress, A3_CALLS)
            if not torch.equal(without, with_radc):
                problems.append(("A3", f"at progress {progress} a config forced to radc draws other timesteps"))
    if modes:
        problems.append(("A3", f"the draw calls TimestepSampler (modes {sorted(set(modes))})"))
    return problems


def _outcome(defn) -> Result:
    try:
        if not _caps(defn).get("supports_flow_timestep_sampling", True):
            return Result("declared", _no_flow_problems(defn))
        return Result("checked", _focus_problems(defn))
    except Exception as exc:  # noqa: BLE001 -- a definition that can't be built or drawn is reported by name
        return Result("undrawable", [("build", f"could not be built or drawn: {type(exc).__name__}: {exc}")])


@functools.lru_cache(maxsize=None)
def _result(definition_id: str) -> Result:
    return _outcome(registry.get_definition(definition_id))


def _coverage_failures(definitions, outcome) -> list[str]:
    """A4: every definition passed A1 and A2, or passed A3. Returns each one that did neither."""
    failures = []
    for defn in definitions:
        result = outcome(defn)
        if result.problems:
            failures.append(f"{_name(defn)} ({result.kind}): " + "; ".join(t for _, t in result.problems))
    return failures


# ── The registry-wide tests ──────────────────────────────────────────────────

def _params(offering_only: bool = False):
    return [
        pytest.param(defn.id, id=_name(defn))
        for defn in _definitions()
        if not offering_only or _caps(defn).get("supports_flow_timestep_sampling", True)
    ]


def _assert_clean(definition_id: str, *lines: str) -> None:
    result = _result(definition_id)
    found = result.lines(*lines, "build")
    if result.kind == "declared":
        found += result.lines("A3")
    assert not found, f"{_name(registry.get_definition(definition_id))} ({result.kind}):\n" + "\n".join(found)


@pytest.mark.parametrize("definition_id", _params())
def test_radc_trains_at_its_set_focus(definition_id):
    _assert_clean(definition_id, "A1")


@pytest.mark.parametrize("definition_id", _params())
def test_radc_resolution_term_moves_the_focus_or_is_declared_absent(definition_id):
    _assert_clean(definition_id, "A2")


def test_a_declared_absent_resolution_term_is_true_for_the_real_latents():
    """A definition that declares no resolution term is drawn with its real latent shape (an audio
    family's 3D latents, never a spatial grid) at influence 0.2, and lands on the centre without it."""
    absent = [d for d in _definitions()
              if _caps(d).get("supports_flow_timestep_sampling", True) and not _caps(d).get("supports_radc_resolution", True)]
    assert absent, "no offering definition declares supports_radc_resolution false: this check would be vacuous"
    failures = [f"{_name(d)}: {line}" for d in absent for line in _result(d.id).lines("A2", "build")]
    assert not failures, "\n".join(failures)
    audio = [d for d in absent if _caps(d).get("is_audio_family")]
    assert audio and all(len(_latent_shape(d, SMALL_GRID)) == 3 for d in audio)


def test_a_family_without_radc_hides_it_and_ignores_it():
    declared = [d for d in _definitions() if _result(d.id).kind == "declared"]
    assert declared, "no definition declares supports_flow_timestep_sampling false: this check would be vacuous"
    failures = [f"{_name(d)}: {line}" for d in declared for line in _result(d.id).lines("A3", "build")]
    assert not failures, "\n".join(failures)


def test_every_definition_is_checked_or_declared():
    definitions = _definitions()
    assert len(definitions) >= 32, f"only {len(definitions)} definitions: the registry did not load"
    failures = _coverage_failures(definitions, lambda d: _result(d.id))
    assert not failures, f"{len(failures)} of {len(definitions)} definitions neither on focus nor declared:\n" + "\n".join(failures)


# ── A5: the uniform share ────────────────────────────────────────────────────

SHARES = (1.0, 0.25)
SHARE_PROGRESS = (0.0, 1.0)


def _share_problems(defn) -> list[str]:
    problems = []
    small = torch.zeros(_latent_shape(defn, SMALL_GRID))
    for share in SHARES:
        trainer = _build(defn, **_radc(0.0, share))
        for progress in SHARE_PROGRESS:
            got = _weighted_quartiles(*_trained(trainer, defn, small, progress))
            want = _curve_quartiles(_set_centre(progress), WIDTH, share)
            if _off_by(got, want) > TOLERANCE:
                problems.append(f"share {share} at progress {progress}: the trained quartiles are {_fmt(got)}, "
                                f"the mixture's {_fmt(want)}")
    return problems


@pytest.mark.parametrize("definition_id", _params(offering_only=True))
def test_radc_uniform_share_is_a_mixture_on_the_noise_scale(definition_id):
    problems = _share_problems(registry.get_definition(definition_id))
    assert not problems, "\n".join(problems)


def test_share_zero_leaves_the_random_stream_untouched():
    """Share 0 (the default) consumes the stream as the draw always did: one ``rand`` per batch."""
    cpu = torch.device("cpu")
    config = _radc(0.0)
    without_key = {k: v for k, v in config.items() if k != "radc_uniform_prob"}
    after = []
    for cfg in (config, without_key):
        torch.manual_seed(0)
        drawn = TimestepSampler.sample("radc", 8, cpu, cfg, progress=0.5)
        after.append((drawn, torch.rand(4)))
    torch.manual_seed(0)
    torch.rand(8)
    one_draw = torch.rand(4)
    assert torch.equal(after[0][0], after[1][0])
    assert torch.equal(after[0][1], after[1][1]) and torch.equal(after[0][1], one_draw)


def test_share_uses_the_local_generator():
    cpu = torch.device("cpu")
    config = _radc(0.0, 0.5)
    state = torch.random.get_rng_state()
    draws = [TimestepSampler.sample("radc", 4096, cpu, config, progress=0.0,
                                    generator=torch.Generator().manual_seed(1)) for _ in range(2)]
    assert torch.equal(torch.random.get_rng_state(), state), "the share drew from the global RNG"
    assert torch.equal(draws[0], draws[1])
    # The curve alone (centre 0.9, width 0.3) puts about 0.4 % below 0.5; half of U(0, 1) puts 25 %.
    assert float((draws[0] < 0.5).float().mean()) > 0.2, "the share did not act"


def test_only_radc_sets_the_noise_level_itself():
    assert [m for m in TimestepSampler.SUPPORTED_MODES if timestep_sampling.sets_noise_level_itself(m)] == ["radc"]


# ── Controls: break one link on a passing family, the check must report it ───


def _base_path(defn) -> bool:
    """The family draws through the shared base ``sample_timesteps`` on the ``[0, 1000]`` clock."""
    trainer = _build(defn)
    return (
        type(trainer).sample_timesteps is PipelineBaseMixin.sample_timesteps
        and trainer._driver_hook_override("sample_timesteps") is None
        and type(trainer).compute_loss_weight is PipelineBaseMixin.compute_loss_weight
        and not _caps(defn).get("dual_expert")
    )


@functools.lru_cache(maxsize=1)
def _control_id() -> str:
    for defn in _definitions():
        if _caps(defn).get("supports_flow_timestep_sampling", True) and _base_path(defn) and not _result(defn.id).problems:
            return defn.id
    raise AssertionError("no passing base-path definition to break")


def _control():
    return registry.get_definition(_control_id())


def test_the_fixture_gives_a_base_path_family_a_noise_oracle():
    defn = _control()
    trainer = _build(defn)
    fractions = noise_fraction_of(trainer, torch.tensor([0.0, 250.0, 750.0, 1000.0]), _reference(defn))
    assert torch.allclose(fractions, torch.tensor([0.0, 0.25, 0.75, 1.0]), atol=1e-6), fractions


def _trainer_class(defn):
    return type(_build(defn))


def test_positive_control_a_shifted_draw_is_reported(monkeypatch):
    defn = _control()
    cls = _trainer_class(defn)
    original = cls.sample_timesteps

    def shifted(self, batch_size, latents=None):
        t = original(self, batch_size, latents) / 1000.0
        return 3.0 * t / (1.0 + 2.0 * t) * 1000.0

    monkeypatch.setattr(cls, "sample_timesteps", shifted)
    assert _outcome(defn).lines("A1"), "a draw shifted after the fact was not reported"


def test_positive_control_a_reversed_clock_is_reported(monkeypatch):
    defn = _control()
    cls = _trainer_class(defn)
    original = cls.sample_timesteps
    monkeypatch.setattr(cls, "sample_timesteps", lambda self, bs, latents=None: 1000.0 - original(self, bs, latents))
    assert _outcome(defn).lines("A1"), "a reversed clock was not reported"


def test_positive_control_a_curve_counted_twice_is_reported(monkeypatch):
    defn = _control()
    cls = _trainer_class(defn)

    def weighted(self, timesteps):
        progress = self.global_step / self.max_train_steps
        centre = timestep_sampling._radc_center(self.config, progress, None)
        return _make_radc_pdf(centre, WIDTH)[timesteps.long().clamp(1, 1000) - 1]

    monkeypatch.setattr(cls, "compute_loss_weight", weighted)
    assert _outcome(defn).lines("A1"), "a loss weighted by the curve the draw already follows was not reported"


def test_positive_control_a_constant_added_to_the_resolution_centre_is_reported(monkeypatch):
    defn = _control()
    real = timestep_sampling._radc_center

    def off_centre(config, progress, latents):
        centre = real(config, progress, latents)
        if float(timestep_sampling.timestep_setting(config, "radc_res_influence")) > 0:
            centre = min(1.0, centre + 0.1)
        return centre

    monkeypatch.setattr(timestep_sampling, "_radc_center", off_centre)
    result = _outcome(defn)
    assert result.lines("A2") and not result.lines("A1"), result


def test_positive_control_a_falsely_disabled_flow_capability_is_reported(monkeypatch):
    defn = _control()
    family = registry.get_family_class(defn.family)
    overrides = {**getattr(family, "capability_overrides", {}), "supports_flow_timestep_sampling": False}
    # A family with nothing to override carries no dict; resolve_capabilities reads it by getattr.
    monkeypatch.setattr(family, "capability_overrides", overrides, raising=False)
    result = _outcome(defn)
    assert result.kind == "declared"
    assert any("draws other timesteps" in line for line in result.lines("A3")), result
    assert _coverage_failures([defn], _outcome), "the coverage test counted a false declaration"


def test_an_undrawable_family_fails_by_name(monkeypatch):
    defn = _control()
    cls = _trainer_class(defn)

    def broken(self, batch_size, latents=None):
        raise RuntimeError("no draw")

    monkeypatch.setattr(cls, "sample_timesteps", broken)
    failures = _coverage_failures([defn], _outcome)
    assert len(failures) == 1 and failures[0].startswith(f"{_name(defn)} (undrawable)"), failures


def test_dual_expert_mix_uses_the_draws_latents():
    """The router's ``p_high`` is computed for the draw's own latents; mixed by the router's
    reference shape instead, the resolution check must go red (the mix is not vacuous)."""
    dual = [d for d in _definitions()
            if _caps(d).get("dual_expert") and _caps(d).get("supports_flow_timestep_sampling", True)]
    assert dual, "no dual-expert definition offers radc: this check would be vacuous"
    failures = []
    for defn in dual:
        router = _router(_build(defn, **_radc(INFLUENCE)), defn)
        reference = tuple(router._reference_latent_shape)
        if _resolution_problems(defn, progresses=(0.5,)):
            failures.append(f"{_name(defn)}: the mix by the draw's latents fails the resolution check")
        if not _resolution_problems(defn, router_shape=reference, progresses=(0.5,)):
            failures.append(f"{_name(defn)}: a mix by the reference shape {reference} is not reported")
    assert not failures, "\n".join(failures)
