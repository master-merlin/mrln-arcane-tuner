"""LANE-168 RED: every engine field the training form SHOWS for a family is observed on that
family's training path, or is hidden by a declared capability.

Before this lane the form showed every ENGINE field to every family while some were dropped on the
way to the trainer: ``bernini_r`` and ``hidream_o1`` drew their own timesteps and never read
``timestep_sampling``; ``hidream_o1`` drew fresh noise and never applied ``noise_offset``;
``min_snr_gamma`` was applied by the SDXL trainer alone. Nothing in the suite compared "shown in
``field_visibility``" with "read by the trainer", so each of those stayed invisible.

CPU only, no weights, no network. Every family trainer is the real class built with
``object.__new__`` plus its real ``_setup_family`` (real driver, no model); only the heavy model
parts are stubbed, and the recorders wrap the REAL sinks (they record the call and still draw).
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import structlog
import torch
from torch import nn

from app.engine.core.archetypes import resolve_capabilities
from app.engine.core.config_allowlist import apply_capability_allowlist
from app.engine.factories.optimizer import LRSchedulerFactory, OptimizerFactory
from app.engine.models.registry import registry
from app.engine.strategies.timestep_sampling import TimestepSampler

BASE_CONFIG = {
    "mixed_precision": "fp32",
    "te_quantization": "none",
    "cache_text_embeddings": True,
    "sample_prompts": [],
}
ENGINE_FIELDS = ("timestep_sampling", "noise_offset", "min_snr_gamma", "optimizer_type", "lr_scheduler")
# Sentinels no family default equals, so a call that carries one proves the config value arrived.
TIMESTEP_SENTINEL = "radc"
OPTIMIZER_SENTINEL = "Lion"
SCHEDULER_SENTINEL = "polynomial"


@pytest.fixture(scope="module", autouse=True)
def _registry():
    registry.initialize()
    return registry


# ── Shared helpers ───────────────────────────────────────────────────────────


def _definitions():
    """Every definition of every family (unavailable ones included -- a hidden family is the defect)."""
    registry.initialize()
    return list(registry._definitions.values())


def _trainer(defn, **config):
    """The real trainer class of ``defn``'s family around its real driver (no weights loaded)."""
    family = object.__new__(registry.get_family_class(defn.family))
    family.definition = defn
    family.config = {}
    trainer = object.__new__(family.get_trainer_class())
    trainer.definition = defn
    # minimax_h3 refuses any precision but bf16 (its own explicit validation), whatever the field under test
    base = {**BASE_CONFIG, "mixed_precision": "bf16"} if defn.family == "minimax_h3" else BASE_CONFIG
    trainer.config = {**base, **config}
    trainer.device = torch.device("cpu")
    trainer.logger = structlog.get_logger("lane-168-test")
    trainer.components = {}
    trainer.epoch = 0
    trainer.global_step = 30
    trainer.max_train_steps = 60
    trainer.optimizer = None
    trainer._setup_family()
    try:  # SDXL's discrete draw and Min-SNR read the driver's DDPM scheduler
        trainer.scheduler = trainer.driver.init_scheduler()
    except Exception:  # noqa: BLE001 -- families without a driver scheduler never read self.scheduler
        trainer.scheduler = None
    return trainer


class _SamplerSpy:
    """Records every ``TimestepSampler.sample`` / ``sample_scaled`` call and still draws for real."""

    def __init__(self, monkeypatch):
        self.calls: list[SimpleNamespace] = []
        real_sample = TimestepSampler.sample
        real_scaled = TimestepSampler.sample_scaled

        def sample(mode, bs, device, config, *, latents=None, progress=0.0):
            self.calls.append(SimpleNamespace(
                kind="sample", mode=mode, bs=bs, device=device, config=config, progress=progress))
            return real_sample(mode, bs, device, config, latents=latents, progress=progress)

        def sample_scaled(mode, bs, device, config, *, scale=1000.0, latents=None, progress=0.0):
            self.calls.append(SimpleNamespace(
                kind="sample_scaled", mode=mode, bs=bs, device=device, config=config, progress=progress))
            return real_scaled(mode, bs, device, config, scale=scale, latents=latents, progress=progress)

        monkeypatch.setattr(TimestepSampler, "sample", staticmethod(sample))
        monkeypatch.setattr(TimestepSampler, "sample_scaled", staticmethod(sample_scaled))

    def draws(self, mode=None):
        """The ``sample`` calls (``sample_scaled`` delegates to ``sample``, so this counts each draw once)."""
        return [c for c in self.calls if c.kind == "sample" and (mode is None or c.mode == mode)]


@pytest.fixture
def spy(monkeypatch):
    return _SamplerSpy(monkeypatch)


# ── 1. bernini_r ─────────────────────────────────────────────────────────────


def test_bernini_r_timestep_mode_reaches_the_factory(spy):
    defn = registry.get_definition("bernini-r-1.3b")
    assert defn is not None, "definition bernini-r-1.3b is not in the registry"
    trainer = _trainer(defn, timestep_sampling="radc", radc_start=0.8)
    trainer.global_step, trainer.max_train_steps = 30, 60

    timesteps = trainer.sample_timesteps(4)

    draws = spy.draws("radc")
    assert draws, "TimestepSampler.sample was not called for mode 'radc' (bernini_r draws its own torch.rand)"
    assert [(c.mode, c.bs, c.progress) for c in spy.draws()] == [("radc", 4, 0.5)], (
        f"expected exactly one factory draw (radc, batch 4, progress 0.5), got "
        f"{[(c.mode, c.bs, c.progress) for c in spy.draws()]}")
    assert draws[0].device == trainer.device, f"the factory got device {draws[0].device}, not the trainer's"
    assert draws[0].config.get("radc_start") == 0.8, "the factory did not receive the trainer's config"
    assert timesteps.shape == (4,) and bool((timesteps >= 0).all() and (timesteps <= 1000).all()), (
        f"timesteps must be 4 values in [0, 1000], got {timesteps.tolist()}")


# ── 2. hidream_o1 ────────────────────────────────────────────────────────────


class _SigmaCapturingModel(nn.Module):
    """Stands in for the Qwen3-VL pixel model: returns a grad-carrying x_pred and keeps the timestep."""

    def __init__(self, patch_dim: int):
        super().__init__()
        self.config = MagicMock()
        self.patch_dim = patch_dim
        self.timesteps: list[torch.Tensor] = []

    def forward(self, **kwargs):
        self.timesteps.append(kwargs["timestep"].detach().clone())
        out = MagicMock()
        out.x_pred = torch.randn(1, kwargs["vinputs"].shape[1], self.patch_dim, requires_grad=True)
        return out


def _hidream_trainer_and_batch(**config):
    from app.engine.models.families.hidream_o1.trainer import PATCH_SIZE
    from tests.test_hidream_o1_integration import _make_mock_text_sample

    defn = registry.get_definition("hidream_o1_image")
    assert defn is not None, "definition hidream_o1_image is not in the registry"
    trainer = _trainer(defn, **config)
    model = _SigmaCapturingModel(3 * PATCH_SIZE * PATCH_SIZE)
    trainer.driver.assign_components({"unet": model})
    trainer.components["unet"] = model
    trainer.processor = MagicMock()
    trainer.tokenizer = MagicMock()
    trainer.autocast_dtype = torch.float32
    side = PATCH_SIZE * 2
    n_patches = (side // PATCH_SIZE) ** 2
    batch = {"images": torch.randn(8, 3, side, side), "captions": ["a caption"] * 8}
    text_sample = _make_mock_text_sample(seq_len=n_patches, n_visual=n_patches)
    return trainer, model, batch, text_sample


def test_hidream_o1_timestep_mode_reaches_the_factory(spy):
    from app.engine.models.families.hidream_o1.trainer import T_EPS

    target = "app.engine.models.families.hidream_o1.trainer.build_t2i_text_sample"
    configs = (
        ("logit_normal", dict(timestep_sampling="logit_normal", logit_normal_mu=0.0, logit_normal_sigma=1.0)),
        ("radc", dict(timestep_sampling="radc", radc_start=0.8)),
    )
    for mode, config in configs:
        trainer, model, batch, text_sample = _hidream_trainer_and_batch(**config)
        trainer.global_step, trainer.max_train_steps = 30, 60
        drivers = (("forward_pass", lambda: trainer.forward_pass(batch["images"], None, None, batch)),
                   ("compute_loss", lambda: trainer.compute_loss(batch)))
        for name, run in drivers:
            spy.calls.clear()
            model.timesteps.clear()
            torch.manual_seed(0)
            with patch(target, return_value=text_sample):
                run()
            assert spy.draws(mode), (
                f"TimestepSampler.sample was not called for mode '{mode}' (hidream_o1 draws its own _sample_sigma)")
            assert [(c.mode, c.bs) for c in spy.draws()] == [(mode, 8)], (
                f"{name}: expected one factory draw ({mode}, batch 8), got "
                f"{[(c.mode, c.bs) for c in spy.draws()]}")
            if mode == "radc":
                assert spy.draws()[0].progress == 0.5, (
                    f"{name}: the factory got progress {spy.draws()[0].progress}, expected 0.5 (step 30 of 60)")
            sigma = 1.0 - model.timesteps[-1]
            assert sigma.shape == (8,), f"{name}: expected 8 sigmas, got shape {tuple(sigma.shape)}"
            assert bool((sigma >= T_EPS - 1e-6).all() and (sigma <= 0.9999 + 1e-6).all()), (
                f"{name}: sigmas leave [{T_EPS}, 0.9999]: {sigma.tolist()}")


# ── 3. min_snr_gamma capability rule ─────────────────────────────────────────


def test_min_snr_gamma_is_hidden_for_every_family_but_sdxl():
    failures: list[str] = []
    checked = 0
    for defn in _definitions():
        checked += 1
        rule = resolve_capabilities(defn)["field_visibility"].get("min_snr_gamma")
        is_sdxl = defn.family == "sdxl"
        where = f"{defn.family}/{defn.id} min_snr_gamma"
        if rule is None:
            failures.append(f"{where}: no field_visibility rule at all (a missing rule counts as supported)")
        elif bool(rule.get("supported")) != is_sdxl:
            failures.append(f"{where}: supported={rule.get('supported')} but must be {is_sdxl}")
        elif not is_sdxl and "sdxl" not in str(rule.get("reason", "")).lower():
            failures.append(f"{where}: hidden without a reason naming SDXL (reason={rule.get('reason')!r})")
        config = {"min_snr_gamma": 7.0}
        dropped = apply_capability_allowlist(config, defn)
        if is_sdxl and (dropped or config.get("min_snr_gamma") != 7.0):
            failures.append(f"{where}: the allowlist dropped min_snr_gamma for an sdxl definition")
        if not is_sdxl and ("min_snr_gamma" in config or dropped != ["min_snr_gamma"]):
            failures.append(f"{where}: the allowlist kept min_snr_gamma (dropped={dropped}, config={config})")
    assert checked >= 32, f"only {checked} definitions in the registry -- the registry did not load"
    assert not failures, f"{len(failures)} min_snr_gamma visibility failure(s):\n" + "\n".join(failures)


# ── 4. the registry-wide contract ────────────────────────────────────────────

_RANDOM_NOISE_CALLS = {"randn", "randn_like", "normal", "normal_"}
_NOISE_METHODS = ("forward_pass", "compute_loss", "_compute_step_loss", "prepare_noise_for_training")


def _own_noise_methods(trainer_cls) -> list[str]:
    """Methods of the FAMILY trainer classes that draw fresh gaussian noise (so discard the shared
    loop's noise, offset included). The shared loop draws and offsets noise at one site
    (``pipeline_train.py``), so a family whose trainer redraws it is the only way to lose the offset."""
    found: list[str] = []
    for klass in trainer_cls.__mro__:
        if not klass.__module__.startswith("app.engine.models.families."):
            continue
        for name in _NOISE_METHODS:
            fn = klass.__dict__.get(name)
            if fn is None:
                continue
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                    if called in _RANDOM_NOISE_CALLS:
                        found.append(f"{klass.__name__}.{name}")
                        break
    return found


def _discards_shared_timesteps(trainer_cls) -> list[str]:
    """Family-trainer ``forward_pass``/``compute_loss`` overrides that neither read the ``timesteps`` the
    shared loop drew via ``sample_timesteps`` nor call the factory themselves: ``sample_timesteps`` can
    then reach the factory while the sigma the model trains on comes from somewhere else."""
    found: list[str] = []
    for klass in trainer_cls.__mro__:
        if not klass.__module__.startswith("app.engine.models.families."):
            continue
        for name in ("forward_pass", "compute_loss"):
            fn = klass.__dict__.get(name)
            if fn is None:
                continue
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            reads = ("timesteps" in names) or ("sample_timesteps" in attrs) or ("TimestepSampler" in names)
            if not reads:
                found.append(f"{klass.__name__}.{name}")
    return found


def _observe_timestep(defn, spy) -> str | None:
    spy.calls.clear()
    trainer = _trainer(defn, timestep_sampling=TIMESTEP_SENTINEL, radc_start=0.8)
    trainer.sample_timesteps(4)
    discarded = _discards_shared_timesteps(type(trainer))
    if discarded:
        return f"{', '.join(discarded)} neither read the loop's timesteps nor call the factory (own sigma draw)"
    if spy.draws(TIMESTEP_SENTINEL):
        return None
    return f"TimestepSampler.sample/sample_scaled never saw mode {TIMESTEP_SENTINEL!r} (modes seen: {[c.mode for c in spy.draws()]})"


def _observe_noise_offset(defn, spy) -> str | None:
    if defn.family == "minimax_h3":
        quiet = _trainer(defn, noise_offset=0.0)
        quiet._refuse_unproven_settings()  # the negative: no offset, no refusal
        try:  # the refusal may fire while the trainer is built or when it validates; either is explicit
            _trainer(defn, noise_offset=0.1)._refuse_unproven_settings()
        except ValueError as exc:
            if "noise_offset" in str(exc):
                return None
            return f"the validation error does not name noise_offset: {exc}"
        return "noise_offset=0.1 raised no validation error"
    trainer = _trainer(defn, noise_offset=0.1)
    redraws = _own_noise_methods(type(trainer))
    if redraws:
        return f"{', '.join(redraws)} draw fresh noise, so the shared loop's noise offset (pipeline_train.py) never reaches the model"
    return None


def _observe_min_snr(defn, spy) -> str | None:
    timesteps = torch.tensor([10, 300, 800, 990])
    weights = []
    for gamma in (1.0, 50.0):
        trainer = _trainer(defn, min_snr_gamma=gamma)
        weights.append(trainer.compute_loss_weight(timesteps))
    if any(w is None for w in weights):
        return "compute_loss_weight returns None (uniform weighting): min_snr_gamma changes no loss weight"
    if torch.equal(weights[0].float(), weights[1].float()):
        return "compute_loss_weight is identical for min_snr_gamma 1.0 and 50.0"
    return None


def _configure_optimization(defn, **config):
    trainer = _trainer(defn, **config)
    model = nn.Linear(2, 2)
    trainer._get_primary_model = lambda: model
    trainer._name_optimizer_params = lambda params: [f"p{i}" for i in range(len(params))]
    trainer._get_text_encoders = lambda: {}
    try:
        trainer.driver.assign_components({"unet": model})
    except Exception:  # noqa: BLE001 -- drivers that need no model for the optimizer step ignore it
        pass
    # bernini_r (14B) and wan22 wrap the shared method around their expert pair and call super();
    # the shared implementation is the sink the contract is about, so drive it directly.
    shared = next(k for k in type(trainer).__mro__
                  if "_configure_optimization" in k.__dict__
                  and not k.__module__.startswith("app.engine.models.families."))
    shared.__dict__["_configure_optimization"](trainer, 10)


def _observe_factory(defn, spy, field) -> str | None:
    seen: dict[str, object] = {}

    def optimizer_create(optimizer_type, params, lr, weight_decay, **kwargs):
        seen["optimizer_type"] = optimizer_type
        return torch.optim.SGD(params, lr=lr)

    def scheduler_create(sched_type, optimizer, warmup, total):
        seen["lr_scheduler"] = sched_type
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    sentinel = {"optimizer_type": OPTIMIZER_SENTINEL, "lr_scheduler": SCHEDULER_SENTINEL}[field]
    with patch.object(OptimizerFactory, "create", staticmethod(optimizer_create)), \
            patch.object(LRSchedulerFactory, "create", staticmethod(scheduler_create)):
        _configure_optimization(defn, **{field: sentinel})
    if seen.get(field) == sentinel:
        return None
    return f"{'OptimizerFactory' if field == 'optimizer_type' else 'LRSchedulerFactory'}.create saw {seen.get(field)!r}, not {sentinel!r}"


def test_every_shown_engine_field_is_observed(spy):
    observers = {
        "timestep_sampling": _observe_timestep,
        "noise_offset": _observe_noise_offset,
        "min_snr_gamma": _observe_min_snr,
        "optimizer_type": lambda d, s: _observe_factory(d, s, "optimizer_type"),
        "lr_scheduler": lambda d, s: _observe_factory(d, s, "lr_scheduler"),
    }
    failures: list[str] = []
    skipped: list[str] = []
    unverified: list[str] = []
    observed = 0
    definitions = _definitions()
    for defn in definitions:
        visibility = resolve_capabilities(defn)["field_visibility"]
        for field in ENGINE_FIELDS:
            rule = visibility.get(field)
            if rule is not None and not rule.get("supported", True):
                skipped.append(f"{defn.family}/{defn.id} {field}: hidden ({rule.get('reason')})")
                continue
            try:
                problem = observers[field](defn, spy)
            except Exception as exc:  # noqa: BLE001 -- a family that cannot be driven is reported by name
                unverified.append(f"{defn.family}/{defn.id} {field}: UNVERIFIED, driving raised {type(exc).__name__}: {exc}")
                continue
            if problem is None:
                observed += 1
            else:
                failures.append(f"{defn.family}/{defn.id} {field}: shown but never reached its sink -- {problem}")

    report = [f"{len(failures)} shown-but-unobserved, {len(unverified)} unverified, {len(skipped)} skipped "
              f"(hidden by name), {observed} observed over {len(definitions)} definitions"]
    report += failures + unverified + [f"SKIPPED {s}" for s in skipped]
    print("\n".join(report))
    assert len(definitions) >= 32, f"only {len(definitions)} definitions in the registry -- it did not load"
    assert not failures and not unverified, "\n".join(report)
