"""LANE-168 task 3: hidream_o1 takes its sigmas from the shared ``TimestepSampler`` on both draws.

Before this lane ``forward_pass`` and ``compute_loss`` each drew sigma through a private
``_sample_sigma`` keyed on the non-form ``timestep_type``, so the configured ``timestep_sampling``
(and its sub-fields) never reached the trainer. Now the configured mode is the draw (progress
forwarded on BOTH paths), clamped to the pixel-space bounds ``[T_EPS, 0.9999]``. The two modes
hidream_o1 cannot honour in pixel space are replaced under a logged warning, never silently:
``flux_shift`` / ``model_shift`` -> ``uniform``, and a ``radc_res_influence > 0`` -> ``0.0``.

CPU only, no weights: the real trainer class around its real driver, a tiny stand-in model that
records the timestep it is called with (sigma = 1 - timestep), and a recorder on the real factory.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from app.engine.models.registry import registry
from app.engine.strategies.timestep_sampling import TimestepSampler

TEXT_SAMPLE_TARGET = "app.engine.models.families.hidream_o1.trainer.build_t2i_text_sample"
BATCH = 8


class _Recorder:
    """A logger double that records ``(level, event, fields)`` -- avoids structlog caching traps."""

    def __init__(self):
        self.events: list[tuple[str, str, dict]] = []

    def info(self, event, **kw):
        self.events.append(("info", event, kw))

    def warning(self, event, **kw):
        self.events.append(("warning", event, kw))

    def debug(self, event, **kw):
        self.events.append(("debug", event, kw))

    def named(self, event):
        return [e for e in self.events if e[1] == event]


class _Model(nn.Module):
    """Stands in for the Qwen3-VL pixel model: returns a grad-carrying x_pred, keeps the timestep."""

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


@pytest.fixture(scope="module", autouse=True)
def _registry():
    registry.initialize()
    return registry


@pytest.fixture
def draws(monkeypatch):
    """Every ``TimestepSampler.sample`` call (mode, bs, progress, config); still draws for real."""
    calls: list[SimpleNamespace] = []
    real = TimestepSampler.sample

    def sample(mode, bs, device, config, *, latents=None, progress=0.0):
        calls.append(SimpleNamespace(mode=mode, bs=bs, progress=progress, config=dict(config)))
        return real(mode, bs, device, config, latents=latents, progress=progress)

    monkeypatch.setattr(TimestepSampler, "sample", staticmethod(sample))
    return calls


def _setup(step=30, max_steps=60, **config):
    from app.engine.models.families.hidream_o1.trainer import PATCH_SIZE
    from tests.test_hidream_o1_integration import _make_mock_text_sample

    defn = registry.get_definition("hidream_o1_image")
    assert defn is not None, "definition hidream_o1_image is not in the registry"
    family = object.__new__(registry.get_family_class(defn.family))
    family.definition = defn
    family.config = {}
    trainer = object.__new__(family.get_trainer_class())
    trainer.definition = defn
    trainer.config = {"mixed_precision": "fp32", **config}
    trainer.device = torch.device("cpu")
    trainer.logger = _Recorder()
    trainer.components = {}
    trainer.global_step = step
    trainer.max_train_steps = max_steps
    trainer._setup_family()
    model = _Model(3 * PATCH_SIZE * PATCH_SIZE)
    trainer.driver.assign_components({"unet": model})
    trainer.components["unet"] = model
    trainer.processor = MagicMock()
    trainer.tokenizer = MagicMock()
    trainer.autocast_dtype = torch.float32
    side = PATCH_SIZE * 2
    n_patches = (side // PATCH_SIZE) ** 2
    batch = {"images": torch.randn(BATCH, 3, side, side), "captions": ["a caption"] * BATCH}
    text_sample = _make_mock_text_sample(seq_len=n_patches, n_visual=n_patches)
    return trainer, model, batch, text_sample


def _run(path, trainer, model, batch, text_sample, seed=0) -> torch.Tensor:
    """One draw through ``path``; returns the sigmas the model saw."""
    torch.manual_seed(seed)
    with patch(TEXT_SAMPLE_TARGET, return_value=text_sample):
        if path == "forward_pass":
            trainer.forward_pass(batch["images"], None, None, batch)
        else:
            trainer.compute_loss(batch)
    return 1.0 - model.timesteps[-1]


PATHS = ("forward_pass", "compute_loss")


# ── Scenario 4: the configured mode is the draw; timestep_type is ignored ───────


@pytest.mark.parametrize("path", PATHS)
def test_logit_normal_reaches_the_factory_and_timestep_type_is_ignored(draws, path):
    from app.engine.models.families.hidream_o1.trainer import T_EPS

    # ``timestep_type: sigmoid`` was the old private switch; it must change nothing now.
    trainer, model, batch, ts = _setup(timestep_sampling="logit_normal", logit_normal_mu=0.0,
                                       logit_normal_sigma=1.0, timestep_type="sigmoid")
    sigma = _run(path, trainer, model, batch, ts)

    assert [(c.mode, c.bs) for c in draws] == [("logit_normal", BATCH)], (
        f"{path}: expected one factory draw (logit_normal, {BATCH}), got {[(c.mode, c.bs) for c in draws]}")
    assert sigma.shape == (BATCH,)
    assert bool((sigma >= T_EPS - 1e-6).all() and (sigma <= 0.9999 + 1e-6).all()), (
        f"{path}: sigmas leave [{T_EPS}, 0.9999]: {sigma.tolist()}")


def test_the_private_sampler_and_timestep_type_are_gone():
    import app.engine.models.families.hidream_o1.trainer as mod

    assert not hasattr(mod, "_sample_sigma"), "the private _sample_sigma still exists beside the factory"
    trainer, *_ = _setup(timestep_sampling="uniform")
    assert "timestep_type" not in trainer.config, "_setup_family still seeds the non-form timestep_type"


# ── Scenario 5: radc progress on both draws ────────────────────────────────────


@pytest.mark.parametrize("path", PATHS)
def test_radc_gets_the_progress_on_both_draws(draws, path):
    trainer, model, batch, ts = _setup(step=30, max_steps=60, timestep_sampling="radc",
                                       radc_res_influence=0.0)
    _run(path, trainer, model, batch, ts)
    assert [(c.mode, c.progress) for c in draws] == [("radc", 0.5)], (
        f"{path}: expected one radc draw at progress 0.5 (step 30 of 60), got "
        f"{[(c.mode, c.progress) for c in draws]}")


@pytest.mark.parametrize("path", PATHS)
def test_radc_draws_at_progress_0_and_0_9_differ(draws, path):
    early = _run(path, *_setup(step=0, max_steps=60, timestep_sampling="radc", radc_res_influence=0.0))
    late = _run(path, *_setup(step=54, max_steps=60, timestep_sampling="radc", radc_res_influence=0.0))
    assert [c.progress for c in draws] == [0.0, 0.9]
    assert not torch.allclose(early, late), f"{path}: the radc curriculum did not move the sigmas"


# ── Scenario 6: the YAML preset ────────────────────────────────────────────────


def test_definition_presets_uniform():
    defn = registry.get_definition("hidream_o1_image")
    assert (defn.defaults or {}).get("timestep_sampling") == "uniform", (
        f"hidream_o1_image presets timestep_sampling={(defn.defaults or {}).get('timestep_sampling')!r}")


# ── Scenario 7: the two replacements are logged once per job, never silent ─────


@pytest.mark.parametrize("mode", ["flux_shift", "model_shift"])
def test_resolution_shift_modes_become_uniform_under_one_warning(draws, mode):
    trainer, model, batch, ts = _setup(timestep_sampling=mode)
    replaced = [_run(p, trainer, model, batch, ts, seed=7) for p in PATHS]
    ref_trainer, ref_model, ref_batch, ref_ts = _setup(timestep_sampling="uniform")
    expected = [_run(p, ref_trainer, ref_model, ref_batch, ref_ts, seed=7) for p in PATHS]

    assert [c.mode for c in draws] == ["uniform"] * 4, f"{mode} reached the factory: {[c.mode for c in draws]}"
    for got, want in zip(replaced, expected):
        assert torch.equal(got, want), f"{mode} -> uniform draws differ from a uniform job at the same seed"
    warnings = trainer.logger.named("hidream_o1_timestep_mode_replaced")
    assert len(warnings) == 1 and warnings[0][0] == "warning", f"expected ONE warning, got {warnings}"
    assert warnings[0][2].get("configured") == mode and warnings[0][2].get("used") == "uniform"
    info = trainer.logger.named("timestep_sampling_mode")
    assert len(info) == 1 and info[0][2]["configured"] == mode and info[0][2]["effective"] == "uniform"


def test_radc_res_influence_is_forced_to_zero_under_one_warning(draws):
    trainer, model, batch, ts = _setup(timestep_sampling="radc", radc_res_influence=0.15)
    forced = [_run(p, trainer, model, batch, ts, seed=3) for p in PATHS]
    ref = _setup(timestep_sampling="radc", radc_res_influence=0.0)
    expected = [_run(p, *ref, seed=3) for p in PATHS]

    assert [c.config.get("radc_res_influence") for c in draws] == [0.0] * 4, (
        "the factory did not receive radc_res_influence 0.0")
    assert trainer.config["radc_res_influence"] == 0.15, "the job config itself must not be rewritten"
    for got, want in zip(forced, expected):
        assert torch.equal(got, want)
    warnings = trainer.logger.named("hidream_o1_radc_res_influence_ignored")
    assert len(warnings) == 1 and warnings[0][0] == "warning", f"expected ONE warning, got {warnings}"
    assert warnings[0][2].get("configured") == 0.15 and warnings[0][2].get("used") == 0.0
    info = trainer.logger.named("timestep_sampling_mode")
    assert len(info) == 1 and info[0][2]["res_influence"] == 0.0


def test_radc_at_zero_influence_logs_the_mode_and_no_replacement(draws):
    """Plan review 2.02: an honoured radc names itself and raises no warning."""
    trainer, model, batch, ts = _setup(timestep_sampling="radc", radc_res_influence=0.0)
    for p in PATHS:
        _run(p, trainer, model, batch, ts)
    info = trainer.logger.named("timestep_sampling_mode")
    assert len(info) == 1, f"expected the mode logged once per job, got {info}"
    assert info[0][0] == "info"
    assert info[0][2] == {"family": "hidream_o1", "configured": "radc", "effective": "radc",
                          "progress_forwarded": True, "res_influence": 0.0}
    assert not [e for e in trainer.logger.events if e[0] == "warning"], trainer.logger.events
