"""LANE-167 RED: moss_image's timesteps come from the shared ``TimestepSampler`` and its
``scheduler.shift`` survives definition enrichment.

Before this lane the trainer drew its own ``sigmoid(z)`` shifted by ``scheduler.training_shift`` and
ignored the run config's ``timestep_sampling``; both definition YAMLs left ``scheduler.shift`` open to
the harvest of the repo's grid shift 1.0. The trainer shell and the stubbed driver are the LANE-165
contract test's own helpers (CPU only, no weights).
"""

from __future__ import annotations

import importlib
import shutil

import pytest
import structlog
import torch
import yaml

from app.engine.models.registry import registry
from app.engine.strategies.timestep_sampling import TimestepSampler
from tests.test_moss_image_contracts import BASE_ID, FAMILY_DIR, SFT_ID, _bare_trainer, _driver

TRAINER_MODULE = "app.engine.models.families.moss_image.trainer"
FAKE_SIGMAS = [0.1, 0.4, 0.6, 0.9]


def _trainer(**config):
    driver, _te, _proc, _spy = _driver()
    base = {"mixed_precision": "fp32", "te_quantization": "none",
            "cache_text_embeddings": True, "sample_prompts": []}
    return _bare_trainer(driver, {**base, **config})


def _median_sigma(trainer, n: int = 4096) -> float:
    torch.manual_seed(0)
    return float((trainer.sample_timesteps(n) / 1000.0).median())


def test_moss_image_timestep_mode_reaches_factory(monkeypatch):
    calls: list[tuple] = []

    def recorder(mode, bs, device, config, *, latents=None, progress=0.0):
        calls.append((mode, bs, progress, latents, config.get("flux_shift_patchify_factor")))
        return torch.tensor(FAKE_SIGMAS)

    monkeypatch.setattr(TimestepSampler, "sample", staticmethod(recorder))
    latents = torch.zeros(4, 32, 64, 64)
    trainer = _trainer(timestep_sampling="radc", radc_start=0.8)
    trainer.global_step, trainer.max_train_steps = 30, 60

    timesteps = trainer.sample_timesteps(4, latents)

    assert [c[:3] for c in calls] == [("radc", 4, 0.5)], (
        "TimestepSampler.sample was not called for mode 'radc' (moss_image draws its own sigmoid)")
    assert calls[0][3] is latents, "the batch latents were not handed to the factory"
    assert calls[0][4] == 2, f"patchify factor default is {calls[0][4]}, expected 2 (2x2 patchify)"
    assert torch.allclose(timesteps, torch.tensor(FAKE_SIGMAS) * 1000.0, atol=1e-3), (
        f"timesteps must be the factory's sigmas * 1000, got {timesteps.tolist()}")

    # a user-set factor is kept (setdefault, never an override)
    calls.clear()
    user = _trainer(timestep_sampling="radc", flux_shift_patchify_factor=1)
    user.sample_timesteps(4, latents)
    assert calls and calls[0][4] == 1, (
        f"TimestepSampler.sample was not called with the user's flux_shift_patchify_factor 1: {calls}")


def test_moss_image_default_is_report_logit_normal():
    trainer = _trainer()
    median = _median_sigma(trainer)
    assert abs(median - 0.748) <= 0.02, (
        f"default median sigma {median:.4f} is not the report's logit-normal median 0.748 "
        "(1 - sigmoid(Z), Z ~ N(-1.09, 0.95^2))")
    # The median alone cannot tell the old shifted N(0,1) draw (median 0.75 too) from the report's
    # distribution; the spread of logit(sigma) can: 0.95 here, 1.0 for the old draw.
    torch.manual_seed(0)
    sigma = (trainer.sample_timesteps(4096) / 1000.0).double()
    z = torch.log(sigma / (1.0 - sigma))
    assert abs(float(z.std()) - 0.95) <= 0.025, (
        f"default logit(sigma) std {float(z.std()):.4f} is not the report's 0.95 (median {median:.4f})")
    assert abs(float(z.mean()) - 1.09) <= 0.05, (
        f"default logit(sigma) mean {float(z.mean()):.4f} is not the report's 1.09")


def test_moss_image_user_logit_normal_wins():
    trainer = _trainer(timestep_sampling="logit_normal", logit_normal_mu=0.0, logit_normal_sigma=1.0)
    median = _median_sigma(trainer)
    assert abs(median - 0.5) <= 0.02, (
        f"median sigma {median:.4f} with the user's mu 0 / sigma 1 is not 0.5: a family default or "
        "shift overrode the user's values")


def test_moss_image_no_family_shift_on_flux_shift():
    trainer = _trainer(timestep_sampling="flux_shift")
    lat = torch.zeros(2, 32, 64, 64)
    torch.manual_seed(0)
    expected = TimestepSampler.sample(
        "flux_shift", 2, trainer.device,
        {**trainer.config, "timestep_sampling": "flux_shift", "flux_shift_patchify_factor": 2},
        latents=lat)
    torch.manual_seed(0)
    with structlog.testing.capture_logs() as logs:
        got = trainer.sample_timesteps(2, lat) / 1000.0

    assert torch.allclose(got, expected), (
        f"sigmas are not the factory's flux_shift draw (a family shift or own sampler): "
        f"got {got.tolist()}, expected {expected.tolist()}")
    replaced = [e for e in logs if "replaced" in str(e.get("event", ""))]
    assert not replaced, f"a replaced-mode warning was logged: {replaced}"


@pytest.fixture
def enrich_copy(tmp_path):
    """The real enricher on a COPY of a shipped definition (never rewrites the tracked YAML)."""
    registry.initialize()
    saved: dict[str, tuple] = {}

    def run(did: str, filename: str, harvested: dict):
        import app.engine.utils.config_harvester as harvester

        copy = tmp_path / filename
        shutil.copyfile(FAMILY_DIR / "definitions" / filename, copy)
        saved.setdefault(did, (registry._definitions[did], registry._paths[did]))
        registry.load_definition(str(copy))
        original = harvester.harvest
        harvester.harvest = lambda _root: dict(harvested)
        try:
            registry.enrich_definition(did, {}, root_path="fake-root-no-checkpoint")
        finally:
            harvester.harvest = original
        return registry.get_definition(did)

    try:
        yield run
    finally:
        for did, (defn, path) in saved.items():
            registry._definitions[did] = defn
            registry._paths[did] = path


def test_moss_image_scheduler_keys_pinned(enrich_copy):
    for did, filename in ((BASE_ID, "base.yaml"), (SFT_ID, "sft.yaml")):
        data = yaml.safe_load((FAMILY_DIR / "definitions" / filename).read_text(encoding="utf-8"))
        assert "scheduler.shift" in (data.get("enrich_pinned_keys") or []), (
            f"'scheduler.shift' not in enrich_pinned_keys of {did}")
        assert "scheduler.training_shift" not in data["architecture_params"], (
            f"{did} still declares scheduler.training_shift")

    for did, filename in ((BASE_ID, "base.yaml"), (SFT_ID, "sft.yaml")):
        with structlog.testing.capture_logs() as logs:
            defn = enrich_copy(did, filename, {"scheduler.shift": 1.0})
        assert defn.architecture_params["scheduler.shift"] == 5.0, (
            f"{did}: enrichment replaced scheduler.shift 5.0 with the harvest: "
            f"{defn.architecture_params['scheduler.shift']}")
        drift = [e for e in logs if e.get("event") == "config_drift_detected"
                 and e.get("key") == "scheduler.shift" and not e.get("pinned")]
        assert not drift, f"{did}: unpinned config_drift_detected for scheduler.shift: {drift}"


def test_moss_image_harvested_training_shift_is_inert(enrich_copy):
    """Enrichment appends the repo scheduler config; the appended training_shift is dead data."""
    harvested = {"scheduler.shift": 1.0, "scheduler.training_shift": 1.0}
    for did, filename in ((BASE_ID, "base.yaml"), (SFT_ID, "sft.yaml")):
        shipped = yaml.safe_load((FAMILY_DIR / "definitions" / filename).read_text(encoding="utf-8"))
        plain_defn = registry.get_definition(did)
        driver, *_ = _driver(plain_defn)
        plain = _bare_trainer(driver, {"mixed_precision": "fp32", "te_quantization": "none",
                                       "cache_text_embeddings": True, "sample_prompts": []})
        torch.manual_seed(0)
        before = plain.sample_timesteps(256)

        enriched = enrich_copy(did, filename, harvested)
        assert enriched.architecture_params["scheduler.shift"] == shipped["architecture_params"][
            "scheduler.shift"] == 5.0, f"{did}: the pin did not hold through the harvest"
        assert enriched.architecture_params.get("scheduler.training_shift") == 1.0, (
            f"{did}: the harvest did not append scheduler.training_shift (test premise broken)")

        driver2, *_ = _driver(enriched)
        after_trainer = _bare_trainer(driver2, {"mixed_precision": "fp32", "te_quantization": "none",
                                                "cache_text_embeddings": True, "sample_prompts": []})
        torch.manual_seed(0)
        after = after_trainer.sample_timesteps(256)
        assert torch.equal(before, after), (
            f"{did}: an appended scheduler.training_shift changed sample_timesteps "
            f"(max abs diff {float((before - after).abs().max()):.4f})")


def test_moss_image_private_sampler_gone():
    module = importlib.import_module(TRAINER_MODULE)
    assert not hasattr(module, "shift_sigma"), "trainer still defines the private shift_sigma"
    assert not hasattr(module, "DEFAULT_TRAINING_SHIFT"), "trainer still defines DEFAULT_TRAINING_SHIFT"
    assert not hasattr(module.MossImageTrainer, "sample_sigmas"), (
        "MossImageTrainer still defines the private sample_sigmas")
