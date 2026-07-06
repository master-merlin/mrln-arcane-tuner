"""BL1-6: Kontext no-control guidance fallback must match Kontext's own default.

``Flux1Trainer._sync_driver_forward_state`` reads ``config["guidance_scale"]``
with a class-level ``_GUIDANCE_DEFAULT`` fallback when the config omits the
key. ``Flux1KontextTrainer.forward_pass`` itself defaults to 1.0 on its own
(non-super) path, but before this fix the SYNCED driver default was
hard-coded to 3.5 — so a no-control batch that fell through to
``super().forward_pass`` (which reads the driver-synced value) would train
at 3.5 while every other Kontext forward used 1.0.

These pin the fallback with a bare config (no ``guidance_scale`` key) on
purpose — YAML defaults normally supply the key and would mask the bug.
"""

from __future__ import annotations

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.flux1.driver import Flux1Driver
from app.engine.models.families.flux1.trainer import Flux1Trainer
from app.engine.models.families.flux1.trainer_kontext import Flux1KontextTrainer


def _defn() -> ModelDefinition:
    return ModelDefinition(
        id="flux1-dev", family="flux1", name="FLUX.1 Dev", defaults={}, components={},
    )


def _bare_trainer(cls) -> Flux1Trainer:
    """A trainer instance with a config lacking ``guidance_scale`` entirely."""
    t = object.__new__(cls)
    t.device = torch.device("cpu")
    t.config = {}  # deliberately no "guidance_scale" key
    t.autocast_dtype = torch.float32
    t.driver = Flux1Driver(_defn(), torch.device("cpu"))
    return t


def test_flux1_driver_defaults_to_3_5_without_config_key():
    t = _bare_trainer(Flux1Trainer)
    latents = torch.randn(1, 16, 8, 8)

    t.prepare_latents_for_training(latents)

    assert t.driver.guidance_scale == 3.5


def test_kontext_driver_defaults_to_1_0_without_config_key():
    """Kontext overrides ``_GUIDANCE_DEFAULT`` so the driver-synced fallback
    (used when a no-control batch falls through to ``super().forward_pass``)
    agrees with Kontext's own forward_pass default (1.0), not flux1's 3.5."""
    t = _bare_trainer(Flux1KontextTrainer)
    latents = torch.randn(1, 16, 8, 8)

    t.prepare_latents_for_training(latents)

    assert t.driver.guidance_scale == 1.0
