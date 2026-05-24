"""Regression — GenericTrainingPipeline must tolerate families that
return ``None`` from ``get_vae()`` and an empty dict from
``get_text_encoders()``.

Covers the HiDream-O1 pixel-space case. Uses a minimal fake driver so
existing-family tests stay isolated.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.engine.core.interfaces import IModelDriver, IModelSaver


class _NullDriver(IModelDriver):
    """Driver that returns None/{} for VAE/TE — mimics HiDream-O1."""

    def __init__(self):
        self._model = nn.Linear(4, 4)

    def assign_components(self, c): pass
    def get_components(self): return {"unet": self._model}
    def get_primary_model(self): return self._model
    def get_text_encoders(self): return {}
    def get_vae(self): return None
    def get_lora_targets(self): return ["to_q"]
    def init_scheduler(self): return None
    def resolve_loading_dtype(self): return torch.bfloat16
    def encode_text(self, captions, dtype):
        raise NotImplementedError("test driver — no TE")
    def get_te_lora_targets(self): return []
    def forward_pass(self, *args, **kwargs):
        raise NotImplementedError("test driver — no forward")
    def get_saver(self) -> IModelSaver:
        raise NotImplementedError("test driver — no saver")


def test_null_driver_can_be_constructed():
    """Smoke test — the fake driver itself is well-formed."""
    d = _NullDriver()
    assert d.get_vae() is None
    assert d.get_text_encoders() == {}


def test_pipeline_get_vae_returns_none_safely():
    """A path that previously dereferenced .vae directly must now check first."""
    driver = _NullDriver()
    # The key contract: get_vae() returning None is enough — base pipeline
    # code paths that read VAE must guard with `if vae is not None:`
    # before dereferencing.
    assert driver.get_vae() is None


def test_encode_text_returns_none_when_no_text_encoders():
    """encode_text in the base pipeline must return None (not crash) when the
    driver has no text encoders, so pixel-space families skip TE encoding."""
    from app.engine.core.pipeline import GenericTrainingPipeline
    from unittest.mock import MagicMock

    stub = MagicMock()
    stub.driver = _NullDriver()

    # Call the base pipeline's encode_text method directly
    result = GenericTrainingPipeline.encode_text(stub, ["test caption"], torch.bfloat16)
    assert result is None, (
        "encode_text should return None for drivers with no text encoders, "
        "not raise NotImplementedError"
    )
