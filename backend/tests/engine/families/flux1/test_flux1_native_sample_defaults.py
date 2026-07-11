"""W3-1: FLUX.1 native in-training-preview defaults, definition-sourced.

The Flux1Sampler (and the Kontext subclass that inherits it) fills the native
``num_inference_steps`` from each definition's ``defaults`` block instead of
the generic base's off-distribution 20 steps:

* Dev / Kontext: 28 steps (FluxPipeline / FluxKontextPipeline __call__ default).
* Schnell: 4 steps (timestep-distilled).

Guidance is intentionally NOT definition-sourced: ``defaults.guidance_scale``
is the TRAINING guidance-embed value (Kontext trains at 1.0), while the native
SAMPLE guidance for Dev/Kontext (3.5) already matches the generic passthrough.
Sourcing it would sample Kontext at a weak 1.0 — so the sampler leaves
``guidance_scale`` alone and only overrides the step count + resolution.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from app.engine.core.sampling import GenericSamplingPipeline
from app.engine.models.families.flux1.sampler import Flux1Sampler
from app.engine.models.families.flux1.sampler_kontext import Flux1KontextSampler


def _build_sampler(cls, defaults: dict):
    pipeline = MagicMock()
    pipeline.config = {"sample_every_n_steps": 50, "sample_negative_prompt": ""}
    pipeline.device = torch.device("cpu")
    defn = MagicMock()
    defn.defaults = dict(defaults)
    pipeline.definition = defn
    return cls(pipeline)


def _capture(sampler, prompt_cfg, monkeypatch):
    captured: dict = {}

    def _fake_base(self, cfg, step):
        captured.update(cfg)
        return MagicMock()

    monkeypatch.setattr(GenericSamplingPipeline, "_sample_single", _fake_base)
    sampler._sample_single(prompt_cfg, 0)
    return captured


class TestFlux1NativeSampleDefaults:
    def test_dev_fills_28_steps_and_leaves_guidance_untouched(self, monkeypatch):
        sampler = _build_sampler(Flux1Sampler, {"num_inference_steps": 28, "resolution": 1024})
        captured = _capture(sampler, {"prompt": "dev"}, monkeypatch)
        assert captured["num_inference_steps"] == 28
        assert captured["width"] == 1024 and captured["height"] == 1024
        # Guidance is NOT filled by the sampler (generic base supplies its own).
        assert "guidance_scale" not in captured

    def test_schnell_fills_4_steps(self, monkeypatch):
        sampler = _build_sampler(Flux1Sampler, {"num_inference_steps": 4, "resolution": 1024})
        captured = _capture(sampler, {"prompt": "schnell"}, monkeypatch)
        assert captured["num_inference_steps"] == 4

    def test_kontext_inherits_and_fills_28_steps(self, monkeypatch):
        sampler = _build_sampler(
            Flux1KontextSampler,
            {"num_inference_steps": 28, "guidance_scale": 1.0, "resolution": 1024},
        )
        captured = _capture(sampler, {"prompt": "edit"}, monkeypatch)
        assert captured["num_inference_steps"] == 28
        # The TRAINING guidance (1.0) in defaults must NOT leak into the sample
        # config — the sampler leaves guidance to the native 3.5 passthrough.
        assert "guidance_scale" not in captured

    def test_definition_sourced_not_hardcoded(self, monkeypatch):
        """A sentinel step count in defaults passes through (proves sourcing)."""
        sampler = _build_sampler(Flux1Sampler, {"num_inference_steps": 17, "resolution": 512})
        captured = _capture(sampler, {"prompt": "sourced"}, monkeypatch)
        assert captured["num_inference_steps"] == 17
        assert captured["width"] == 512

    def test_explicit_values_win(self, monkeypatch):
        sampler = _build_sampler(Flux1Sampler, {"num_inference_steps": 28, "resolution": 1024})
        captured = _capture(
            sampler,
            {"prompt": "x", "num_inference_steps": 9, "width": 640, "height": 640},
            monkeypatch,
        )
        assert captured["num_inference_steps"] == 9
        assert captured["width"] == 640

    def test_falls_back_to_constant_when_defaults_missing(self, monkeypatch):
        sampler = _build_sampler(Flux1Sampler, {})
        captured = _capture(sampler, {"prompt": "bare"}, monkeypatch)
        assert captured["num_inference_steps"] == 28  # _FLUX1_DEFAULT_STEPS


@pytest.mark.parametrize(
    "def_id,expected_steps",
    [("dev.yaml", 28), ("schnell.yaml", 4), ("kontext_dev.yaml", 28)],
)
def test_shipped_yaml_carries_native_step_count(def_id, expected_steps):
    """Each shipped flux1 definition pins its native preview step count."""
    import pathlib

    import yaml

    path = (
        pathlib.Path(__file__).resolve().parents[4]  # .../backend
        / "app" / "engine" / "models" / "families" / "flux1" / "definitions" / def_id
    )
    defaults = yaml.safe_load(path.read_text(encoding="utf-8"))["defaults"]
    assert defaults["num_inference_steps"] == expected_steps
