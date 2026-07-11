"""W3-1: FLUX.2 native in-training-preview defaults, definition-sourced.

Flux2Sampler._sample_single sources the native 50 steps / 4.0 guidance /
1024^2 (FLUX.2 / Klein pipeline __call__ defaults) from the definition's
``defaults`` block, replacing the generic base's off-distribution 20 / 3.5.
Explicit per-prompt values still win.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import pytest
import torch

from app.engine.core.sampling import GenericSamplingPipeline
from app.engine.models.families.flux2.sampler import Flux2Sampler


def _build_sampler(defaults: dict):
    sampler = object.__new__(Flux2Sampler)
    pipeline = MagicMock()
    pipeline.config = {"sample_every_n_steps": 50, "sample_negative_prompt": ""}
    pipeline.device = torch.device("cpu")
    defn = MagicMock()
    defn.defaults = dict(defaults)
    pipeline.definition = defn
    sampler.pipeline = pipeline
    sampler.config = pipeline.config
    sampler.device = pipeline.device
    return sampler


def _capture(sampler, prompt_cfg, monkeypatch):
    captured: dict = {}

    def _fake_base(self, cfg, step):
        captured.update(cfg)
        return MagicMock()

    monkeypatch.setattr(GenericSamplingPipeline, "_sample_single", _fake_base)
    sampler._sample_single(prompt_cfg, 0)
    return captured


class TestFlux2NativeSampleDefaults:
    def test_fills_native_50_steps_4_guidance_1024(self, monkeypatch):
        sampler = _build_sampler(
            {"num_inference_steps": 50, "guidance_scale": 4.0, "resolution": 1024}
        )
        captured = _capture(sampler, {"prompt": "flux2"}, monkeypatch)
        assert captured["num_inference_steps"] == 50
        assert captured["guidance_scale"] == 4.0
        assert captured["width"] == 1024 and captured["height"] == 1024

    def test_definition_sourced_not_hardcoded(self, monkeypatch):
        sampler = _build_sampler(
            {"num_inference_steps": 23, "guidance_scale": 1.7, "resolution": 768}
        )
        captured = _capture(sampler, {"prompt": "sourced"}, monkeypatch)
        assert captured["num_inference_steps"] == 23
        assert captured["guidance_scale"] == 1.7
        assert captured["width"] == 768

    def test_explicit_values_win(self, monkeypatch):
        sampler = _build_sampler(
            {"num_inference_steps": 50, "guidance_scale": 4.0, "resolution": 1024}
        )
        captured = _capture(
            sampler,
            {"prompt": "x", "num_inference_steps": 6, "guidance_scale": 1.0,
             "width": 512, "height": 512},
            monkeypatch,
        )
        assert captured["num_inference_steps"] == 6
        assert captured["guidance_scale"] == 1.0
        assert captured["width"] == 512

    def test_falls_back_to_constants_when_defaults_missing(self, monkeypatch):
        sampler = _build_sampler({})
        captured = _capture(sampler, {"prompt": "bare"}, monkeypatch)
        assert captured["num_inference_steps"] == 50
        assert captured["guidance_scale"] == 4.0


@pytest.mark.parametrize("def_id", ["dev.yaml", "klein_4b.yaml", "klein_9b.yaml"])
def test_shipped_yaml_carries_native_defaults(def_id):
    import yaml

    path = (
        pathlib.Path(__file__).resolve().parents[4]  # .../backend
        / "app" / "engine" / "models" / "families" / "flux2"
        / "definitions" / def_id
    )
    defaults = yaml.safe_load(path.read_text(encoding="utf-8"))["defaults"]
    assert defaults["num_inference_steps"] == 50
    assert defaults["guidance_scale"] == 4.0
    assert defaults["resolution"] == 1024
