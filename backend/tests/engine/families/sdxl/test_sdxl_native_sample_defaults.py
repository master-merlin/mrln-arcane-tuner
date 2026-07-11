"""W3-1: SDXL native in-training-preview defaults, definition-sourced.

The generic base fills unset previews with 20 steps / 3.5 guidance, which is
off-distribution for SDXL. SDXLSampler._sample_single now sources the native
50 steps / 5.0 guidance / 1024^2 (StableDiffusionXLPipeline.__call__'s own
defaults) from the definition's ``defaults`` block. Explicit per-prompt values
still win.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from app.engine.core.sampling import GenericSamplingPipeline
from app.engine.models.families.sdxl.sampler import SDXLSampler


def _build_sampler(defaults: dict):
    # SDXLSampler.__init__ builds a DDIMScheduler from pipeline.scheduler.config;
    # bypass __init__ and set only what _sample_single touches.
    sampler = object.__new__(SDXLSampler)
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


class TestSDXLNativeSampleDefaults:
    def test_fills_native_50_steps_5_guidance_1024(self, monkeypatch):
        sampler = _build_sampler(
            {"num_inference_steps": 50, "guidance_scale": 5.0, "resolution": 1024}
        )
        captured = _capture(sampler, {"prompt": "sdxl"}, monkeypatch)
        assert captured["num_inference_steps"] == 50
        assert captured["guidance_scale"] == 5.0
        assert captured["width"] == 1024 and captured["height"] == 1024

    def test_definition_sourced_not_hardcoded(self, monkeypatch):
        sampler = _build_sampler(
            {"num_inference_steps": 31, "guidance_scale": 2.3, "resolution": 768}
        )
        captured = _capture(sampler, {"prompt": "sourced"}, monkeypatch)
        assert captured["num_inference_steps"] == 31
        assert captured["guidance_scale"] == 2.3
        assert captured["width"] == 768

    def test_explicit_values_win(self, monkeypatch):
        sampler = _build_sampler(
            {"num_inference_steps": 50, "guidance_scale": 5.0, "resolution": 1024}
        )
        captured = _capture(
            sampler,
            {"prompt": "x", "num_inference_steps": 8, "guidance_scale": 1.0,
             "width": 512, "height": 512},
            monkeypatch,
        )
        assert captured["num_inference_steps"] == 8
        assert captured["guidance_scale"] == 1.0
        assert captured["width"] == 512

    def test_falls_back_to_constants_when_defaults_missing(self, monkeypatch):
        sampler = _build_sampler({})
        captured = _capture(sampler, {"prompt": "bare"}, monkeypatch)
        assert captured["num_inference_steps"] == 50
        assert captured["guidance_scale"] == 5.0

    def test_shipped_yaml_carries_native_defaults(self):
        import pathlib

        import yaml

        path = (
            pathlib.Path(__file__).resolve().parents[4]  # .../backend
            / "app" / "engine" / "models" / "families" / "sdxl"
            / "definitions" / "sdxl_base_1.0.yaml"
        )
        defaults = yaml.safe_load(path.read_text(encoding="utf-8"))["defaults"]
        assert defaults["num_inference_steps"] == 50
        assert defaults["guidance_scale"] == 5.0
        assert defaults["resolution"] == 1024
