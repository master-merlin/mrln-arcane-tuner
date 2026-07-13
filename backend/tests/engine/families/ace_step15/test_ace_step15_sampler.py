"""ACE-Step 1.5 sampler unit tests — fp32 Euler trajectory, noise shape,
sigma schedule endpoints, and WAV artifact persistence via the shared
``core/sampling.py`` machinery (AudioSampleArtifact).

Uses the same tiny meta-instantiable components as test_ace_step15_driver.py.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import torch

from app.engine.core.sampling import AudioSampleArtifact
from app.engine.models.families.ace_step15.sampler import (
    ACE_STEP15_DEFAULT_SHIFT,
    AceStep15Sampler,
)

from .test_ace_step15_driver import _make_driver


def _make_sampler(driver, **config) -> AceStep15Sampler:
    pipeline = SimpleNamespace(
        driver=driver,
        config={"duration_s": 6.0, **config},
        device=torch.device("cpu"),
        definition=SimpleNamespace(architecture_params={}),
    )
    sampler = object.__new__(AceStep15Sampler)
    sampler.pipeline = pipeline
    sampler.config = pipeline.config
    sampler.device = pipeline.device
    sampler.logger = SimpleNamespace(info=lambda *a, **k: None)
    return sampler


# ── sigma schedule ────────────────────────────────────────────────────────


def test_build_sigmas_endpoints_and_length():
    driver = _make_driver()
    sampler = _make_sampler(driver)
    sigmas = sampler._build_sigmas(4)
    assert sigmas.shape == (5,)
    assert math.isclose(sigmas[0].item(), 1.0, abs_tol=1e-6)
    assert math.isclose(sigmas[-1].item(), 0.0, abs_tol=1e-6)
    assert torch.all(sigmas[:-1] >= sigmas[1:])  # monotonically descending


def test_build_sigmas_uses_model_shift_fixed_override():
    driver = _make_driver()
    sampler = _make_sampler(driver, model_shift_fixed=1.0)
    sigmas = sampler._build_sigmas(4)
    # shift=1.0 -> plain linear schedule (no warping).
    expected = torch.linspace(1.0, 0.0, 5)
    assert torch.allclose(sigmas, expected, atol=1e-6)


def test_build_sigmas_default_shift_matches_turbo_constant():
    assert ACE_STEP15_DEFAULT_SHIFT == 3.0


# ── initial noise / per-prompt duration override ─────────────────────────


def test_create_initial_noise_shape_from_config_duration():
    driver = _make_driver()
    sampler = _make_sampler(driver)
    sampler._active_prompt_cfg = {}
    noise = sampler._create_initial_noise(0, 0, torch.Generator().manual_seed(0))
    expected_len = math.ceil(6.0 * driver.latents_per_second)
    assert noise.shape == (1, expected_len, driver.audio_acoustic_hidden_dim)
    assert noise.dtype == torch.float32


def test_create_initial_noise_per_prompt_duration_override():
    driver = _make_driver()
    sampler = _make_sampler(driver)
    sampler._active_prompt_cfg = {"duration_s": 2.0}
    noise = sampler._create_initial_noise(0, 0, torch.Generator().manual_seed(0))
    expected_len = math.ceil(2.0 * driver.latents_per_second)
    assert noise.shape[1] == expected_len


# ── denoise / decode_latents ──────────────────────────────────────────────


def test_denoise_and_decode_shapes():
    driver = _make_driver()
    driver.transformer.eval()
    sampler = _make_sampler(driver)
    sampler._active_prompt_cfg = {"lyrics": "la la"}

    eh, mask = driver.encode_condition(["x"], ["la la"], torch.float32, audio_duration=6.0)
    noise = sampler._create_initial_noise(0, 0, torch.Generator().manual_seed(0))

    with torch.no_grad():
        latents = sampler.denoise(noise, (eh, mask), num_steps=2, guidance_scale=1.0, seed=0)
    assert latents.shape == noise.shape

    artifact = sampler.decode_latents(latents)
    assert isinstance(artifact, AudioSampleArtifact)
    assert artifact.waveform.ndim == 2  # [C, T]
    assert artifact.waveform.shape[0] == 2  # stereo (tiny VAE audio_channels=2)
    assert artifact.sample_rate == 1000  # tiny VAE sampling_rate
    assert artifact.waveform.abs().max().item() <= 1.0 + 1e-5


def test_denoise_turbo_ignores_cfg():
    """is_turbo=True (the shipped default) must NEVER engage the CFG branch,
    even if guidance_scale > 1.0 is passed — matches the pipeline's own
    turbo-coercion behavior."""
    driver = _make_driver()
    assert driver.transformer.config.is_turbo is True
    driver.transformer.eval()
    sampler = _make_sampler(driver)
    sampler._active_prompt_cfg = {}

    eh, mask = driver.encode_condition(["x"], [""], torch.float32, audio_duration=6.0)
    noise = sampler._create_initial_noise(0, 0, torch.Generator().manual_seed(0))

    with torch.no_grad():
        out_cfg = sampler.denoise(noise, (eh, mask), num_steps=2, guidance_scale=7.0, seed=0)
        out_no_cfg = sampler.denoise(noise, (eh, mask), num_steps=2, guidance_scale=1.0, seed=0)
    assert torch.allclose(out_cfg, out_no_cfg)


# ── encode_prompt seam (lyrics reach the condition encoder) ──────────────


def test_sample_single_stashes_prompt_cfg_before_encode(monkeypatch):
    driver = _make_driver()
    sampler = _make_sampler(driver)

    seen_cfg_during_encode = {}

    def _fake_encode_prompt(prompt):
        seen_cfg_during_encode["lyrics"] = sampler._active_prompt_cfg.get("lyrics")
        return "embedding"

    monkeypatch.setattr(sampler, "encode_prompt", _fake_encode_prompt)
    monkeypatch.setattr(sampler, "_ensure_on_gpu", lambda names: [])
    monkeypatch.setattr(sampler, "_offload_to_cpu", lambda names: None)
    monkeypatch.setattr(sampler, "_create_initial_noise", lambda w, h, g: torch.zeros(1))
    monkeypatch.setattr(sampler, "denoise", lambda *a, **k: torch.zeros(1))
    monkeypatch.setattr(sampler, "decode_latents", lambda latents: "artifact")
    sampler.pipeline.driver = driver

    prompt_cfg = {"prompt": "a song", "lyrics": "la la la", "seed": 1}
    result = sampler._sample_single(prompt_cfg, step=0)

    assert result == "artifact"
    assert seen_cfg_during_encode["lyrics"] == "la la la"


def test_persist_wav_uses_expected_filename(tmp_path):
    """Filename convention: sample_XX_stepNNNNNN.wav (brief requirement)."""
    from pathlib import Path

    from app.engine.core.sampling import GenericSamplingPipeline

    class _P(GenericSamplingPipeline):
        def encode_prompt(self, prompt):
            raise NotImplementedError

        def denoise(self, *a, **k):
            raise NotImplementedError

        def decode_latents(self, latents):
            raise NotImplementedError

        def _create_initial_noise(self, width, height, generator):
            raise NotImplementedError

    p = object.__new__(_P)
    artifact = AudioSampleArtifact(waveform=torch.zeros(2, 100), sample_rate=8000)
    path = p._persist_artifact(artifact, Path(tmp_path), index=0, displayed_step=50, final=False)
    assert path.name == "sample_00_step000050.wav"
    assert os.path.isfile(path)
