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
from diffusers.guiders.adaptive_projected_guidance import (
    MomentumBuffer,
    normalized_guidance,
)

from app.engine.core.sampling import AudioSampleArtifact
from app.engine.models.families.ace_step15.sampler import (
    ACE_STEP15_APG_ETA,
    ACE_STEP15_APG_MOMENTUM,
    ACE_STEP15_APG_NORM_DIM,
    ACE_STEP15_APG_NORM_THRESHOLD,
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


# ── CFG-path validation (task C2): real APG, not a plain linear blend ────


def _manual_forward(driver, x_full, t01, cond, context_latents, model_dtype, b):
    """Independent re-implementation of sampler._forward — used as the
    ground truth for the APG cross-check tests below (deliberately NOT
    calling into sampler.py's own helper, so a bug in both wouldn't
    coincidentally cancel out)."""
    t_tensor = t01.reshape(1).expand(b).to(model_dtype)
    with torch.no_grad():
        out = driver.transformer(
            hidden_states=x_full.to(model_dtype),
            timestep=t_tensor,
            timestep_r=t_tensor,
            encoder_hidden_states=cond.to(x_full.device, model_dtype),
            context_latents=context_latents,
            return_dict=False,
        )
    return out[0] if isinstance(out, (tuple, list)) else out


def _manual_apg_denoise(sampler, driver, noise, encoder_hidden_states, num_steps, guidance_scale):
    """Independently replicates the expected APG denoise loop (own
    MomentumBuffer instance, own normalized_guidance calls) to cross-check
    sampler.denoise()'s real implementation bit-for-bit."""
    x = noise.to(torch.float32)
    b, t_len, _ = x.shape
    model_dtype = driver.transformer.dtype
    context_latents = driver._build_context_latents(b, t_len, x.device, model_dtype)
    null_emb = driver.condition_encoder.null_condition_emb.to(
        x.device, model_dtype
    ).expand_as(encoder_hidden_states)

    momentum_buffer = MomentumBuffer(momentum=ACE_STEP15_APG_MOMENTUM)
    sigmas = sampler._build_sigmas(num_steps)
    for i in range(len(sigmas) - 1):
        v_c = _manual_forward(
            driver, x, sigmas[i], encoder_hidden_states, context_latents, model_dtype, b
        ).to(torch.float32)
        v_u = _manual_forward(
            driver, x, sigmas[i], null_emb, context_latents, model_dtype, b
        ).to(torch.float32)
        v = normalized_guidance(
            pred_cond=v_c,
            pred_uncond=v_u,
            guidance_scale=guidance_scale - 1.0,
            momentum_buffer=momentum_buffer,
            eta=ACE_STEP15_APG_ETA,
            norm_threshold=ACE_STEP15_APG_NORM_THRESHOLD,
            use_original_formulation=True,
            norm_dim=ACE_STEP15_APG_NORM_DIM,
        )
        dt = sigmas[i + 1] - sigmas[i]
        x = x + dt * v
    return x


def test_denoise_base_apg_matches_upstream_formula():
    """is_turbo=False + guidance_scale>1.0 must run the REAL APG blend
    (diffusers' own normalized_guidance/MomentumBuffer, momentum=-0.75,
    eta=0, norm_threshold=2.5, use_original_formulation=True, norm_dim=(1,))
    — byte-verified against `AceStepPipeline.__call__`'s own denoise loop
    (task C2 recon) — not the simplified linear blend this sampler shipped
    with initially. 3 steps so the STATEFUL momentum buffer's carry-over
    is actually exercised (a per-step-reset buffer would diverge from this
    reference by step 3)."""
    driver = _make_driver(is_turbo=False)
    assert driver.transformer.config.is_turbo is False
    driver.transformer.eval()
    sampler = _make_sampler(driver)
    sampler._active_prompt_cfg = {"lyrics": "la la"}

    eh, mask = driver.encode_condition(["x"], ["la la"], torch.float32, audio_duration=6.0)
    noise = sampler._create_initial_noise(0, 0, torch.Generator().manual_seed(0))

    with torch.no_grad():
        actual = sampler.denoise(noise, (eh, mask), num_steps=3, guidance_scale=7.0, seed=0)
        expected = _manual_apg_denoise(sampler, driver, noise, eh, num_steps=3, guidance_scale=7.0)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_denoise_base_apg_differs_from_plain_cfg_blend():
    """Proves the shipped implementation is NOT the old simplified
    ``v_uncond + gs*(v_cond - v_uncond)`` blend — APG's norm-clamped,
    direction-projected update must diverge from a plain linear blend for
    a guidance_scale large enough to trigger the norm_threshold clamp."""
    driver = _make_driver(is_turbo=False)
    driver.transformer.eval()
    sampler = _make_sampler(driver)
    sampler._active_prompt_cfg = {"lyrics": "la la"}

    eh, mask = driver.encode_condition(["x"], ["la la"], torch.float32, audio_duration=6.0)
    noise = sampler._create_initial_noise(0, 0, torch.Generator().manual_seed(0))

    with torch.no_grad():
        actual = sampler.denoise(noise, (eh, mask), num_steps=2, guidance_scale=12.0, seed=0)

        # Plain linear CFG blend, single momentum-free step-by-step loop —
        # the OLD (pre-C2) behavior this sampler must no longer produce.
        x = noise.to(torch.float32)
        b, t_len, _ = x.shape
        model_dtype = driver.transformer.dtype
        context_latents = driver._build_context_latents(b, t_len, x.device, model_dtype)
        null_emb = driver.condition_encoder.null_condition_emb.to(
            x.device, model_dtype
        ).expand_as(eh)
        sigmas = sampler._build_sigmas(2)
        for i in range(len(sigmas) - 1):
            v_c = _manual_forward(driver, x, sigmas[i], eh, context_latents, model_dtype, b).to(torch.float32)
            v_u = _manual_forward(driver, x, sigmas[i], null_emb, context_latents, model_dtype, b).to(torch.float32)
            v = v_u + 12.0 * (v_c - v_u)
            x = x + (sigmas[i + 1] - sigmas[i]) * v
    assert not torch.allclose(actual, x)


def test_denoise_base_guidance_scale_1_skips_apg():
    """guidance_scale=1.0 on a non-turbo checkpoint must still skip CFG
    entirely (matches `do_classifier_free_guidance`'s `gs > 1.0` gate) —
    output must equal a plain single-forward Euler loop with no guidance
    blend at all."""
    driver = _make_driver(is_turbo=False)
    driver.transformer.eval()
    sampler = _make_sampler(driver)
    sampler._active_prompt_cfg = {"lyrics": "la la"}

    eh, mask = driver.encode_condition(["x"], ["la la"], torch.float32, audio_duration=6.0)
    noise = sampler._create_initial_noise(0, 0, torch.Generator().manual_seed(0))

    with torch.no_grad():
        actual = sampler.denoise(noise, (eh, mask), num_steps=2, guidance_scale=1.0, seed=0)

        x = noise.to(torch.float32)
        b, t_len, _ = x.shape
        model_dtype = driver.transformer.dtype
        context_latents = driver._build_context_latents(b, t_len, x.device, model_dtype)
        sigmas = sampler._build_sigmas(2)
        for i in range(len(sigmas) - 1):
            v = _manual_forward(driver, x, sigmas[i], eh, context_latents, model_dtype, b).to(torch.float32)
            x = x + (sigmas[i + 1] - sigmas[i]) * v
    assert torch.allclose(actual, x, atol=1e-6)


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
