"""WAN 2.2 TI2V-5B ``expand_timesteps`` I2V conditioning — fake-tensor pins.

Recon (diffusers 0.39 ``pipelines/wan/pipeline_wan_i2v.py``, gated by
``self.config.expand_timesteps``, the REAL pipeline path for this checkpoint):
inference substitutes a CLEAN encoded condition into frame 0 of the model
input at every step and feeds a PER-TOKEN timestep (zero on frame-0 tokens,
the scalar sigma elsewhere). Training reproduces this by pinning frame 0's
flow-match noise SCALE to zero (self-referential: the "condition" is the same
clip's own clean frame-0 latent) — see ``driver.py``'s module docstring for
the full equivalence argument.

These tests exercise the REAL driver methods with tiny fake tensors — no
weights, no GPU (``wan21``/``wan22``'s ``test_wan21_i2v_conditioning.py`` /
``test_wan_i2v_still_guard.py`` precedent).
"""

from __future__ import annotations

import torch

from app.engine.models.families.wan22_ti2v_5b.driver import Wan22Ti2v5bDriver

Z_DIM = 48


class _Defn:
    architecture_params = {"mode": "both", "te.max_length": 512}
    lora_targetable_modules: list[str] = []


class _RecordingTransformer:
    """Fake WAN transformer: records hidden_states/timestep, echoes 48ch."""

    def __init__(self):
        self.seen_hidden_states = None
        self.seen_timestep = None
        self.seen_image_embed = "unset"

    def __call__(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image=None,
        return_dict=False,
    ):
        self.seen_hidden_states = hidden_states
        self.seen_timestep = timestep
        self.seen_image_embed = encoder_hidden_states_image
        return (hidden_states.clone(),)


def _make_driver() -> tuple[Wan22Ti2v5bDriver, _RecordingTransformer]:
    driver = Wan22Ti2v5bDriver(_Defn(), torch.device("cpu"))
    fake = _RecordingTransformer()
    driver.transformer = fake
    return driver, fake


# ── add_noise: frame-0 scale pinned to 0 only when engaged ─────────────────


def test_add_noise_not_engaged_matches_base_scalar_lerp():
    driver, _ = _make_driver()
    assert driver._i2v_active is False  # default off
    latents = torch.randn(2, Z_DIM, 3, 4, 4)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([300.0, 700.0])

    noisy = driver.add_noise(latents, noise, timesteps)

    t = (timesteps / 1000.0).reshape(2, 1, 1, 1, 1)
    expected = t * noise + (1.0 - t) * latents
    assert torch.allclose(noisy, expected)


def test_add_noise_engaged_pins_frame_zero_clean():
    driver, _ = _make_driver()
    driver._i2v_active = True
    latents = torch.randn(2, Z_DIM, 3, 4, 4)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([500.0, 500.0])

    noisy = driver.add_noise(latents, noise, timesteps)

    # Frame 0 is UNCHANGED (scale 0 == the clean latent, self-referential cond).
    assert torch.equal(noisy[:, :, 0], latents[:, :, 0])
    # Frames 1+ are noised at the full sampled timestep (unchanged formula).
    t = (timesteps / 1000.0).reshape(2, 1, 1, 1)
    expected_rest = t.unsqueeze(2) * noise[:, :, 1:] + (1.0 - t.unsqueeze(2)) * latents[:, :, 1:]
    assert torch.allclose(noisy[:, :, 1:], expected_rest)


def test_add_noise_engaged_f1_still_matches_base_scalar_lerp():
    """F=1 still guard: engaged flag alone does not pin anything for a still."""
    driver, _ = _make_driver()
    driver._i2v_active = True
    latents = torch.randn(2, Z_DIM, 1, 4, 4)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([500.0, 500.0])

    noisy = driver.add_noise(latents, noise, timesteps)

    t = (timesteps / 1000.0).reshape(2, 1, 1, 1, 1)
    expected = t * noise + (1.0 - t) * latents
    assert torch.allclose(noisy, expected)


# ── forward_pass: 48ch input unchanged; per-token timestep only when engaged ──


def test_forward_not_engaged_uses_scalar_timestep_no_concat():
    driver, fake = _make_driver()
    latents = torch.randn(2, Z_DIM, 1, 4, 4)  # t2v-shaped
    timesteps = torch.tensor([500.0, 500.0])
    text = torch.randn(2, 8, 16)

    pred = driver.forward_pass(latents, timesteps, text, {})

    assert fake.seen_hidden_states.shape == (2, Z_DIM, 1, 4, 4)
    assert torch.equal(fake.seen_hidden_states, latents)  # no concat, ever
    assert fake.seen_timestep.ndim == 1  # scalar-per-batch
    assert torch.equal(fake.seen_timestep, timesteps)
    assert fake.seen_image_embed is None  # no CLIP image encoder, ever
    assert pred.shape == latents.shape


def test_forward_engaged_multi_frame_uses_per_token_timestep_grid():
    driver, fake = _make_driver()
    driver._i2v_active = True
    latents = torch.randn(2, Z_DIM, 3, 4, 4)  # F=3, H=W=4 -> patch/2 -> gh=gw=2
    timesteps = torch.tensor([700.0, 250.0])
    text = torch.randn(2, 8, 16)

    driver.forward_pass(latents, timesteps, text, {})

    seen_ts = fake.seen_timestep
    assert seen_ts.ndim == 2
    # seq_len = F * (H//2) * (W//2) = 3 * 2 * 2 = 12
    assert seen_ts.shape == (2, 12)
    grid = seen_ts.view(2, 3, 2, 2)
    assert torch.all(grid[:, 0] == 0.0)  # frame 0 tokens: clean (t=0)
    assert torch.all(grid[0, 1:] == 700.0)
    assert torch.all(grid[1, 1:] == 250.0)
    # hidden_states is STILL the raw 48ch latent — no concat, no substitution
    # (the substitution already happened inside add_noise upstream).
    assert torch.equal(fake.seen_hidden_states, latents)
    assert fake.seen_image_embed is None


def test_forward_engaged_f1_still_falls_back_to_scalar_timestep():
    """F=1 still guard mirrored in forward_pass: no per-token grid for a still
    even when the run is i2v-active."""
    driver, fake = _make_driver()
    driver._i2v_active = True
    latents = torch.randn(2, Z_DIM, 1, 4, 4)
    timesteps = torch.tensor([500.0, 500.0])
    text = torch.randn(2, 8, 16)

    driver.forward_pass(latents, timesteps, text, {})

    assert fake.seen_timestep.ndim == 1
    assert torch.equal(fake.seen_timestep, timesteps)
    assert torch.equal(fake.seen_hidden_states, latents)


# ── _conditioning_engaged gate itself ───────────────────────────────────────


def test_conditioning_engaged_requires_active_flag_and_multi_frame():
    driver, _ = _make_driver()
    clip = torch.randn(1, Z_DIM, 3, 4, 4)
    still = torch.randn(1, Z_DIM, 1, 4, 4)

    driver._i2v_active = False
    assert driver._conditioning_engaged(clip) is False
    assert driver._conditioning_engaged(still) is False

    driver._i2v_active = True
    assert driver._conditioning_engaged(clip) is True
    assert driver._conditioning_engaged(still) is False  # F=1 still guard
