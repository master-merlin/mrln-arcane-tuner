"""WAN i2v F=1 still-guard — a still on an i2v run trains as t2v (ltx2/k5 parity).

RECON VERDICT (2026-07-12): on a WAN i2v run an F=1 still is DEGENERATE, not
NaN. Unlike ltx2/k5 (per-token / per-frame loss masks that empty → NaN), WAN
computes the flow-match target (``noise - latents``) and MSE loss uniformly over
ALL 16 noise channels of the single frame, so nothing crashes. But
``forward_pass`` hands the model its OWN clean latent back in the ``cond(16)``
channels with ``mask=1`` on the only frame, then asks it to predict
``noise - latents`` — which is a closed-form function of ``noisy`` + ``cond``.
The i2v conditioning path can drive that loss to ~0 by copying the answer,
teaching the shared denoising LoRA nothing about the still's content: a silent
zero-information / answer-leaked step.

THE GUARD: a single still (post-lift ``F == 1``) on an i2v-mode WAN driver must
take the t2v path — NO first-frame conditioning stashed, and the 36-channel
transformer input built with the ``mask(4)`` + ``cond(16)`` channels ZEROED (the
36-in-channel ``patch_embedding`` still needs 36 channels; zeroing them means
"no conditioning frame", so the still is denoised from scratch with no leak).
Multi-frame (F>1) i2v behaviour is unchanged.

Parity precedents: ltx2 ``_i2v_conditioning_engaged`` (driver.py:287-300, F>1),
kandinsky5 (driver.py:433-443).
"""

import pytest
import torch

from app.engine.models.families.wan21.driver import Wan21Driver
from app.engine.models.families.wan22.driver import Wan22Driver
from app.engine.models.families.wan_shared.i2v_conditioning import (
    COND_CHANNELS,
    I2V_IN_CHANNELS,
    MASK_CHANNELS,
    NOISE_CHANNELS,
    build_still_t2v_input,
)


class _Defn:
    architecture_params = {"mode": "i2v", "te.max_length": 512}
    lora_targetable_modules: list[str] = []


class _RecordingTransformer:
    """Fake WAN i2v transformer: records the input it saw, returns 16-ch pred."""

    def __init__(self):
        self.seen_hidden_states = None
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
        self.seen_image_embed = encoder_hidden_states_image
        return (hidden_states[:, :NOISE_CHANNELS].clone(),)


def _make_driver(cls):
    driver = cls(_Defn(), torch.device("cpu"))
    fake = _RecordingTransformer()
    driver.transformer = fake
    assert driver.is_i2v is True
    return driver, fake


DRIVER_CLASSES = [Wan21Driver, Wan22Driver]


# ── build_still_t2v_input: the zero-padded 36-ch t2v input ─────────────────────


def test_build_still_t2v_input_is_36ch_with_zeroed_mask_and_cond():
    noisy = torch.randn(2, NOISE_CHANNELS, 1, 4, 4)
    out = build_still_t2v_input(noisy)
    assert out.shape[1] == I2V_IN_CHANNELS == 36
    # First 16 channels are exactly the noised latent (target space intact).
    assert torch.equal(out[:, :NOISE_CHANNELS], noisy)
    # mask(4) + cond(16) are ALL zero — no conditioning frame, no answer leak.
    assert torch.all(out[:, NOISE_CHANNELS:] == 0.0)
    mask = out[:, NOISE_CHANNELS : NOISE_CHANNELS + MASK_CHANNELS]
    cond = out[:, NOISE_CHANNELS + MASK_CHANNELS :]
    assert mask.shape[1] == MASK_CHANNELS and cond.shape[1] == COND_CHANNELS


def test_build_still_t2v_input_rejects_wrong_channels():
    with pytest.raises(ValueError):
        build_still_t2v_input(torch.randn(2, 8, 1, 4, 4))


# ── attach_conditioning: F=1 does not stash; F>1 does ──────────────────────────


@pytest.mark.parametrize("cls", DRIVER_CLASSES)
def test_attach_conditioning_skips_stash_for_f1_still(cls):
    driver, _ = _make_driver(cls)
    still = torch.randn(2, NOISE_CHANNELS, 1, 4, 4)  # F=1
    batch: dict = {}
    driver.attach_conditioning(batch, still)
    assert driver.BATCH_FIRST_FRAME_LATENT not in batch


@pytest.mark.parametrize("cls", DRIVER_CLASSES)
def test_attach_conditioning_skips_stash_for_4d_still(cls):
    """A still is cached 4D [B,C,H,W]; after lift it is F=1 → no stash."""
    driver, _ = _make_driver(cls)
    still_4d = torch.randn(2, NOISE_CHANNELS, 4, 4)  # 4D, lifts to F=1
    batch: dict = {}
    driver.attach_conditioning(batch, still_4d)
    assert driver.BATCH_FIRST_FRAME_LATENT not in batch


@pytest.mark.parametrize("cls", DRIVER_CLASSES)
def test_attach_conditioning_stashes_for_multi_frame_clip(cls):
    driver, _ = _make_driver(cls)
    clip = torch.randn(2, NOISE_CHANNELS, 3, 4, 4)  # F=3 video
    batch: dict = {}
    driver.attach_conditioning(batch, clip)
    assert driver.BATCH_FIRST_FRAME_LATENT in batch
    stashed = batch[driver.BATCH_FIRST_FRAME_LATENT]
    assert stashed.shape[2] == 1  # only frame 0


# ── forward_pass: F=1 still takes the zeroed-conditioning t2v path ─────────────


@pytest.mark.parametrize("cls", DRIVER_CLASSES)
def test_forward_f1_still_uses_zeroed_conditioning_no_leak(cls):
    driver, fake = _make_driver(cls)
    latents = torch.randn(2, NOISE_CHANNELS, 1, 4, 4)  # F=1 still
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([500.0, 500.0])
    noisy = driver.add_noise(latents, noise, timesteps)

    # Real path: attach_conditioning ran first (and stashed nothing for F=1).
    batch: dict = {}
    driver.attach_conditioning(batch, latents)
    text = torch.randn(2, 8, 16)

    pred = driver.forward_pass(noisy, timesteps, text, batch)

    seen = fake.seen_hidden_states
    assert seen.shape[1] == I2V_IN_CHANNELS == 36  # architecture still needs 36
    # First 16 channels are the noisy latent; mask+cond ZEROED (no leak).
    assert torch.equal(seen[:, :NOISE_CHANNELS], noisy)
    assert torch.all(seen[:, NOISE_CHANNELS:] == 0.0)
    # No CLIP image embed on the t2v path.
    assert fake.seen_image_embed is None
    # Prediction lives over the 16 noise channels (same space as target).
    assert pred.shape == latents.shape


@pytest.mark.parametrize("cls", DRIVER_CLASSES)
def test_forward_multi_frame_still_conditions_on_first_frame(cls):
    """F>1 i2v is unchanged: mask=1 on frame 0, cond carries the first frame."""
    driver, fake = _make_driver(cls)
    latents = torch.randn(2, NOISE_CHANNELS, 3, 4, 4)  # F=3
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([500.0, 500.0])
    noisy = driver.add_noise(latents, noise, timesteps)

    batch: dict = {}
    driver.attach_conditioning(batch, latents)
    first_frame = batch[driver.BATCH_FIRST_FRAME_LATENT]
    text = torch.randn(2, 8, 16)

    driver.forward_pass(noisy, timesteps, text, batch)
    seen = fake.seen_hidden_states
    assert seen.shape[1] == I2V_IN_CHANNELS == 36
    mask = seen[:, NOISE_CHANNELS : NOISE_CHANNELS + MASK_CHANNELS]
    cond = seen[:, NOISE_CHANNELS + MASK_CHANNELS :]
    # Conditioning ENGAGED: mask=1 on frame 0, cond frame 0 == first frame.
    assert torch.all(mask[:, :, 0] == 1.0)
    assert torch.all(mask[:, :, 1:] == 0.0)
    assert torch.equal(cond[:, :, 0], first_frame[:, :, 0])
