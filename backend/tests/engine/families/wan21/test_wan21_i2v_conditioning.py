"""WAN 2.1 I2V conditioning tests.

The 36-channel transformer input is ``[noisy(16), mask(4), cond(16)]``:
- channel count is exactly 36,
- the mask is 1.0 on the first latent frame and 0.0 elsewhere,
- the first 16 channels are exactly the noised latent, so the velocity target
  (computed by the trainer over those 16 channels) and the model prediction
  share the same space — the mask/cond channels are conditioning only.
"""

import torch

from app.engine.models.families.wan21.driver import Wan21Driver
from app.engine.models.families.wan_shared.i2v_conditioning import (
    COND_CHANNELS,
    I2V_IN_CHANNELS,
    MASK_CHANNELS,
    NOISE_CHANNELS,
    build_i2v_conditioning,
    build_temporal_mask,
)


def _shape(b=2, c=16, f=3, h=4, w=4):
    return (b, c, f, h, w)


def test_concat_is_36_channels():
    noisy = torch.randn(_shape())
    first_frame = torch.randn(2, 16, 1, 4, 4)
    out = build_i2v_conditioning(noisy, first_frame)
    assert out.shape[1] == I2V_IN_CHANNELS == 36
    assert out.shape[1] == NOISE_CHANNELS + MASK_CHANNELS + COND_CHANNELS
    # Spatial/temporal dims preserved.
    assert out.shape[0] == 2 and out.shape[2:] == (3, 4, 4)


def test_first_16_channels_are_the_noised_latent():
    noisy = torch.randn(_shape())
    first_frame = torch.randn(2, 16, 1, 4, 4)
    out = build_i2v_conditioning(noisy, first_frame)
    # The diffusion variable must survive untouched in the first 16 channels.
    assert torch.equal(out[:, :NOISE_CHANNELS], noisy)


def test_mask_layout_first_latent_frame_only():
    mask = build_temporal_mask(latent_frames=3, batch=2, height=4, width=4)
    assert mask.shape == (2, MASK_CHANNELS, 3, 4, 4)
    # Frame 0 is all ones, the rest all zeros.
    assert torch.all(mask[:, :, 0] == 1.0)
    assert torch.all(mask[:, :, 1:] == 0.0)

    # And inside the concat, mask occupies channels [16:20].
    noisy = torch.randn(_shape())
    first_frame = torch.randn(2, 16, 1, 4, 4)
    out = build_i2v_conditioning(noisy, first_frame)
    cat_mask = out[:, NOISE_CHANNELS : NOISE_CHANNELS + MASK_CHANNELS]
    assert torch.all(cat_mask[:, :, 0] == 1.0)
    assert torch.all(cat_mask[:, :, 1:] == 0.0)


def test_cond_channels_only_first_frame_nonzero():
    noisy = torch.zeros(_shape())
    first_frame = torch.randn(2, 16, 1, 4, 4)
    out = build_i2v_conditioning(noisy, first_frame)
    cond = out[:, NOISE_CHANNELS + MASK_CHANNELS :]
    assert cond.shape[1] == COND_CHANNELS
    # First latent frame holds the encoded first frame; rest is zero.
    assert torch.equal(cond[:, :, 0], first_frame[:, :, 0])
    assert torch.all(cond[:, :, 1:] == 0.0)


def test_velocity_target_only_over_16_noise_channels_via_forward():
    """End-to-end: forward_pass builds the 36-ch concat internally; the target
    (noise - latents) lives over the 16 noise channels only.

    We use a fake transformer that ASSERTS it received 36 input channels and
    returns a 16-channel prediction, mirroring the real WAN I2V contract.
    """

    class _Defn:
        architecture_params = {"mode": "i2v", "te.max_length": 512}
        lora_targetable_modules: list[str] = []

    class _FakeI2VTransformer:
        def __init__(self):
            self.seen_in_channels = None

        def __call__(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image=None,
            return_dict=False,
        ):
            # Confirm the 36-channel concat reached the model.
            self.seen_in_channels = hidden_states.shape[1]
            # Predict over the 16 noise channels only (real WAN out_channels=16).
            return (hidden_states[:, :NOISE_CHANNELS].clone(),)

    driver = Wan21Driver(_Defn(), torch.device("cpu"))
    fake = _FakeI2VTransformer()
    driver.transformer = fake

    latents = torch.randn(_shape())  # 16-channel clean latent
    noise = torch.randn(_shape())  # 16-channel noise
    timesteps = torch.tensor([500.0, 500.0])
    noisy = driver.add_noise(latents, noise, timesteps)  # 16-channel
    assert noisy.shape[1] == NOISE_CHANNELS

    first_frame = torch.randn(2, 16, 1, 4, 4)
    batch = {driver.BATCH_FIRST_FRAME_LATENT: first_frame}
    text = torch.randn(2, 8, 16)

    pred = driver.forward_pass(noisy, timesteps, text, batch)

    # The transformer saw 36 channels; the prediction is 16 channels.
    assert fake.seen_in_channels == I2V_IN_CHANNELS == 36
    assert pred.shape[1] == NOISE_CHANNELS == 16

    # The velocity target lives over the 16 noise channels and matches the
    # prediction's shape exactly (same space) — no mask/cond leakage.
    target = noise - latents
    assert target.shape == pred.shape == noisy.shape
