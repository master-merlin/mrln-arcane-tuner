"""hv15 i2v F=1 still-guard — a still on an i2v run trains as t2v (WAN parity).

RECON VERDICT (2026-07-12): on an hv15 i2v run an F=1 still is DEGENERATE — the
EXACT degeneracy WAN just fixed (commit 11173c2c), and the same silent
zero-information / answer-leaked class as k5 I2V frame-0.

Mechanism (verified against the real driver): ``attach_conditioning`` stashes the
clip's own clean frame-0 latent UNCONDITIONALLY; ``build_i2v_cond_and_mask``
places that clean still into the ``cond(32)`` channels at slot 0 with ``mask=1``
on frame 0. For an F=1 still ALL frames ARE frame 0, so the model is handed its
own clean latent in the cond channels with ``mask=1`` on the only frame. hv15
computes the flow-match target (``noise - latents``) and MSE loss UNIFORMLY over
all frames (no frame-0 exclusion, no per-frame mask, unlike ltx2/k5 → no NaN) —
so the target is a closed-form function of ``noisy`` + ``cond`` and the i2v path
can drive the loss to ~0 by copying the answer, teaching the shared denoising
LoRA nothing about the still's content.

SEMANTICS CHECK (why zeroing is the fix, not off-distribution garbage): hv15's
own T2V encoding (``build_t2v_cond_and_mask``) is ZERO cond + ZERO mask, and the
transformer detects the unconditioned image stream via
``torch.all(image_embeds == 0)`` and masks it out. So ``mask=0`` IS the model's
built-in "generate this frame / no reference" encoding — identical semantics to
WAN's mask. Routing an F=1 still through that path (zero cond/mask + zero
``image_embeds``) is the model's designed unconditioned fallback, and it strips
the leak.

THE GUARD (WAN parity, ``wan_shared`` F>1 gate): a single still (post-lift
``F == 1``) on an i2v-mode hv15 driver takes the t2v path — NO first-frame stash,
and the 65-channel input built with the ``cond(32)`` + ``mask(1)`` channels
ZEROED plus zero ``image_embeds``. Multi-frame (F>1) i2v behaviour is unchanged.
"""

import pytest
import torch

from app.engine.models.families.hunyuan_video15.driver import (
    COND_CHANNELS,
    Hv15Driver,
    NOISE_CHANNELS,
)
from app.engine.models.families.hunyuan_video15.trainer import Hv15Trainer


class _Defn:
    def __init__(self, mode: str = "i2v"):
        self.architecture_params = {"mode": mode, "transformer.num_layers": 1}
        self.lora_targetable_modules: list[str] = []


class _CaptureTransformer(torch.nn.Module):
    """Records the input it saw; returns a velocity over the first 32 channels."""

    def __init__(self):
        super().__init__()
        self.calls: list[dict] = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return (kwargs["hidden_states"][:, :NOISE_CHANNELS].clone(),)


def _text4(b=2, l1=6, d1=16, l2=4, d2=8):
    return (
        torch.randn(b, l1, d1),
        torch.ones(b, l1, dtype=torch.int64),
        torch.zeros(b, l2, d2),
        torch.zeros(b, l2, dtype=torch.int64),
    )


def _driver(mode: str = "i2v") -> tuple[Hv15Driver, _CaptureTransformer]:
    driver = Hv15Driver(_Defn(mode), torch.device("cpu"))
    model = _CaptureTransformer()
    driver.assign_components({"unet": model})
    assert driver.is_i2v is (mode == "i2v")
    return driver, model


# ── attach_conditioning: F=1 does not stash; F>1 does ──────────────────────────


def test_attach_conditioning_skips_stash_for_f1_still():
    driver, _ = _driver("i2v")
    still = torch.randn(2, NOISE_CHANNELS, 1, 4, 4)  # F=1
    batch: dict = {}
    driver.attach_conditioning(batch, still)
    assert Hv15Driver.BATCH_FIRST_FRAME_LATENT not in batch


def test_attach_conditioning_skips_stash_for_4d_still():
    """A still is cached 4D [B,C,H,W]; after lift it is F=1 → no stash."""
    driver, _ = _driver("i2v")
    still_4d = torch.randn(2, NOISE_CHANNELS, 4, 4)  # 4D, lifts to F=1
    batch: dict = {}
    driver.attach_conditioning(batch, still_4d)
    assert Hv15Driver.BATCH_FIRST_FRAME_LATENT not in batch


def test_attach_conditioning_stashes_for_multi_frame_clip():
    driver, _ = _driver("i2v")
    clip = torch.randn(2, NOISE_CHANNELS, 3, 4, 4)  # F=3 video
    batch: dict = {}
    driver.attach_conditioning(batch, clip)
    assert Hv15Driver.BATCH_FIRST_FRAME_LATENT in batch
    stashed = batch[Hv15Driver.BATCH_FIRST_FRAME_LATENT]
    assert stashed.shape[2] == 1  # only frame 0


# ── forward_pass: F=1 still takes the zeroed-conditioning t2v path ─────────────


def test_forward_f1_still_uses_zeroed_conditioning_no_leak():
    driver, model = _driver("i2v")
    latents = torch.randn(2, NOISE_CHANNELS, 1, 4, 4)  # F=1 still
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([500.0, 500.0])
    # Match the base component's lerp so the noised latent is realistic.
    t = (timesteps / 1000.0).view(-1, 1, 1, 1, 1)
    noisy = t * noise + (1.0 - t) * latents

    batch: dict = {}
    driver.attach_conditioning(batch, latents)  # stashes nothing for F=1
    driver.forward_pass(noisy, timesteps, _text4(2), batch)

    x = model.calls[0]["hidden_states"]
    assert x.shape[1] == 65
    # First 32 channels are the noisy latent; cond(32)+mask(1) ZEROED (no leak).
    assert torch.equal(x[:, :NOISE_CHANNELS], noisy)
    assert torch.all(x[:, NOISE_CHANNELS:] == 0.0)
    # image_embeds fed as the all-zero stream → transformer masks it out.
    img = model.calls[0]["image_embeds"]
    assert img.shape == (2, 729, 1152)
    assert torch.all(img == 0)


def test_forward_multi_frame_still_conditions_on_first_frame():
    """F>1 i2v is unchanged: mask=1 on frame 0, cond carries the first frame."""
    driver, model = _driver("i2v")
    latents = torch.randn(2, NOISE_CHANNELS, 3, 4, 4)  # F=3
    noisy = torch.randn_like(latents)
    timesteps = torch.tensor([500.0, 500.0])

    batch: dict = {}
    driver.attach_conditioning(batch, latents)
    first_frame = batch[Hv15Driver.BATCH_FIRST_FRAME_LATENT]
    driver.forward_pass(noisy, timesteps, _text4(2), batch)

    x = model.calls[0]["hidden_states"]
    cond = x[:, NOISE_CHANNELS : NOISE_CHANNELS + COND_CHANNELS]
    mask = x[:, NOISE_CHANNELS + COND_CHANNELS :]
    # Conditioning ENGAGED: cond frame 0 == first frame, mask=1 on frame 0.
    assert torch.equal(cond[:, :, 0], first_frame[:, :, 0])
    assert torch.all(cond[:, :, 1:] == 0.0)
    assert torch.all(mask[:, :, 0] == 1.0)
    assert torch.all(mask[:, :, 1:] == 0.0)


# ── REAL trainer/pipeline dispatch path (MRO) — the leak, end to end ───────────


def _trainer_shell() -> tuple[Hv15Trainer, _CaptureTransformer]:
    """REAL Hv15Trainer + REAL Hv15Driver, no model/config load.

    Drives exactly the two hooks ``pipeline_train.py`` calls per step —
    ``PipelineBaseMixin._attach_conditioning`` → ``driver.attach_conditioning``
    and ``PipelineBaseMixin.forward_pass`` → ``driver.forward_pass`` — through the
    trainer instance (MRO), never a driver method directly.
    """
    tr = object.__new__(Hv15Trainer)
    tr.device = torch.device("cpu")
    driver, model = _driver("i2v")
    tr.driver = driver
    return tr, model


def test_dispatch_path_f1_still_routes_to_zeroed_t2v_no_leak():
    """RED before the fix: through the REAL trainer MRO an F=1 still on an i2v run
    must reach the zeroed-conditioning t2v path — cond(32)+mask(1) all zero.

    Before the guard, ``attach_conditioning`` stashes the still's own clean
    frame-0 latent and ``forward_pass`` places it into the cond channels with
    ``mask=1`` on the only frame — the answer leak this pins against. This is the
    dispatch-regression pin: it fails if anyone removes the F=1 guard from the
    real path.
    """
    tr, model = _trainer_shell()
    latents = torch.randn(2, NOISE_CHANNELS, 1, 4, 4)  # F=1 still
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([500.0, 500.0])
    t = (timesteps / 1000.0).view(-1, 1, 1, 1, 1)
    noisy = t * noise + (1.0 - t) * latents

    batch: dict = {}
    tr._attach_conditioning(batch, latents)
    assert Hv15Driver.BATCH_FIRST_FRAME_LATENT not in batch  # no stash for F=1

    tr.forward_pass(noisy, timesteps, _text4(2), batch)

    x = model.calls[0]["hidden_states"]
    assert x.shape[1] == 65
    assert torch.equal(x[:, :NOISE_CHANNELS], noisy)
    # cond(32) + mask(1) ZEROED — the clean still is NOT handed back (no leak).
    assert torch.all(x[:, NOISE_CHANNELS:] == 0.0)
    assert torch.all(model.calls[0]["image_embeds"] == 0)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
