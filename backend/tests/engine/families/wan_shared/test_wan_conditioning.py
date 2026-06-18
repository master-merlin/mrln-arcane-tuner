import torch
from app.engine.models.families.wan_shared.driver_base import WanDriverBase
from app.engine.models.families.wan_shared.i2v_conditioning import build_i2v_conditioning


class _Drv(WanDriverBase):
    """Concrete WAN driver stub — just exercises prepare_latents/attach_conditioning."""
    def __init__(self, is_i2v):
        # Bypass the real __init__ (needs definition + device) by setting
        # only the attributes the methods under test actually read.
        self.is_i2v = is_i2v
        self.BATCH_FIRST_FRAME_LATENT = WanDriverBase.BATCH_FIRST_FRAME_LATENT

    def get_saver(self):
        return None


class TestPrepareLatents5DLift:
    def test_4d_still_lifted_to_5d(self):
        d = _Drv(is_i2v=False)
        out = d.prepare_latents(torch.randn(2, 16, 8, 8))
        assert out.shape == (2, 16, 1, 8, 8)

    def test_5d_video_unchanged(self):
        d = _Drv(is_i2v=False)
        x = torch.randn(2, 16, 5, 8, 8)
        assert d.prepare_latents(x).shape == (2, 16, 5, 8, 8)

    def test_noise_lifted_same_as_latents(self):
        d = _Drv(is_i2v=False)
        assert d.prepare_noise(torch.randn(1, 16, 8, 8)).shape == (1, 16, 1, 8, 8)


class TestAttachConditioning:
    def test_t2v_is_noop(self):
        d = _Drv(is_i2v=False)
        batch = {}
        d.attach_conditioning(batch, torch.randn(1, 16, 5, 8, 8))
        assert d.BATCH_FIRST_FRAME_LATENT not in batch

    def test_i2v_stashes_first_frame(self):
        d = _Drv(is_i2v=True)
        batch = {}
        latents = torch.randn(1, 16, 5, 8, 8)
        d.attach_conditioning(batch, latents)
        ff = batch[d.BATCH_FIRST_FRAME_LATENT]
        assert ff.shape == (1, 16, 1, 8, 8)
        assert torch.equal(ff[:, :, 0], latents[:, :, 0])

    def test_i2v_first_frame_feeds_build_i2v_conditioning(self):
        # End-to-end: stashed first frame builds the 36-ch input without raising.
        d = _Drv(is_i2v=True)
        batch = {}
        latents = torch.randn(1, 16, 5, 8, 8)
        d.attach_conditioning(batch, latents)
        noisy = torch.randn(1, 16, 5, 8, 8)
        out = build_i2v_conditioning(noisy, batch[d.BATCH_FIRST_FRAME_LATENT])
        assert out.shape == (1, 36, 5, 8, 8)
