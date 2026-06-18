import torch
from app.engine.strategies.sigma_schedule import shifted_sigmas
from app.engine.strategies.sigma_tracker import SigmaTracker


class TestShiftedSigmas:
    def test_identity_when_shift_one(self):
        out = shifted_sigmas(10, 1.0)
        assert torch.allclose(out, torch.linspace(1.0, 0.0, 11))

    def test_endpoints_fixed(self):
        out = shifted_sigmas(8, 3.0)
        assert abs(float(out[0]) - 1.0) < 1e-6
        assert abs(float(out[-1]) - 0.0) < 1e-6

    def test_shift_gt_one_pushes_toward_high_noise(self):
        lin = shifted_sigmas(100, 1.0)
        sh = shifted_sigmas(100, 3.0)
        # At every interior point the shifted schedule sits ABOVE linear (more
        # high-noise mass) for s>1.
        assert (sh[1:-1] >= lin[1:-1] - 1e-6).all()
        assert float(sh[1:-1].mean()) > float(lin[1:-1].mean())

    def test_monotonic_descending(self):
        out = shifted_sigmas(50, 5.0)
        assert (out[:-1] - out[1:] >= -1e-6).all()


class TestSigmaTracker:
    def test_mean_and_bins(self):
        tr = SigmaTracker(bins=10)
        # timesteps in [0,1000]; a batch all at 0.9*1000 → decile index 8.
        tr.update(torch.full((100,), 900.0), scale=1000.0)
        s = tr.summary()
        assert s["n"] == 100
        assert abs(s["mean"] - 0.9) < 1e-3
        assert s["deciles"][8] == 1.0

    def test_empty_safe(self):
        assert SigmaTracker().summary()["n"] == 0
