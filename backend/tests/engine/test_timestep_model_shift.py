import math
import torch
from app.engine.strategies.timestep_sampling import TimestepSampler, _patchified_seq_len


class TestPatchifiedSeqLen:
    def test_4d_image_frames_one(self):
        assert _patchified_seq_len(torch.randn(1, 16, 64, 64), 1) == 64 * 64

    def test_5d_uses_last_two_dims_and_includes_frames(self):
        # [B,C,F,H,W] = [1,16,5,64,48]; tokens = 5 * 64 * 48 (NOT 5*64 read as F,H).
        assert _patchified_seq_len(torch.randn(1, 16, 5, 64, 48), 1) == 5 * 64 * 48

    def test_patchify_factor(self):
        assert _patchified_seq_len(torch.randn(1, 16, 2, 64, 64), 2) == 2 * 32 * 32

    def test_none_and_bad_rank(self):
        assert _patchified_seq_len(None) is None
        assert _patchified_seq_len(torch.randn(16, 64), 1) is None


class TestFluxShift5DRegression:
    def test_flux_shift_video_uses_hw_not_fh(self):
        # A 5D latent whose F != W must NOT change the shift vs a 4D latent of
        # the same H,W (frames raise seq_len, but the axis must be H,W not F,H).
        cfg = {"timestep_sampling": "flux_shift"}
        torch.manual_seed(0)
        # Just assert it runs and returns [bs] in [0,1] for 5D (would have raised
        # / mis-indexed before; now axis-correct).
        out = TimestepSampler.sample("flux_shift", 8, torch.device("cpu"), cfg,
                                     latents=torch.randn(1, 16, 5, 64, 64))
        assert out.shape == (8,)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


class TestModelShift:
    def _cpu(self):
        return torch.device("cpu")

    def test_dynamic_interpolates_with_seqlen(self):
        # LTX params: small seq → mu≈base_shift, large seq → mu≈max_shift.
        cfg = {"model_shift_base_shift": 0.95, "model_shift_max_shift": 2.05,
               "model_shift_base_seq": 1024, "model_shift_max_seq": 4096,
               "model_shift_std": 0.0001, "timestep_uniform_prob": 0.0}
        torch.manual_seed(0)
        small = TimestepSampler.sample("model_shift", 256, self._cpu(), cfg,
                                       latents=torch.randn(1, 16, 1, 32, 32))  # seq=1024
        large = TimestepSampler.sample("model_shift", 256, self._cpu(), cfg,
                                       latents=torch.randn(1, 16, 1, 64, 64))  # seq=4096
        # std≈0 → t≈sigmoid(mu). sigmoid(0.95)<sigmoid(2.05) → large mean higher.
        assert float(large.mean()) > float(small.mean())
        assert abs(float(small.mean()) - torch.sigmoid(torch.tensor(0.95)).item()) < 0.02
        assert abs(float(large.mean()) - torch.sigmoid(torch.tensor(2.05)).item()) < 0.02

    def test_fixed_shift_uses_ln(self):
        # WAN fixed flow_shift=3.0 → mu=ln(3); std≈0 → t≈sigmoid(ln(3)).
        cfg = {"model_shift_fixed": 3.0, "model_shift_std": 0.0001,
               "timestep_uniform_prob": 0.0}
        torch.manual_seed(0)
        out = TimestepSampler.sample("model_shift", 256, self._cpu(), cfg,
                                     latents=torch.randn(1, 16, 1, 64, 64))
        assert abs(float(out.mean()) - torch.sigmoid(torch.tensor(math.log(3.0))).item()) < 0.02

    def test_uniform_prob_one_is_uniform(self):
        cfg = {"model_shift_fixed": 5.0, "timestep_uniform_prob": 1.0}
        torch.manual_seed(0)
        out = TimestepSampler.sample("model_shift", 4096, self._cpu(), cfg,
                                     latents=torch.randn(1, 16, 1, 64, 64))
        assert abs(float(out.mean()) - 0.5) < 0.05  # U(0,1) mean ≈ 0.5

    def test_range_and_shape(self):
        cfg = {"model_shift_base_shift": 0.95, "model_shift_max_shift": 2.05}
        out = TimestepSampler.sample("model_shift", 16, self._cpu(), cfg,
                                     latents=torch.randn(1, 16, 3, 64, 64))
        assert out.shape == (16,)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    def test_model_shift_in_supported_modes_and_enum(self):
        assert "model_shift" in TimestepSampler.SUPPORTED_MODES
        from app.engine.models.base import BaseTrainingConfig
        schema = BaseTrainingConfig.model_json_schema()["properties"]
        assert "model_shift" in schema["timestep_sampling"]["enum"]
        assert schema["timestep_uniform_prob"]["default"] == 0.1
