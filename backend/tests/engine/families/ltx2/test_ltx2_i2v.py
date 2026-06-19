"""Tests for LTX-2 image-to-video (i2v) first-frame conditioning.

Tests the per-token timestep helper, add_noise first-frame clean behaviour,
and the trainer's i2v gate + loss masking.  Uses object.__new__ + manual attr
assignment so no model weights need to be loaded.
"""

from __future__ import annotations

import types

import torch

from app.engine.models.families.ltx2.driver import Ltx2Driver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bare_driver(latent_shape=(2, 2, 2), i2v=True):
    """Build a Ltx2Driver shell without loading any model weights."""
    d = object.__new__(Ltx2Driver)
    d.patch_size = 1
    d.patch_size_t = 1
    d._latent_shape = latent_shape  # (post_f, post_h, post_w)
    d._i2v_active = i2v
    d.train_audio = False
    # Attrs read by forward_pass / compute_loss but not exercised here
    d.audio_in_channels = 128
    d.caption_channels = 3840
    d.frame_rate = 24.0
    return d


# ---------------------------------------------------------------------------
# 1. Per-token timestep helper
# ---------------------------------------------------------------------------

class TestPerTokenTimestep:
    def test_first_frame_tokens_zero(self):
        d = _bare_driver((3, 2, 2))           # num_tokens = 12, tpf = 4
        t = d._i2v_per_token_timestep(torch.full((1,), 900.0), 12, 4)
        assert t.shape == (1, 12)
        assert (t[:, :4] == 0).all(), "first frame tokens must be 0 (clean)"
        assert (t[:, 4:] == 900.0).all(), "remaining tokens must carry sigma"

    def test_batch_dim(self):
        d = _bare_driver((2, 2, 2))           # num_tokens = 8, tpf = 4
        sigmas = torch.tensor([200.0, 800.0])
        t = d._i2v_per_token_timestep(sigmas, 8, 4)
        assert t.shape == (2, 8)
        assert (t[:, :4] == 0).all()
        assert torch.equal(t[:, 4:], sigmas.unsqueeze(1).expand(-1, 4))

    def test_zero_timestep_all_zero(self):
        """t=0 → all tokens 0 (trivially clean everywhere)."""
        d = _bare_driver((3, 2, 2))
        t = d._i2v_per_token_timestep(torch.zeros(1), 12, 4)
        assert (t == 0).all()

    def test_full_noise_non_first(self):
        """t=1000 → non-first-frame tokens 1000, first frame tokens still 0."""
        d = _bare_driver((3, 2, 2))
        t = d._i2v_per_token_timestep(torch.full((1,), 1000.0), 12, 4)
        assert (t[:, :4] == 0).all()
        assert (t[:, 4:] == 1000.0).all()


# ---------------------------------------------------------------------------
# 2. add_noise — i2v path
# ---------------------------------------------------------------------------

class TestAddNoiseLeavesFirstFrameClean:
    def test_first_frame_clean_rest_noised(self):
        d = _bare_driver((3, 2, 2))           # num_tokens 12, tpf 4
        torch.manual_seed(0)
        latents = torch.randn(1, 12, 8)
        noise = torch.randn(1, 12, 8)
        out = d.add_noise(latents, noise, torch.full((1,), 500.0))
        # First 4 tokens == clean latents (t=0 → 0*noise + 1*latents)
        assert torch.allclose(out[:, :4], latents[:, :4]), \
            "conditioning frame must be byte-identical to clean latents"
        # Remaining tokens differ from clean (t=0.5 → 0.5*noise + 0.5*latents)
        assert not torch.allclose(out[:, 4:], latents[:, 4:]), \
            "non-conditioning tokens must be noised"

    def test_first_frame_lerp_value(self):
        """At t=500 the non-conditioning tokens should match the flow-match lerp."""
        d = _bare_driver((2, 2, 2))           # num_tokens 8, tpf 4
        latents = torch.ones(1, 8, 4)
        noise = torch.zeros(1, 8, 4)
        # frac = 0.5 → noisy = 0.5*0 + 0.5*1 = 0.5
        out = d.add_noise(latents, noise, torch.full((1,), 500.0))
        assert torch.allclose(out[:, :4], latents[:, :4])    # frame 0 clean
        assert torch.allclose(out[:, 4:], torch.full_like(out[:, 4:], 0.5), atol=1e-5)

    def test_t2v_unchanged(self):
        """Non-i2v (t2v) path must be byte-identical to the scalar flow-match."""
        d = _bare_driver((3, 2, 2), i2v=False)
        torch.manual_seed(1)
        latents = torch.randn(1, 12, 8)
        noise = torch.randn(1, 12, 8)
        out = d.add_noise(latents, noise, torch.full((1,), 500.0))
        # scalar lerp: frac = 0.5 for ALL tokens
        expected = 0.5 * noise + 0.5 * latents
        assert torch.allclose(out, expected, atol=1e-5), \
            "t2v path must be byte-identical to old scalar behaviour"

    def test_i2v_inactive_flag_bypasses_conditioning(self):
        """_i2v_active=False must fall through to scalar path regardless of mode."""
        d = _bare_driver((3, 2, 2), i2v=False)
        latents = torch.ones(1, 12, 4)
        noise = torch.zeros(1, 12, 4)
        out = d.add_noise(latents, noise, torch.full((1,), 1000.0))
        # frac=1 → noisy = 1*noise + 0*latents = 0
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_single_frame_still_fully_noised(self):
        """F=1 still under i2v: conditioning is bypassed → the WHOLE latent is
        noised (t2v). If i2v engaged on a still, the first-frame tokens (== ALL
        tokens) would stay clean and the loss would mask everything → NaN.
        """
        d = _bare_driver((1, 2, 2), i2v=True)   # F=1 → num_tokens=4, tpf=4
        latents = torch.ones(1, 4, 4)
        noise = torch.zeros(1, 4, 4)
        # t=1000 → frac=1 → fully noised → zeros. i2v-engaged would leave ones.
        out = d.add_noise(latents, noise, torch.full((1,), 1000.0))
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5), \
            "single-frame still must be fully noised (i2v conditioning bypassed)"


# ---------------------------------------------------------------------------
# 3. forward_pass — per-token timestep is passed to the transformer (i2v)
#    The transformer interface is mocked; we only check the timestep shape/values.
# ---------------------------------------------------------------------------

class _RecordingTransformer(torch.nn.Module):
    """Minimal fake transformer that records kwarg shapes."""

    def __init__(self, num_tokens: int, d_model: int = 4):
        super().__init__()
        self._num_tokens = num_tokens
        self._d_model = d_model
        self.last_timestep = None
        self.last_audio_timestep = None

    def forward(self, hidden_states, **kwargs):
        self.last_timestep = kwargs.get("timestep")
        self.last_audio_timestep = kwargs.get("audio_timestep")
        B = hidden_states.shape[0]
        return (torch.zeros(B, self._num_tokens, self._d_model),)


def _driver_for_forward(latent_shape=(3, 2, 2), i2v=True):
    """Build a driver wired with a fake transformer, ready for forward_pass."""
    post_f, post_h, post_w = latent_shape
    num_tokens = post_f * post_h * post_w
    d = _bare_driver(latent_shape, i2v=i2v)
    d.transformer = _RecordingTransformer(num_tokens)
    # forward_pass needs these helpers; stub them directly on the instance
    d._batch_frame_rate = lambda batch: 24.0
    # _video_embeddings is a static method, but we can override on the instance:
    # It only needs to return a tensor shaped [B, L, C] — shape doesn't matter for this test
    def _fake_video_emb(te):
        return torch.zeros(1, 1, 3840)
    # override via types.MethodType so it works as a bound method
    d._video_embeddings = _fake_video_emb
    return d


class TestForwardPassTimestepShape:
    def test_i2v_passes_per_token_timestep(self):
        d = _driver_for_forward((3, 2, 2), i2v=True)
        num_tokens = 3 * 2 * 2  # 12
        tpf = 2 * 2              # 4

        noisy = torch.zeros(1, num_tokens, 4)
        # Use a SimpleNamespace as a minimal TextEncoderOutput stand-in
        text_emb = types.SimpleNamespace(
            embeddings=torch.zeros(1, 1, 3840),
            pooled=None,
            attention_mask=None,
        )
        batch = {}

        d.forward_pass(noisy, torch.full((1,), 600.0), text_emb, batch)

        recorded = d.transformer.last_timestep
        assert recorded is not None, "transformer must receive a timestep kwarg"
        assert recorded.shape == (1, num_tokens), \
            f"i2v timestep must be [B, num_tokens], got {recorded.shape}"
        # First tpf tokens must be 0 (conditioning frame)
        assert (recorded[:, :tpf] == 0).all(), "first frame tokens must be 0"
        # Rest must carry the sampled sigma
        assert (recorded[:, tpf:] == 600.0).all(), "non-first-frame tokens must be sigma"

        # REGRESSION (GPU smoke ltx2_i2v): the isolated dummy-audio stream must NOT
        # inherit the per-token VIDEO timestep. diffusers LTX2 defaults
        # audio_timestep to `timestep`; a per-token [B, num_tokens] video timestep
        # then sizes the audio modulation/RoPE to the video token count and crashes
        # against the 1-token dummy audio ("tensor a (N) must match tensor b (1)").
        # The driver must pass a per-BATCH scalar audio_timestep [B].
        audio_t = d.transformer.last_audio_timestep
        assert audio_t is not None, "transformer must receive an explicit audio_timestep"
        assert audio_t.shape == (1,), \
            f"audio_timestep must be [B] scalar (not per-token), got {audio_t.shape}"
        assert (audio_t == 600.0).all(), "audio_timestep must be the per-batch sigma"

    def test_t2v_passes_scalar_timestep(self):
        d = _driver_for_forward((3, 2, 2), i2v=False)
        num_tokens = 3 * 2 * 2

        noisy = torch.zeros(1, num_tokens, 4)
        text_emb = types.SimpleNamespace(
            embeddings=torch.zeros(1, 1, 3840),
            pooled=None,
            attention_mask=None,
        )
        batch = {}

        d.forward_pass(noisy, torch.full((1,), 400.0), text_emb, batch)

        recorded = d.transformer.last_timestep
        assert recorded is not None
        # t2v path should pass the original [B] timestep, NOT [B, num_tokens]
        assert recorded.shape == (1,), \
            f"t2v timestep must be [B] (scalar), got {recorded.shape}"
        assert (recorded == 400.0).all()
        # The dummy-audio stream also takes the per-batch scalar timestep.
        audio_t = d.transformer.last_audio_timestep
        assert audio_t is not None and audio_t.shape == (1,), \
            f"t2v audio_timestep must be [B] scalar, got " \
            f"{None if audio_t is None else audio_t.shape}"

    def test_i2v_single_frame_passes_scalar_timestep(self):
        """F=1 still under i2v must pass a scalar [B] timestep (conditioning
        bypassed) — there is no subsequent frame to predict, so per-token
        conditioning is meaningless and would NaN the loss."""
        d = _driver_for_forward((1, 2, 2), i2v=True)   # F=1 → num_tokens=4
        noisy = torch.zeros(1, 4, 4)
        text_emb = types.SimpleNamespace(
            embeddings=torch.zeros(1, 1, 3840), pooled=None, attention_mask=None,
        )
        d.forward_pass(noisy, torch.full((1,), 600.0), text_emb, {})
        recorded = d.transformer.last_timestep
        assert recorded.shape == (1,), \
            f"single-frame i2v must pass scalar timestep, got {recorded.shape}"


# ---------------------------------------------------------------------------
# 4. Trainer _attach_conditioning gate
# ---------------------------------------------------------------------------

def _bare_trainer(video_mode="i2v", prob=0.5, train_audio=False):
    """Build a minimal Ltx2Trainer-shaped object without a real pipeline."""
    from app.engine.models.families.ltx2.trainer import Ltx2Trainer

    t = object.__new__(Ltx2Trainer)
    t.config = {
        "video_mode": video_mode,
        "first_frame_conditioning_probability": prob,
        "train_audio": train_audio,
    }
    d = _bare_driver((3, 2, 2), i2v=False)   # start inactive
    t.driver = d
    return t


class TestAttachConditioningGate:
    def test_t2v_never_activates(self):
        trainer = _bare_trainer(video_mode="t2v", prob=1.0)
        # Even with p=1.0, t2v must never activate
        for _ in range(20):
            trainer._attach_conditioning({}, torch.zeros(1, 12, 8))
            assert trainer.driver._i2v_active is False, "t2v must never set _i2v_active"

    def test_i2v_p1_always_active(self):
        trainer = _bare_trainer(video_mode="i2v", prob=1.0)
        for _ in range(10):
            trainer._attach_conditioning({}, torch.zeros(1, 12, 8))
            assert trainer.driver._i2v_active is True, "p=1.0 must always activate"

    def test_i2v_p0_never_active(self):
        trainer = _bare_trainer(video_mode="i2v", prob=0.0)
        for _ in range(10):
            trainer._attach_conditioning({}, torch.zeros(1, 12, 8))
            assert trainer.driver._i2v_active is False, "p=0.0 must never activate"

    def test_audio_batch_suppresses_i2v(self):
        """Audio batches must not activate i2v conditioning."""
        trainer = _bare_trainer(video_mode="i2v", prob=1.0)
        audio_batch = {"audio_clean": torch.zeros(1, 10, 128)}
        trainer._attach_conditioning(audio_batch, torch.zeros(1, 12, 8))
        assert trainer.driver._i2v_active is False, \
            "audio batch must suppress i2v conditioning"

    def test_stochastic_activation(self):
        """With p=0.5, roughly half the calls should activate (with a wide margin)."""
        import random
        trainer = _bare_trainer(video_mode="i2v", prob=0.5)
        random.seed(42)
        activations = 0
        n = 200
        for _ in range(n):
            trainer._attach_conditioning({}, torch.zeros(1, 12, 8))
            if trainer.driver._i2v_active:
                activations += 1
        # Expect 50% ± 15%
        ratio = activations / n
        assert 0.35 <= ratio <= 0.65, \
            f"Stochastic gate with p=0.5 gave {ratio:.2f} activation rate"


# ---------------------------------------------------------------------------
# 5. Trainer _compute_step_loss — conditioning frame masked out
# ---------------------------------------------------------------------------

def _bare_trainer_for_loss(latent_shape=(3, 2, 2)):
    """Build a minimal Ltx2Trainer-shaped object for loss tests."""
    from app.engine.models.families.ltx2.trainer import Ltx2Trainer

    t = object.__new__(Ltx2Trainer)
    t.config = {"video_mode": "i2v", "first_frame_conditioning_probability": 1.0}
    d = _bare_driver(latent_shape, i2v=True)
    t.driver = d
    return t


class TestComputeStepLossI2V:
    def test_loss_excludes_first_frame_tokens(self):
        """When i2v is active, the first tpf tokens must NOT contribute to loss.

        We fabricate a pred that is WRONG only for the first-frame tokens and
        CORRECT for the rest. The i2v loss should be near-zero (ignores frame 0).
        The t2v loss would be large.
        """
        # latent_shape=(3, 2, 2) → num_tokens=12, tpf=4
        trainer = _bare_trainer_for_loss((3, 2, 2))
        num_tokens = 12
        tpf = 4

        target = torch.zeros(1, num_tokens, 4)
        pred = torch.zeros_like(target)
        # Corrupt only the FIRST frame tokens
        pred[:, :tpf] = 100.0

        # grad_accum=1, no audio
        loss_i2v = trainer._compute_step_loss(pred, target, torch.full((1,), 500.0), {}, 1)
        # If loss ignores first frame, it should be 0 (the non-first tokens are perfect)
        assert loss_i2v.item() < 1e-6, \
            f"i2v loss must ignore first-frame tokens, got {loss_i2v.item()}"

    def test_t2v_includes_all_tokens(self):
        """When i2v is inactive, all tokens contribute to loss."""
        from app.engine.models.families.ltx2.trainer import Ltx2Trainer

        t = object.__new__(Ltx2Trainer)
        t.config = {"video_mode": "t2v"}
        d = _bare_driver((3, 2, 2), i2v=False)
        t.driver = d

        target = torch.zeros(1, 12, 4)
        pred = torch.zeros_like(target)
        pred[:, :4] = 100.0  # Corrupt first frame — should be counted now

        loss_t2v = t._compute_step_loss(pred, target, torch.full((1,), 500.0), {}, 1)
        assert loss_t2v.item() > 1.0, \
            f"t2v must include first-frame tokens in loss, got {loss_t2v.item()}"

    def test_audio_batch_bypasses_i2v_mask(self):
        """When audio is present, loss falls back to full-token audio path."""
        trainer = _bare_trainer_for_loss((3, 2, 2))
        # Simulate audio-on step: driver reports i2v active BUT audio is present
        # The task spec says 'audio path and non-i2v path unchanged'
        trainer.driver._i2v_active = True
        trainer.driver.train_audio = False  # audio OFF at driver level

        batch_with_audio = {"audio_clean": torch.zeros(1, 10, 128)}
        target = torch.zeros(1, 12, 4)
        pred = torch.zeros_like(target)
        pred[:, :4] = 100.0  # corrupt first frame

        # With audio_clean present in batch but train_audio=False on driver,
        # the loss path should fall back to full video MSE (audio_pred is None).
        # The i2v mask is skipped when audio_clean is in the batch.
        loss = trainer._compute_step_loss(
            pred, target, torch.full((1,), 500.0), batch_with_audio, 1
        )
        # Loss should include ALL tokens because the audio branch bypasses the i2v mask
        assert loss.item() > 1.0, \
            f"audio batch must bypass i2v loss mask, got {loss.item()}"

    def test_grad_accum_division(self):
        """Loss must be divided by grad_accum."""
        trainer = _bare_trainer_for_loss((3, 2, 2))
        pred = torch.ones(1, 12, 4)
        target = torch.zeros(1, 12, 4)

        loss1 = trainer._compute_step_loss(pred, target, torch.full((1,), 500.0), {}, 1)
        loss2 = trainer._compute_step_loss(pred, target, torch.full((1,), 500.0), {}, 2)
        # loss2 should be half of loss1 (mod floating point)
        assert abs(loss2.item() * 2 - loss1.item()) < 1e-5, \
            f"grad_accum=2 must halve the loss: {loss1.item()} vs {loss2.item()}"

    def test_single_frame_still_full_token_loss(self):
        """F=1 still under i2v: loss must use ALL tokens (t2v) and stay finite.

        Regression for the GPU smoke ltx2_i2v NaN: masking the (only) frame
        slices pred/target to an EMPTY tensor → mean over 0 elements → NaN.
        A mixed stills+video i2v dataset must train every step.
        """
        trainer = _bare_trainer_for_loss((1, 2, 2))   # F=1 → num_tokens=4, tpf=4
        target = torch.zeros(1, 4, 4)
        pred = torch.zeros_like(target)
        pred[:, :2] = 100.0   # corrupt some tokens — must be counted
        loss = trainer._compute_step_loss(pred, target, torch.full((1,), 500.0), {}, 1)
        assert torch.isfinite(loss).all(), \
            f"single-frame i2v loss must be finite, got {loss.item()}"
        assert loss.item() > 1.0, \
            "still must use full-token loss (i2v mask bypassed)"
