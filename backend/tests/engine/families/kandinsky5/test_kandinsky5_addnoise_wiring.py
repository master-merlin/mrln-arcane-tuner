"""Kandinsky 5.0 I2V — REAL-path ``add_noise`` wiring (recon-found bug).

SYMPTOM (recon, no live incident yet): ``Kandinsky5Driver.add_noise``
(``driver.py:445-467``) carries an I2V frame-0-clean pin — engaged I2V steps
must NEVER noise frame 0, because ``Kandinsky5Trainer._compute_step_loss``
(``trainer.py:95-115``) drops frame 0 from the loss on the assumption that it
stays the clean conditioning frame (mirroring the upstream I2V pipeline,
which never lets its scheduler touch frame 0).

ROOT CAUSE: the REAL training loop (``pipeline_train.py``) never calls
``driver.add_noise`` directly — it calls the TRAINER's own ``self.add_noise``
(family hook, MRO-resolved). ``Kandinsky5Trainer`` does not override
``add_noise``, so the call resolves to ``PipelineBaseMixin.add_noise``
(``pipeline_base.py:129-140``), which unconditionally delegates to
``self.noise_interpolation.add_noise`` (:class:`NoiseInterpolation`,
mode ``"linear"``) — a component with NO knowledge of Kandinsky5's I2V
frame-0 pin. On a real I2V run, frame 0 gets noised like every other frame,
directly contradicting the loss-exclusion logic's assumption.

Every existing ``add_noise`` test (``test_kandinsky5_driver.py``) calls
``drv.add_noise(...)`` directly — never through ``trainer.add_noise``, so the
MRO-resolution gap was invisible to unit tests (the historical krea2/boogu
bug class: "only the real path — or here, its exact call sequence —
surfaces it").

THE FIX (``trainer.py``): ``Kandinsky5Trainer.add_noise`` now overrides the
``PipelineBaseMixin`` hook to delegate to ``self.driver.add_noise`` — mirrors
``boogu_image``'s convention-delegation precedent
(``families/boogu_image/trainer.py:239-254``). Verified SAFE for T2V: the
driver's T2V math (``t*noise + (1-t)*latents``, RAW ``[0,1000]`` scale) is
algebraically IDENTICAL to ``NoiseInterpolation._linear``'s
``(1-t)*latents + t*noise`` (same terms, commutative sum) — the delegation
changes ZERO T2V training behavior, only wires the I2V pin that was already
dead code on the real path.

This test walks the REAL sequence ``pipeline_train.py`` executes on every
training step: ``trainer._attach_conditioning(batch, latents)`` (sets the
per-step i2v flag + stashes the clean frame) ->
``trainer.prepare_latents_for_training(latents)`` (delegates to
``driver.prepare_latents``, records the latent grid) ->
``trainer.add_noise(prepared_latents, noise, timesteps)`` — the exact family
hook the loop calls. Nothing in this test calls ``driver.add_noise``
directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from app.engine.models.families.kandinsky5.driver import (
    FLOWMATCH_SCALE,
    Kandinsky5Driver,
)
from app.engine.models.families.kandinsky5.trainer import Kandinsky5Trainer
from app.engine.strategies.noise_interpolation import NoiseInterpolation


def _i2v_trainer(prob: float = 1.0) -> Kandinsky5Trainer:
    """Real trainer + real driver, i2v mode — mirrors
    ``test_kandinsky5_text_cache.py``'s ``_i2v_trainer`` helper."""
    t = object.__new__(Kandinsky5Trainer)
    t.device = torch.device("cpu")
    t.config = {
        "video_mode": "i2v",
        "first_frame_conditioning_probability": prob,
    }
    d = MagicMock()
    d.family = "kandinsky5"
    d.id = "k5-test"
    d.lora_targetable_modules = []
    d.architecture_params = {"mode": "i2v"}
    t.driver = Kandinsky5Driver(d, torch.device("cpu"))
    # The real setup path always assigns this (pipeline_loading.py) before
    # any add_noise call — without it the base MRO's fallback would AttributeError
    # instead of silently misbehaving, masking the real bug.
    t.noise_interpolation = NoiseInterpolation("linear")
    return t


def _t2v_trainer() -> Kandinsky5Trainer:
    t = object.__new__(Kandinsky5Trainer)
    t.device = torch.device("cpu")
    t.config = {"video_mode": "t2v"}
    d = MagicMock()
    d.family = "kandinsky5"
    d.id = "k5-test-t2v"
    d.lora_targetable_modules = []
    d.architecture_params = {"mode": "t2v"}
    t.driver = Kandinsky5Driver(d, torch.device("cpu"))
    t.noise_interpolation = NoiseInterpolation("linear")
    return t


class TestRealPathI2VFrame0Wiring:
    """Load-bearing regression test — FAILS RED without the trainer override."""

    def test_engaged_i2v_real_step_sequence_keeps_frame0_clean(self) -> None:
        t = _i2v_trainer(prob=1.0)
        batch: dict = {}
        # Channels-first raw latents [B, C, F, H, W], F=3 (multi-frame — engages).
        latents = torch.randn(2, 4, 3, 8, 8)

        # Exact pipeline_train.py sequence: attach BEFORE prepare/noise.
        t._attach_conditioning(batch, latents)
        assert t.driver._i2v_active is True  # sanity: gate actually engaged

        prepared_latents = t.prepare_latents_for_training(latents)
        assert t.driver._i2v_conditioning_engaged() is True  # sanity: F>1

        noise = torch.randn_like(prepared_latents)
        timesteps = torch.tensor([700.0, 700.0])

        # THE REAL CALL — exactly what pipeline_train.py:543 makes.
        noisy_input = t.add_noise(prepared_latents, noise, timesteps)

        assert torch.equal(noisy_input[:, :1], prepared_latents[:, :1]), (
            "frame 0 must stay the CLEAN conditioning frame through the "
            "REAL trainer.add_noise() call — got it noised, which "
            "contradicts Kandinsky5Trainer._compute_step_loss's frame-0 "
            "exclusion assumption (loss-excluded but noised == wrong "
            "training objective)."
        )
        assert not torch.equal(noisy_input[:, 1:], prepared_latents[:, 1:]), (
            "sanity: frames 1+ must actually be noised (not a no-op bug)"
        )

    def test_single_frame_i2v_step_noises_frame0_normally(self) -> None:
        """A still (F=1) disengages i2v conditioning — frame 0 IS noised
        (mirrors the driver-level guard; no clean-everything degenerate)."""
        t = _i2v_trainer(prob=1.0)
        batch: dict = {}
        latents = torch.randn(1, 4, 1, 8, 8)  # single frame

        t._attach_conditioning(batch, latents)
        prepared_latents = t.prepare_latents_for_training(latents)
        assert t.driver._i2v_conditioning_engaged() is False

        noise = torch.randn_like(prepared_latents)
        noisy_input = t.add_noise(prepared_latents, noise, torch.tensor([500.0]))
        assert not torch.equal(noisy_input, prepared_latents)

    def test_t2v_real_step_matches_base_noise_interpolation_exactly(self) -> None:
        """Delegation must not silently change T2V training: the driver's
        add_noise (routed through the trainer override) must be BIT-IDENTICAL
        to what the un-overridden base ``PipelineBaseMixin.add_noise`` would
        have produced via ``NoiseInterpolation('linear')`` — same formula,
        same [0,1000] scale, just commuted term order."""
        t = _t2v_trainer()
        latents = torch.randn(2, 4, 2, 8, 8)
        prepared_latents = t.prepare_latents_for_training(latents)
        noise = torch.randn_like(prepared_latents)
        timesteps = torch.tensor([333.0, 812.0])

        via_trainer = t.add_noise(prepared_latents, noise, timesteps)
        via_base_component = t.noise_interpolation.add_noise(
            prepared_latents, noise, timesteps
        )

        assert torch.equal(via_trainer, via_base_component), (
            "kandinsky5 T2V add_noise delegation must be numerically "
            "identical to the base NoiseInterpolation path — any "
            "divergence would silently change T2V training math."
        )

    def test_flowmatch_scale_preserved_through_trainer_call(self) -> None:
        """The trainer-level call must still divide by FLOWMATCH_SCALE exactly
        once — not zero times (pure-noise gotcha) and not twice."""
        t = _t2v_trainer()
        latents = torch.zeros(1, 2, 1, 4, 4)
        prepared_latents = t.prepare_latents_for_training(latents)
        noise = torch.ones_like(prepared_latents)
        noisy = t.add_noise(prepared_latents, noise, torch.tensor([500.0]))
        assert torch.allclose(noisy, torch.full_like(noisy, 500.0 / FLOWMATCH_SCALE))
