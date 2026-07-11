"""LTX-2 I2V — REAL-path ``add_noise`` wiring (recon-found dead-dispatch bug).

SYMPTOM (recon, no live incident yet): ``Ltx2Driver.add_noise``
(``driver.py:300-329``) carries an I2V frame-0-clean pin — the first
``tokens_per_frame`` tokens must receive ``t=0`` (stay the clean conditioning
frame) whenever i2v conditioning is engaged, because
``Ltx2Trainer._compute_step_loss`` (``trainer.py:371-418``) drops those same
tokens from the loss on the assumption that they were never noised (mirrors
the upstream LTX-2 i2v pipeline, which never lets its scheduler touch the
conditioning frame).

ROOT CAUSE: the REAL training loop (``pipeline_train.py``) never calls
``driver.add_noise`` directly for the VIDEO stream — it calls the TRAINER's
own ``self.add_noise`` (family hook, MRO-resolved,
``pipeline_train.py:543``). ``Ltx2Trainer`` does not override ``add_noise``,
so the call resolves to ``PipelineBaseMixin.add_noise``
(``pipeline_base.py:129-140``), which unconditionally delegates to
``self.noise_interpolation.add_noise`` (:class:`NoiseInterpolation`, mode
``"linear"``) — a component with NO knowledge of LTX-2's i2v frame-0 pin. On
a real i2v run, the conditioning frame's tokens get noised like every other
token, directly contradicting ``_compute_step_loss``'s frame-0-exclusion
assumption: the model is trained against a loss that ignores those tokens
while never actually being SHOWN the clean conditioning signal the upstream
architecture expects — a strictly worse outcome than either "always clean"
or "always noised + always in the loss".

NOTE: ``Ltx2Driver.add_noise`` is NOT fully dead code — ``forward_pass``
(``driver.py:459``) calls ``self.add_noise(audio_clean, audio_noise,
timesteps)`` for the AUDIO stream, but that is a DRIVER-internal call (``self``
is the driver, not the trainer) and resolves correctly regardless of this
bug. Only the VIDEO stream's noising — driven by the trainer-level hook the
train loop actually calls — was dead.

THE FIX (``trainer.py``): ``Ltx2Trainer.add_noise`` now overrides the
``PipelineBaseMixin`` hook to delegate to ``self.driver.add_noise`` — mirrors
the ``kandinsky5``/``boogu_image`` convention-delegation precedent. Verified
SAFE for T2V (and i2v-inactive steps): the driver's un-engaged math (``frac =
timesteps / _FLOWMATCH_SCALE; frac*noise + (1-frac)*latents``) is
algebraically IDENTICAL to ``NoiseInterpolation._linear``'s
``(1-t)*latents + t*noise`` (same terms, commutative sum, same ``/1000``
scale) — the delegation changes ZERO non-i2v training behavior, only wires
the i2v pin that was already dead code on the real path.

This test walks the REAL sequence ``pipeline_train.py`` executes on every
training step: ``trainer._attach_conditioning(batch, latents)`` (sets the
per-step i2v flag) -> ``trainer.prepare_latents_for_training(latents)``
(delegates to ``driver.prepare_latents``, records the post-patch latent grid
+ packs to token space) -> ``trainer.add_noise(prepared_latents, noise,
timesteps)`` — the exact family hook the loop calls. Nothing in this test
calls ``driver.add_noise`` directly.
"""

from __future__ import annotations

import torch

from app.engine.models.families.ltx2.driver import Ltx2Driver, _FLOWMATCH_SCALE
from app.engine.models.families.ltx2.trainer import Ltx2Trainer
from app.engine.strategies.noise_interpolation import NoiseInterpolation


def _i2v_trainer(prob: float = 1.0) -> Ltx2Trainer:
    """Real trainer + real driver, i2v mode, patch_size=1 (token == voxel)."""
    t = object.__new__(Ltx2Trainer)
    t.device = torch.device("cpu")
    t.config = {
        "video_mode": "i2v",
        "first_frame_conditioning_probability": prob,
        "train_audio": False,
    }
    d = object.__new__(Ltx2Driver)
    d.patch_size = 1
    d.patch_size_t = 1
    d.train_audio = False
    d._latent_shape = None
    t.driver = d
    # The real setup path always assigns this (pipeline_loading.py) before any
    # add_noise call — without it the base MRO's fallback would AttributeError
    # instead of silently misbehaving, masking the real bug.
    t.noise_interpolation = NoiseInterpolation("linear")
    return t


def _t2v_trainer() -> Ltx2Trainer:
    t = object.__new__(Ltx2Trainer)
    t.device = torch.device("cpu")
    t.config = {"video_mode": "t2v", "train_audio": False}
    d = object.__new__(Ltx2Driver)
    d.patch_size = 1
    d.patch_size_t = 1
    d.train_audio = False
    d._latent_shape = None
    t.driver = d
    t.noise_interpolation = NoiseInterpolation("linear")
    return t


class TestRealPathI2VFrame0Wiring:
    """Load-bearing regression test — FAILS RED without the trainer override."""

    def test_engaged_i2v_real_step_sequence_keeps_frame0_clean(self) -> None:
        t = _i2v_trainer(prob=1.0)
        batch: dict = {}
        # [B, C, F, H, W], F=3 (multi-frame — engages i2v conditioning).
        latents = torch.randn(2, 4, 3, 8, 8)

        # Exact pipeline_train.py sequence: attach BEFORE prepare/noise.
        t._attach_conditioning(batch, latents)
        assert t.driver._i2v_active is True  # sanity: gate actually engaged

        prepared_latents = t.prepare_latents_for_training(latents)
        assert t.driver._i2v_conditioning_engaged() is True  # sanity: F>1
        _, h, w = t.driver._latent_grid()
        tpf = h * w  # tokens_per_frame (patch_size=1 → post_h*post_w)

        noise = torch.randn_like(prepared_latents)
        timesteps = torch.tensor([700.0, 700.0])

        # THE REAL CALL — exactly what pipeline_train.py:543 makes.
        noisy_input = t.add_noise(prepared_latents, noise, timesteps)

        assert torch.equal(noisy_input[:, :tpf], prepared_latents[:, :tpf]), (
            "the conditioning frame's tokens must stay CLEAN through the REAL "
            "trainer.add_noise() call — got them noised, which contradicts "
            "Ltx2Trainer._compute_step_loss's frame-0-token exclusion "
            "(loss-excluded but noised == wrong training objective, and the "
            "model never sees a clean conditioning signal)."
        )
        assert not torch.equal(noisy_input[:, tpf:], prepared_latents[:, tpf:]), (
            "sanity: non-conditioning tokens must actually be noised (not a "
            "no-op bug)"
        )

    def test_single_frame_i2v_step_noises_all_tokens(self) -> None:
        """A still (F=1) disengages i2v conditioning — every token IS noised
        (mirrors the driver-level guard; no clean-everything degenerate that
        would NaN the loss mean over an empty slice)."""
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
            "ltx2 T2V add_noise delegation must be numerically identical to "
            "the base NoiseInterpolation path — any divergence would "
            "silently change T2V training math."
        )

    def test_flowmatch_scale_preserved_through_trainer_call(self) -> None:
        """The trainer-level call must still divide by _FLOWMATCH_SCALE exactly
        once — not zero times (pure-noise gotcha) and not twice."""
        t = _t2v_trainer()
        latents = torch.zeros(1, 2, 1, 4, 4)
        prepared_latents = t.prepare_latents_for_training(latents)
        noise = torch.ones_like(prepared_latents)
        noisy = t.add_noise(prepared_latents, noise, torch.tensor([500.0]))
        assert torch.allclose(
            noisy, torch.full_like(noisy, 500.0 / _FLOWMATCH_SCALE)
        )


class TestComputeTargetVideoDispatchIsHarmlesslyDead:
    """``Ltx2Driver.compute_target`` (driver.py:331-338) is a THIRD MRO-dead
    method on the real VIDEO path (same shape as ``add_noise``): the train
    loop calls ``self.compute_target(...)`` on the TRAINER
    (``pipeline_train.py:551``), and ``Ltx2Trainer`` does not override it, so
    it resolves to ``PipelineBaseMixin.compute_target``
    (``pipeline_base.py:98-106``) — NOT ``Ltx2Driver.compute_target``.

    Unlike ``add_noise``, this is proven HARMLESS, not a bug: both formulas
    are the UNCONDITIONAL, t-independent flow-matching velocity ``noise -
    latents`` — ``Ltx2Driver.compute_target`` has no i2v (or any other)
    branch, so there is nothing for the trainer to fail to wire. No trainer
    override was added; this test pins the equivalence so a future edit that
    adds real special-casing to ONE side without the other gets caught.

    NOTE: ``Ltx2Driver.compute_target`` is still live code — the driver's own
    ``forward_pass`` calls ``self.compute_target(...)`` directly for the
    AUDIO stream (driver.py:460, a driver-internal call unaffected by this
    MRO gap) — so it must NOT be deleted.
    """

    def test_real_dispatch_matches_driver_method_for_video(self) -> None:
        t = _t2v_trainer()
        latents = torch.randn(2, 4, 2, 8, 8)
        prepared_latents = t.prepare_latents_for_training(latents)
        noise = torch.randn_like(prepared_latents)
        timesteps = torch.tensor([333.0, 812.0])

        via_real_dispatch = t.compute_target(prepared_latents, noise, timesteps)
        via_driver_method = t.driver.compute_target(prepared_latents, noise, timesteps)

        assert torch.equal(via_real_dispatch, via_driver_method), (
            "compute_target must stay t-independent (noise - latents) on "
            "BOTH the real (base-default) dispatch and the driver method — "
            "any divergence here would mean the driver grew i2v-specific "
            "compute_target logic that the trainer never wires, the same "
            "bug class as the add_noise gap this file otherwise pins."
        )

    def test_i2v_engaged_still_matches_despite_no_wiring(self) -> None:
        """Even with i2v conditioning engaged, compute_target must match —
        proving the (currently nonexistent) special-casing gap is moot."""
        t = _i2v_trainer(prob=1.0)
        latents = torch.randn(2, 4, 3, 8, 8)
        t._attach_conditioning({}, latents)
        prepared_latents = t.prepare_latents_for_training(latents)
        assert t.driver._i2v_conditioning_engaged() is True
        noise = torch.randn_like(prepared_latents)
        timesteps = torch.tensor([700.0, 700.0])

        via_real_dispatch = t.compute_target(prepared_latents, noise, timesteps)
        via_driver_method = t.driver.compute_target(prepared_latents, noise, timesteps)
        assert torch.equal(via_real_dispatch, via_driver_method)
