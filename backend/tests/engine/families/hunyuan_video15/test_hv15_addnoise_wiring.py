"""hv15 — ``add_noise`` dead-dispatch audit (A0-4), EQUIVALENT verdict.

CONTEXT (the bug class): the real train loop calls ``add_noise`` on the
TRAINER via MRO (``pipeline_train.py:543``: ``self.add_noise(...)``), not on
the driver directly. A family that puts special-casing in a DRIVER-level
``add_noise`` override without the trainer delegating to it ships dead code
that never runs on the real path — the class of bug fixed for ``boogu_image``
(scheduler clobber) and ``kandinsky5`` (I2V frame-0 noised; see
``test_kandinsky5_addnoise_wiring.py``) and ``ltx2`` (this same audit,
``test_ltx2_addnoise_wiring.py``).

VERDICT FOR hv15: EQUIVALENT, not a bug. ``Hv15Driver`` used to carry an
``add_noise`` override (driver.py, formerly lines 482-497) with the formula
``t = timesteps/1000.0; while t.ndim < latents.ndim: t = t.unsqueeze(-1);
return t*noise + (1-t)*latents`` — algebraically IDENTICAL to
``NoiseInterpolation._linear``'s ``(1-t)*latents + t*noise`` (same terms,
commutative sum, same ``/1000`` scale, same broadcast shape since both start
from a ``[B]`` timestep and unsqueeze to the latent's ndim). Critically, that
override had NO i2v branch at all — hv15's i2v conditioning is carried
entirely through the SEPARATE cond/mask channels
(``build_i2v_cond_and_mask`` / ``build_model_input``, concatenated onto the
noised latent for the 65-channel transformer input), never by suppressing
noise on any particular frame. So there was no special-casing for the
trainer to fail to wire — the override was PURE dead code, safe to delete
outright rather than wire-and-pin.

THE FIX: ``Hv15Driver.add_noise`` was deleted. ``Hv15Trainer`` carries no
override either, so the real path now runs
``PipelineBaseMixin.add_noise`` → ``NoiseInterpolation('linear')`` directly
— exactly what the (deleted) driver method computed, with zero code
duplication. ``test_hv15_precision_contracts.py`` was updated to pin the
REAL dispatch path (a trainer instance) instead of calling the driver method
directly.

This file pins two things: (1) the deletion actually happened (no
resurrected dead code), and (2) the real dispatch path's [0,1000] lerp
matches a fresh ``NoiseInterpolation('linear')`` instance bit-for-bit, for
both T2V-shaped and I2V-shaped (multi-frame) latents — since hv15's add_noise
never branches on i2v, both shapes must match identically.
"""

from __future__ import annotations

import torch

from app.engine.models.families.hunyuan_video15.driver import Hv15Driver
from app.engine.models.families.hunyuan_video15.trainer import Hv15Trainer
from app.engine.strategies.noise_interpolation import NoiseInterpolation


def test_driver_carries_no_add_noise_override() -> None:
    """Guard against resurrecting the dead-dispatch trap: if a future edit
    re-adds a driver-level ``add_noise`` override, it MUST also wire trainer
    delegation (kandinsky5/boogu_image/ltx2 pattern) — this test forces that
    conversation to happen rather than silently shipping dead code again."""
    assert "add_noise" not in Hv15Driver.__dict__, (
        "Hv15Driver defines add_noise again — if it has real i2v/family "
        "special-casing, Hv15Trainer.add_noise must delegate to it (the "
        "real training loop calls the TRAINER's add_noise via MRO, not the "
        "driver's); if it's still just the generic lerp, it's dead code — "
        "delete it instead."
    )


def _real_dispatch_trainer() -> Hv15Trainer:
    """A trainer shell exercising the REAL ``self.add_noise(...)`` call
    ``pipeline_train.py:543`` makes. ``Hv15Trainer`` has no override, so this
    resolves through MRO to ``PipelineBaseMixin.add_noise``."""
    t = object.__new__(Hv15Trainer)
    t.noise_interpolation = NoiseInterpolation("linear")
    return t


class TestRealDispatchMatchesBaseComponent:
    def test_t2v_shaped_latents(self) -> None:
        t = _real_dispatch_trainer()
        latents = torch.randn(2, 32, 1, 4, 5)  # T2V: 1-frame lift (WAN precedent)
        noise = torch.randn_like(latents)
        timesteps = torch.tensor([250.0, 900.0])

        via_real_dispatch = t.add_noise(latents, noise, timesteps)
        via_base_component = t.noise_interpolation.add_noise(latents, noise, timesteps)

        assert torch.equal(via_real_dispatch, via_base_component), (
            "hv15 T2V add_noise real dispatch must be numerically identical "
            "to the base NoiseInterpolation path."
        )

    def test_i2v_shaped_multiframe_latents_no_frame0_special_case(self) -> None:
        """hv15 has NO frame-0-clean pin (unlike wan/ltx2/kandinsky5) — even a
        multi-frame i2v-shaped latent must noise ALL frames uniformly,
        matching the base component exactly with zero exceptions."""
        t = _real_dispatch_trainer()
        latents = torch.randn(2, 32, 3, 4, 5)  # multi-frame, i2v-shaped
        noise = torch.randn_like(latents)
        timesteps = torch.tensor([500.0, 500.0])

        via_real_dispatch = t.add_noise(latents, noise, timesteps)
        via_base_component = t.noise_interpolation.add_noise(latents, noise, timesteps)

        assert torch.equal(via_real_dispatch, via_base_component)
        # Sanity: frame 0 IS noised (no clean-conditioning-frame pin here —
        # hv15 carries i2v conditioning via separate cond/mask channels).
        assert not torch.equal(via_real_dispatch[:, :, 0], latents[:, :, 0])
