"""Regression: Lumina2 must REVERSE + NEGATE around the transformer call.

Root cause class this guards against (see ``driver.py`` module docstring
§3 — the family's #1 silent-LoRA-killer risk): ``pipeline_lumina2.py``
lines 720-724 and 758 establish that the Lumina2 TRANSFORMER's OWN
``timestep`` input uses the OPPOSITE convention (0=noise, 1=image) from
this project's shared flow-match convention (``TimestepSampler``/
``NoiseInterpolation``: raw ``[0,1000]``, high=noisy, matching the
scheduler's own ``sigma*1000``), and the raw model output must be NEGATED
before it matches the standard ``noise - latents`` velocity target. Get
either half wrong and the LoRA trains against reversed/mis-signed targets
while every loss curve still looks plausible — training silently
converges to a pure-noise or inverted-denoising LoRA (the exact failure
mode this project's memory of the WAN raw-timestep bug and the flow-match
timestep-scale gotcha both warn about).

Two independent guards:
1. ``Lumina2Driver`` must NOT override ``add_noise``/``sample_timesteps``/
   ``compute_target`` — the reversal is entirely internal to
   ``forward_pass``. This matters structurally: this project's auto-
   delegation mechanism (``_driver_hook_override`` /
   ``driver_meaningfully_overrides``, ``pipeline_base.py``) only calls a
   driver's own ``add_noise`` when it MEANINGFULLY differs from
   ``IModelDriver``'s base implementation — and that base implementation
   does NOT itself divide by the ``[0,1000]`` scale (it assumes an
   already-normalized ``t``). If ``Lumina2Driver`` ever grew its OWN
   same-shaped ``add_noise`` override, it would silently replace the
   correctly-scaled shared ``NoiseInterpolation.add_noise`` the pipeline
   currently falls back to — a real historical bug class this project's
   ``test_wan_timestep_scale.py`` documents for exactly this reason. This
   test pins that lumina2 stays on the safe (non-overridden) path.
2. A recording fake transformer pins the EXACT reversal formula
   (``timestep_seen == 1 - t/1000``) and the output negation
   (``forward_pass output == -raw_model_output``) at several points across
   the ``[0, 1000]`` range, including both endpoints.
"""

from __future__ import annotations

import torch

from app.engine.core.hook_dispatch import driver_meaningfully_overrides
from app.engine.models.families.lumina2.driver import Lumina2Driver


def _make_driver() -> Lumina2Driver:
    definition = type(
        "FakeDefinition", (), {"architecture_params": {}, "family": "lumina2"},
    )()
    return Lumina2Driver(definition, torch.device("cpu"))


class TestNoStrayOverrides:
    def test_add_noise_not_overridden(self):
        """The reversal lives ONLY in forward_pass — add_noise must stay on
        the shared NoiseInterpolation default (see module docstring)."""
        assert driver_meaningfully_overrides(Lumina2Driver, "add_noise") is False

    def test_sample_timesteps_not_overridden(self):
        assert driver_meaningfully_overrides(Lumina2Driver, "sample_timesteps") is False

    def test_compute_target_not_overridden(self):
        """compute_target stays the standard noise-latents default — the
        negation that makes forward_pass's output match it lives entirely
        inside forward_pass (see driver.py module docstring §3)."""
        assert driver_meaningfully_overrides(Lumina2Driver, "compute_target") is False


class _RecordingLumina2(torch.nn.Module):
    """Fake Lumina2 transformer that records every ``timestep`` it is
    called with and returns a KNOWN, non-trivial (non-zero) prediction so
    the negation is observable."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_timesteps: list[torch.Tensor] = []

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        return_dict=False,
    ):
        self.seen_timesteps.append(timestep.detach().clone().float())
        # A deterministic, non-zero, non-symmetric "prediction" so negation
        # is trivially observable (all-ones would make -x == x under abs()
        # comparisons hide a sign bug that only flips even elements).
        pred = torch.full_like(hidden_states, 0.75)
        return (pred,)


class TestForwardPassReversalAndNegation:
    def _run(self, t_raw: float) -> tuple[torch.Tensor, torch.Tensor]:
        fake = _RecordingLumina2()
        drv = _make_driver()
        drv.transformer = fake

        noisy = torch.zeros(1, 2, 2, 2)
        emb = torch.zeros(1, 3, 4)
        mask = torch.ones(1, 3, dtype=torch.long)

        pred = drv.forward_pass(
            noisy_input=noisy,
            timesteps=torch.tensor([t_raw]),
            text_embeddings=(emb, mask),
            batch={},
        )
        return fake.seen_timesteps[-1], pred

    def test_reversal_formula_pinned_at_several_points(self):
        """timestep_seen == 1 - t_raw/1000 at t=0, 250, 500, 750, 1000."""
        for t_raw, expected in (
            (0.0, 1.0), (250.0, 0.75), (500.0, 0.5), (750.0, 0.25), (1000.0, 0.0),
        ):
            seen, _ = self._run(t_raw)
            assert torch.allclose(seen, torch.tensor([expected]), atol=1e-6), (
                f"t={t_raw}: expected transformer timestep {expected}, got {seen.item()}"
            )

    def test_output_is_negated_raw_prediction(self):
        """forward_pass's return value must be exactly -raw_model_output —
        the fake always predicts a constant 0.75 tensor, so the returned
        prediction must be a constant -0.75 tensor."""
        _, pred = self._run(500.0)
        assert torch.allclose(pred, torch.full_like(pred, -0.75))

    def test_not_reversed_would_fail_this_pin(self):
        """Sanity check on the test itself: an UN-reversed (bug) formula
        (feeding t_raw/1000 directly) would NOT satisfy the reversal pin at
        a non-symmetric point — proves the assertion actually discriminates
        the two conventions rather than passing vacuously."""
        seen, _ = self._run(250.0)
        wrong_unreversed = torch.tensor([250.0 / 1000.0])  # 0.25, NOT reversed
        assert not torch.allclose(seen, wrong_unreversed), (
            "the pinned reversal value must differ from the un-reversed "
            "(buggy) formula at a non-symmetric timestep"
        )
