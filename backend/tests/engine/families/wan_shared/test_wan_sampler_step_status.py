"""FAM-2: WAN samplers emit per-step ``Sampling {i}/{N}`` status (audit P1b).

Every image family streams ``Sampling {i}/{N}`` (1-based, once per denoise
step) through ``self._log_writer.status(...)`` so the UI shows live sampling
progress via job_log.jsonl → LogTailer. The WAN video Euler loop emitted
nothing. The emit lives in the SHARED :meth:`WanVideoSamplerBase.euler_integrate`
seam so BOTH wan21 (base ``denoise``) and wan22 (dual-expert override) emit
without duplicating the logic.

The string format must stay byte-identical to the image families' (e.g.
``krea2/sampler.py``): ``Sampling {i}/{N}``.
"""

from __future__ import annotations

import torch

from app.engine.models.families.wan21.sampler import Wan21Sampler
from app.engine.models.families.wan22.sampler import Wan22Sampler


class _ZeroVelocityFakeWan(torch.nn.Module):
    """Minimal WAN-forward-compatible fake: velocity = linear proj of input."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(channels, channels)

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image=None,
        return_dict=False,
    ):
        h = self.proj(hidden_states.movedim(1, -1)).movedim(-1, 1)
        return (h,)


class _StatusRecorder:
    """JobLogWriter stand-in recording every status(...) call."""

    def __init__(self) -> None:
        self.statuses: list[str] = []

    def status(self, message: str) -> None:
        self.statuses.append(message)


class _DriverStub:
    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model

    def get_primary_model(self) -> torch.nn.Module:
        return self._model


class _Pipeline:
    def __init__(self, model: torch.nn.Module) -> None:
        self.config = {"resolutions": [480], "sample_num_frames": 5}
        self.device = torch.device("cpu")
        self.autocast_dtype = torch.bfloat16
        self.driver = _DriverStub(model)


def _noise() -> torch.Tensor:
    return torch.randn(1, 16, 2, 4, 4, dtype=torch.float32)


def test_wan21_denoise_emits_per_step_sampling_status():
    """The base (wan21) denoise path emits Sampling 1/N … N/N, 1-based."""
    sampler = Wan21Sampler(_Pipeline(_ZeroVelocityFakeWan()))
    lw = _StatusRecorder()
    sampler._log_writer = lw

    sampler.denoise(
        _noise(), torch.zeros(1, 8, 16), num_steps=3, guidance_scale=1.0, seed=0
    )

    assert lw.statuses == ["Sampling 1/3", "Sampling 2/3", "Sampling 3/3"]


def test_wan21_denoise_without_log_writer_is_safe():
    """No _log_writer attached (e.g. precision-contract tests) → no crash."""
    sampler = Wan21Sampler(_Pipeline(_ZeroVelocityFakeWan()))

    out = sampler.denoise(
        _noise(), torch.zeros(1, 8, 16), num_steps=2, guidance_scale=1.0, seed=0
    )

    assert out.shape == (1, 16, 2, 4, 4)


def test_wan22_dual_expert_denoise_emits_per_step_sampling_status():
    """wan22 OVERRIDES denoise (boundary expert switch) — a separate code path
    that must emit through the same shared seam, once per step, no duplicates.
    """
    high = _ZeroVelocityFakeWan()
    low = _ZeroVelocityFakeWan()

    class _Wan22Driver:
        boundary = 0.5
        transformer_high = high
        transformer_low = low

        def get_primary_model(self):
            return high

    class _Wan22Pipeline:
        def __init__(self):
            self.config = {"resolutions": [480], "sample_num_frames": 5}
            self.device = torch.device("cpu")
            self.autocast_dtype = torch.bfloat16
            self.driver = _Wan22Driver()

    sampler = Wan22Sampler(_Wan22Pipeline())
    lw = _StatusRecorder()
    sampler._log_writer = lw

    sampler.denoise(
        _noise(), torch.zeros(1, 8, 16), num_steps=4, guidance_scale=1.0, seed=0
    )

    assert lw.statuses == [
        "Sampling 1/4",
        "Sampling 2/4",
        "Sampling 3/4",
        "Sampling 4/4",
    ]
