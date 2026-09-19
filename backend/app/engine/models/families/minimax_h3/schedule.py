"""minimax_h3 — the dual-shift sigma schedule (plan row 1.1; ECOSYSTEM §6 `H3SigmaSchedule`).

# Method re-implemented from ostris/ai-toolkit@561a0236 (MIT) `src/packing.py`
# (`shift_sigma` / `remap_sigma`) and diffusers 0.40.0
# `scheduling_minimax_h3.py:157` (Apache-2.0); no line copied.

H3 runs TWO coupled rectified-flow schedules per step — one per modality
(`shift=12.0` video, `shift=3.0` audio) — driven by ONE uniform draw ``u``:

    σ = s·u / (1 + (s−1)·u)

applied twice on the same ``u``. Two independent draws would train the audio
and video streams at unrelated noise levels; ``test_one_u_drives_both_curves``
pins that ``σ_a`` is a deterministic function of ``σ_v`` (un-shift by the
video shift, re-shift by the audio shift — the reference identity).

The shifts come from :class:`H3EffectiveSettings` (resolved by
``settings.py`` from the run config or the definition's ``video.sigma_shift``
/ ``audio.sigma_shift``), never a literal — the PR0 Task-7 defect was a
literal that no YAML edit could reach.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .settings import H3EffectiveSettings


def sigma_to_t(sigma: torch.Tensor) -> torch.Tensor:
    """H3's clock: ``t = 1 − σ`` in ``[0, 1]``, ``t = 1`` clean, UNSCALED (no
    ×1000 — the transformer's ``time_proj`` consumes ``[0, 1]`` directly;
    diffusers ``scheduling_minimax_h3.py:170-171``). The ONE σ→t site in the
    family; ``test_trainer_and_sampler_share_one_conversion_module`` pins it."""
    return 1.0 - sigma


def t_to_sigma(t: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`sigma_to_t` (the noise weight of ``x_t``)."""
    return 1.0 - t


def shift_sigma(u: torch.Tensor, shift: float) -> torch.Tensor:
    """The exponential sigma shift ``s·u / (1 + (s−1)·u)``; fixes 0 and 1."""
    return (shift * u) / (1.0 + (shift - 1.0) * u)


def unshift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """Inverse of :func:`shift_sigma`: ``u = σ / (s − (s−1)·σ)``."""
    return sigma / (shift - (shift - 1.0) * sigma)


def remap_sigma(sigma: torch.Tensor, from_shift: float, to_shift: float) -> torch.Tensor:
    """Move a sigma from one shifted curve to another through the shared ``u``."""
    return shift_sigma(unshift_sigma(sigma, from_shift), to_shift)


@dataclass(frozen=True)
class H3SigmaSchedule:
    """The two shifts, read off the effective settings; ``draw`` maps ONE ``u``
    through both curves."""

    sigma_shift_video: float
    sigma_shift_audio: float

    @classmethod
    def from_settings(cls, settings: H3EffectiveSettings) -> H3SigmaSchedule:
        # The resolver has already refused an absent or non-positive shift.
        return cls(
            sigma_shift_video=float(settings.sigma_shift_video),
            sigma_shift_audio=float(settings.sigma_shift_audio),
        )

    def draw(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(σ_v, σ_a)`` for the SAME ``u`` in ``[0, 1]``."""
        u = torch.as_tensor(u)
        return shift_sigma(u, self.sigma_shift_video), shift_sigma(u, self.sigma_shift_audio)
