"""Which timestep means which noise level, measured through the family's own add_noise.

Clocks differ across families: some run reversed, some take ``[0, 1]`` and others
``[0, 1000]``. A hand-kept table would drift from the code that really adds the noise,
so the fraction is measured: ``add_noise`` with latents of 0 and noise of 1 returns it.
The legal range comes from the family's declared ``validation_timestep_domain``, never
from ``sample_timesteps``, whose draws depend on the configured sampler.
"""
import torch

_END_TOLERANCE = 0.02
_CENTRE_TOLERANCE = 0.01
_GRID_POINTS = 129
_BISECTIONS = 48
_MONOTONIC_SLACK = 1e-5


def band_centres(count: int) -> torch.Tensor:
    """The centres of ``count`` bands of equal width in noise fraction, from 0 to 1."""
    return (torch.arange(count, dtype=torch.float64) + 0.5) / count


def timesteps_for_bands(family, count: int, like: torch.Tensor | None = None) -> torch.Tensor:
    """The family's timestep for each band centre, in order of rising noise.

    ``like`` gives the shape, device and dtype of the family's prepared latents; a
    family's ``add_noise`` may depend on them. A frame the family pins to clean stays
    at 0, so the fraction is the largest value in the result.
    """
    name = getattr(family, "family_name", None) or type(family).__name__
    domain = getattr(family, "validation_timestep_domain", None)
    if domain is None:
        raise ValueError(f"Validation is unavailable for the {name} family in this version.")

    reference = torch.zeros(1, 1) if like is None else like[:1]
    latents = torch.zeros_like(reference)
    noise = torch.ones_like(reference)
    device = reference.device

    def fraction(timestep: torch.Tensor) -> float:
        return float(family.add_noise(latents, noise, timestep.reshape(1).to(device)).max())

    low, high = float(domain[0]), float(domain[1])
    grid = torch.linspace(low, high, _GRID_POINTS, dtype=torch.float32)
    measured = [fraction(t) for t in grid]

    rising = measured[-1] >= measured[0]
    ordered = measured if rising else measured[::-1]
    steps_ok = all(b >= a - _MONOTONIC_SLACK for a, b in zip(ordered, ordered[1:]))
    if not steps_ok or ordered[0] > _END_TOLERANCE or ordered[-1] < 1.0 - _END_TOLERANCE:
        raise ValueError(
            f"Validation is unavailable for the {name} family: add_noise over {domain} "
            f"goes from {measured[0]:.3f} to {measured[-1]:.3f}, not monotonically from "
            "0 to 1 in noise fraction."
        )

    found = []
    for centre in band_centres(count).tolist():
        a, b = (low, high) if rising else (high, low)
        for _ in range(_BISECTIONS):
            middle = (a + b) / 2.0
            if fraction(torch.tensor(middle, dtype=torch.float32)) < centre:
                a = middle
            else:
                b = middle
        timestep = (a + b) / 2.0
        reached = fraction(torch.tensor(timestep, dtype=torch.float32))
        if abs(reached - centre) > _CENTRE_TOLERANCE:
            raise ValueError(
                f"Validation is unavailable for the {name} family: the nearest timestep "
                f"to noise fraction {centre:.3f} reaches {reached:.3f}."
            )
        found.append(timestep)
    return torch.tensor(found, dtype=torch.float32)
