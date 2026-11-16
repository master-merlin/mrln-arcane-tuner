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
_CHUNK = 1024


def noise_fraction_of(family, timesteps: torch.Tensor, like: torch.Tensor | None = None) -> torch.Tensor:
    """The noise fraction (0 clean, 1 pure noise) of each of ``timesteps`` on the family's clock.

    Measured through ``add_noise`` with latents of 0 and noise of 1, shaped like ``like``'s
    first item: the fraction is the item's largest value, since a frame the family pins to
    clean stays at 0. A family whose forward adds the noise itself, so that its ``add_noise``
    never runs (``validation_timestep_domain`` None), answers through its ``noise_fraction``.
    """
    flat = timesteps.reshape(-1)
    if getattr(family, "validation_timestep_domain", None) is None:
        try:
            return family.noise_fraction(flat).float().reshape(-1)
        except (AttributeError, NotImplementedError):
            pass
    reference = torch.zeros(1, 1) if like is None else like[:1]
    out = []
    for chunk in flat.split(_CHUNK):
        latents = torch.zeros((chunk.shape[0], *reference.shape[1:]), dtype=reference.dtype, device=reference.device)
        noisy = family.add_noise(latents, torch.ones_like(latents), chunk.to(reference.device))
        out.append(noisy.reshape(chunk.shape[0], -1).amax(dim=1).float())
    return torch.cat(out) if out else torch.zeros(0)


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

    def fraction(timestep: torch.Tensor) -> float:
        return float(noise_fraction_of(family, timestep, like))

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
