"""Flow-match sigma schedule with the model's inference-time shift.

Linear sigmas in [1, 0] are reshaped by the flow-match shift ``s``:
    sigma' = s * sigma / (1 + (s - 1) * sigma)
``s > 1`` pushes mass toward high noise (structure), matching the LTX/WAN
inference schedulers. ``s == 1`` is the identity (linear), so callers that pass
no shift are byte-identical to the old behaviour.
"""

from __future__ import annotations

import torch


def shifted_sigmas(num_steps: int, shift: float = 1.0, device=None) -> torch.Tensor:
    """Descending sigma schedule of length ``num_steps + 1`` from 1.0 → 0.0.

    Applies the flow-match shift ``s`` (s=1 → linear). The endpoints stay exactly
    1.0 and 0.0 (the transform fixes 0 and 1), so it remains a valid 1→0 schedule.
    """
    sigmas = torch.linspace(1.0, 0.0, int(num_steps) + 1, device=device)
    s = float(shift)
    if s == 1.0:
        return sigmas
    return s * sigmas / (1.0 + (s - 1.0) * sigmas)
