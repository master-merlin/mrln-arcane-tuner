"""Lightweight histogram of sampled timesteps over training.

Lets us VERIFY the timestep distribution (e.g. that model_shift biases toward
high noise) without a debugger. Pure-Python accumulation; emit a compact summary.
"""

from __future__ import annotations

import torch


class SigmaTracker:
    """Accumulate sampled timesteps (in [0,1]) into fixed deciles."""

    def __init__(self, bins: int = 10) -> None:
        self.bins = int(bins)
        self.counts = [0] * self.bins
        self.total = 0
        self._sum = 0.0

    def update(self, timesteps: torch.Tensor, scale: float = 1000.0) -> None:
        """Record a batch of timesteps. ``scale`` is the [0,scale] range they live in."""
        t = (timesteps.detach().float() / float(scale)).clamp(0.0, 1.0)
        self._sum += float(t.sum().item())
        self.total += int(t.numel())
        # Right-closed bucket: bucket i covers (i/bins, (i+1)/bins].
        # Using ceil-1 so that exactly-representable boundaries (e.g. 0.9 →
        # float32 rounds to 9.0 after ×10) fall in the lower bucket, matching
        # the intuitive decile (0.9 → decile 8, not 9).
        idx = (t * self.bins).ceil().long().sub(1).clamp(0, self.bins - 1)
        binc = torch.bincount(idx.flatten(), minlength=self.bins)
        for i in range(self.bins):
            self.counts[i] += int(binc[i].item())

    def summary(self) -> dict:
        """Return {mean, n, deciles:[fractions]} (empty-safe)."""
        if self.total == 0:
            return {"mean": 0.0, "n": 0, "deciles": [0.0] * self.bins}
        return {
            "mean": round(self._sum / self.total, 4),
            "n": self.total,
            "deciles": [round(c / self.total, 4) for c in self.counts],
        }
