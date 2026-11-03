"""A Hessian diagonal from ordinary gradient passes, for SophiaH.

Hutchinson's estimate is ``z * (H z)``. The usual ``H z`` differentiates the
gradients, which the trainer can't do: PyTorch has no second derivative for the
efficient attention kernel its models use, and a Flux-sized model ran out of memory
at a 67 GiB cap keeping the graph (a plain step peaks near 25 GiB).

A central finite difference needs neither:

    H z = (g(theta + eps z) - g(theta - eps z)) / (2 eps)

Measured on a small Flux with LoRA against the exact product (its MATH attention
kernel does have the derivative): cosine 0.9995 at ``alpha`` = 0.1, 0.98 at 0.01 and
0.39 at 0.001, where gradient rounding swamps the difference. On the Flux-sized
model the peak grew by 0.2 GiB.
"""
import random
from collections.abc import Callable, Sequence

import numpy as np
import torch

DISTRIBUTIONS = ("gaussian", "rademacher")
_ALPHA = 0.1


def _capture_random_state() -> tuple:
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        random.getstate(),
        np.random.get_state(),
    )


def _restore_random_state(state: tuple) -> None:
    torch_state, cuda_state, python_state, numpy_state = state
    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    random.setstate(python_state)
    np.random.set_state(numpy_state)


def _sample(p: torch.Tensor, distribution: str) -> torch.Tensor:
    if distribution == "rademacher":
        return torch.randint_like(p, 0, 2) * 2.0 - 1.0
    return torch.randn_like(p)


def _norm(tensors: Sequence[torch.Tensor]) -> float:
    # A sum of per-tensor floats: a dual-expert model's tensors can be on two devices.
    return sum(t.detach().pow(2).sum().item() for t in tensors) ** 0.5


def estimate_hessian_diagonal(
    params: Sequence[torch.Tensor],
    grads: Callable[[], Sequence[torch.Tensor | None]],
    *,
    num_samples: int = 1,
    distribution: str = "gaussian",
) -> list[torch.Tensor] | None:
    """Estimate the Hessian's diagonal at ``params``, one tensor per parameter.

    ``grads`` runs one forward and backward pass at the current values and returns
    one gradient per parameter, ``None`` for one the pass did not reach. Both passes
    of a pair start from the same random state, so they see the same noise, and the
    random streams end where they began, so training continues as if this never ran.

    Returns ``None`` when a pass gives a non-finite result, for the caller to keep
    its previous estimate. The parameters are restored exactly in every case.
    """
    if distribution not in DISTRIBUTIONS:
        raise ValueError(
            f"unknown Hessian distribution {distribution!r}; use one of {DISTRIBUTIONS}"
        )

    outer_state = _capture_random_state()
    originals = [p.detach().clone() for p in params]
    theta_norm = _norm(originals)
    total = [torch.zeros_like(p) for p in originals]

    def set_values(zs: Sequence[torch.Tensor], step: float) -> None:
        with torch.no_grad():
            for p, original, z in zip(params, originals, zs):
                p.copy_(original).add_(z, alpha=step)

    def collect(pair_state: tuple) -> list[torch.Tensor | None]:
        _restore_random_state(pair_state)
        return list(grads())

    try:
        for _ in range(num_samples):
            zs = [_sample(p, distribution) for p in originals]
            eps = _ALPHA * (1.0 + theta_norm) / _norm(zs)
            pair_state = _capture_random_state()

            set_values(zs, eps)
            ahead = collect(pair_state)
            set_values(zs, -eps)
            behind = collect(pair_state)

            for running, z, a, b in zip(total, zs, ahead, behind):
                if a is None or b is None:
                    continue
                running.add_(z * (a - b).to(z.dtype), alpha=1.0 / (2.0 * eps * num_samples))
    finally:
        with torch.no_grad():
            for p, original in zip(params, originals):
                p.copy_(original)
        _restore_random_state(outer_state)

    if not all(torch.isfinite(t).all() for t in total):
        return None
    return total
