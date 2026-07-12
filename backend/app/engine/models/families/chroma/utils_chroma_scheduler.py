"""Shared Chroma scheduler helpers — sampler + tests use the SAME code path.

Chroma1-Base and Chroma1-HD ship materially different
``scheduler/scheduler_config.json`` files:

- HD: static/dynamic-shift fields fully specified, ``use_dynamic_shifting:
  false`` + ``shift: 3.0`` (a static logistic shift, NOT resolution-
  dependent), ``use_beta_sigmas: false``.
- Base: only ``num_train_timesteps`` + ``use_beta_sigmas: true`` are set;
  every other field falls back to ``FlowMatchEulerDiscreteScheduler``'s own
  class defaults (notably ``shift: 1.0`` — i.e. NO static shift at all —
  and ``use_dynamic_shifting: false``), so Base samples via a Karras-style
  beta-resampled sigma schedule instead of any logistic shift.

Rather than re-deriving this branching by hand (dynamic-shift math,
static-shift math, beta/karras/exponential sigma remap), we build a REAL
``diffusers.FlowMatchEulerDiscreteScheduler`` from the definition's
``architecture_params`` and let the library's own ``set_timesteps`` handle
it — see ``venv/Lib/site-packages/diffusers/schedulers/
scheduling_flow_match_euler_discrete.py`` lines 309, 348-362.
"""

from __future__ import annotations

from typing import Any


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Resolution-dependent mu — copied verbatim from
    ``pipeline_chroma.py``'s module-level ``calculate_shift`` (itself
    ``# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift``).
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def build_scheduler(arch: dict[str, Any]):
    """Build a ``FlowMatchEulerDiscreteScheduler`` from architecture_params.

    Every kwarg defaults to the diffusers scheduler class's own default
    (not a Chroma-specific assumption), so an definition that omits a key
    (like Chroma1-Base's sparse scheduler config) gets the SAME effective
    scheduler diffusers itself would construct from that checkpoint.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=int(arch.get("scheduler.num_train_timesteps", 1000)),
        shift=float(arch.get("scheduler.shift", 1.0)),
        use_dynamic_shifting=bool(arch.get("scheduler.use_dynamic_shifting", False)),
        base_shift=float(arch.get("scheduler.base_shift", 0.5)),
        max_shift=float(arch.get("scheduler.max_shift", 1.15)),
        base_image_seq_len=int(arch.get("scheduler.base_image_seq_len", 256)),
        max_image_seq_len=int(arch.get("scheduler.max_image_seq_len", 4096)),
        invert_sigmas=bool(arch.get("scheduler.invert_sigmas", False)),
        shift_terminal=arch.get("scheduler.shift_terminal", None),
        use_karras_sigmas=bool(arch.get("scheduler.use_karras_sigmas", False)),
        use_exponential_sigmas=bool(arch.get("scheduler.use_exponential_sigmas", False)),
        use_beta_sigmas=bool(arch.get("scheduler.use_beta_sigmas", False)),
        time_shift_type=str(arch.get("scheduler.time_shift_type", "exponential")),
        stochastic_sampling=bool(arch.get("scheduler.stochastic_sampling", False)),
    )
