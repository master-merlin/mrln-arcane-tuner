"""The run-config keys of the timestep sampler, and their schema defaults.

Torch-free, so a process that must not load torch (a tool that watches a run) reads the same
names the sampler reads. ``timestep_sampling`` re-exports every name here.
"""
from __future__ import annotations

from typing import Any

# The run-config keys that parameterize a timestep mode. A definition may state a family default for
# each as ``timestep.<key>`` in ``architecture_params`` (or as ``<key>`` in its ``defaults:``).
TIMESTEP_PARAMETERS: tuple[str, ...] = (
    "logit_normal_mu",
    "logit_normal_sigma",
    "mode_scale",
    "flux_shift_base",
    "flux_shift_max",
    "model_shift_std",
    "timestep_uniform_prob",
    "radc_start",
    "radc_end",
    "radc_width",
    "radc_res_influence",
    "radc_uniform_prob",
    "fixed_sigmas",
)

# Older names the sampler still reads for a parameter: a config that carries one has set it.
PARAMETER_ALIASES: dict[str, tuple[str, ...]] = {
    "logit_normal_mu": ("logit_normal_mean",),
    "logit_normal_sigma": ("logit_normal_std",),
}

# The run-config schema's own defaults for the timestep keys (``BaseTrainingConfig``'s; a test
# pins the two equal): the value for a config that names none, of a definition that names none.
TIMESTEP_SCHEMA_DEFAULTS: dict[str, Any] = {
    "timestep_sampling": "logit_normal",
    "logit_normal_mu": 0.0,
    "logit_normal_sigma": 1.0,
    "mode_scale": 1.5,
    "flux_shift_base": 0.5,
    "flux_shift_max": 1.16,
    "model_shift_std": 1.0,
    "timestep_uniform_prob": 0.1,
    "radc_start": 0.8,
    "radc_end": 0.2,
    "radc_width": 0.5,
    "radc_res_influence": 0.15,
    "radc_uniform_prob": 0.0,
}

# Every run-config key that sets the timestep draw: the mode, each parameter and its older names.
TIMESTEP_KEYS: frozenset[str] = frozenset({
    "timestep_sampling", *TIMESTEP_PARAMETERS, *(name for names in PARAMETER_ALIASES.values() for name in names),
})
