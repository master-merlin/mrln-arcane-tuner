"""POSITIVE CONTROLS for ``test_minimax_h3_convention.py`` tests 5 and 8.

NOT production code; never imported by the engine. Two deliberate offenders
the source guards are REQUIRED to flag (CONVENTIONS "Tests" rule 11):

* a ``* 1000`` timestep rescale — the flow-match timestep-scale gotcha that
  once shipped a pure-noise LoRA (test 5 ``no_thousand_scaling``);
* a second ``1 - sigma`` conversion site outside ``schedule.py`` — training
  and sampling conventions diverging (test 8 ``share_one_conversion_module``).

Plan row 1.2 (``_harness/plans/2026-09-12-minimax-h3-pr1.md``).
"""

from __future__ import annotations

import torch


def rescaled_timestep(t: torch.Tensor) -> torch.Tensor:
    # Offender 1: the ad-hoc rescale (H3 consumes t in [0, 1] unscaled).
    return t * 1000


def second_conversion_site(sigma: torch.Tensor) -> torch.Tensor:
    # Offender 2: a private sigma -> t conversion instead of schedule.sigma_to_t.
    t = 1.0 - sigma
    return t
