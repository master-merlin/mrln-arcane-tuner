# Vendored from boogu-project/Boogu-Image @ ac9e40c1350fd60c502137a678ad1001d51e2ae7 (2026-07-10)
# Source: boogu/cache_functions/force_scheduler.py
# vendored for boogu_image family — local diffusers 0.39.0

# Copyright (C) 2026 Boogu Team.
# This repository is a fork by Boogu Team; modifications have been made.
#
# Original work: TaylorSeer (Shenyi-Z), taylorseer_flux/cache_functions/force_scheduler.py
# Source: https://github.com/Shenyi-Z/TaylorSeer/blob/main/TaylorSeers-xDiT/taylorseer_flux/cache_functions/force_scheduler.py

import torch


def force_scheduler(cache_dic, current):
    """
    Update `cache_dic['cal_threshold']` for the current denoising step.

    Args:
        cache_dic: Mutable cache state dict. Expected keys include
            `fresh_ratio` and `fresh_threshold`.
        current: Per-step state dict. Expected keys include
            `step` and `num_steps`.
    """
    if cache_dic["fresh_ratio"] == 0:
        # FORA
        linear_step_weight = 0.0
    else:
        # TokenCache
        linear_step_weight = 0.0
    # Scale threshold by step position when linear weighting is enabled.
    step_factor = torch.tensor(
        1
        - linear_step_weight
        + 2 * linear_step_weight * current["step"] / current["num_steps"]
    )
    threshold = torch.round(cache_dic["fresh_threshold"] / step_factor)

    # no force constrain for sensitive steps, cause the performance is good enough.
    # you may have a try.

    cache_dic["cal_threshold"] = threshold
