"""Contrastive guidance: training a guidance-distilled model toward its own guided output.

A guidance-distilled model computes at guidance 1 what its teacher computed under CFG.
Plain flow-matching MSE pulls the conditional output toward the data velocity at guidance
1, so the baked-in guidance decays (outputs flatten). Contrastive guidance trains toward a
target that carries the guidance (ostris/ai-toolkit ``do_guidance_loss``; the maths, in our
sign ``v = x0 - noise``):

    u     = f(x_t, sigma, E_blank)            no_grad, the user's LoRA on (and the adapter)
    s_eff = s                                 schedule ``constant``
          = 1 + (s - 1) * sigma               schedule ``sigma`` (weaker toward clean)
          = 1                                 for an item whose caption was dropped
    y     = u + s_eff * (v_data - u)          fp32, detached
    loss  = the family's own loss on (p, y)   p = f(x_t, sigma, E_caption), with grad

whose minimiser is ``p* = u + s * (E[v | x_t, c] - u)``: the conditional output learns CFG
at scale ``s`` by itself. An item with ``s_eff == 1`` takes ``v_data`` itself as its target,
never ``u + 1 * (v_data - u)``, so it trains bit for bit as without guidance.

The shared loop (``pipeline_train.forward_loss_step``) runs it between the family's
``compute_target`` and ``_compute_step_loss``, on the same ``noisy_input`` and timesteps as the
grad pass; a family supplies nothing new but ``noise_fraction`` (its clock -> sigma) for the
``sigma`` schedule.
"""

from __future__ import annotations

from typing import Any

import torch


def guidance_scales(
    scale: float,
    schedule: str,
    batch_size: int,
    *,
    sigma: torch.Tensor | None = None,
    dropped: Any = None,
) -> torch.Tensor:
    """``s_eff`` per item, fp32 ``[B]``: ``s`` (``constant``) or ``1 + (s - 1) * sigma``
    (``sigma``, ``sigma`` in ``[0, 1]`` per item), and exactly 1 for every item ``dropped``
    marks (the batch's caption-dropout mask)."""
    if schedule == "constant":
        s_eff = torch.full((batch_size,), float(scale), dtype=torch.float32)
    elif schedule == "sigma":
        if sigma is None:
            raise ValueError("contrastive guidance: the sigma schedule needs each item's sigma")
        sigma = sigma.detach().float().reshape(-1).cpu()
        if sigma.numel() == 1 and batch_size > 1:
            sigma = sigma.expand(batch_size)
        if sigma.numel() != batch_size:
            raise ValueError(f"contrastive guidance: {sigma.numel()} sigmas for a batch of {batch_size}")
        if bool(((sigma < 0) | (sigma > 1)).any()):
            raise ValueError(f"contrastive guidance: sigma outside [0, 1]: {sigma.tolist()}")
        s_eff = 1.0 + (float(scale) - 1.0) * sigma
    else:
        raise ValueError(f"contrastive guidance: unknown schedule {schedule!r}")
    if dropped is not None:
        mask = torch.as_tensor(list(dropped), dtype=torch.bool).reshape(-1)
        if mask.numel() != batch_size:
            raise ValueError(f"contrastive guidance: a dropout mask of {mask.numel()} for a batch of {batch_size}")
        s_eff = torch.where(mask, torch.ones_like(s_eff), s_eff)
    return s_eff


def guided_target(blank: torch.Tensor, data: torch.Tensor, s_eff: torch.Tensor) -> torch.Tensor:
    """``y = u + s_eff * (v_data - u)`` in fp32 and detached; an item with ``s_eff == 1`` gets
    ``v_data`` itself (no cancellation error)."""
    if blank.shape != data.shape:
        raise ValueError(f"contrastive guidance: blank prediction {tuple(blank.shape)} vs target {tuple(data.shape)}")
    u = blank.detach().float()
    v = data.detach().float()
    shape = (-1,) + (1,) * (v.dim() - 1)
    s = s_eff.to(device=v.device, dtype=torch.float32).reshape(shape)
    guided = u + s * (v - u)
    plain = (s_eff.to(v.device) == 1.0).reshape(shape)
    return torch.where(plain, v, guided)
