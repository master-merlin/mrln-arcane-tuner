"""Shared PRX transformer forward adapter — the normalized-timestep seam.

PRX's timestep convention differs from every other flow-match family here:
``PRXPipeline`` normalizes the raw scheduler timestep BEFORE the transformer
(``t_cont = t.float() / scheduler.config.num_train_timesteps``, see
``pipeline_prx.py``) — the model's ``time_factor=1000.0`` re-scales
internally for the sinusoidal embedding. So the flow-match "÷1000 exactly
once" rule lands HERE, in the driver-side adapter, while the scheduler side
keeps raw ``[0, 1000]`` timesteps.

Family-agnostic: parameterized by ``num_train_timesteps``; shared by the
latent ``prx`` family and the future pixel-space sibling (whose transformer
has the identical forward signature — patchify/unpatchify live INSIDE the
model, so callers always pass unpacked ``[B, C, H, W]`` tensors).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def prx_transformer_forward(
    model: nn.Module,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    """Run a PRX transformer forward with raw ``[0, num_train_timesteps]`` t.

    Args:
        model: A ``PRXTransformer2DModel`` (or PEFT-wrapped equivalent).
        noisy_latents: Unpacked input ``[B, C, H, W]`` (H/W divisible by
            the model's ``patch_size`` — patchify happens inside).
        timesteps: RAW timesteps on the ``[0, num_train_timesteps]`` scale;
            normalized to ``[0, 1]`` here — exactly once, never twice.
        encoder_hidden_states: Text conditioning ``[B, L, context_in_dim]``.
        attention_mask: Boolean text mask ``[B, L]`` (0 = padding) or None.
        num_train_timesteps: Scheduler scale (default 1000).

    Returns:
        Velocity prediction ``[B, C, H, W]``.
    """
    # Pipeline-verbatim normalization: t.float() / num_train_timesteps.
    t_cont = timesteps.float() / float(num_train_timesteps)
    t_cont = t_cont.to(device=noisy_latents.device)

    output = model(
        hidden_states=noisy_latents,
        timestep=t_cont,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        return_dict=False,
    )
    return output[0] if isinstance(output, tuple) else output
