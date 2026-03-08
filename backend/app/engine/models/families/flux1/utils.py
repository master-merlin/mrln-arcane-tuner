"""FLUX.1 latent packing utilities.

Converts between spatial latents [B, C, H, W] and the packed sequence
format [B, L, D] expected by diffusers ``FluxTransformer2DModel``.

The transformer expects 2×2 spatial patches packed into the channel
dimension: [B, 16, H, W] → [B, (H/2)*(W/2), 64].  Position IDs use
3 columns (t, h, w) over the *patched* grid.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor


def pack_latents(x: Tensor) -> tuple[Tensor, Tensor]:
    """Pack spatial latents via 2×2 patching for FluxTransformer2DModel.

    Groups every 2×2 spatial block into the channel dimension so the
    transformer receives tokens of width ``C * 4`` (= 64 for 16-ch VAE).

    Args:
        x: Image latents ``[B, C, H, W]`` (H, W must be even).

    Returns:
        Tuple of:
        - packed tokens ``[B, (H/2)*(W/2), C*4]``
        - position IDs ``[L, 3]`` with columns ``(t, h, w)``
          (2-D, no batch dim — matches diffusers expectation)
    """
    B, C, H, W = x.shape

    # 2×2 patch packing: [B, C, H, W] → [B, C, H/2, 2, W/2, 2]
    #                   → [B, H/2, W/2, C*2*2] → [B, L, C*4]
    packed = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    # Position IDs over the patched grid (H/2 × W/2)
    h_patched = H // 2
    w_patched = W // 2
    h_ids = torch.arange(h_patched, device=x.device)
    w_ids = torch.arange(w_patched, device=x.device)
    grid_h, grid_w = torch.meshgrid(h_ids, w_ids, indexing="ij")
    img_ids = torch.stack(
        [
            torch.zeros_like(grid_h),  # t = 0
            grid_h,
            grid_w,
        ],
        dim=-1,
    ).reshape(-1, 3)  # [(H/2)*(W/2), 3]

    return packed, img_ids


def unpack_latents(
    packed: Tensor, height: int, width: int
) -> Tensor:
    """Unpack patched sequence format back to spatial latents.

    Args:
        packed: Packed tokens ``[B, (H/2)*(W/2), C*4]``.
        height: Spatial height of latents (H, before patching).
        width: Spatial width of latents (W, before patching).

    Returns:
        Spatial latents ``[B, C, H, W]``.
    """
    return rearrange(
        packed,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=height // 2,
        w=width // 2,
        ph=2,
        pw=2,
    )
