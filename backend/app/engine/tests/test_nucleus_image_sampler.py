"""Tests for NucleusImageSampler CFG combine semantics.

The load-bearing pin here is the RENORM GROUPING: the real
``NucleusMoEImagePipeline`` performs its whole denoising loop — including
the CFG combine + renormalize at ``pipeline_nucleusmoe_image.py`` lines
594-597 — in PACKED space (``prepare_latents`` packs at line 356 via
``_pack_latents``; the only unpack is AFTER the loop at line 627), so its
``torch.norm(noise_pred, dim=-1)`` is a per-spatial-token norm over the
``C*p*p`` packed channel values. Our sampler operates on UNPACKED
``[B, C, H, W]`` driver outputs, where a naive ``dim=-1`` norm is over
WIDTH — the wrong grouping (lumina2's convention, whose pipeline does CFG
unpacked; copying it here was a real reviewed-and-fixed bug). These tests
pin the pipeline-equivalent grouping by explicitly replicating the
pipeline's own ``_pack_latents`` math as the reference computation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _pack_latents(latents: Tensor, patch_size: int) -> Tensor:
    """Verbatim replication of ``NucleusMoEImagePipeline._pack_latents``
    (pipeline_nucleusmoe_image.py lines 303-311): ``[B, C, H, W] ->
    [B, (H/p)*(W/p), C*p*p]``."""
    B, C, H, W = latents.shape
    p = patch_size
    x = latents.view(B, C, H // p, p, W // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // p) * (W // p), C * p * p)


def _unpack_latents(packed: Tensor, C: int, H: int, W: int, patch_size: int) -> Tensor:
    """Inverse of ``_pack_latents`` (mirrors the pipeline's
    ``_unpack_latents`` lines 314-328, minus the frame dim)."""
    B = packed.shape[0]
    p = patch_size
    x = packed.view(B, H // p, W // p, C, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, C, H, W)


def _pipeline_reference_cfg(
    pos: Tensor, neg: Tensor, guidance_scale: float, patch_size: int,
) -> Tensor:
    """The ground-truth computation: run the EXACT pipeline formula
    (pipeline_nucleusmoe_image.py lines 594-597) in PACKED space using the
    pipeline's own pack math, then unpack — byte-for-byte what the real
    pipeline's loop does per step (modulo the shared final negation, which
    distributes linearly and is applied inside driver.forward_pass in our
    stack)."""
    B, C, H, W = pos.shape
    pos_p = _pack_latents(pos, patch_size)
    neg_p = _pack_latents(neg, patch_size)

    comb = neg_p + guidance_scale * (pos_p - neg_p)
    cond_norm = torch.norm(pos_p, dim=-1, keepdim=True)
    noise_norm = torch.norm(comb, dim=-1, keepdim=True)
    out_p = comb * (cond_norm / noise_norm)

    return _unpack_latents(out_p, C, H, W, patch_size)


def test_cfg_combine_matches_pipeline_packed_space_computation():
    """_combine_cfg on unpacked tensors must equal pack -> pipeline formula
    (dim=-1 over the C*p*p token vector) -> unpack, on generic random
    inputs."""
    from app.engine.models.families.nucleus_image.sampler import _combine_cfg

    torch.manual_seed(7)
    B, C, H, W = 2, 4, 8, 12  # non-square W != H to catch axis mixups
    p = 2
    pos = torch.randn(B, C, H, W)
    neg = torch.randn(B, C, H, W)
    gs = 4.0

    out = _combine_cfg(pos, neg, gs, patch_size=p)
    expected = _pipeline_reference_cfg(pos, neg, gs, patch_size=p)

    assert torch.allclose(out, expected, atol=1e-6), (
        "unpacked-space _combine_cfg must be numerically identical to the "
        "pipeline's packed-space combine+renorm"
    )
    # Sanity: renormalization actually changed something vs the raw combine.
    assert not torch.allclose(out, neg + gs * (pos - neg))


def test_cfg_renorm_grouping_is_per_packed_token_not_width():
    """THE pin for the reviewed bug: construct inputs where the per-packed-
    token grouping and a naive width-wise (lumina2-style ``dim=-1`` on the
    unpacked tensor) grouping give DIFFERENT results, and assert we produce
    the pipeline-equivalent one and NOT the width-norm one.

    Construction: make the conditional prediction's magnitude vary sharply
    along HEIGHT within each 2x2 patch column so that width-rows have very
    different norms from packed 2x2-token groups.
    """
    from app.engine.models.families.nucleus_image.sampler import _combine_cfg

    torch.manual_seed(11)
    B, C, H, W = 1, 4, 4, 4
    p = 2
    pos = torch.randn(B, C, H, W)
    # Amplify alternating rows: rows 0/2 x100 — a width-wise norm groups an
    # entire row together, while the packed-token norm groups each (C, 2x2)
    # block, mixing one amplified and one small row per token.
    pos[:, :, 0::2, :] *= 100.0
    neg = torch.randn(B, C, H, W)
    gs = 4.0

    out = _combine_cfg(pos, neg, gs, patch_size=p)

    expected_packed = _pipeline_reference_cfg(pos, neg, gs, patch_size=p)

    # The WRONG (width-norm / lumina2-style) computation on the unpacked
    # tensor — what the pre-review sampler did.
    comb = neg + gs * (pos - neg)
    wrong = comb * (
        torch.norm(pos, dim=-1, keepdim=True)
        / torch.norm(comb, dim=-1, keepdim=True)
    )

    # First prove the construction actually discriminates the two groupings.
    assert not torch.allclose(expected_packed, wrong, atol=1e-4), (
        "test construction failed: width-norm and per-token-norm must "
        "disagree for this pin to mean anything"
    )
    assert torch.allclose(out, expected_packed, atol=1e-6), (
        "_combine_cfg must renormalize per PACKED TOKEN (C*p*p channel "
        "group), matching the pipeline's packed-space dim=-1 norm"
    )
    assert not torch.allclose(out, wrong, atol=1e-4), (
        "_combine_cfg must NOT use the width-wise dim=-1 grouping "
        "(lumina2's unpacked-CFG convention does not apply to Nucleus)"
    )


def test_cfg_combine_formula_basic():
    """Uniform-tensor sanity check of the affine combine itself (mirrors
    the lumina2 sampler test style): velocity = neg + g*(pos - neg), then
    renormalized to the conditional norm — with uniform values every
    grouping agrees, so this isolates the formula from the grouping."""
    from app.engine.models.families.nucleus_image.sampler import _combine_cfg

    pos = torch.ones(1, 4, 4, 4)
    neg = -torch.ones(1, 4, 4, 4)
    out = _combine_cfg(pos, neg, 2.0, patch_size=2)

    raw = neg + 2.0 * (pos - neg)  # == 3.0 everywhere
    # Renorm scales it back to the conditional's norm: uniform 1.0.
    assert torch.allclose(out, torch.ones_like(pos), atol=1e-6)
    assert not torch.allclose(out, raw)
