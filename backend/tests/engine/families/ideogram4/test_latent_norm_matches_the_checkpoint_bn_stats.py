"""LANE-150: does the driver's LATENT_SHIFT/LATENT_SCALE agree with the checkpoint's VAE ``bn`` statistics?

diffusers 0.41.0's ``Ideogram4Pipeline`` (de)normalises with ``bn.running_mean`` and
``sqrt(bn.running_var + batch_norm_eps)`` (pipeline_ideogram4.py:731-734, eps 1e-4 from the VAE config);
the driver keeps the upstream ideogram-oss constants (``utils.LATENT_SHIFT`` / ``LATENT_SCALE``). This test
reads ONLY those two 128-vectors from the local checkpoint and records their (measured) disagreement.
Skips naming the missing path when the checkpoint is not on the host (CI).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from app.engine.models.families.ideogram4.utils import LATENT_SCALE, LATENT_SHIFT

BN_EPS = 1e-4  # ideogram4_fp8.yaml vae.batch_norm_eps
AGREEMENT_TOL = 1e-3
RECORD_TOL = 1e-4
RECORDED_SHIFT_DIFF = 0.4283
RECORDED_SCALE_DIFF = 0.2933
_REPO = "models--ideogram-ai--ideogram-4-fp8"


def _candidates() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("MRLN_IDEOGRAM4_CHECKPOINT")
    if env:
        out.append(Path(env) / "vae" / "diffusion_pytorch_model.safetensors")
    hubs = [os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME") and os.path.join(os.environ["HF_HOME"], "hub"),
            str(Path.home() / ".cache" / "huggingface" / "hub")]
    for hub in filter(None, hubs):
        snaps = Path(hub) / _REPO / "snapshots"
        if snaps.is_dir():
            out += [s / "vae" / "diffusion_pytorch_model.safetensors" for s in sorted(snaps.iterdir())]
    return out


def _vae_weights() -> Path:
    cands = _candidates()
    for p in cands:
        if p.is_file():
            return p
    tried = cands[0] if cands else Path(f"<HF hub>/{_REPO}/snapshots/*/vae/diffusion_pytorch_model.safetensors")
    pytest.skip(f"ideogram4 checkpoint not on this host: {tried} (set MRLN_IDEOGRAM4_CHECKPOINT)")


def test_latent_shift_and_scale_differences_from_the_checkpoint_bn_stats_are_the_recorded_ones() -> None:
    """Pins the MEASURED state: the two normalisations are NOT the same statistics.

    Measured 2026-10-12 on snapshot ee79a723 of ideogram-ai/ideogram-4-fp8: max|bn.running_mean - LATENT_SHIFT|
    = 0.4283 and max|sqrt(bn.running_var + 1e-4) - LATENT_SCALE| = 0.2933, both far above the 1e-3 agreement
    bar. The driver keeps the upstream ideogram-oss constants (no training math changes in a retirement lane);
    the difference is a UAT item and a follow-up lane for the user's decision (RULE-17). The test fails if the
    checkpoint or the constants drift from the recorded state, and says so by number.
    """
    from safetensors import safe_open

    with safe_open(str(_vae_weights()), framework="pt") as f:
        mean = f.get_tensor("bn.running_mean").to(torch.float32)
        var = f.get_tensor("bn.running_var").to(torch.float32)
    assert mean.shape == (128,) and var.shape == (128,)
    shift_diff = float((mean - torch.tensor(LATENT_SHIFT)).abs().max())
    scale_diff = float((torch.sqrt(var + BN_EPS) - torch.tensor(LATENT_SCALE)).abs().max())
    print(f"LANE-150 ideogram4 bn-stats: max|shift diff| = {shift_diff:.4f}, max|scale diff| = {scale_diff:.4f}")
    assert abs(shift_diff - RECORDED_SHIFT_DIFF) <= RECORD_TOL, (
        f"LANE-150 bn-stats: max|bn.running_mean - LATENT_SHIFT| = {shift_diff:.6f}, recorded {RECORDED_SHIFT_DIFF}"
        f" (agreement bar {AGREEMENT_TOL:g}); constants or checkpoint changed"
    )
    assert abs(scale_diff - RECORDED_SCALE_DIFF) <= RECORD_TOL, (
        f"LANE-150 bn-stats: max|sqrt(bn.running_var+eps) - LATENT_SCALE| = {scale_diff:.6f}, recorded "
        f"{RECORDED_SCALE_DIFF} (agreement bar {AGREEMENT_TOL:g}); constants or checkpoint changed"
    )
    assert shift_diff > AGREEMENT_TOL and scale_diff > AGREEMENT_TOL  # the recorded state is a disagreement
