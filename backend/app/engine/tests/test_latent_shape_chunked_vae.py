"""The 5D latent-shape validator knows a CHUNKED video VAE (LANE-92 real-job
run: ``latent_video_shape_mismatch expected=90x32x32 actual=27x32x32
temporal_downscale=1`` on every job of a family whose VAE was doing exactly
what its definition says).

Production caller: ``LatentManager._validate_shape`` (called by
``encode_and_cache_batch`` on every encoded batch). The expectation used to come
from a class-name list alone; a VAE outside that list got ``t = 1`` and so
"expected F latent frames". A definition that declares ``vae.clip_length``
states a VAE that encodes the clip in chunks of that many pixel frames, each
chunk by the ``video.vae_temporal`` factor: ``17n+5`` frames -> ``5n+2``
latent frames (definition ``minimax_h3_t2va.yaml`` ``video.frame_rule``
comment; ``sampler.latent_frames_for``).

Every family that declares no ``vae.clip_length`` keeps the expectation it had:
the two positive controls pin one ``4n+1`` and one ``8n+1`` family, message
fields included.
"""

from __future__ import annotations

import pytest
import torch
from structlog.testing import capture_logs
from torch import nn

from app.engine.components.latents import LatentManager
from app.engine.models.families.minimax_h3.pixel_adapter import H3PixelAdaptedVAE
from app.engine.models.families.minimax_h3.sampler import latent_frames_for
from app.engine.models.registry import ModelRegistry
from app.engine.tests.h3_text_stubs import StubVisualVAE

EVENT = "latent_video_shape_mismatch"


def _arch(def_id: str) -> dict:
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    return dict(ModelRegistry._definitions[def_id].architecture_params)


class AutoencoderKLWan(nn.Module):
    """Matched by CLASS NAME, which is how production recognises the WAN VAE."""

    spatial_compression_ratio = 8


class AutoencoderKLLTX2Video(nn.Module):
    spatial_compression_ratio = 32


def _h3_manager() -> LatentManager:
    inner = StubVisualVAE()
    inner.spatial_compression_ratio = 16
    return LatentManager(H3PixelAdaptedVAE(inner), device="cpu", arch_params=_arch("minimax-h3-t2va"))


def _mismatches(manager: LatentManager, latent_shape, input_shape) -> list[dict]:
    with capture_logs() as logs:
        manager._validate_shape(torch.zeros(latent_shape), torch.Size(input_shape))
    return [e for e in logs if e.get("event") == EVENT]


@pytest.mark.parametrize("frames", [5, 22, 90, 107])
def test_h3_latent_of_the_stated_length_raises_no_warning(frames):
    lat_f = latent_frames_for(frames)
    got = _mismatches(_h3_manager(), (1, 24, lat_f, 32, 32), (1, 3, frames, 512, 512))
    assert got == [], f"false warning for {frames} -> {lat_f} latent frames: {got}"


def test_h3_the_real_job_shape_90_frames_27_latents():
    """The exact pair the real job logged a WARNING for."""
    assert _mismatches(_h3_manager(), (1, 24, 27, 32, 32), (1, 3, 90, 512, 512)) == []


def test_h3_a_truly_wrong_latent_length_still_warns():
    """The check is not switched off for the family: 23 latent frames for 90
    pixel frames is wrong and says what was expected."""
    got = _mismatches(_h3_manager(), (1, 24, 23, 32, 32), (1, 3, 90, 512, 512))
    assert len(got) == 1
    assert got[0]["expected"] == "27x32x32" and got[0]["actual"] == "23x32x32", got[0]
    assert got[0]["temporal_downscale"] == 4, got[0]


@pytest.mark.parametrize(
    ("def_id", "vae_cls", "frames", "size", "ok", "td"),
    [
        ("wan2.1-t2v-1.3b", AutoencoderKLWan, 81, 64, (21, 8, 8), 4),
        ("ltx2-3-base",AutoencoderKLLTX2Video, 121, 64, (16, 2, 2), 8),
    ],
)
def test_other_families_expectation_does_not_move(def_id, vae_cls, frames, size, ok, td):
    m = LatentManager(vae_cls(), device="cpu", arch_params=_arch(def_id))
    assert _mismatches(m, (1, 16, *ok), (1, 3, frames, size, size)) == []
    wrong = _mismatches(m, (1, 16, ok[0] + 1, ok[1], ok[2]), (1, 3, frames, size, size))
    assert len(wrong) == 1
    assert {k: wrong[0][k] for k in ("expected", "actual", "input", "temporal_downscale")} == {
        "expected": f"{ok[0]}x{ok[1]}x{ok[2]}",
        "actual": f"{ok[0] + 1}x{ok[1]}x{ok[2]}",
        "input": f"{frames}x{size}x{size}",
        "temporal_downscale": td,
    }
