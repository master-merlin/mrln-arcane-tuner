"""Cross-boundary integration: Ideogram4's VAE <-> shared LatentManager.

Since LANE-150 the VAE is diffusers 0.41.0's ``AutoencoderKLFlux2`` (the
vendored ``Ideogram4AutoEncoder`` is retired). The shared
:class:`~app.engine.components.latents.LatentManager` (used by the training
latent pre-cache) and the sampler rely on:

* ``vae.dtype`` -- read to cast the input image batch;
* ``vae.encode(x).latent_dist`` -- LatentManager takes ``.mode()`` for an
  ``AutoencoderKLFlux2`` (latents.py), the deterministic posterior mean.

For ideogram4 the cached latent MUST be the **32-channel** posterior latent
(``latent_channels=32``), so the pipeline's 2x2 patchify yields 128-dim tokens
that match the DiT's ``in_channels=128``. The per-channel
LATENT_SHIFT/LATENT_SCALE normalization happens LATER in
``driver.prepare_latents``; the cached latent must therefore be the RAW VAE
posterior, so LatentManager's scaling_factor must stay 1.0
(``AutoencoderKLFlux2`` has no ``scaling_factor``).
"""

from __future__ import annotations

import torch
from diffusers import AutoencoderKLFlux2

from app.engine.components.latents import LatentManager

# Two blocks -> one downsample -> spatial factor 2; latent_channels=4 keeps the
# latent small while still exercising the mean-vs-(mean+logvar) split.
LATENT_CHANNELS = 4
TINY_SPATIAL_FACTOR = 2


def _tiny_vae() -> AutoencoderKLFlux2:
    return AutoencoderKLFlux2(
        in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2, block_out_channels=(32, 64),
        layers_per_block=1, latent_channels=LATENT_CHANNELS, norm_num_groups=32, sample_size=32,
    ).eval()


def test_vae_exposes_dtype_property() -> None:
    """LatentManager reads ``vae.dtype`` to cast the image batch."""
    vae = _tiny_vae()
    assert vae.dtype == next(vae.parameters()).dtype


def test_latent_manager_produces_z_channel_latent() -> None:
    """Driving the VAE through LatentManager yields a z_channels latent.

    This is the exact bug C1: the cached latent must be the posterior MEAN
    (latent_channels), NOT the 2x encoder head. With latent_channels=4 the
    production analogue is 32 (-> patchify 128, matching the DiT). We assert
    the latent has ``z_channels`` channels and that LatentManager applies NO
    scaling (scaling_factor == 1.0) so it does not double up with the later
    LATENT_SHIFT/LATENT_SCALE normalization.
    """
    vae = _tiny_vae()
    # No arch_params -> scaling_factor falls back to 1.0 (AutoencoderKLFlux2's
    # config has no scaling_factor), spatial_downscale is unused here.
    lm = LatentManager(vae=vae, device=torch.device("cpu"))
    assert lm.scaling_factor == 1.0

    h = w = 16
    image_batch = torch.randn(1, 3, h, w)

    latents = lm.encode_and_cache_batch(image_batch, ids=["img0"])

    # The cached latent is the z_channels posterior, NOT 2*z_channels.
    assert latents.shape[1] == LATENT_CHANNELS, (
        f"expected {LATENT_CHANNELS} latent channels (posterior mean), "
        f"got {latents.shape[1]} (raw encoder head would be "
        f"{2 * LATENT_CHANNELS})"
    )
    assert latents.shape[0] == 1
    assert latents.shape[2] == h // TINY_SPATIAL_FACTOR
    assert latents.shape[3] == w // TINY_SPATIAL_FACTOR


def test_latent_roundtrips_through_decode() -> None:
    """A z_channels latent from encode() round-trips back through decode()."""
    vae = _tiny_vae()
    lm = LatentManager(vae=vae, device=torch.device("cpu"))

    h = w = 16
    image_batch = torch.randn(1, 3, h, w)
    latents = lm.encode_and_cache_batch(image_batch, ids=["img0"])

    assert latents.shape[1] == LATENT_CHANNELS
    with torch.no_grad():
        decoded = vae.decode(latents).sample

    assert decoded.shape[0] == 1
    assert decoded.shape[1] == 3
    assert decoded.shape[2] == h
    assert decoded.shape[3] == w


def test_sampler_decode_latents_does_not_raise_on_dtype() -> None:
    """Sampler ``decode_latents`` casts via ``vae.dtype`` without raising (C2).

    We invoke the unbound method with a minimal stub pipeline so we avoid
    standing up the whole trainer; the only behavior under test is that the
    ``vae.dtype`` cast + decode no longer raises and returns a PIL image.
    """
    from PIL import Image

    from app.engine.models.families.ideogram4.sampler import IdeogramV4Sampler

    vae = _tiny_vae()

    class _StubPipeline:
        pass

    stub = _StubPipeline()
    stub.vae = vae

    sampler = object.__new__(IdeogramV4Sampler)
    sampler.pipeline = stub
    sampler.device = torch.device("cpu")

    # A z_channels latent at the tiny resolution, [1, z_channels, H, W].
    latents = torch.randn(1, LATENT_CHANNELS, 16, 16)
    img = sampler.decode_latents(latents)
    assert isinstance(img, Image.Image)
    # The decoded pixels, not the DecoderOutput wrapper: 16 x 2 = 32 per side.
    assert img.size == (16 * TINY_SPATIAL_FACTOR, 16 * TINY_SPATIAL_FACTOR)
