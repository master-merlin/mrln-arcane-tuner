"""Cross-boundary integration: Ideogram4 custom VAE <-> shared LatentManager.

The vendored ``Ideogram4AutoEncoder`` is a bare ``nn.Module`` (NOT a diffusers
autoencoder). The shared :class:`~app.engine.components.latents.LatentManager`
(used by the training latent pre-cache) and the sampler both assume a minimal
VAE contract:

* ``vae.dtype`` — read to cast the input image batch
  (``latents.py``: ``image_batch.to(self.device, dtype=self.vae.dtype)``);
* ``vae.encode(x)`` returning either a raw latent Tensor (consumed AS the
  cached latent, no extra scaling) or a diffusers ``.latent_dist`` object.

For ideogram4 the cached latent MUST be the **32-channel** posterior latent
(``z_channels=32``), so the pipeline's 2x2 patchify yields 128-dim tokens that
match the DiT's ``in_channels=128``. Taking the raw 64-channel encoder head
(mean+logvar concat) un-split would patchify to 256 and crash the DiT.

The per-channel LATENT_SHIFT/LATENT_SCALE normalization happens LATER in
``driver.prepare_latents``; the cached latent must therefore be the RAW VAE
posterior (no diffusers scaling_factor), so LatentManager's scaling_factor
must stay 1.0 (no ``vae.config.scaling_factor``, no ``vae.scaling_factor``).

These tests instantiate a TINY real VAE (random weights, CPU) and run it
through the SAME code path LatentManager uses. They FAIL against the pre-fix
code, where ``encode()`` returns 64 channels and the module has no ``.dtype``.
"""

from __future__ import annotations

import torch

from app.engine.components.latents import LatentManager
from app.engine.models.families.ideogram4.vendor.autoencoder_ideogram4 import (
    AutoEncoderParams,
    Ideogram4AutoEncoder,
)

# Tiny config: ch_mult length 2 -> 1 downsample -> spatial factor 2.
# z_channels=4 keeps the latent small while still exercising the 32-vs-64
# channel split (encoder head emits 2 * z_channels = 8).
TINY_PARAMS = AutoEncoderParams(
    resolution=32,
    in_channels=3,
    ch=32,
    out_ch=3,
    ch_mult=[1, 2],
    num_res_blocks=1,
    z_channels=4,
)
TINY_SPATIAL_FACTOR = 2


def _tiny_vae() -> Ideogram4AutoEncoder:
    return Ideogram4AutoEncoder(TINY_PARAMS).eval()


def test_vae_exposes_dtype_property() -> None:
    """LatentManager reads ``vae.dtype`` to cast the image batch."""
    vae = _tiny_vae()
    assert vae.dtype == next(vae.parameters()).dtype


def test_latent_manager_produces_z_channel_latent() -> None:
    """Driving the VAE through LatentManager yields a z_channels latent.

    This is the exact bug C1: the cached latent must be the posterior MEAN
    (z_channels), NOT the 2*z_channels encoder head. With z_channels=4 the
    production analogue is 32 (-> patchify 128, matching the DiT). We assert
    the latent has ``z_channels`` channels and that LatentManager applies NO
    scaling (scaling_factor == 1.0) so it does not double up with the later
    LATENT_SHIFT/LATENT_SCALE normalization.
    """
    vae = _tiny_vae()
    # No arch_params -> scaling_factor falls back to 1.0 (no vae.config,
    # no vae.scaling_factor), spatial_downscale -> 8 (unused here).
    lm = LatentManager(vae=vae, device=torch.device("cpu"))
    assert lm.scaling_factor == 1.0

    h = w = 16
    image_batch = torch.randn(1, TINY_PARAMS.in_channels, h, w)

    latents = lm.encode_and_cache_batch(image_batch, ids=["img0"])

    # The cached latent is the z_channels posterior, NOT 2*z_channels.
    assert latents.shape[1] == TINY_PARAMS.z_channels, (
        f"expected {TINY_PARAMS.z_channels} latent channels (posterior mean), "
        f"got {latents.shape[1]} (raw encoder head would be "
        f"{2 * TINY_PARAMS.z_channels})"
    )
    assert latents.shape[0] == 1
    assert latents.shape[2] == h // TINY_SPATIAL_FACTOR
    assert latents.shape[3] == w // TINY_SPATIAL_FACTOR


def test_latent_roundtrips_through_decode() -> None:
    """A z_channels latent from encode() round-trips back through decode()."""
    vae = _tiny_vae()
    lm = LatentManager(vae=vae, device=torch.device("cpu"))

    h = w = 16
    image_batch = torch.randn(1, TINY_PARAMS.in_channels, h, w)
    latents = lm.encode_and_cache_batch(image_batch, ids=["img0"])

    assert latents.shape[1] == TINY_PARAMS.z_channels
    with torch.no_grad():
        decoded = vae.decode(latents)

    assert decoded.shape[0] == 1
    assert decoded.shape[1] == TINY_PARAMS.out_ch
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
    latents = torch.randn(1, TINY_PARAMS.z_channels, 16, 16)
    img = sampler.decode_latents(latents)
    assert isinstance(img, Image.Image)
