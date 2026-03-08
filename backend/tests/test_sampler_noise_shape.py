"""
Tests for FLUX.1 / FLUX.2 sampler noise shape correctness.

Verifies that ``_create_initial_noise`` produces noise in **VAE latent
space** (pre-packing), so that ``pack_latents`` in ``denoise()`` yields
the correct packed sequence dimensions expected by the transformer.

Pure tensor-shape tests — no model loading required.
"""

import torch
import pytest


# ── FLUX.1 ───────────────────────────────────────────────────────────────


class TestFlux1SamplerNoiseShape:
    """Verify Flux1Sampler._create_initial_noise + pack_latents shapes."""

    def test_noise_channels_are_vae_latent_channels(self):
        """Initial noise must have 16 channels (VAE latent_channels)."""
        from app.engine.models.families.flux1.sampler import Flux1Sampler

        # Call the unbound method via __func__ with a minimal stub
        stub = type("Stub", (), {"device": "cpu"})()
        gen = torch.Generator(device="cpu").manual_seed(0)
        noise = Flux1Sampler._create_initial_noise(stub, 1024, 1024, gen)

        assert noise.shape[1] == 16, (
            f"Expected 16 VAE latent channels, got {noise.shape[1]}"
        )

    def test_noise_spatial_dims_vae_scale(self):
        """Spatial dims must be H/8 × W/8 (VAE downscale factor)."""
        from app.engine.models.families.flux1.sampler import Flux1Sampler

        stub = type("Stub", (), {"device": "cpu"})()
        gen = torch.Generator(device="cpu").manual_seed(0)
        noise = Flux1Sampler._create_initial_noise(stub, 1024, 1024, gen)

        assert noise.shape[2] == 1024 // 8, (
            f"Expected H={1024 // 8}, got {noise.shape[2]}"
        )
        assert noise.shape[3] == 1024 // 8, (
            f"Expected W={1024 // 8}, got {noise.shape[3]}"
        )

    def test_packed_shape_matches_transformer_expectation(self):
        """After pack_latents, feature dim must be 64 (= 16 * 4)."""
        from app.engine.models.families.flux1.sampler import Flux1Sampler
        from app.engine.models.families.flux1.utils import pack_latents

        stub = type("Stub", (), {"device": "cpu"})()
        gen = torch.Generator(device="cpu").manual_seed(0)
        noise = Flux1Sampler._create_initial_noise(stub, 1024, 1024, gen)
        packed, img_ids = pack_latents(noise)

        assert packed.shape == (1, 4096, 64), (
            f"Expected packed [1, 4096, 64], got {list(packed.shape)}"
        )
        assert img_ids.shape == (4096, 3), (
            f"Expected img_ids [4096, 3], got {list(img_ids.shape)}"
        )

    @pytest.mark.parametrize("width,height", [(512, 512), (768, 1024), (1024, 768)])
    def test_packed_shape_various_resolutions(self, width: int, height: int):
        """Packed feature dim must always be 64, regardless of resolution."""
        from app.engine.models.families.flux1.sampler import Flux1Sampler
        from app.engine.models.families.flux1.utils import pack_latents

        stub = type("Stub", (), {"device": "cpu"})()
        gen = torch.Generator(device="cpu").manual_seed(0)
        noise = Flux1Sampler._create_initial_noise(stub, width, height, gen)
        packed, _ = pack_latents(noise)

        expected_seq_len = (height // 16) * (width // 16)
        assert packed.shape == (1, expected_seq_len, 64), (
            f"For {width}×{height}: expected [1, {expected_seq_len}, 64], "
            f"got {list(packed.shape)}"
        )


# ── FLUX.2 ───────────────────────────────────────────────────────────────


class TestFlux2SamplerNoiseShape:
    """Verify Flux2Sampler._create_initial_noise produces packed noise.

    After the BatchNorm normalization fix, ``_create_initial_noise``
    returns noise directly in packed sequence format ``[1, L, 128]``
    (patchified 128-ch at H/16 × W/16, then packed to sequence).
    """

    def test_noise_is_packed_sequence(self):
        """Noise must be a 3-D packed tensor [B, L, D]."""
        from app.engine.models.families.flux2.sampler import Flux2Sampler

        stub = type("Stub", (), {"device": "cpu"})()
        gen = torch.Generator(device="cpu").manual_seed(0)
        noise = Flux2Sampler._create_initial_noise(stub, 1024, 1024, gen)

        assert noise.ndim == 3, (
            f"Expected 3-D packed tensor, got {noise.ndim}-D"
        )

    def test_noise_feature_dim_is_128(self):
        """Packed feature dim must be 128 (= 32 VAE ch × 4 from 2×2 patchify)."""
        from app.engine.models.families.flux2.sampler import Flux2Sampler

        stub = type("Stub", (), {"device": "cpu"})()
        gen = torch.Generator(device="cpu").manual_seed(0)
        noise = Flux2Sampler._create_initial_noise(stub, 1024, 1024, gen)

        assert noise.shape[2] == 128, (
            f"Expected feature dim 128, got {noise.shape[2]}"
        )

    def test_noise_sequence_length_matches_patchified_grid(self):
        """Sequence length must be (H/16) × (W/16), the patchified grid."""
        from app.engine.models.families.flux2.sampler import Flux2Sampler

        stub = type("Stub", (), {"device": "cpu"})()
        gen = torch.Generator(device="cpu").manual_seed(0)
        noise = Flux2Sampler._create_initial_noise(stub, 1024, 1024, gen)

        # 1024/8 = 128 (VAE), 128/2 = 64 (patchify), 64*64 = 4096
        expected_seq_len = (1024 // 16) * (1024 // 16)
        assert noise.shape[1] == expected_seq_len, (
            f"Expected seq_len {expected_seq_len}, got {noise.shape[1]}"
        )

    @pytest.mark.parametrize("width,height", [(512, 512), (768, 1024), (1024, 768)])
    def test_packed_shape_various_resolutions(self, width: int, height: int):
        """Packed shape must be correct at various resolutions."""
        from app.engine.models.families.flux2.sampler import Flux2Sampler

        stub = type("Stub", (), {"device": "cpu"})()
        gen = torch.Generator(device="cpu").manual_seed(0)
        noise = Flux2Sampler._create_initial_noise(stub, width, height, gen)

        expected_seq_len = (height // 16) * (width // 16)
        assert noise.shape == (1, expected_seq_len, 128), (
            f"For {width}×{height}: expected [1, {expected_seq_len}, 128], "
            f"got {list(noise.shape)}"
        )

