"""
Latent Creation Validation — Diagnostic tests for Flux VAE encoding pipeline.

Tests the full encode → decode cycle at every stage using native diffusers VAEs:
  1. Image preprocessing (normalization to [-1, 1])
  2. Diffusers AutoencoderKL encode → latent_dist.mode()
  3. Scale + shift: (z - shift_factor) * scaling_factor
  4. Pixel unshuffle (32ch → 128ch)
  5. Decode roundtrip (reverse all steps)
  6. Latent distribution validation

Requires: FLUX_VAE_PATH env var pointing to a local Flux VAE directory.
Run: python -m pytest tests/test_latent_validation.py -v -s
"""

import os
import math
import torch
import pytest
import numpy as np
from PIL import Image
from torchvision import transforms
from einops import rearrange

# ---------------------------------------------------------------------------
# Skip everything if model paths don't resolve
# ---------------------------------------------------------------------------
FLUX_VAE_PATH = os.environ.get(
    "FLUX_VAE_PATH",
    r"D:\AI\huggingface\hub\hub\models--black-forest-labs--FLUX.2-klein-base-9B\snapshots\17c3b160520b7dd44665dbf0b9ed9dd30c15cd06\vae",
)
TEST_IMAGE_DIR = os.environ.get(
    "TEST_IMAGE_DIR",
    r"D:\MRLN Arcane Tuner\backend\tests\fixtures",
)
_can_run = os.path.exists(FLUX_VAE_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_test_image(width: int = 1024, height: int = 1024) -> Image.Image:
    """Create a vibrant test image or load one from fixtures."""
    if os.path.isdir(TEST_IMAGE_DIR):
        for f in os.listdir(TEST_IMAGE_DIR):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                return Image.open(os.path.join(TEST_IMAGE_DIR, f)).convert("RGB").resize(
                    (width, height), Image.Resampling.LANCZOS
                )

    # Fallback: procedural gradient image
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x] = [
                int(255 * x / width),
                int(255 * y / height),
                int(255 * (1 - x / width)),
            ]
    return Image.fromarray(arr)


def _print_stats(label: str, t: torch.Tensor):
    """Print detailed tensor statistics."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Shape:  {list(t.shape)}")
    print(f"  Dtype:  {t.dtype}")
    print(f"  Min:    {t.min().item():.6f}")
    print(f"  Max:    {t.max().item():.6f}")
    print(f"  Mean:   {t.mean().item():.6f}")
    print(f"  Std:    {t.std().item():.6f}")
    print(f"  Abs Mean: {t.abs().mean().item():.6f}")
    if t.ndim == 4:
        ch = t.shape[1]
        print(f"  Per-channel means (first 8 of {ch}):")
        for c in range(min(8, ch)):
            print(f"    ch[{c}]: mean={t[0, c].mean().item():.6f}  std={t[0, c].std().item():.6f}")


# ---------------------------------------------------------------------------
# Core Validation Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _can_run, reason=f"VAE not found at {FLUX_VAE_PATH}")
class TestLatentCreation:
    """Validate the Flux latent creation pipeline end-to-end."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load VAE once for all tests."""
        from diffusers import AutoencoderKL

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusers_vae = AutoencoderKL.from_pretrained(
            FLUX_VAE_PATH, torch_dtype=torch.float32
        ).to(self.device).eval()

        # Read the actual config values
        cfg = self.diffusers_vae.config
        self.scaling_factor = getattr(cfg, "scaling_factor", None) or 0.3611
        self.shift_factor = getattr(cfg, "shift_factor", None) or 0.1159
        print(f"\n  VAE Config: scaling_factor={self.scaling_factor}, shift_factor={self.shift_factor}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def test_01_image_preprocessing(self):
        """Verify image normalization produces [-1, 1] range."""
        img = _make_test_image(512, 512)
        tensor = self.transform(img)

        _print_stats("Image after ToTensor + Normalize([0.5],[0.5])", tensor.unsqueeze(0))

        assert tensor.min() >= -1.0, f"Min {tensor.min()} < -1.0"
        assert tensor.max() <= 1.0, f"Max {tensor.max()} > 1.0"
        assert tensor.shape == (3, 512, 512)
        print("  [OK] Image correctly normalized to [-1, 1]")

    def test_02_raw_vae_encode(self):
        """Test raw Diffusers VAE encode (before our scaling)."""
        img = _make_test_image(512, 512)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            encoded = self.diffusers_vae.encode(tensor)
            raw_z = encoded.latent_dist.mode()

        _print_stats("Raw VAE encode (latent_dist.mode())", raw_z)

        assert raw_z.shape == (1, 32, 64, 64), f"Expected (1,32,64,64) got {raw_z.shape}"
        print(f"  [OK] Raw encode shape correct: {list(raw_z.shape)}")

    def test_03_scaled_latents(self):
        """Test latents after shift + scale."""
        img = _make_test_image(512, 512)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_z = self.diffusers_vae.encode(tensor).latent_dist.mode()
            scaled_z = (raw_z - self.shift_factor) * self.scaling_factor

        _print_stats("After (z - shift) * scale", scaled_z)
        print(f"\n  Scaling applied: (z - {self.shift_factor}) * {self.scaling_factor}")

    def test_04_pixel_unshuffle(self):
        """Test pixel unshuffle (32ch → 128ch, /2 spatial)."""
        img = _make_test_image(512, 512)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_z = self.diffusers_vae.encode(tensor).latent_dist.mode()
            scaled_z = (raw_z - self.shift_factor) * self.scaling_factor
            unshuffled = rearrange(scaled_z, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)

        _print_stats("After pixel unshuffle (128ch)", unshuffled)

        assert unshuffled.shape == (1, 128, 32, 32), f"Expected (1,128,32,32) got {unshuffled.shape}"
        # Verify reshape is lossless
        reshuffled = rearrange(unshuffled, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=2, p2=2, c=32)
        assert torch.allclose(reshuffled, scaled_z), "Pixel unshuffle is not reversible!"
        print("  [OK] Pixel unshuffle correct and reversible")

    def test_05_direct_vae_encode_decode(self):
        """Test native diffusers VAE encode → decode roundtrip."""
        img = _make_test_image(512, 512)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_z = self.diffusers_vae.encode(tensor).latent_dist.mode()
            decoded = self.diffusers_vae.decode(raw_z).sample

        _print_stats("Original image tensor", tensor)
        _print_stats("Decoded image tensor", decoded)

        # L1 reconstruction error
        l1_error = (tensor - decoded).abs().mean().item()
        mse = ((tensor - decoded) ** 2).mean().item()
        psnr = 10 * math.log10(4.0 / mse) if mse > 0 else float("inf")

        print(f"\n  L1 Reconstruction Error: {l1_error:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  PSNR: {psnr:.2f} dB")

        assert l1_error < 0.15, f"L1 error {l1_error:.4f} too high — VAE roundtrip is broken!"
        assert psnr > 20.0, f"PSNR {psnr:.1f} too low — VAE roundtrip is broken!"

    def test_06_latent_distribution_check(self):
        """Verify latent distribution after scale+shift."""
        img = _make_test_image(1024, 1024)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_z = self.diffusers_vae.encode(tensor).latent_dist.mode()
            latents = (raw_z - self.shift_factor) * self.scaling_factor

        _print_stats("Scaled latents (1024x1024)", latents)

        mean = latents.mean().item()
        std = latents.std().item()

        print("\n  Expected for well-scaled latents:")
        print(f"  - Mean close to 0 (got {mean:.4f})")
        print(f"  - Std close to 1 (got {std:.4f})")

        if abs(mean) > 0.5:
            print(f"  [WARN] Mean {mean:.4f} is far from 0 -- possible scaling issue!")
        if std < 0.1 or std > 5.0:
            print(f"  [WARN] Std {std:.4f} is abnormal -- possible scaling issue!")

    def test_07_noise_vs_latent_scale(self):
        """Verify that noise and latents have compatible scales for training.

        In flow-matching, the noisy input is: x_t = (1-t)*latent + t*noise
        If noise ~ N(0,1) and latents have very different scale, the model
        will struggle at certain timesteps.
        """
        img = _make_test_image(1024, 1024)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_z = self.diffusers_vae.encode(tensor).latent_dist.mode()
            latents = (raw_z - self.shift_factor) * self.scaling_factor
            # Pack to sequence format via pixel unshuffle
            packed = rearrange(latents, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)
            packed = packed.flatten(2).transpose(1, 2)  # [B, L, C]

        noise = torch.randn_like(packed)

        _print_stats("Packed latents [B, L, D]", packed)
        _print_stats("Noise [B, L, D]", noise)

        latent_std = packed.std().item()
        noise_std = noise.std().item()
        ratio = latent_std / noise_std

        print(f"\n  Latent std / Noise std = {ratio:.4f}")
        print("  Ideal: ≈ 1.0 (latents and noise at same scale)")

        if ratio < 0.1 or ratio > 10.0:
            print(f"  [FAIL] CRITICAL: Scale mismatch ratio {ratio:.4f}")
        elif ratio < 0.3 or ratio > 3.0:
            print(f"  [WARN] Scale mismatch ratio {ratio:.4f}")
        else:
            print("  [OK] Scale ratio is acceptable")

    def test_08_compare_mode_vs_sample(self):
        """Compare mode() vs sample() for the latent distribution."""
        img = _make_test_image(512, 512)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            encoded = self.diffusers_vae.encode(tensor)
            z_mode = encoded.latent_dist.mode()
            z_sample = encoded.latent_dist.sample()
            z_mean = encoded.latent_dist.mean
            z_std_latent = encoded.latent_dist.std

        _print_stats("latent_dist.mode()", z_mode)
        _print_stats("latent_dist.sample()", z_sample)

        mode_sample_diff = (z_mode - z_sample).abs().mean().item()
        mode_mean_diff = (z_mode - z_mean).abs().mean().item()

        print(f"\n  mode() vs sample() mean abs diff: {mode_sample_diff:.6f}")
        print(f"  mode() vs mean:   mean abs diff: {mode_mean_diff:.6f}")
        print(f"  mode() == mean? {torch.allclose(z_mode, z_mean)}")
        print(f"  latent std (stochasticity): mean={z_std_latent.mean().item():.6f}")
