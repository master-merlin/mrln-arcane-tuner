"""LTX-2 audio feature/latent core — mel contract + pack/normalize round-trips.

CPU-only: exercises the real torchaudio mel transform and the verbatim
LTX2Pipeline packing/normalization, so the *target-latent* recipe is pinned
without needing the real audio VAE or a GPU.
"""

from __future__ import annotations

import torch

from app.engine.models.families.ltx2.audio_mel import (
    AudioMelExtractor,
    denormalize_audio_latents,
    encode_clean_audio_latents,
    normalize_audio_latents,
    pack_audio_latents,
    unpack_audio_latents,
)


def test_mel_extractor_shape_and_pinned_params():
    ext = AudioMelExtractor()
    # Pinned to the Lightricks audio VAE training contract.
    assert ext.mel.n_fft == 1024
    assert ext.mel.hop_length == 160
    assert ext.target_sample_rate == 16000

    waveform = torch.randn(2, 2, 16000)  # [B, stereo, 1s @ 16k]
    mel = ext.waveform_to_mel(waveform, sample_rate=16000)

    # [B, C, time, n_mels] with n_mels pinned to 64.
    assert mel.shape[0] == 2 and mel.shape[1] == 2 and mel.shape[3] == 64
    assert mel.shape[2] > 0
    assert torch.isfinite(mel).all()


def test_mel_extractor_resamples_non_native_rate():
    ext = AudioMelExtractor()
    # 8 kHz input → resampled to 16 kHz; should not raise and stays finite.
    mel = ext.waveform_to_mel(torch.randn(1, 2, 8000), sample_rate=8000)
    assert mel.shape[1] == 2 and mel.shape[3] == 64


def test_pack_unpack_roundtrip():
    latent = torch.randn(2, 8, 5, 16)  # [B, C, L, M]
    packed = pack_audio_latents(latent)
    assert packed.shape == (2, 5, 128)  # [B, L, C*M]
    restored = unpack_audio_latents(packed, num_mel_bins=16)
    assert restored.shape == latent.shape
    assert torch.allclose(restored, latent)


def test_normalize_denormalize_roundtrip():
    packed = torch.randn(2, 5, 128)
    mean = torch.randn(128)
    std = torch.rand(128) + 0.5
    norm = normalize_audio_latents(packed, mean, std)
    back = denormalize_audio_latents(norm, mean, std)
    assert torch.allclose(back, packed, atol=1e-5)


class _FakeDist:
    def __init__(self, t):
        self._t = t

    def mode(self):
        return self._t


class _FakeAudioVAE:
    """Encodes a mel [B, C, T, 64] → latent [B, 8, T//4, 16]; 128-wide stats."""

    def __init__(self):
        self.latents_mean = torch.zeros(128)
        self.latents_std = torch.full((128,), 2.0)

    def encode(self, mel):
        b, _, t, _ = mel.shape
        lat = torch.ones(b, 8, max(t // 4, 1), 16)
        return type("O", (), {"latent_dist": _FakeDist(lat)})()


def test_encode_clean_audio_latents_packs_then_normalizes():
    vae = _FakeAudioVAE()
    mel = torch.zeros(2, 2, 8, 64)  # → latent [2, 8, 2, 16]
    clean = encode_clean_audio_latents(vae, mel)

    # packed [B, L=2, 128], then (1 - 0)/2 = 0.5 per feature.
    assert clean.shape == (2, 2, 128)
    assert torch.allclose(clean, torch.full_like(clean, 0.5))


def test_driver_encode_audio_clean_end_to_end_with_fake_vae():
    """The driver wires waveform → (real) mel → fake VAE → packed/normalized."""
    from app.engine.models.families.ltx2.driver import Ltx2Driver

    drv = object.__new__(Ltx2Driver)
    drv.audio_vae = _FakeAudioVAE()
    drv.audio_sampling_rate = 16000
    drv.device = torch.device("cpu")
    drv._audio_mel = None

    clean = drv.encode_audio_clean(torch.randn(2, 2, 16000), sample_rate=16000)
    assert clean.ndim == 3 and clean.shape[0] == 2 and clean.shape[2] == 128
