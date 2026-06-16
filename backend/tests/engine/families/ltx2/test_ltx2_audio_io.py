"""LTX-2 training-side audio decode (no GPU, no model weights).

Covers ``audio_io.load_audio_waveform``: the contract that a clip's audio trim
window decodes to a DETERMINISTIC-length mono waveform at the target sample rate
(so equal-duration clips collate to equal-length audio latents), zero-padding a
short source, downmixing stereo, and returning ``None`` when there is no audio
stream. Uses tiny WAVs written with the stdlib ``wave`` module + a video-only
mp4 written via the project's own encoder — both go through PyAV, the same
decoder the data path uses.
"""

from __future__ import annotations

import math
import struct
import wave

import torch

from app.engine.models.families.ltx2.audio_io import load_audio_waveform


def _write_sine_wav(path, *, sr, duration_s, freq=440.0, channels=1):
    n = int(sr * duration_s)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)  # 16-bit PCM
        w.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            val = int(32767 * 0.5 * math.sin(2 * math.pi * freq * i / sr))
            for _ in range(channels):
                frames += struct.pack("<h", val)
        w.writeframes(bytes(frames))


def test_returns_fixed_length_stereo_at_target_sr(tmp_path):
    p = tmp_path / "a.wav"
    _write_sine_wav(p, sr=8000, duration_s=2.0)

    out = load_audio_waveform(str(p), trim_start_s=0.0, duration_s=1.0, target_sr=16000)

    assert out is not None
    wav, sr = out
    assert sr == 16000
    assert wav.shape == (2, 16000)  # 2 channels (LTX-2 audio VAE is stereo), N = 16000
    assert wav.dtype == torch.float32
    assert wav.abs().max() <= 1.0


def test_short_source_is_zero_padded_to_target_length(tmp_path):
    p = tmp_path / "short.wav"
    _write_sine_wav(p, sr=8000, duration_s=0.25)  # only 0.25s of audio

    out = load_audio_waveform(str(p), trim_start_s=0.0, duration_s=1.0, target_sr=16000)

    assert out is not None
    wav, _ = out
    assert wav.shape == (2, 16000)  # padded out to a full second, both channels
    # ~0.25s → ~4000 samples at 16k; everything well past that must be silence.
    assert torch.count_nonzero(wav[:, 6000:]) == 0


def test_mono_source_upmixed_to_stereo(tmp_path):
    p = tmp_path / "mono.wav"
    _write_sine_wav(p, sr=8000, duration_s=1.0, channels=1)

    out = load_audio_waveform(str(p), trim_start_s=0.0, duration_s=0.5, target_sr=16000)

    assert out is not None
    wav, _ = out
    assert wav.shape == (2, 8000)  # mono duplicated into both channels
    assert torch.allclose(wav[0], wav[1])


def test_stereo_source_preserved_as_two_channels(tmp_path):
    p = tmp_path / "stereo.wav"
    _write_sine_wav(p, sr=8000, duration_s=1.0, channels=2)

    out = load_audio_waveform(str(p), trim_start_s=0.0, duration_s=0.5, target_sr=16000)

    assert out is not None
    wav, _ = out
    assert wav.shape == (2, 8000)  # both channels, 0.5s @ 16k


def test_trim_start_offsets_the_window(tmp_path):
    p = tmp_path / "a.wav"
    _write_sine_wav(p, sr=16000, duration_s=2.0)

    full = load_audio_waveform(str(p), trim_start_s=0.0, duration_s=2.0, target_sr=16000)
    offset = load_audio_waveform(str(p), trim_start_s=1.0, duration_s=1.0, target_sr=16000)

    assert full is not None and offset is not None
    # The offset window equals the back half of the full decode (per channel).
    assert torch.allclose(offset[0][0], full[0][0, 16000:], atol=1e-4)


def test_guards_and_missing_audio_return_none(tmp_path):
    p = tmp_path / "a.wav"
    _write_sine_wav(p, sr=8000, duration_s=1.0)

    assert load_audio_waveform(str(p), trim_start_s=0.0, duration_s=0.0, target_sr=16000) is None
    assert load_audio_waveform(str(p), trim_start_s=0.0, duration_s=1.0, target_sr=0) is None
    assert load_audio_waveform(str(tmp_path / "nope.wav"), trim_start_s=0.0, duration_s=1.0, target_sr=16000) is None


def test_video_only_file_has_no_audio_stream(tmp_path):
    from app.engine.components.video import VideoFrameLoader

    p = tmp_path / "v.mp4"
    frames = torch.zeros(3, 3, 16, 16)  # [C, F, H, W] in [-1, 1], video only
    VideoFrameLoader().encode_video(frames, None, 8.0, str(p))

    assert load_audio_waveform(str(p), trim_start_s=0.0, duration_s=1.0, target_sr=16000) is None
