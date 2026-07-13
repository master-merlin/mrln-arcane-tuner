"""Tests for ``app.core.audio.probe.probe_audio``.

Tiny synthetic clips are generated with ``soundfile`` (no fixture binaries
committed) so the suite runs identically on Windows and Docker. Covers all
five contract extensions (``.wav .mp3 .flac .ogg .opus``) — the decoder
recon that picked ``soundfile`` as the primary decoder (docker's torchaudio
2.11 trio is installed ``--no-deps`` and may carry different backends than
the local 2.12.1 install) verified every one round-trips via libsndfile.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from app.core.audio import AudioProbe, AudioProbeError, probe_audio


# ── Fixture helpers ──────────────────────────────────────────────────────


def _write_tone(
    path: Path,
    *,
    sample_rate: int = 8000,
    duration_s: float = 0.25,
    channels: int = 1,
    format: str | None = None,
    subtype: str | None = None,
) -> Path:
    """Write a tiny sine-wave tone to *path* and return it."""
    n = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    if channels > 1:
        tone = np.stack([tone] * channels, axis=-1)
    kwargs = {}
    if format:
        kwargs["format"] = format
    if subtype:
        kwargs["subtype"] = subtype
    sf.write(str(path), tone.astype(np.float32), sample_rate, **kwargs)
    return path


# ── Tests ────────────────────────────────────────────────────────────────


def test_probe_wav_mono(tmp_path):
    clip = _write_tone(tmp_path / "clip.wav", sample_rate=8000, duration_s=0.25)

    probe = probe_audio(clip)

    assert isinstance(probe, AudioProbe)
    assert probe.sample_rate == 8000
    assert probe.channels == 1
    assert probe.duration_s == pytest.approx(0.25, abs=0.01)


def test_probe_wav_stereo(tmp_path):
    clip = _write_tone(tmp_path / "stereo.wav", channels=2, sample_rate=16000, duration_s=0.1)

    probe = probe_audio(clip)

    assert probe.channels == 2
    assert probe.sample_rate == 16000


def test_probe_flac(tmp_path):
    clip = _write_tone(tmp_path / "clip.flac", sample_rate=22050, duration_s=0.2, format="FLAC")

    probe = probe_audio(clip)

    assert probe.sample_rate == 22050
    assert probe.channels == 1
    assert probe.duration_s == pytest.approx(0.2, abs=0.01)


def test_probe_ogg_vorbis(tmp_path):
    clip = _write_tone(
        tmp_path / "clip.ogg", sample_rate=8000, duration_s=0.15,
        format="OGG", subtype="VORBIS",
    )

    probe = probe_audio(clip)

    assert probe.sample_rate == 8000
    assert probe.duration_s == pytest.approx(0.15, abs=0.03)


def test_probe_opus(tmp_path):
    """.opus (OGG container, Opus subtype) — verified decodable, not excluded."""
    clip = _write_tone(
        tmp_path / "clip.opus", sample_rate=8000, duration_s=0.15,
        format="OGG", subtype="OPUS",
    )

    probe = probe_audio(clip)

    assert probe.duration_s == pytest.approx(0.15, abs=0.03)
    assert probe.channels == 1


def test_probe_mp3(tmp_path):
    clip = _write_tone(tmp_path / "clip.mp3", sample_rate=8000, duration_s=0.2)

    probe = probe_audio(clip)

    assert probe.duration_s == pytest.approx(0.2, abs=0.03)
    assert probe.channels == 1


def test_probe_is_immutable(tmp_path):
    """AudioProbe is frozen — fields can't be reassigned after construction."""
    probe = probe_audio(_write_tone(tmp_path / "frozen.wav"))
    with pytest.raises(ValidationError):
        probe.sample_rate = 99  # type: ignore[misc]


def test_probe_bad_file_raises_probe_error(tmp_path):
    """A non-audio file raises AudioProbeError (callers fall back gracefully)."""
    bad = tmp_path / "not_audio.wav"
    bad.write_bytes(b"this is not a valid audio container")

    with pytest.raises(AudioProbeError):
        probe_audio(bad)


def test_probe_missing_file_raises_probe_error(tmp_path):
    """A missing path raises AudioProbeError rather than a bare OSError."""
    with pytest.raises(AudioProbeError):
        probe_audio(tmp_path / "does_not_exist.wav")
