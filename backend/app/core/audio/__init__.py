"""Audio utilities for the dataset layer.

Pure, side-effect-free helpers for probing audio files (duration, sample
rate, channels) used by the scanner to populate per-file metadata.
``soundfile``-backed (bundled libsndfile) — chosen over torchaudio as the
primary decoder because the docker image's torchaudio 2.11.0 and the local
install's (2.11.1, or 2.11.0 on the cu126 set) are built against different
torch releases and may carry different backends; soundfile is a single,
consistent dependency across all of them.
"""

from __future__ import annotations

from app.core.audio.probe import AudioProbe, AudioProbeError, probe_audio

__all__ = ["AudioProbe", "AudioProbeError", "probe_audio"]
