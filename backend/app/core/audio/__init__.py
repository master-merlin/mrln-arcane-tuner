"""Audio utilities for the dataset layer.

Pure, side-effect-free helpers for probing audio files (duration, sample
rate, channels) used by the scanner to populate per-file metadata.
``soundfile``-backed (bundled libsndfile) — chosen over torchaudio as the
primary decoder because the docker image's torchaudio 2.11 trio is
installed ``--no-deps`` and may carry different backends than the local
2.12.1 install; soundfile is a single, consistent dependency across both.
"""

from __future__ import annotations

from app.core.audio.probe import AudioProbe, AudioProbeError, probe_audio

__all__ = ["AudioProbe", "AudioProbeError", "probe_audio"]
