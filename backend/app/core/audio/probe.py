"""soundfile-backed audio probing.

``probe_audio(path)`` opens an audio file and extracts the metadata the
dataset scanner needs — duration, sample rate, channel count — without
decoding the full waveform (``soundfile.info`` reads only the header).

The probe is deliberately tolerant: a malformed/undecodable file raises
:class:`AudioProbeError` (logged), which callers catch to fall back to
best-effort metadata rather than failing a whole dataset scan — mirrors
``app.core.video.probe.probe_video``.
"""

from __future__ import annotations

from pathlib import Path

import structlog

try:  # pragma: no cover - import guard
    from pydantic import BaseModel, ConfigDict
except Exception:  # pragma: no cover
    BaseModel = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)


class AudioProbeError(RuntimeError):
    """Raised when an audio file cannot be opened or probed."""


class AudioProbe(BaseModel):  # type: ignore[misc]
    """Immutable per-file probe result."""

    model_config = ConfigDict(frozen=True)

    duration_s: float
    sample_rate: int
    channels: int


def probe_audio(path: Path) -> AudioProbe:
    """Probe *path* and return an :class:`AudioProbe`.

    Raises:
        AudioProbeError: the file is missing, has no readable header, or
            is not a format libsndfile understands. The original error is
            chained and logged.
    """
    import soundfile as sf  # imported lazily so non-audio code paths never pull this in

    path = Path(path)
    try:
        info = sf.info(str(path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("audio_probe_failed", path=str(path), error=str(exc))
        raise AudioProbeError(f"failed to probe audio: {path}") from exc

    duration_s = float(info.duration) if info.duration is not None else 0.0
    sample_rate = int(info.samplerate or 0)
    channels = int(info.channels or 0)

    return AudioProbe(
        duration_s=duration_s,
        sample_rate=sample_rate,
        channels=channels,
    )
