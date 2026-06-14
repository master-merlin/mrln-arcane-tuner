"""Video utilities for the dataset layer.

Pure, side-effect-free helpers for probing video clips (framerate,
duration, frame count, audio presence, codec) used by the scanner to
populate per-clip metadata. PyAV-backed (bundled ffmpeg), so this works
identically on Windows and inside the Docker image.
"""

from __future__ import annotations

from app.core.video.probe import VideoProbe, VideoProbeError, probe_video

__all__ = ["VideoProbe", "VideoProbeError", "probe_video"]
