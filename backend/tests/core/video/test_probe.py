"""Tests for ``app.core.video.probe.probe_video``.

Synthetic clips are generated with PyAV (which bundles ffmpeg) so the suite
needs no fixture media on disk and runs identically on Windows and Docker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import av

from app.core.video import VideoProbe, VideoProbeError, probe_video


# ── Fixture helpers ──────────────────────────────────────────────────────


def _write_clip(
    path: Path,
    *,
    codec: str = "libx264",
    n_frames: int = 12,
    fps: int = 24,
    width: int = 64,
    height: int = 64,
    with_audio: bool = False,
) -> Path:
    """Encode a tiny solid-color clip; optionally add a silent audio track.

    Returns *path* for convenience. Each video frame is a flat color whose
    value advances per frame (a cheap visible counter).
    """
    with av.open(str(path), mode="w") as container:
        vstream = container.add_stream(codec, rate=fps)
        vstream.width = width
        vstream.height = height
        vstream.pix_fmt = "yuv420p"

        astream = None
        sample_rate = 44100
        if with_audio:
            astream = container.add_stream("aac", rate=sample_rate)

        for i in range(n_frames):
            arr = np.full((height, width, 3), (i * 20) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in vstream.encode(frame):
                container.mux(packet)

        if astream is not None:
            # ~0.5s of silence (matches 12 frames @ 24fps).
            n_samples = sample_rate // 2
            samples = np.zeros((1, n_samples), dtype=np.int16)
            aframe = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
            aframe.sample_rate = sample_rate
            aframe.pts = 0
            for packet in astream.encode(aframe):
                container.mux(packet)
            for packet in astream.encode():
                container.mux(packet)

        for packet in vstream.encode():
            container.mux(packet)

    return path


# ── Tests ────────────────────────────────────────────────────────────────


def test_probe_basic_mp4_no_audio(tmp_path):
    """A 12-frame 24fps 64x64 h264 clip probes to exact metadata, no audio."""
    clip = _write_clip(tmp_path / "clip.mp4")

    probe = probe_video(clip)

    assert isinstance(probe, VideoProbe)
    assert probe.width == 64
    assert probe.height == 64
    assert probe.fps == pytest.approx(24.0, abs=0.1)
    # h264 carries an exact frame count → not estimated.
    assert probe.frame_count == 12
    assert probe.frame_count_estimated is False
    assert probe.duration_s == pytest.approx(0.5, abs=0.05)
    assert probe.has_audio is False
    assert probe.video_codec is not None
    assert "h264" in probe.video_codec


def test_probe_mp4_with_audio(tmp_path):
    """A clip with a silent AAC track reports has_audio True."""
    clip = _write_clip(tmp_path / "with_audio.mp4", with_audio=True)

    probe = probe_video(clip)

    assert probe.has_audio is True
    assert probe.width == 64
    assert probe.height == 64
    assert "h264" in (probe.video_codec or "")


def test_probe_webm_estimated_frame_count(tmp_path):
    """A VP9 webm exposes no exact frame count, so it is estimated from
    duration × fps (round(0.5 * 24) == 12) and flagged accordingly."""
    clip = _write_clip(tmp_path / "clip.webm", codec="libvpx-vp9")

    probe = probe_video(clip)

    assert probe.video_codec == "vp9"
    assert probe.fps == pytest.approx(24.0, abs=0.1)
    assert probe.duration_s == pytest.approx(0.5, abs=0.05)
    assert probe.frame_count_estimated is True
    assert probe.frame_count == 12


def test_probe_is_immutable(tmp_path):
    """VideoProbe is frozen — fields can't be reassigned after construction."""
    probe = probe_video(_write_clip(tmp_path / "frozen.mp4"))
    with pytest.raises(Exception):
        probe.fps = 99.0  # type: ignore[misc]


def test_probe_bad_file_raises_probe_error(tmp_path):
    """A non-video file raises VideoProbeError (callers fall back gracefully)."""
    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"this is not a valid video container")

    with pytest.raises(VideoProbeError):
        probe_video(bad)


def test_probe_missing_file_raises_probe_error(tmp_path):
    """A missing path raises VideoProbeError rather than a bare OSError."""
    with pytest.raises(VideoProbeError):
        probe_video(tmp_path / "does_not_exist.mp4")
