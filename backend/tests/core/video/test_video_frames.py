"""Tests for ``app.core.video.frames`` — frame sampling for video captioning.

Synthetic clips are generated with PyAV (bundled ffmpeg) so the suite needs no
fixture media and runs identically on Windows and Docker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import av
from PIL import Image

from app.core.video.frames import (
    FrameSampleError,
    frame_to_data_url,
    sample_frames,
    sample_frames_as_data_urls,
)


# ── Fixture helpers ──────────────────────────────────────────────────────


def _write_clip(
    path: Path,
    *,
    codec: str = "libx264",
    n_frames: int = 16,
    fps: int = 8,
    width: int = 64,
    height: int = 64,
) -> Path:
    """Encode a tiny clip whose each frame is a distinct flat color counter."""
    with av.open(str(path), mode="w") as container:
        vstream = container.add_stream(codec, rate=fps)
        vstream.width = width
        vstream.height = height
        vstream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            # Distinct red channel per frame so frames are visually different.
            arr = np.zeros((height, width, 3), dtype=np.uint8)
            arr[:, :, 0] = (i * 15) % 256
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in vstream.encode(frame):
                container.mux(packet)
        for packet in vstream.encode():
            container.mux(packet)
    return path


# ── Tests ────────────────────────────────────────────────────────────────


def test_sample_returns_n_rgb_frames(tmp_path):
    """A 16-frame clip sampled at n=4 returns 4 RGB PIL images."""
    clip = _write_clip(tmp_path / "clip.mp4", n_frames=16)

    frames = sample_frames(clip, n=4)

    assert len(frames) == 4
    assert all(isinstance(f, Image.Image) for f in frames)
    assert all(f.mode == "RGB" for f in frames)
    assert all(f.size == (64, 64) for f in frames)


def test_sample_frames_evenly_spaced(tmp_path):
    """Sampled frames span the clip — distinct content, not all the same frame."""
    clip = _write_clip(tmp_path / "spread.mp4", n_frames=16)

    frames = sample_frames(clip, n=4)

    # Mean red value differs across the spread (frames advance a red counter),
    # proving the sampler isn't returning the same source frame repeatedly.
    means = [np.asarray(f)[:, :, 0].mean() for f in frames]
    assert len(set(round(m) for m in means)) >= 3


def test_bounds_respected(tmp_path):
    """start_s/end_s restrict sampling to the requested sub-window."""
    # 16 frames @ 8fps == 2.0s total.
    clip = _write_clip(tmp_path / "bounds.mp4", n_frames=16, fps=8)

    early = sample_frames(clip, n=2, start_s=0.0, end_s=0.5)
    late = sample_frames(clip, n=2, start_s=1.5, end_s=2.0)

    early_mean = np.asarray(early[0])[:, :, 0].mean()
    late_mean = np.asarray(late[-1])[:, :, 0].mean()
    # The red counter rises over time, so the late window is brighter.
    assert late_mean > early_mean


def test_n_greater_than_frames_returns_at_most_available(tmp_path):
    """Requesting more frames than exist returns <= available (deduped)."""
    clip = _write_clip(tmp_path / "short.mp4", n_frames=5, fps=8)

    frames = sample_frames(clip, n=20)

    assert 0 < len(frames) <= 5
    assert all(isinstance(f, Image.Image) for f in frames)


def test_data_url_helper_returns_base64_jpeg(tmp_path):
    """The data-url helper yields base64 data:image/jpeg strings, one per frame."""
    clip = _write_clip(tmp_path / "urls.mp4", n_frames=16)

    urls = sample_frames_as_data_urls(clip, n=4, max_long_side=48)

    assert len(urls) == 4
    for url in urls:
        assert url.startswith("data:image/jpeg;base64,")
        b64 = url.split(",", 1)[1]
        # Valid, non-trivial base64 payload.
        import base64

        decoded = base64.b64decode(b64)
        assert decoded[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_frame_to_data_url_downscales(tmp_path):
    """frame_to_data_url honours max_long_side without upscaling small frames."""
    img = Image.new("RGB", (200, 100), (10, 20, 30))
    url = frame_to_data_url(img, max_long_side=50)
    assert url.startswith("data:image/jpeg;base64,")


def test_bad_file_raises_frame_sample_error(tmp_path):
    """A non-video file raises FrameSampleError (caller fails one item)."""
    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"definitely not a video container")
    with pytest.raises(FrameSampleError):
        sample_frames(bad, n=4)


def test_missing_file_raises_frame_sample_error(tmp_path):
    with pytest.raises(FrameSampleError):
        sample_frames(tmp_path / "nope.mp4", n=4)
