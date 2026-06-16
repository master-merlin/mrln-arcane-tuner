"""Tests for VideoFrameLoader (decode/resample + encode round-trip).

CPU-only; a tiny counter-pattern mp4 is synthesized in tmp_path via PyAV.
"""

import math

import av
import numpy as np
import pytest
import torch

from app.engine.components.video import VideoClipTooShort, VideoFrameLoader


# ── Fixture: synthesize a deterministic counter-pattern clip ─────────────


def _make_counter_clip(path, n_frames=30, fps=30, w=64, h=64):
    """Write an mp4 where frame k is a flat field of intensity ~k.

    Returns the per-frame nominal intensity list so tests can map a decoded
    frame back to its source index via mean brightness.
    """
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    # Use a high-quality, near-lossless setting so the flat intensity survives
    # encoding well enough to recover the source frame index by brightness.
    stream.options = {"crf": "0", "preset": "veryfast"}

    intensities = []
    for k in range(n_frames):
        val = (k * 8) % 256  # spread across the range; distinct per frame
        intensities.append(val)
        arr = np.full((h, w, 3), val, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return intensities


@pytest.fixture
def counter_clip(tmp_path):
    path = tmp_path / "counter.mp4"
    intensities = _make_counter_clip(path)
    return str(path), intensities


def _frame_brightness_to_index(brightness_0_1, intensities):
    """Map a recovered [-1,1]->[0,255] brightness to the nearest source index."""
    val = (brightness_0_1 + 1.0) * 127.5
    return min(range(len(intensities)), key=lambda i: abs(intensities[i] - val))


# ── load_clip ────────────────────────────────────────────────────────────


class TestLoadClip:
    def test_output_shape_and_range(self, counter_clip):
        path, _ = counter_clip
        clip = VideoFrameLoader().load_clip(
            path,
            target_frames=9,
            target_fps=16,
            trim_start_s=0.0,
            trim_end_s=None,
            target_w=64,
            target_h=64,
        )
        assert clip.shape == (3, 9, 64, 64)
        assert clip.min() >= -1.0001 and clip.max() <= 1.0001

    def test_frame_index_selection(self, counter_clip):
        """16fps/9-frame resample of a 30fps clip selects ~30/16-spaced frames."""
        path, intensities = counter_clip
        clip = VideoFrameLoader().load_clip(
            path,
            target_frames=9,
            target_fps=16,
            trim_start_s=0.0,
            trim_end_s=None,
            target_w=64,
            target_h=64,
        )
        # Wanted timestamps: k/16 for k in 0..8 → nearest 30fps source frame is
        # round(t * 30) = round(k * 30/16). Recover indices via brightness.
        for k in range(9):
            mean_brightness = float(clip[:, k].mean())
            recovered = _frame_brightness_to_index(mean_brightness, intensities)
            expected = round(k * 30 / 16)
            # Allow ±1 source-frame slack (nearest-frame + codec rounding).
            assert abs(recovered - expected) <= 1, (
                f"frame {k}: recovered src {recovered}, expected ~{expected}"
            )

    def test_trim_window_respected(self, counter_clip):
        """A trim starting at 0.5s should begin near source frame 15 (30fps)."""
        path, intensities = counter_clip
        clip = VideoFrameLoader().load_clip(
            path,
            target_frames=4,
            target_fps=8,
            trim_start_s=0.5,
            trim_end_s=None,
            target_w=64,
            target_h=64,
        )
        first_brightness = float(clip[:, 0].mean())
        recovered = _frame_brightness_to_index(first_brightness, intensities)
        # 0.5s * 30fps = frame 15.
        assert abs(recovered - 15) <= 1

    def test_h_flip_flips_width(self, tmp_path):
        """h_flip mirrors along the width axis."""
        # A clip with a left/right asymmetric pattern so flip is observable.
        path = tmp_path / "asym.mp4"
        container = av.open(str(path), mode="w")
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        stream.options = {"crf": "0"}
        for _ in range(10):
            arr = np.zeros((64, 64, 3), dtype=np.uint8)
            arr[:, :32, :] = 255  # bright LEFT half
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for p in stream.encode(frame):
                container.mux(p)
        for p in stream.encode():
            container.mux(p)
        container.close()

        loader = VideoFrameLoader()
        normal = loader.load_clip(
            str(path),
            3,
            10,
            0.0,
            None,
            64,
            64,
            h_flip=False,
        )
        flipped = loader.load_clip(
            str(path),
            3,
            10,
            0.0,
            None,
            64,
            64,
            h_flip=True,
        )
        # Normal: left brighter than right. Flipped: right brighter than left.
        assert normal[:, 0, :, :16].mean() > normal[:, 0, :, 48:].mean()
        assert flipped[:, 0, :, 48:].mean() > flipped[:, 0, :, :16].mean()

    def test_too_short_raises(self, counter_clip):
        """Requesting more frames than the trimmed window can supply raises."""
        path, _ = counter_clip
        # Clip is 1s (30 frames @30fps). Ask for 100 frames @60fps from a 0.1s
        # window → impossible.
        with pytest.raises(VideoClipTooShort):
            VideoFrameLoader().load_clip(
                path,
                target_frames=100,
                target_fps=60,
                trim_start_s=0.0,
                trim_end_s=0.1,
                target_w=64,
                target_h=64,
            )


# ── encode_video round-trip ──────────────────────────────────────────────


class TestEncodeVideo:
    def test_roundtrip_frame_count_and_fps(self, tmp_path):
        """Encode 8 frames @16fps → probe back → 8 frames, fps ≈ 16."""
        out = tmp_path / "out.mp4"
        # [C, F, H, W] float in [-1, 1].
        frames = torch.rand(3, 8, 64, 64) * 2 - 1
        VideoFrameLoader().encode_video(frames, None, fps=16, out_path=str(out))
        assert out.exists()

        container = av.open(str(out))
        try:
            stream = container.streams.video[0]
            decoded = sum(1 for _ in container.decode(stream))
            fps = float(stream.average_rate)
        finally:
            container.close()
        assert decoded == 8
        assert math.isclose(fps, 16, rel_tol=0.05)

    def test_roundtrip_uint8_fhwc_input(self, tmp_path):
        """The alternate [F, H, W, C] uint8 input layout also round-trips."""
        out = tmp_path / "out_u8.mp4"
        frames = (torch.rand(6, 64, 64, 3) * 255).to(torch.uint8)
        VideoFrameLoader().encode_video(frames, None, fps=12, out_path=str(out))

        container = av.open(str(out))
        try:
            decoded = sum(1 for _ in container.decode(container.streams.video[0]))
        finally:
            container.close()
        assert decoded == 6

    def test_roundtrip_with_audio(self, tmp_path):
        """A waveform muxes an audio stream alongside the video."""
        out = tmp_path / "out_audio.mp4"
        frames = torch.rand(3, 8, 64, 64) * 2 - 1
        # 0.5s of mono noise at 44100 Hz.
        wav = torch.rand(22050) * 2 - 1
        VideoFrameLoader().encode_video(frames, wav, fps=16, out_path=str(out))

        container = av.open(str(out))
        try:
            assert len(container.streams.video) == 1
            assert len(container.streams.audio) == 1
        finally:
            container.close()

    def test_roundtrip_with_stereo_audio(self, tmp_path):
        """A 2-channel waveform muxes a STEREO AAC stream.

        Regression: LTX-2 samples produce stereo audio ([2, N]); the muxer fed
        that planar layout to a PACKED ``s16`` ``from_ndarray`` → "Expected packed
        array.shape[0] to equal 1 but got 2" (caught → silent mp4).
        """
        out = tmp_path / "out_stereo.mp4"
        frames = torch.rand(3, 8, 64, 64) * 2 - 1
        wav = torch.rand(2, 22050) * 2 - 1  # [2, N] stereo, 0.5s @ 44100
        VideoFrameLoader().encode_video(frames, wav, fps=16, out_path=str(out))

        container = av.open(str(out))
        try:
            assert len(container.streams.audio) == 1
            assert container.streams.audio[0].codec_context.layout.nb_channels == 2
        finally:
            container.close()
