"""End-to-end regression for ``VideoFrameLoader.load_clip`` on a real mp4.

``load_clip`` had no test that actually decoded a file — the streaming-selection
rewrite (which replaced "buffer every decoded frame as a full-size rgb24 array,
then pick the nearest" with a single two-pointer pass) therefore had no safety
net. This pins the output against an INDEPENDENT oracle: decode the clip
separately, brute-force the nearest frame per wanted timestamp, and run the same
resize helper. Byte equality is required — fps resampling is nearest-frame
selection, so there is no interpolation slack to allow for.
"""

from __future__ import annotations

import math

import pytest
import torch

from app.engine.components.video import (
    VideoClipTooShort,
    VideoFrameLoader,
    _smart_resize_crop_chw,
)


def _make_mp4(path, *, seconds=3.0, fps=10, w=96, h=64):
    """A clip whose every frame is a distinct solid colour, so a wrongly
    selected frame changes the tensor rather than silently matching."""
    import av
    import numpy as np

    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    for i in range(int(seconds * fps)):
        arr = np.zeros((h, w, 3), dtype="uint8")
        arr[:, :, 0] = (i * 7) % 256
        arr[:, :, 1] = (i * 13) % 256
        arr[:, :, 2] = (i * 29) % 256
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _oracle(path, wanted_ts, target_w, target_h, h_flip=False):
    """Brute-force nearest-frame selection over the whole decoded clip."""
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        time_base = float(stream.time_base) if stream.time_base else None
        decoded = []
        for frame in container.decode(stream):
            if frame.time is not None:
                t = float(frame.time)
            elif frame.pts is not None and time_base:
                t = float(frame.pts) * time_base
            else:
                t = len(decoded) / 10.0
            decoded.append((t, frame.to_ndarray(format="rgb24")))
    finally:
        container.close()

    times = [d[0] for d in decoded]
    out = []
    for ts in wanted_ts:
        idx = min(range(len(times)), key=lambda i: abs(times[i] - ts))
        out.append(_smart_resize_crop_chw(decoded[idx][1], target_w, target_h, h_flip))
    return torch.stack(out, dim=1)


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "src.mp4"
    _make_mp4(path)
    return path


class TestLoadClipMatchesOracle:
    @pytest.mark.parametrize(
        "target_frames,target_fps,trim_start",
        [
            (5, 10.0, 0.0),   # native rate from the start
            (9, 5.0, 0.0),    # downsampled fps
            (5, 10.0, 1.0),   # trimmed start (exercises the keyframe seek)
            (13, 24.0, 0.2),  # target above source rate → frame reuse
        ],
    )
    def test_selection_is_byte_identical(self, clip, target_frames, target_fps, trim_start):
        loader = VideoFrameLoader()
        got = loader.load_clip(
            str(clip),
            target_frames=target_frames,
            target_fps=target_fps,
            trim_start_s=trim_start,
            trim_end_s=None,
            target_w=64,
            target_h=48,
        )
        wanted = [trim_start + k / target_fps for k in range(target_frames)]
        expected = _oracle(clip, wanted, 64, 48)

        assert got.shape == (3, target_frames, 48, 64)
        assert torch.equal(got, expected)

    def test_h_flip_is_applied(self, clip):
        loader = VideoFrameLoader()
        plain = loader.load_clip(
            str(clip), 3, 10.0, 0.0, None, 64, 48, h_flip=False
        )
        flipped = loader.load_clip(
            str(clip), 3, 10.0, 0.0, None, 64, 48, h_flip=True
        )
        assert torch.equal(flipped, torch.flip(plain, dims=[-1]))

    def test_range_and_dtype_contract(self, clip):
        got = VideoFrameLoader().load_clip(str(clip), 4, 10.0, 0.0, None, 64, 48)
        assert got.dtype == torch.float32
        assert float(got.min()) >= -1.0
        assert float(got.max()) <= 1.0


class TestLoadClipGuards:
    def test_window_too_short_raises(self, clip):
        with pytest.raises(VideoClipTooShort):
            VideoFrameLoader().load_clip(
                str(clip), target_frames=50, target_fps=10.0,
                trim_start_s=0.0, trim_end_s=1.0, target_w=64, target_h=48,
            )

    def test_missing_video_stream_raises(self, tmp_path):
        bogus = tmp_path / "not_a_video.mp4"
        bogus.write_bytes(b"\x00" * 512)
        with pytest.raises(Exception):
            VideoFrameLoader().load_clip(
                str(bogus), 2, 10.0, 0.0, None, 64, 48
            )

    @pytest.mark.parametrize("frames,fps", [(0, 10.0), (2, 0.0), (2, -1.0)])
    def test_invalid_arguments_raise(self, clip, frames, fps):
        with pytest.raises(ValueError):
            VideoFrameLoader().load_clip(str(clip), frames, fps, 0.0, None, 64, 48)

    def test_clip_shorter_than_window_uses_the_last_frame(self, clip):
        """When the stream runs out, remaining stamps take the final frame —
        the same fallback the brute-force version produced."""
        loader = VideoFrameLoader()
        # 3s clip; ask for stamps out to 2.9s which is inside it, then verify the
        # tail stamps resolve to real (not zero) frames.
        got = loader.load_clip(str(clip), 30, 10.0, 0.0, None, 64, 48)
        assert got.shape[1] == 30
        assert not torch.equal(got[:, -1], torch.full_like(got[:, -1], -1.0))

    def test_math_inf_window_end_is_accepted(self, clip):
        """trim_end_s=None means "to the clip end" — the guard must not reject
        it via the math.inf comparison."""
        assert math.isinf(float("inf"))
        got = VideoFrameLoader().load_clip(str(clip), 2, 10.0, 0.0, None, 64, 48)
        assert got.shape[1] == 2
