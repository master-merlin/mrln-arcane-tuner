"""Video clip loading + encoding for the training/sampling data pipeline.

``VideoFrameLoader`` decodes a clip with PyAV, resamples it to a target frame
count at a target fps by nearest-frame selection (no re-encode), and applies the
SAME smart-resize + center-crop math the still-image path uses, yielding a
``[C, F, H, W]`` float tensor in ``[-1, 1]``.

It also provides ``encode_video`` for muxing a generated clip back to an mp4
(H.264 video, optional AAC audio) — used later by the sampler. Both directions
go through PyAV, which bundles its own ffmpeg, so behavior is identical on
Windows and in Docker.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import structlog
import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = structlog.get_logger(__name__)

# ── Module-level codec defaults ──────────────────────────────────────────
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_AUDIO_CODEC = "aac"
DEFAULT_PIXEL_FORMAT = "yuv420p"
# H.264 needs even dimensions for yuv420p; encode guards against odd W/H.


class VideoClipTooShort(RuntimeError):
    """Raised when a trimmed clip cannot supply the requested frames at fps.

    Bucketing is responsible for never asking for more frames than the clip
    can provide; this is a runtime guard that fails loudly rather than padding
    with duplicate/black frames.
    """


def _smart_resize_crop_chw(
    frame_hwc: "np.ndarray", target_w: int, target_h: int, h_flip: bool
) -> torch.Tensor:
    """Resize+center-crop one RGB frame to ``[3, target_h, target_w]`` in [-1,1].

    Mirrors the still-image path in ``pipeline_data._load_image_to``: scale by
    ``max(tw/w, th/h)`` (cover), then center-crop. PIL LANCZOS is used for the
    resample so the result matches the image pipeline byte-for-byte given the
    same source pixels.
    """
    import numpy as np
    from PIL import Image

    img = Image.fromarray(frame_hwc, mode="RGB")
    scale = max(target_w / img.width, target_h / img.height)
    nw, nh = int(img.width * scale), int(img.height * scale)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left, top = (nw - target_w) // 2, (nh - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))
    if h_flip:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # ToTensor + Normalize([0.5],[0.5]) == (x/255 - 0.5)/0.5 == x/127.5 - 1.
    t = torch.from_numpy(np.asarray(img, dtype="float32"))  # [H,W,3]
    t = t.permute(2, 0, 1).contiguous()  # [3, H, W]
    t = t / 127.5 - 1.0
    return t


class VideoFrameLoader:
    """Decode, resample, and tensorize video clips for training/sampling."""

    def load_clip(
        self,
        path: str,
        target_frames: int,
        target_fps: float,
        trim_start_s: float,
        trim_end_s: float | None,
        target_w: int,
        target_h: int,
        h_flip: bool = False,
    ) -> torch.Tensor:
        """Decode a clip and return ``[C, F, H, W]`` float in ``[-1, 1]``.

        Frames are selected at uniform target-fps timestamps inside the trim
        window: ``trim_start_s + k / target_fps`` for ``k`` in
        ``range(target_frames)``. For each timestamp the nearest decoded frame
        is used (fps resampling by nearest source-frame index — no re-encode).

        Args:
            path: Source video file.
            target_frames: Number of frames to emit (F).
            target_fps: Output framerate used to space the sample timestamps.
            trim_start_s: Start of the usable window, in seconds.
            trim_end_s: End of the usable window (``None`` → clip end).
            target_w / target_h: Output spatial size after resize+crop.
            h_flip: Horizontal flip (on width) when ``True``.

        Returns:
            ``torch.Tensor`` of shape ``[3, target_frames, target_h, target_w]``.

        Raises:
            VideoClipTooShort: the trim window cannot supply the requested
                frames at ``target_fps``.
        """
        import av

        if target_frames < 1:
            raise ValueError("target_frames must be >= 1")
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")

        # Desired sample timestamps within the trim window.
        wanted_ts = [trim_start_s + k / target_fps for k in range(target_frames)]
        window_end = trim_end_s if trim_end_s is not None else math.inf
        # The last sample must land inside the usable window. A tiny epsilon
        # absorbs float rounding on the boundary.
        if wanted_ts[-1] > window_end + 1e-6:
            raise VideoClipTooShort(
                f"clip window [{trim_start_s}, {trim_end_s}] cannot supply "
                f"{target_frames} frames @ {target_fps}fps (needs "
                f"{wanted_ts[-1]:.3f}s)"
            )

        container = av.open(str(path))
        try:
            if not container.streams.video:
                raise VideoClipTooShort(f"no video stream in {path}")
            stream = container.streams.video[0]
            time_base = float(stream.time_base) if stream.time_base else None

            # Seek near the first wanted timestamp to avoid decoding from 0 on
            # long clips. Seek is backward to a keyframe; we then walk forward.
            if time_base:
                try:
                    seek_pts = int(max(trim_start_s, 0.0) / time_base)
                    container.seek(
                        seek_pts, stream=stream, backward=True, any_frame=False
                    )
                except (OSError, ValueError):
                    pass

            decoded: list[tuple[float, "np.ndarray"]] = []
            max_wanted = wanted_ts[-1]
            for frame in container.decode(stream):
                # Frame presentation time in seconds (fallback to pts*time_base).
                if frame.time is not None:
                    t = float(frame.time)
                elif frame.pts is not None and time_base:
                    t = float(frame.pts) * time_base
                else:
                    t = len(decoded) / (target_fps or 1.0)
                decoded.append((t, frame.to_ndarray(format="rgb24")))
                # Stop once we have a frame at/after the last wanted timestamp.
                if t >= max_wanted:
                    break

            if not decoded:
                raise VideoClipTooShort(f"decoded 0 frames from {path}")

            times = [d[0] for d in decoded]
            frames_out: list[torch.Tensor] = []
            for ts in wanted_ts:
                idx = min(range(len(times)), key=lambda i: abs(times[i] - ts))
                frames_out.append(
                    _smart_resize_crop_chw(decoded[idx][1], target_w, target_h, h_flip)
                )

            clip = torch.stack(frames_out, dim=1)  # [3, F, H, W]
            logger.debug(
                "video_clip_loaded",
                path=str(path),
                frames=clip.shape[1],
                size=f"{target_w}x{target_h}",
                fps=target_fps,
            )
            return clip
        finally:
            try:
                container.close()
            except Exception:  # noqa: BLE001
                pass

    def encode_video(
        self,
        frames: torch.Tensor,
        audio_waveform_or_none,
        fps: float,
        out_path: str,
    ) -> str:
        """Encode frames to an mp4 (H.264; AAC audio when a waveform is given).

        Canonical input: ``frames`` is a ``[C, F, H, W]`` float tensor in
        ``[-1, 1]`` (the same layout ``load_clip`` returns). A ``[F, H, W, C]``
        ``uint8`` tensor is also accepted and converted internally.

        ``audio_waveform_or_none`` — when not ``None`` — is a 1-D or 2-D float
        tensor/array of samples in ``[-1, 1]`` muxed as an AAC stream at the
        same sample rate it was captured (assumed 44100 Hz unless it is a
        2-tuple ``(waveform, sample_rate)``).

        Returns the output path.
        """
        import av

        frames_u8 = self._to_fhwc_uint8(frames)  # [F, H, W, C] uint8
        f, h, w, _ = frames_u8.shape

        # yuv420p requires even dimensions.
        if w % 2 or h % 2:
            w -= w % 2
            h -= h % 2
            frames_u8 = frames_u8[:, :h, :w, :]

        out_fps = float(fps) if fps and fps > 0 else 1.0
        container = av.open(str(out_path), mode="w")
        try:
            vstream = container.add_stream(
                DEFAULT_VIDEO_CODEC, rate=round(out_fps) or 1
            )
            vstream.width = w
            vstream.height = h
            vstream.pix_fmt = DEFAULT_PIXEL_FORMAT

            # Register the audio stream BEFORE muxing any video packets. Adding a
            # stream after the muxer has written the header (first mux) leaves
            # its time_base unresolved and crashes the AAC muxer.
            astream = None
            if audio_waveform_or_none is not None:
                astream = self._add_audio_stream(container, audio_waveform_or_none)

            for i in range(f):
                av_frame = av.VideoFrame.from_ndarray(frames_u8[i], format="rgb24")
                for packet in vstream.encode(av_frame):
                    container.mux(packet)
            for packet in vstream.encode():  # flush
                container.mux(packet)

            if astream is not None:
                self._encode_audio(container, astream, audio_waveform_or_none)

            logger.debug(
                "video_encoded",
                path=str(out_path),
                frames=f,
                size=f"{w}x{h}",
                fps=out_fps,
            )
        finally:
            try:
                container.close()
            except Exception:  # noqa: BLE001
                pass
        return str(out_path)

    @staticmethod
    def _to_fhwc_uint8(frames: torch.Tensor) -> "np.ndarray":
        """Normalize the two accepted input layouts to ``[F, H, W, C]`` uint8."""
        import numpy as np

        if not isinstance(frames, torch.Tensor):
            frames = torch.as_tensor(frames)

        if frames.dtype == torch.uint8:
            # Assumed [F, H, W, C] uint8.
            arr = frames.detach().cpu().numpy()
        else:
            # Assumed [C, F, H, W] float in [-1, 1] → [F, H, W, C] uint8.
            t = frames.detach().cpu().float()
            if t.ndim != 4:
                raise ValueError(f"expected 4D frames, got shape {tuple(t.shape)}")
            t = ((t.clamp(-1.0, 1.0) + 1.0) * 127.5).round()
            t = t.permute(1, 2, 3, 0).contiguous()  # [F, H, W, C]
            arr = t.numpy().astype(np.uint8)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _audio_sample_rate(waveform) -> int:
        """Extract the sample rate from a waveform-or-(waveform, sr) input."""
        if isinstance(waveform, tuple) and len(waveform) == 2:
            return int(waveform[1])
        return 44100

    @staticmethod
    def _audio_channels(waveform) -> int:
        """Channel count of a waveform-or-(waveform, sr) input (1-D → mono)."""
        if isinstance(waveform, tuple) and len(waveform) == 2:
            waveform = waveform[0]
        t = torch.as_tensor(waveform)
        return int(t.shape[0]) if t.ndim >= 2 else 1

    @classmethod
    def _add_audio_stream(cls, container, waveform):
        """Register an AAC stream with a resolved time_base + channel layout."""
        from fractions import Fraction

        sample_rate = cls._audio_sample_rate(waveform)
        astream = container.add_stream(DEFAULT_AUDIO_CODEC, rate=sample_rate)
        # The stream's own time_base must be set before any mux or PyAV raises
        # "Cannot rebase to zero time" (it stays unset until header write).
        astream.time_base = Fraction(1, sample_rate)
        # Match the encoder's channel layout to the waveform (stereo for LTX-2
        # audio samples); otherwise a stereo frame meets a default-mono encoder.
        try:
            astream.layout = "stereo" if cls._audio_channels(waveform) >= 2 else "mono"
        except (AttributeError, ValueError):  # older PyAV — default layout stands
            pass
        return astream

    @classmethod
    def _encode_audio(cls, container, astream, waveform) -> None:
        """Encode + mux a float waveform in [-1, 1] into ``astream``.

        Samples are chunked to the AAC frame size (1024) with explicit,
        monotonically increasing ``pts``.
        """
        import av
        from fractions import Fraction

        import numpy as np

        sample_rate = cls._audio_sample_rate(waveform)
        if isinstance(waveform, tuple) and len(waveform) == 2:
            waveform = waveform[0]

        wav = torch.as_tensor(waveform).detach().cpu().float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)  # [1, N] mono
        n_ch = wav.shape[0]
        samples = (wav.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16).numpy()
        samples = np.ascontiguousarray(samples)

        layout = "mono" if n_ch == 1 else "stereo"
        time_base = Fraction(1, sample_rate)

        chunk = 1024  # AAC frame size (samples PER CHANNEL)
        total = samples.shape[1]
        pts = 0
        for start in range(0, total, chunk):
            block = samples[:, start : start + chunk]  # [n_ch, frames]
            frames_in_block = block.shape[1]
            # ``s16`` is a PACKED format → PyAV wants interleaved [1, frames*n_ch]
            # (L,R,L,R,…). Passing the planar [n_ch, frames] raised "Expected
            # packed array.shape[0] to equal 1 but got 2" for stereo audio.
            interleaved = np.ascontiguousarray(block.T.reshape(1, -1))
            aframe = av.AudioFrame.from_ndarray(interleaved, format="s16", layout=layout)
            aframe.sample_rate = sample_rate
            aframe.pts = pts
            aframe.time_base = time_base
            pts += frames_in_block
            for packet in astream.encode(aframe):
                container.mux(packet)
        for packet in astream.encode():  # flush
            container.mux(packet)
