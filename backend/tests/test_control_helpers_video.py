"""Tests for video control pairs in the dataset layer — Task BR0.

Bernini-R (video-EDIT) pairs a target video with a stem-matched control
VIDEO in ``control/``, exactly like the existing image-control convention.
Covers:
- ``CONTROL_MEDIA_EXTS`` = ``CONTROL_IMAGE_EXTS`` + the four video containers
- ``detect_control_slots`` probing a video control (width/height/num_frames/fps)
- Image-only regression pin: byte-identical results to pre-BR0 behavior
- Ext priority: an image control still wins over a same-stem video control
"""

from __future__ import annotations

import os

import av
import numpy as np
import pytest
from PIL import Image

from app.core.dataset.control_helpers import (
    CONTROL_IMAGE_EXTS,
    CONTROL_MEDIA_EXTS,
    CONTROL_SLOTS,
    detect_control_slots,
    list_control_stem_maps,
)


# ── Fixture helpers ──────────────────────────────────────────────────────


def _create_image(path: str, width: int = 64, height: int = 64, color: str = "red"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (width, height), color).save(path)


def _create_video(
    path: str, *, n_frames: int = 12, fps: int = 24, width: int = 32, height: int = 24,
):
    """Encode a tiny solid-color h264 mp4 — mirrors ``test_probe._write_clip``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with av.open(path, mode="w") as container:
        vstream = container.add_stream("libx264", rate=fps)
        vstream.width = width
        vstream.height = height
        vstream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            arr = np.full((height, width, 3), (i * 20) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in vstream.encode(frame):
                container.mux(packet)
        for packet in vstream.encode():
            container.mux(packet)
    return path


# ── CONTROL_MEDIA_EXTS ────────────────────────────────────────────────────


def test_control_media_exts_is_image_exts_plus_video_containers():
    assert CONTROL_MEDIA_EXTS == CONTROL_IMAGE_EXTS + (".mp4", ".webm", ".mkv", ".mov")


# ── Video control detection ──────────────────────────────────────────────


class TestVideoControlDetection:
    def test_detects_video_control_with_dims_and_frame_info(self, tmp_path):
        ds = str(tmp_path / "ds")
        _create_video(
            os.path.join(ds, "control", "clip.mp4"),
            n_frames=12, fps=24, width=32, height=24,
        )
        slots = detect_control_slots(ds, "clip")
        assert list(slots.keys()) == ["control"]
        slot = slots["control"]
        assert slot["rel_path"] == "control/clip.mp4"
        assert slot["width"] == 32
        assert slot["height"] == 24
        assert slot["num_frames"] == 12
        assert slot["fps"] == pytest.approx(24.0, abs=0.5)

    def test_unreadable_video_falls_back_to_zeros(self, tmp_path):
        ds = str(tmp_path / "ds")
        bad = os.path.join(ds, "control", "clip.mp4")
        os.makedirs(os.path.dirname(bad), exist_ok=True)
        with open(bad, "wb") as f:
            f.write(b"not a real video")
        slots = detect_control_slots(ds, "clip")
        slot = slots["control"]
        assert slot["rel_path"] == "control/clip.mp4"
        assert slot["width"] == 0
        assert slot["height"] == 0
        assert slot["num_frames"] == 0
        assert slot["fps"] == 0.0

    def test_multiple_video_containers_recognized(self, tmp_path):
        ds = str(tmp_path / "ds")
        for stem, ext in (("a", ".webm"), ("b", ".mkv"), ("c", ".mov")):
            # mkv/mov re-use the mp4 muxer's payload via a container rename —
            # cheap enough for a probe smoke test; the extension drives detection.
            _create_video(os.path.join(ds, "control", f"{stem}.mp4"))
            os.rename(
                os.path.join(ds, "control", f"{stem}.mp4"),
                os.path.join(ds, "control", f"{stem}{ext}"),
            )
        for stem, ext in (("a", ".webm"), ("b", ".mkv"), ("c", ".mov")):
            slots = detect_control_slots(ds, stem)
            assert slots["control"]["rel_path"] == f"control/{stem}{ext}"

    def test_image_control_wins_ext_priority_over_video(self, tmp_path):
        ds = str(tmp_path / "ds")
        _create_video(os.path.join(ds, "control", "clip.mp4"))
        _create_image(os.path.join(ds, "control", "clip.jpg"))
        slots = detect_control_slots(ds, "clip")
        assert slots["control"]["rel_path"] == "control/clip.jpg"
        assert "num_frames" not in slots["control"]

    def test_mixed_image_and_video_slots_across_stems(self, tmp_path):
        ds = str(tmp_path / "ds")
        _create_image(os.path.join(ds, "control", "img1.jpg"))
        _create_video(os.path.join(ds, "control", "clip1.mp4"))
        img_slots = detect_control_slots(ds, "img1")
        vid_slots = detect_control_slots(ds, "clip1")
        assert img_slots["control"]["rel_path"] == "control/img1.jpg"
        assert vid_slots["control"]["rel_path"] == "control/clip1.mp4"


# ── Regression pin: image-only datasets are byte-identical ──────────────


class TestImageOnlyRegressionPin:
    """detect_control_slots on an image-only dataset must return exactly the
    same shape as before BR0 — no num_frames/fps keys leaking onto image
    entries, same slot ordering, same values."""

    def test_image_entry_shape_unchanged(self, tmp_path):
        ds = str(tmp_path / "ds")
        _create_image(os.path.join(ds, "control", "img1.jpg"), 32, 48)
        slots = detect_control_slots(ds, "img1")
        assert slots == {
            "control": {"rel_path": "control/img1.jpg", "width": 32, "height": 48},
        }

    def test_multiple_image_slots_in_order_unchanged(self, tmp_path):
        ds = str(tmp_path / "ds")
        _create_image(os.path.join(ds, "control", "img1.jpg"))
        _create_image(os.path.join(ds, "control_2", "img1.png"))
        _create_image(os.path.join(ds, "control_3", "img1.webp"))
        slots = detect_control_slots(ds, "img1")
        assert list(slots.keys()) == list(CONTROL_SLOTS)
        for slot in slots.values():
            assert set(slot.keys()) == {"rel_path", "width", "height"}

    def test_no_controls_still_empty(self, tmp_path):
        ds = str(tmp_path / "ds")
        os.makedirs(ds, exist_ok=True)
        assert detect_control_slots(ds, "nothing") == {}


# ── list_control_stem_maps recognizes video files (paired_count contract) ─


class TestListControlStemMapsVideo:
    def test_video_stem_included_in_bulk_scan(self, tmp_path):
        ds = str(tmp_path / "ds")
        _create_video(os.path.join(ds, "control", "clip1.mp4"))
        stem_maps = list_control_stem_maps(ds)
        assert stem_maps["control"] == {"clip1": "clip1.mp4"}

    def test_mixed_image_and_video_stems_in_bulk_scan(self, tmp_path):
        ds = str(tmp_path / "ds")
        _create_image(os.path.join(ds, "control", "img1.jpg"))
        _create_video(os.path.join(ds, "control", "clip1.mp4"))
        stem_maps = list_control_stem_maps(ds)
        assert stem_maps["control"] == {"img1": "img1.jpg", "clip1": "clip1.mp4"}
