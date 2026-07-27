"""Task BR1 — control-VIDEO latents through the training data pipeline.

Bernini-R (video-EDIT) pairs a target video with a stem-matched control
video (BR0 dataset layer). This exercises the ``PipelineDataMixin`` batch
methods directly (drives ``_get_batch`` + ``_load_control_latents`` with a
real ``LatentManager`` + a fake 5D VAE, mirroring ``test_latents_video.py``)
so no GPU / real model is needed:

* A video control at least as long as the target's frame window encodes to
  a 5D latent whose frame axis matches the target's own (``F_lat``).
* A video control SHORTER than the target's window gets padded (not
  errored) to the same frame count.
* An image control (Kontext/Qwen-Edit today) is unaffected — regression pin
  against the pre-BR1 4D cache-key/latent-manager call contract.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import av
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from app.engine.components.latents import LatentManager
from app.engine.core.pipeline.pipeline_data import PipelineDataMixin
from app.engine.core.video_contract import frame_predicate


# ── Fixture builders ──────────────────────────────────────────────────────


def _img(path: str, w: int = 16, h: int = 16, color: str = "red") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (w, h), color).save(path)


def _make_clip(path: str, n_frames: int, fps: int, w: int = 32, h: int = 32) -> None:
    """Write a flat-field mp4 (pixel content is irrelevant to these tests)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    stream.options = {"crf": "0", "preset": "veryfast"}
    for k in range(n_frames):
        arr = np.full((h, w, 3), (k * 8) % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


# ── Fake video VAE (mirrors test_latents_video.FakeWanVAE) ────────────────


class FakeWanVAE(nn.Module):
    """Spatial 8x, temporal 4x: latent_f = (F-1)//4 + 1. Class NAME matters."""

    def __init__(self) -> None:
        super().__init__()
        self.config = MagicMock()
        self.config.scaling_factor = 1.0
        self.config.shift_factor = None
        self.config.latents_mean = None
        self.config.latents_std = None

    @property
    def dtype(self):
        return torch.float32

    def encode(self, x):
        b, c, f, h, w = x.shape
        lf = (f - 1) // 4 + 1
        out = torch.randn(b, 16, lf, h // 8, w // 8)
        result = MagicMock()
        result.latent_dist.sample.return_value = out
        return result


FakeWanVAE.__name__ = "AutoencoderKLWan"
FakeWanVAE.__qualname__ = "AutoencoderKLWan"


class MockImageVAE(nn.Module):
    """Minimal still-image VAE [B,3,H,W] -> [B,4,H//8,W//8]."""

    def __init__(self) -> None:
        super().__init__()
        self.config = MagicMock()
        self.config.scaling_factor = 0.18215
        self.config.shift_factor = None
        self.config.latents_mean = None
        self.config.latents_std = None

    @property
    def dtype(self):
        return torch.float32

    def encode(self, x):
        b, c, h, w = x.shape
        out = torch.randn(b, 4, h // 8, w // 8)
        result = MagicMock()
        result.latent_dist.sample.return_value = out
        return result


class _Harness(PipelineDataMixin):
    """Minimal object exposing the batch methods under test."""

    def __init__(self, latent_manager, frame_rule=None):
        self.device = torch.device("cpu")
        self.autocast_dtype = torch.float32
        self.config = {"cache_latents": True}
        self.latent_manager = latent_manager
        self._video_frame_rule = frame_rule

    def build_batch_extra(self, items):
        return {}


# ── Item builders ──────────────────────────────────────────────────────────


def _image_edit_item(ds: str, stem: str) -> dict:
    """A plain Kontext-style item: still target + still control (pre-BR1)."""
    _img(os.path.join(ds, f"{stem}.png"), color="red")
    rel = f"control/{stem}.jpg"
    _img(os.path.join(ds, rel), color="blue")
    return {
        "path": os.path.join(ds, f"{stem}.png"),
        "id": f"{stem}.png",
        "caption": "make it watercolor",
        "prefix": "",
        "dropout_rate": 0.0,
        "use_captions": True,
        "use_model_aware_captions": False,
        "target_w": 16,
        "target_h": 16,
        "cache_dir": os.path.join(ds, "cache", "original", "16x16"),
        "variant": "original",
        "has_masked": False,
        "control_rel_paths": [rel],
        "control_paths": [os.path.join(ds, rel)],
        "control_dims": [(16, 16)],
        "control_variants": ["control"],
        "control_cache_dirs": [os.path.join(ds, "cache", "control", "16x16")],
        # No "control_is_video" key at all — matches every pre-BR1 item dict.
    }


def _video_edit_item(
    ds: str,
    stem: str,
    target_frames: int,
    target_fps: float,
    control_frames: int,
    control_fps: int | None = None,
) -> dict:
    """A Bernini-R item: video target + video control, stem-paired."""
    control_fps = control_fps or int(target_fps)
    target_path = os.path.join(ds, f"{stem}.mp4")
    _make_clip(target_path, n_frames=target_frames, fps=int(target_fps))
    rel = f"control/{stem}.mp4"
    _make_clip(os.path.join(ds, rel), n_frames=control_frames, fps=control_fps)
    return {
        "path": target_path,
        "id": f"{stem}.mp4",
        "caption": "make the sky stormy",
        "prefix": "",
        "dropout_rate": 0.0,
        "use_captions": True,
        "use_model_aware_captions": False,
        "target_w": 32,
        "target_h": 32,
        "target_frames": target_frames,
        "target_fps": target_fps,
        "trim_start_s": 0.0,
        "trim_end_s": None,
        "cache_dir": os.path.join(ds, "cache", "original", "32x32xTf"),
        "variant": "original",
        "is_video": True,
        "has_masked": False,
        "control_rel_paths": [rel],
        "control_paths": [os.path.join(ds, rel)],
        "control_dims": [(32, 32)],
        "control_variants": ["control"],
        "control_cache_dirs": [os.path.join(ds, "cache", "control", "32x32")],
        "control_is_video": [True],
    }


# ── 5D control-video latents ────────────────────────────────────────────────


class TestControlVideoLatents5D:
    def test_long_control_trims_to_target_frame_count(self, tmp_path):
        """Control longer than the target window -> F_lat == target F_lat."""
        ds = str(tmp_path / "ds")
        lm = LatentManager(FakeWanVAE(), device="cpu")
        h = _Harness(lm, frame_rule="4n+1")
        # 4n+1: target 13 frames @8fps; control clip is much longer (30f).
        item = _video_edit_item(
            ds,
            "clip_a",
            target_frames=13,
            target_fps=8.0,
            control_frames=30,
        )
        assert frame_predicate(h._video_frame_rule)(item["target_frames"])

        batch = h._get_batch([item])
        h._load_control_latents(batch)

        control_lat = batch["control_latents"][0]
        target_f_lat = LatentManager.latent_frames(item["target_frames"], 4)
        assert control_lat.ndim == 5
        assert control_lat.shape[2] == target_f_lat  # [B, C, F_lat, H_lat, W_lat]

    def test_short_control_pads_not_errors(self, tmp_path):
        """A control clip shorter than the target's window is padded, not errored."""
        ds = str(tmp_path / "ds")
        lm = LatentManager(FakeWanVAE(), device="cpu")
        h = _Harness(lm, frame_rule="4n+1")
        # Target wants 13 frames @8fps (1.5s); control clip supplies only 5
        # frames @8fps (~0.625s) — too short to satisfy load_clip directly.
        item = _video_edit_item(
            ds,
            "clip_b",
            target_frames=13,
            target_fps=8.0,
            control_frames=5,
        )

        batch = h._get_batch([item])
        # Must not raise VideoClipTooShort.
        h._load_control_latents(batch)

        control_lat = batch["control_latents"][0]
        target_f_lat = LatentManager.latent_frames(item["target_frames"], 4)
        assert control_lat.ndim == 5
        assert control_lat.shape[2] == target_f_lat

    def test_control_clip_decodes_to_exact_target_frame_count(self, tmp_path):
        """Sanity check one level below the VAE: the decoded pixel clip
        itself already carries exactly the target's frame count, for both
        the long (trim) and short (pad) cases."""
        ds = str(tmp_path / "ds")
        h = _Harness(LatentManager(FakeWanVAE(), device="cpu"))

        long_clip = h._load_control_video_clip(
            _make_and_return(ds, "long", n_frames=30, fps=8),
            target_frames=13,
            target_fps=8.0,
            target_w=32,
            target_h=32,
        )
        assert long_clip.shape == (3, 13, 32, 32)

        short_clip = h._load_control_video_clip(
            _make_and_return(ds, "short", n_frames=5, fps=8),
            target_frames=13,
            target_fps=8.0,
            target_w=32,
            target_h=32,
        )
        assert short_clip.shape == (3, 13, 32, 32)
        # Padded tail repeats the last real decoded frame.
        assert torch.equal(short_clip[:, -1], short_clip[:, -2])

    def test_cache_key_varies_with_target_frame_window(self, tmp_path):
        """Two batches with the same control source but different target
        frame/fps windows must not collide in the content-addressed cache
        (mirrors the target video t{start}-{end} convention)."""
        ds = str(tmp_path / "ds")
        lm = LatentManager(FakeWanVAE(), device="cpu")
        h = _Harness(lm)

        item_a = _video_edit_item(
            ds,
            "same_src",
            target_frames=13,
            target_fps=8.0,
            control_frames=30,
        )
        # Reuse literally the same control file under a different target window.
        item_b = dict(item_a)
        item_b["target_frames"] = 9
        item_b["id"] = "same_src_b.mp4"
        item_b["path"] = item_a["path"]

        batch_a = h._get_batch([item_a])
        h._load_control_latents(batch_a)
        batch_b = h._get_batch([item_b])
        h._load_control_latents(batch_b)

        assert (
            batch_a["control_latents"][0].shape[2]
            != batch_b["control_latents"][0].shape[2]
        )
        # Different cache-key discriminators for the two windows.
        assert batch_a["control_extra_keys"] != batch_b["control_extra_keys"]


def _make_and_return(ds: str, stem: str, n_frames: int, fps: int) -> str:
    path = os.path.join(ds, "control", f"{stem}.mp4")
    _make_clip(path, n_frames=n_frames, fps=fps)
    return path


# ── Finding 2: 4n+1/invariant guard is enforced in production code ─────────


class TestControlVideoFrameCountInvariant:
    def test_control_clip_frame_count_mismatch_raises(self, monkeypatch, tmp_path):
        """If the clip loader ever stops honoring ``target_frames`` exactly,
        ``_load_control_video_clip`` must fail loudly instead of silently
        caching a mismatched control latent (this invariant previously lived
        only in test assertions, never in production code)."""
        from app.engine.components import video as video_mod

        def _bad_load_clip(
            self,
            path,
            target_frames,
            target_fps,
            trim_start_s,
            trim_end_s,
            target_w,
            target_h,
            h_flip=False,
        ):
            return torch.zeros(3, target_frames + 2, target_h, target_w)

        monkeypatch.setattr(video_mod.VideoFrameLoader, "load_clip", _bad_load_clip)

        h = _Harness(LatentManager(FakeWanVAE(), device="cpu"))
        ds = str(tmp_path / "ds")
        path = _make_and_return(ds, "mismatch", n_frames=13, fps=8)

        with pytest.raises(ValueError, match="frame"):
            h._load_control_video_clip(
                path, target_frames=13, target_fps=8.0, target_w=32, target_h=32
            )


# ── Finding 1: sliding temporal coverage + video control is unsupported ────


class TestSlidingControlVideoGuard:
    def test_sliding_target_with_video_control_raises(self, tmp_path):
        """v1 scope: a target cached with ``temporal_mode="sliding"`` holds
        the FULL clip while the control decode targets the per-step window
        (``target_frames``) — a frame-axis mismatch downstream. Must fail
        loudly instead of silently caching mismatched frames."""
        ds = str(tmp_path / "ds")
        lm = LatentManager(FakeWanVAE(), device="cpu")
        h = _Harness(lm)
        item = _video_edit_item(
            ds,
            "clip_slide",
            target_frames=13,
            target_fps=8.0,
            control_frames=13,
        )
        item["temporal_mode"] = "sliding"
        item["cache_frames"] = 61

        with pytest.raises(ValueError, match="sliding"):
            h._attach_control_images({}, [item], None)


# ── Finding 3: still-image target + video control is a diagnostic error ────


class TestStillTargetVideoControlGuard:
    def test_image_target_with_video_control_raises(self, tmp_path):
        """A still-image target paired with a video control slot (e.g. a
        pre-existing image-edit dataset with a same-stem video accidentally
        dropped into control/) must raise a diagnostic error naming the
        dataset item, not die inside VideoFrameLoader with a generic
        'target_fps must be > 0'."""
        ds = str(tmp_path / "ds")
        lm = LatentManager(MockImageVAE(), device="cpu")
        h = _Harness(lm)
        item = _image_edit_item(ds, "a")
        item["control_is_video"] = [True]

        with pytest.raises(ValueError, match="video target"):
            h._attach_control_images({}, [item], None)


# ── Final-review F3: control decode honors the target's temporal window ────


class TestControlTrimWindowAlignment:
    """A control clip must be decoded from the SAME temporal window as its
    paired target — both a user-trimmed target (nonzero ``trim_start_s``) and
    tiled sub-windows (``temporal_coverage='tiled'`` overwrites the per-window
    bounds). Otherwise window k>0 trains target segment [kΔ,(k+1)Δ] against
    control segment [0,Δ], and their cache keys collide."""

    @staticmethod
    def _spy_load_clip(seen: list):
        def _spy(
            self,
            path,
            target_frames,
            target_fps,
            trim_start_s,
            trim_end_s,
            target_w,
            target_h,
            h_flip=False,
        ):
            seen.append(trim_start_s)
            return torch.zeros(3, target_frames, target_h, target_w)

        return _spy

    def test_target_trim_start_flows_to_control_and_varies_cache_key(
        self, monkeypatch, tmp_path
    ):
        from app.engine.components import video as video_mod

        seen: list = []
        monkeypatch.setattr(
            video_mod.VideoFrameLoader, "load_clip", self._spy_load_clip(seen)
        )
        ds = str(tmp_path / "ds")
        h = _Harness(LatentManager(FakeWanVAE(), device="cpu"))

        head = _video_edit_item(
            ds, "clip_head", target_frames=13, target_fps=8.0, control_frames=13
        )
        trimmed = _video_edit_item(
            ds, "clip_trim", target_frames=13, target_fps=8.0, control_frames=13
        )
        trimmed["trim_start_s"] = 0.5

        batch_head: dict = {}
        h._attach_control_images(batch_head, [head], None)
        batch_trim: dict = {}
        h._attach_control_images(batch_trim, [trimmed], None)

        # The control decode honored each target's OWN start offset.
        assert seen == [0.0, 0.5]
        # ...and the cache-key discriminator differs so a trimmed window never
        # collides with the head window of the same source.
        assert batch_head["control_extra_keys"] != batch_trim["control_extra_keys"]

    def test_two_tiled_windows_same_clip_differ(self, monkeypatch, tmp_path):
        """Two tiled windows of the SAME control source get different decode
        start offsets AND different cache keys (else they overwrite each other)."""
        from app.engine.components import video as video_mod

        seen: list = []
        monkeypatch.setattr(
            video_mod.VideoFrameLoader, "load_clip", self._spy_load_clip(seen)
        )
        ds = str(tmp_path / "ds")
        h = _Harness(LatentManager(FakeWanVAE(), device="cpu"))

        win0 = _video_edit_item(
            ds, "tiled_src", target_frames=13, target_fps=8.0, control_frames=40
        )
        # Window k=1 reuses literally the same control source file.
        win1 = dict(win0)
        win1["id"] = "tiled_src_w1.mp4"
        win1["trim_start_s"] = 1.625  # 13/8 s after window 0
        win1["trim_end_s"] = 3.25

        b0: dict = {}
        h._attach_control_images(b0, [win0], None)
        b1: dict = {}
        h._attach_control_images(b1, [win1], None)

        assert seen == [0.0, 1.625]  # distinct decode start offsets
        assert b0["control_extra_keys"] != b1["control_extra_keys"]  # distinct keys


# ── Image-control regression pin (byte-identical to pre-BR1) ──────────────


class TestImageControlRegressionPin:
    """An image control must be completely unaffected by the BR1 video branch."""

    def test_no_control_is_video_key_stays_on_image_path(self, tmp_path):
        ds = str(tmp_path / "ds")
        lm = LatentManager(MockImageVAE(), device="cpu")
        h = _Harness(lm)
        item = _image_edit_item(ds, "a")
        assert "control_is_video" not in item

        batch = h._get_batch([item])
        # No video control -> no new batch key at all (byte-identical dict).
        assert "control_extra_keys" not in batch
        assert batch["control_images"][0].ndim == 4  # [B, C, H, W], not 5D

    def test_cache_filename_matches_pre_br1_hash(self, tmp_path):
        """The saved latent filename is the LEGACY (no extra_key) hash."""
        ds = str(tmp_path / "ds")
        cache_dir = str(tmp_path / "cache")
        lm = LatentManager(MockImageVAE(), device="cpu")
        h = _Harness(lm)
        item = _image_edit_item(ds, "a")
        item["control_cache_dirs"] = [cache_dir]

        batch = h._get_batch([item])
        h._load_control_latents(batch)

        control_id = item["control_rel_paths"][0]
        control_src = item["control_paths"][0]
        expected_name = LatentManager.latent_filename(
            control_id, control_src
        )  # no extra_key
        assert os.path.exists(os.path.join(cache_dir, expected_name))

    def test_load_control_latents_calls_stub_without_extra_keys_kwarg(self, tmp_path):
        """A latent-manager stub with the PRE-BR1 signature (no extra_keys
        parameter at all) must keep working — proves the kwarg stays
        conditional and is never passed for an image-only batch."""

        class _PreBR1Stub:
            def __init__(self):
                self.load_calls = []
                self.encode_calls = []

            def load_cached_latents(self, ids, cache_dirs, source_paths=None):
                self.load_calls.append((list(ids), list(cache_dirs)))
                return None

            def encode_and_cache_batch(
                self, images, ids=None, cache_dirs=None, source_paths=None
            ):
                self.encode_calls.append(list(ids or []))
                return torch.ones(images.shape[0], 4, 1, 1)

        ds = str(tmp_path / "ds")
        stub = _PreBR1Stub()
        h = _Harness(stub)
        item = _image_edit_item(ds, "a")

        batch = h._get_batch([item])
        h._load_control_latents(batch)  # must not raise TypeError

        assert batch["control_latents"][0].shape[0] == 1
        assert stub.encode_calls


# ── Control decode deferral (warm cache skips the per-step clip decode) ────


class TestControlDecodeDeferral:
    """Video-family batches (``decode_pixels=False``) must not pay the control
    clip decode + host→device upload when the control latent is served from the
    warm cache — the same deferral the TARGET video path already has. The
    pixels are re-decoded ONLY on a cache miss, via the stashed plain-data
    decode specs."""

    def test_deferred_batch_carries_specs_not_pixels(self, tmp_path):
        ds = str(tmp_path / "ds")
        h = _Harness(LatentManager(FakeWanVAE(), device="cpu"))
        item = _video_edit_item(
            ds, "defer_a", target_frames=13, target_fps=8.0, control_frames=13
        )

        batch = h._get_batch([item], decode_pixels=False)

        assert batch["control_images"] is None
        specs = batch["control_decode_specs"]
        assert specs[0][0][0] == "video"  # slot 0, item 0 → a video recipe
        # All metadata fields still present (cache lookups need them).
        assert batch["control_ids"] and batch["control_cache_dirs"]
        assert batch["control_extra_keys"]

    def test_warm_cache_never_decodes_control_pixels(self, monkeypatch, tmp_path):
        """THE per-step saving: once the control latent is cached, a deferred
        batch must satisfy ``_load_control_latents`` without ever touching
        the video decoder."""
        ds = str(tmp_path / "ds")
        lm = LatentManager(FakeWanVAE(), device="cpu")
        h = _Harness(lm)
        item = _video_edit_item(
            ds, "defer_warm", target_frames=13, target_fps=8.0, control_frames=13
        )

        # Warm the cache with one deferred cold pass (decodes on the miss).
        cold = h._get_batch([item], decode_pixels=False)
        h._load_control_latents(cold)
        assert cold["control_latents"][0].ndim == 5

        # Now the decoder must be unreachable on the warm path.
        from app.engine.components import video as video_mod

        def _boom(self, *a, **k):  # pragma: no cover - the assertion itself
            raise AssertionError("control clip decoded despite a warm cache")

        monkeypatch.setattr(video_mod.VideoFrameLoader, "load_clip", _boom)

        warm = h._get_batch([item], decode_pixels=False)
        h._load_control_latents(warm)

        assert warm["control_images"] is None
        assert warm["control_latents"][0].ndim == 5
        assert torch.equal(warm["control_latents"][0], cold["control_latents"][0])

    def test_deferred_cold_miss_decodes_and_matches_eager_latent_shape(
        self, tmp_path
    ):
        """A deferred cache miss decodes via the stashed specs and yields the
        same 5D latent geometry as the eager path."""
        ds = str(tmp_path / "ds")
        lm = LatentManager(FakeWanVAE(), device="cpu")
        h = _Harness(lm)
        item = _video_edit_item(
            ds, "defer_cold", target_frames=13, target_fps=8.0, control_frames=13
        )

        eager = h._get_batch([item])  # decode_pixels default True
        h._load_control_latents(eager)

        item2 = _video_edit_item(
            ds, "defer_cold2", target_frames=13, target_fps=8.0, control_frames=13
        )
        deferred = h._get_batch([item2], decode_pixels=False)
        h._load_control_latents(deferred)

        assert deferred["control_latents"][0].shape == eager["control_latents"][0].shape

    def test_guards_still_fire_when_decode_is_deferred(self, tmp_path):
        """The still-target and sliding refusals are metadata checks and must
        keep failing loudly at batch-build time even without a pixel decode."""
        ds = str(tmp_path / "ds")
        h = _Harness(LatentManager(FakeWanVAE(), device="cpu"))

        item = _video_edit_item(
            ds, "defer_guard", target_frames=13, target_fps=8.0, control_frames=13
        )
        item["temporal_mode"] = "sliding"
        with pytest.raises(ValueError, match="sliding"):
            h._attach_control_images({}, [item], None, decode_pixels=False)

        img_item = _image_edit_item(ds, "defer_img")
        img_item["control_is_video"] = [True]
        with pytest.raises(ValueError, match="video target"):
            h._attach_control_images({}, [img_item], None, decode_pixels=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
