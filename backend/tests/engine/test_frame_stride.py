"""Axis B frame stride — effective fps math + batch wiring (no GPU)."""

from __future__ import annotations

import torch
from PIL import Image

from app.engine.core.pipeline.pipeline_data import PipelineDataMixin


class _Host(PipelineDataMixin):
    pass


def _host(config):
    h = _Host()
    h.config = config
    return h


def test_effective_fps_divides_native_by_stride():
    h = _host({"frame_stride": 2})
    # native 24 fps, stride 2 → effective 12 fps
    assert h._effective_fps(native_or_target_fps=24.0) == 12.0


def test_stride_one_is_identity():
    h = _host({"frame_stride": 1})
    assert h._effective_fps(native_or_target_fps=24.0) == 24.0


def test_zero_fps_stays_zero():
    h = _host({"frame_stride": 4})
    assert h._effective_fps(native_or_target_fps=0.0) == 0.0


def test_falsy_stride_coalesces_to_identity():
    # The `or 1` coalescing: a missing key, None, or 0 stride must behave as
    # stride 1 (identity), never divide-by-zero.
    assert _host({})._effective_fps(native_or_target_fps=24.0) == 24.0
    assert _host({"frame_stride": None})._effective_fps(24.0) == 24.0
    assert _host({"frame_stride": 0})._effective_fps(24.0) == 24.0


# ── _get_batch wiring: video items must carry the effective target_fps ──


def _video_inventory_item(tmp_path, fps):
    return {
        "path": str(tmp_path / "clip.mkv"),
        "id": "clip.mkv",
        "caption": "c",
        "prefix": "",
        "dropout_rate": 0.0,
        "use_captions": False,
        "use_model_aware_captions": False,
        "target_w": 64,
        "target_h": 64,
        "cache_dir": str(tmp_path / "cache"),
        "variant": "original",
        "is_video": True,
        "target_frames": 9,
        "target_fps": fps,
        "trim_start_s": 0.0,
        "trim_end_s": 1.0,
        "has_masked": False,
    }


def test_get_batch_sets_target_fps_from_items(tmp_path, monkeypatch):
    import app.engine.components.video as vmod

    class _FakeLoader:
        def load_clip(self, path, target_frames, target_fps, **k):
            return torch.zeros(3, target_frames, 64, 64)

    monkeypatch.setattr(vmod, "VideoFrameLoader", _FakeLoader)

    h = _host({})
    h.device = torch.device("cpu")
    h.is_video_family = True
    h.build_batch_extra = lambda items: {}
    h._select_variant = lambda item: (item["path"], "c", item["cache_dir"])

    batch = h._get_batch([_video_inventory_item(tmp_path, fps=12.0)])
    assert batch["target_fps"] == 12.0


def test_get_batch_image_only_has_no_target_fps(tmp_path, monkeypatch):
    img = tmp_path / "a.png"
    Image.new("RGB", (64, 64), "red").save(img)
    h = _host({})
    h.device = torch.device("cpu")
    h.is_video_family = False
    h.build_batch_extra = lambda items: {}
    h._select_variant = lambda item: (item["path"], "c", item["cache_dir"])
    item = {
        "path": str(img), "id": "a.png", "caption": "c", "prefix": "",
        "dropout_rate": 0.0, "use_captions": False,
        "use_model_aware_captions": False, "target_w": 64, "target_h": 64,
        "cache_dir": str(tmp_path), "variant": "original", "is_video": False,
        "has_masked": False,
    }
    batch = h._get_batch([item])
    assert "target_fps" not in batch
