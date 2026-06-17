"""_get_batch defers the expensive video decode when pixels aren't needed.

A warm latent cache makes the per-step PyAV decode pure waste (the pixels are
discarded for the cached latent), which starves the GPU. ``decode_pixels=False``
must skip the decode entirely while still building every other batch field; the
train loop re-decodes on demand via ``_decode_batch_images`` on a cache miss.

No GPU, no model weights — the clip loader is faked.
"""

from __future__ import annotations

import torch

import app.engine.components.video as video_mod
from app.engine.core.pipeline.pipeline_data import PipelineDataMixin


class _RecordingLoader:
    """Fake VideoFrameLoader recording every clip it's asked to decode."""

    paths: list[str] = []

    def load_clip(self, path, **kw):
        _RecordingLoader.paths.append(path)
        return torch.zeros(3, int(kw["target_frames"]), 8, 8)


class _ExplodingLoader:
    """Fake loader that fails if a decode is ever attempted."""

    def load_clip(self, *a, **kw):  # noqa: D401
        raise AssertionError("video was decoded when it should have been deferred")


def _mixin(*, is_video_family: bool = True) -> PipelineDataMixin:
    inst = object.__new__(PipelineDataMixin)
    inst.config = {}
    inst.device = "cpu"
    inst.is_video_family = is_video_family
    # Shadow the family hook so no real family extras are pulled in.
    inst.build_batch_extra = lambda items: {}
    return inst


def _video_item(idx: int = 0) -> dict:
    return {
        "id": f"v{idx}",
        "path": f"/data/clip{idx}.mp4",
        "cache_dir": "/data/.cache",
        "caption": "a cat",
        "prefix": "",
        "dropout_rate": 0.0,
        "target_w": 16,
        "target_h": 16,
        "is_video": True,
        "target_frames": 2,
        "target_fps": 24.0,
        "trim_start_s": 0.0,
        "trim_end_s": None,
    }


def test_defer_skips_decode_but_builds_metadata(monkeypatch):
    monkeypatch.setattr(video_mod, "VideoFrameLoader", _ExplodingLoader)
    inst = _mixin()
    items = [_video_item(0), _video_item(1)]

    batch = inst._get_batch(items, decode_pixels=False)

    # No pixels decoded …
    assert batch["images"] is None
    # … but everything the cache lookup + forward needs is present.
    assert batch["ids"] == ["v0", "v1"]
    assert batch["cache_dirs"] == ["/data/.cache", "/data/.cache"]
    assert batch["paths"] == ["/data/clip0.mp4", "/data/clip1.mp4"]
    assert batch["captions"] == ["a cat", "a cat"]
    # Trim discriminators + RoPE fps still derived from the items.
    assert batch["extra_keys"] == ["t0.0-None", "t0.0-None"]
    assert batch["target_fps"] == 24.0


def test_decode_pixels_true_decodes_via_loader(monkeypatch):
    _RecordingLoader.paths = []
    monkeypatch.setattr(video_mod, "VideoFrameLoader", _RecordingLoader)
    inst = _mixin()
    items = [_video_item(0), _video_item(1)]

    batch = inst._get_batch(items, decode_pixels=True)

    assert _RecordingLoader.paths == ["/data/clip0.mp4", "/data/clip1.mp4"]
    assert isinstance(batch["images"], torch.Tensor)
    assert batch["images"].shape == (2, 3, 2, 8, 8)  # [B, C, F, H, W]


def test_decode_batch_images_reuses_selected_paths(monkeypatch):
    _RecordingLoader.paths = []
    monkeypatch.setattr(video_mod, "VideoFrameLoader", _RecordingLoader)
    inst = _mixin()
    items = [_video_item(0)]

    # The miss path decodes from the variant-selected paths the batch carries,
    # NOT by re-running variant selection — so the pixels match the cache key.
    out = inst._decode_batch_images(items, ["/picked/variant.mp4"])

    assert _RecordingLoader.paths == ["/picked/variant.mp4"]
    assert out.shape == (1, 3, 2, 8, 8)
