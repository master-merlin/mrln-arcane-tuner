"""Video latent pre-caching contract (no GPU).

Regression for the LTX-2 burn-in crash: ``_pre_cache_latents`` PIL-opened
every item, so a ``.mkv`` clip raised "cannot identify image file". Video
items must instead decode through ``VideoFrameLoader`` and cache under the
SAME trim ``extra_key`` the train loop uses (``video_trim_extra_key``) —
otherwise the pre-cached latent is written under one name and looked up under
another at train time (silent re-encode, risking the VAE-fallback OOM the
pre-cache exists to avoid).
"""

from __future__ import annotations

import asyncio

import structlog
import torch

from app.engine.core.pipeline.pipeline_caching import PipelineCachingMixin
from app.engine.core.pipeline.pipeline_data import video_trim_extra_key


class _Host(PipelineCachingMixin):
    pass


class _RecLatentManager:
    """Records encode calls + the extra_keys handed to the coverage check."""

    def __init__(self) -> None:
        self.encode_calls: list[dict] = []
        self.coverage_extra_keys: list[str] | None = None

    def check_cache_coverage(
        self, ids, cache_dirs, source_paths=None, extra_keys=None
    ):
        self.coverage_extra_keys = list(extra_keys) if extra_keys else None
        return 0, len(ids), list(ids)  # everything missing

    @staticmethod
    def latent_filename(img_id, source_path, extra_key=""):
        return f"{img_id}__{extra_key}.safetensors"  # never exists on disk

    def encode_and_cache_batch(
        self, tensor, ids, cache_dirs=None, source_paths=None, extra_keys=None
    ):
        self.encode_calls.append(
            {
                "ndim": tensor.ndim,
                "shape": tuple(tensor.shape),
                "ids": list(ids),
                "extra_keys": list(extra_keys) if extra_keys else None,
            }
        )
        return tensor


def _host(inventory):
    h = _Host()
    h.logger = structlog.get_logger("test")
    h.device = torch.device("cpu")
    h.config = {"cache_latents": True}
    h.inventory = inventory
    h.latent_manager = _RecLatentManager()
    h._latent_cache_missing = len(inventory)
    h._loop = None
    return h


def _video_item(tmp_path):
    return {
        "path": str(tmp_path / "clip.mkv"),
        "id": "clip.mkv",
        "cache_dir": str(tmp_path / "cache"),
        "target_w": 64,
        "target_h": 64,
        "is_video": True,
        "target_frames": 9,
        "target_fps": 24.0,
        "trim_start_s": 0.0,
        "trim_end_s": None,
        "has_masked": False,
    }


# ── The shared trim cache-key ──────────────────────────────────────────────


def test_video_trim_extra_key_helper():
    assert video_trim_extra_key(
        {"is_video": True, "trim_start_s": 0.0, "trim_end_s": None}
    ) == "t0.0-None"
    assert video_trim_extra_key(
        {"is_video": True, "trim_start_s": 1.5, "trim_end_s": 3.0}
    ) == "t1.5-3.0"
    assert video_trim_extra_key({"is_video": False}) == ""
    assert video_trim_extra_key({}) == ""


# ── Pre-cache routes video through the frame loader, not PIL ───────────────


def test_pre_cache_video_uses_frame_loader_and_trim_key(tmp_path, monkeypatch):
    item = _video_item(tmp_path)
    h = _host([item])

    seen: dict = {}

    class _FakeLoader:
        def load_clip(
            self, path, target_frames, target_fps, trim_start_s,
            trim_end_s, target_w, target_h, h_flip,
        ):
            seen.update(
                path=path, target_frames=target_frames, target_fps=target_fps,
                target_w=target_w, target_h=target_h,
            )
            return torch.zeros(3, target_frames, target_h, target_w)

    import app.engine.components.video as vmod

    monkeypatch.setattr(vmod, "VideoFrameLoader", _FakeLoader)

    # PIL must NEVER touch a video clip (the original crash).
    import app.engine.core.pipeline.pipeline_caching as cmod

    def _boom(*a, **k):
        raise AssertionError("Image.open called on a video clip")

    monkeypatch.setattr(cmod.Image, "open", _boom)

    asyncio.run(h._pre_cache_latents())

    assert seen["target_frames"] == 9 and seen["target_fps"] == 24.0
    assert len(h.latent_manager.encode_calls) == 1
    call = h.latent_manager.encode_calls[0]
    assert call["ndim"] == 5  # [1, C, F, H, W]
    assert call["shape"] == (1, 3, 9, 64, 64)
    assert call["extra_keys"] == ["t0.0-None"]  # matches the train-loop key


def test_pre_cache_image_still_uses_pil_and_empty_key(tmp_path, monkeypatch):
    from PIL import Image as PILImage

    img_path = tmp_path / "a.png"
    PILImage.new("RGB", (80, 80), "red").save(img_path)
    item = {
        "path": str(img_path),
        "id": "a.png",
        "cache_dir": str(tmp_path / "c"),
        "target_w": 64,
        "target_h": 64,
        "is_video": False,
        "has_masked": False,
    }
    h = _host([item])

    import app.engine.components.video as vmod

    class _NoLoader:
        def load_clip(self, *a, **k):
            raise AssertionError("VideoFrameLoader used for an image")

    monkeypatch.setattr(vmod, "VideoFrameLoader", _NoLoader)

    asyncio.run(h._pre_cache_latents())
    call = h.latent_manager.encode_calls[0]
    assert call["ndim"] == 4  # [1, C, H, W]
    assert call["extra_keys"] == [""]


def test_validate_latent_cache_threads_trim_key(tmp_path):
    h = _host([_video_item(tmp_path)])
    h._validate_latent_cache()
    assert h.latent_manager.coverage_extra_keys == ["t0.0-None"]
