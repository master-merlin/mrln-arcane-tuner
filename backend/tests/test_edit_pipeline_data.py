"""PR5 — control-image batch transposition + clean control-latent loading.

Drives the data-mixin batch methods directly with a stub LatentManager so
no VAE / API is needed. The pure inventory helpers and the run-config
validator are covered in test_edit_inventory.py / test_edit_capability_plumbing.py.
"""

from __future__ import annotations

import os

import pytest
import torch
from PIL import Image

from app.engine.core.pipeline.pipeline_data import PipelineDataMixin


def _img(path: str, w: int = 16, h: int = 16, color: str = "red"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (w, h), color).save(path)


class _StubLatentManager:
    """Returns a deterministic latent shaped [B, 4, 1, 1]; records calls."""

    def __init__(self):
        self.load_calls = []
        self.encode_calls = []
        self.cached = False

    def load_cached_latents(self, ids, cache_dirs, source_paths=None):
        self.load_calls.append((list(ids), list(cache_dirs)))
        if not self.cached:
            return None
        return torch.zeros(len(ids), 4, 1, 1)

    def encode_and_cache_batch(self, images, ids=None, cache_dirs=None, source_paths=None):
        self.encode_calls.append(list(ids or []))
        return torch.ones(images.shape[0], 4, 1, 1)


class _Harness(PipelineDataMixin):
    """Minimal object exposing the batch methods under test."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.autocast_dtype = torch.float32
        self.config = {"cache_latents": True}
        self.latent_manager = _StubLatentManager()

    def build_batch_extra(self, items):
        return {}


def _edit_item(ds, stem, n_controls=1):
    _img(os.path.join(ds, f"{stem}.png"), color="red")
    control_paths, control_dims, control_variants, control_cache, control_rel = (
        [], [], [], [], [],
    )
    slots = ["control", "control_2", "control_3"][:n_controls]
    for slot in slots:
        rel = f"{slot}/{stem}.jpg"
        _img(os.path.join(ds, slot, f"{stem}.jpg"), color="blue")
        control_rel.append(rel)
        control_paths.append(os.path.join(ds, rel))
        control_dims.append((16, 16))
        control_variants.append(slot)
        control_cache.append(f"/cache/{slot}/16x16")
    return {
        "path": os.path.join(ds, f"{stem}.png"),
        "id": f"{stem}.png",
        "caption": "make it watercolor",
        "prefix": "",
        "dropout_rate": 0.0,
        "use_captions": True,
        "use_model_aware_captions": False,
        "target_w": 16, "target_h": 16,
        "cache_dir": "/cache/original/16x16",
        "variant": "original",
        "has_masked": False,
        "control_rel_paths": control_rel,
        "control_paths": control_paths,
        "control_dims": control_dims,
        "control_variants": control_variants,
        "control_cache_dirs": control_cache,
    }


class TestControlBatchTransposition:
    def test_get_batch_attaches_per_slot_control_tensors(self, tmp_path):
        ds = str(tmp_path / "ds")
        items = [_edit_item(ds, "a"), _edit_item(ds, "b")]
        batch = _Harness()._get_batch(items)

        assert batch["images"].shape[0] == 2
        # One slot → one stacked control tensor of batch size 2.
        assert len(batch["control_images"]) == 1
        assert batch["control_images"][0].shape[0] == 2
        assert batch["control_ids"] == [["control/a.jpg", "control/b.jpg"]]
        assert batch["control_cache_dirs"] == [
            ["/cache/control/16x16", "/cache/control/16x16"],
        ]

    def test_multi_slot_transpose(self, tmp_path):
        ds = str(tmp_path / "ds")
        items = [_edit_item(ds, "a", n_controls=2), _edit_item(ds, "b", n_controls=2)]
        batch = _Harness()._get_batch(items)
        assert len(batch["control_images"]) == 2
        assert batch["control_ids"][0] == ["control/a.jpg", "control/b.jpg"]
        assert batch["control_ids"][1] == ["control_2/a.jpg", "control_2/b.jpg"]

    def test_non_edit_batch_has_no_control_keys(self, tmp_path):
        ds = str(tmp_path / "ds")
        _img(os.path.join(ds, "a.png"))
        plain = {
            "path": os.path.join(ds, "a.png"), "id": "a.png", "caption": "x",
            "prefix": "", "dropout_rate": 0.0, "use_captions": True,
            "use_model_aware_captions": False, "target_w": 16, "target_h": 16,
            "cache_dir": "/c", "variant": "original", "has_masked": False,
        }
        batch = _Harness()._get_batch([plain])
        assert "control_images" not in batch


class TestLoadControlLatents:
    def test_encodes_on_cache_miss(self, tmp_path):
        ds = str(tmp_path / "ds")
        h = _Harness()
        batch = h._get_batch([_edit_item(ds, "a")])
        h._load_control_latents(batch)
        # Cache miss → encode path; latent batch matches control image count.
        assert len(batch["control_latents"]) == 1
        assert batch["control_latents"][0].shape[0] == 1
        assert h.latent_manager.encode_calls  # encode was used

    def test_uses_cache_when_present(self, tmp_path):
        ds = str(tmp_path / "ds")
        h = _Harness()
        h.latent_manager.cached = True
        batch = h._get_batch([_edit_item(ds, "a"), _edit_item(ds, "b")])
        h._load_control_latents(batch)
        assert batch["control_latents"][0].shape[0] == 2
        assert not h.latent_manager.encode_calls  # served from cache

    def test_noop_without_control_cache_dirs(self):
        h = _Harness()
        batch = {"images": torch.zeros(1, 3, 16, 16)}
        h._load_control_latents(batch)
        assert "control_latents" not in batch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
