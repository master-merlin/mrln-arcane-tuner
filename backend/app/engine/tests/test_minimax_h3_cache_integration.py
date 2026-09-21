"""MiniMax-H3 latent-cache fingerprint — the PRODUCTION seam (plan row 2.3).

Production never consults a driver hook for the cache key: `prepare_data`
(`pipeline_data.py`) writes every inventory item's `cache_dir` with the
literal variant `"original"`, and coverage (`_validate_latent_cache`),
pre-cache (`_pre_cache_latents`) and training (`_get_batch` → the batch's
`cache_dirs`) all read `item["cache_dir"]` from that inventory. So the
fingerprint is only real if `MiniMaxH3Trainer.prepare_data` rewrites the
inventory — these four tests drive the REAL base `prepare_data` on a tiny
on-disk dataset (two real mp4 clips — a still is skipped under the `17n+5`
floor, so it is not a representative H3 item; the dataset API faked at the HTTP client)
and read the dirs back off the seams production reads.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import torch
from app.engine.models.families.minimax_h3.driver import MiniMaxH3Driver
from app.engine.models.families.minimax_h3.pixel_adapter import H3PixelAdaptedVAE
from app.engine.models.families.minimax_h3.trainer import MiniMaxH3Trainer
from app.engine.models.registry import ModelRegistry
from app.engine.tests.h3_real_seams import write_clip
from app.engine.tests.h3_text_stubs import StubVisualVAE

_IDS = ("a.mp4", "b.mp4")


def _definition(def_id: str = "minimax-h3-t2va"):
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    return ModelRegistry._definitions[def_id]


def _dataset(tmp_path) -> str:
    ds = tmp_path / "ds"
    if ds.exists():
        return str(ds)
    ds.mkdir()
    for name in _IDS:
        write_clip(ds / name, 24, soundtrack=False)
    return str(ds)


def _fake_api(monkeypatch, ds_path: str) -> None:
    pairs = [
        {
            "media_file": name,
            "caption_content": f"caption {name}",
            "metadata": {
                "width": 64, "height": 64, "is_video": True, "enabled": True,
                "fps": 24.0, "duration_s": 1.0,
            },
        }
        for name in _IDS
    ]
    info = {"path": ds_path, "version": "1.0.0", "kind": "standard"}

    async def _get(_self, url, *args, **kwargs):
        body = pairs if url.endswith("/pairs") else info
        return SimpleNamespace(status_code=200, json=lambda: json.loads(json.dumps(body)))

    monkeypatch.setattr(httpx.AsyncClient, "get", _get)


def _trainer(tmp_path, monkeypatch) -> MiniMaxH3Trainer:
    t = object.__new__(MiniMaxH3Trainer)
    t.device = torch.device("cpu")
    t.definition = _definition()
    t.driver = MiniMaxH3Driver(t.definition, t.device)
    t.logger = MagicMock()
    t._log_writer = None
    t.config = {
        "resolutions": [64],
        "datasets": [{"dataset_name": "ds"}],
        "cache_latents": True,
        "train_audio": True,
    }
    t.components = {"vae": H3PixelAdaptedVAE(StubVisualVAE())}
    t._assign_components()
    _fake_api(monkeypatch, _dataset(tmp_path))
    return t


def _prepare(t: MiniMaxH3Trainer) -> list[str]:
    asyncio.run(t.prepare_data())
    return [item["cache_dir"] for item in t.inventory]


def _touch_latents(t: MiniMaxH3Trainer) -> None:
    """Create the latent files coverage looks for, so the cache is FULL."""
    from app.engine.core.pipeline.pipeline_data import video_trim_extra_key

    for item in t.inventory:
        fname = t.latent_manager.latent_filename(item["id"], item["path"], video_trim_extra_key(item))
        os.makedirs(item["cache_dir"], exist_ok=True)
        with open(os.path.join(item["cache_dir"], fname), "wb") as fh:
            fh.write(b"\0")


def test_1_unchanged_settings_yield_identical_cache_dirs(tmp_path, monkeypatch):
    first = _prepare(_trainer(tmp_path, monkeypatch))
    second = _prepare(_trainer(tmp_path, monkeypatch))
    assert first and first == second


def test_2_pixel_adapter_bump_changes_every_cache_dir_and_uncovers_it(tmp_path, monkeypatch):
    from app.engine.models.families.minimax_h3 import pixel_adapter

    t = _trainer(tmp_path, monkeypatch)
    before = _prepare(t)
    _touch_latents(t)
    t._validate_latent_cache()
    assert t._latent_cache_missing == 0, "fixture latents not seen as covered"

    monkeypatch.setattr(pixel_adapter, "PIXEL_ADAPTER_VERSION", pixel_adapter.PIXEL_ADAPTER_VERSION + 1)
    bumped = _trainer(tmp_path, monkeypatch)
    after = _prepare(bumped)
    assert len(after) == len(before)
    assert all(a != b for a, b in zip(after, before)), "dirs unchanged"
    bumped._validate_latent_cache()
    assert bumped._latent_cache_missing == len(after), "old latents still counted as coverage"


def test_3_precache_and_training_batch_agree_and_carry_the_fingerprint(tmp_path, monkeypatch):
    t = _trainer(tmp_path, monkeypatch)
    _prepare(t)
    fp = t.driver.latent_cache_fingerprint()

    # What `_pre_cache_latents` iterates: the inventory's cache_dir per id.
    precache = {item["id"]: item["cache_dir"] for item in t.inventory}
    # What a training step reads: the batch's cache_dirs for the same ids.
    batch = t._get_batch(list(t.inventory), decode_pixels=False)
    training = dict(zip(batch["ids"], batch["cache_dirs"]))

    assert precache == training
    # The LAST segment (the resolution, `.../latents/<variant>/<res>`) carries
    # the fingerprint; equality alone would survive a missing rewrite.
    for d in list(precache.values()) + list(training.values()):
        assert os.path.basename(d).endswith(f"-h3{fp}"), (
            f"cache_dir last segment lacks the fingerprint: {d}"
        )
        assert os.path.basename(os.path.dirname(d)) == "original"


def test_4_te_cache_dirs_still_resolve_the_dataset_root(tmp_path, monkeypatch):
    t = _trainer(tmp_path, monkeypatch)
    _prepare(t)
    ds_path = str(tmp_path / "ds")
    model_name = t.definition.id.split("/")[-1]
    assert t._resolve_te_cache_dirs() == [os.path.join(ds_path, ".cache", model_name, "1.0.0")]
    # Same depth as the base layout: the rewrite is INSIDE the last segment.
    for item in t.inventory:
        rel = os.path.relpath(item["cache_dir"], ds_path)
        assert rel.count(os.sep) == 5, rel  # .cache/model/version/latents/res/variant
