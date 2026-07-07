"""hv15 I2V Siglip image-embedding aux cache (LTX-2 audio-cache pattern).

``_pre_cache_aux`` encodes the first frame of each item through the Siglip
image encoder while it is resident (then offloads it); ``build_batch_extra``
serves the cached ``[B, 729, 1152]`` ``image_embeds`` to the training batch.
T2V runs are no-ops on both paths.
"""

import os
from types import SimpleNamespace

import torch

from app.engine.models.families.hunyuan_video15.driver import Hv15Driver
from app.engine.models.families.hunyuan_video15.trainer import Hv15Trainer

_TOKENS, _DIM = 5, 4  # tiny stand-ins for 729 / 1152


class _FakeFeatureExtractor:
    def preprocess(self, images=None, do_resize=True, return_tensors="pt",
                   do_convert_rgb=True):
        return {"pixel_values": torch.zeros(1, 3, 8, 8)}


class _FakeImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 1)  # gives parameters() a dtype

    def forward(self, pixel_values=None):
        return SimpleNamespace(
            last_hidden_state=torch.full((1, _TOKENS, _DIM), 3.0)
        )


class _FakeLatentManager:
    @staticmethod
    def latent_filename(img_id, source_path, extra_key=""):
        return f"{img_id}.safetensors"


def _make_trainer(mode: str, tmp_path, items) -> Hv15Trainer:
    t = object.__new__(Hv15Trainer)
    t.device = torch.device("cpu")
    t.config = {"cache_latents": True}
    t.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    driver = SimpleNamespace(
        is_i2v=(mode == "i2v"),
        image_encoder=_FakeImageEncoder() if mode == "i2v" else None,
        feature_extractor=_FakeFeatureExtractor() if mode == "i2v" else None,
    )
    t.driver = driver
    t.inventory = items
    t.latent_manager = _FakeLatentManager()
    return t


def _still_item(tmp_path, name="img1") -> dict:
    from PIL import Image

    img_path = os.path.join(str(tmp_path), f"{name}.png")
    Image.new("RGB", (16, 16), (128, 64, 32)).save(img_path)
    cache_dir = os.path.join(str(tmp_path), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return {"id": name, "path": img_path, "cache_dir": cache_dir, "is_video": False}


def test_precache_aux_encodes_and_offloads(tmp_path):
    item = _still_item(tmp_path)
    t = _make_trainer("i2v", tmp_path, [item])
    t._pre_cache_aux()

    path = os.path.join(item["cache_dir"], "siglip", "img1.safetensors")
    assert os.path.exists(path)
    from safetensors.torch import load_file

    emb = load_file(path)["image_embeds"]
    assert emb.shape == (_TOKENS, _DIM)
    assert torch.all(emb == 3.0)


def test_precache_aux_skips_existing(tmp_path):
    item = _still_item(tmp_path)
    t = _make_trainer("i2v", tmp_path, [item])
    t._pre_cache_aux()
    mtime = os.path.getmtime(
        os.path.join(item["cache_dir"], "siglip", "img1.safetensors")
    )
    t2 = _make_trainer("i2v", tmp_path, [item])
    t2._pre_cache_aux()
    assert os.path.getmtime(
        os.path.join(item["cache_dir"], "siglip", "img1.safetensors")
    ) == mtime


def test_precache_aux_noop_for_t2v(tmp_path):
    item = _still_item(tmp_path)
    t = _make_trainer("t2v", tmp_path, [item])
    t._pre_cache_aux()
    assert not os.path.exists(os.path.join(item["cache_dir"], "siglip"))


def test_build_batch_extra_serves_cached_embeds(tmp_path):
    item = _still_item(tmp_path)
    t = _make_trainer("i2v", tmp_path, [item])
    t._pre_cache_aux()

    extra = t.build_batch_extra([item])
    emb = extra[Hv15Driver.BATCH_IMAGE_EMBED]
    assert emb.shape == (1, _TOKENS, _DIM)
    assert torch.all(emb == 3.0)


def test_build_batch_extra_zero_fills_missing_items(tmp_path):
    cached = _still_item(tmp_path, "have")
    missing = _still_item(tmp_path, "missing")
    t = _make_trainer("i2v", tmp_path, [cached])
    t._pre_cache_aux()  # only "have" is cached

    extra = t.build_batch_extra([cached, missing])
    emb = extra[Hv15Driver.BATCH_IMAGE_EMBED]
    assert emb.shape == (2, _TOKENS, _DIM)
    assert torch.all(emb[0] == 3.0)
    assert torch.all(emb[1] == 0.0)


def test_build_batch_extra_empty_without_any_cache(tmp_path):
    item = _still_item(tmp_path)
    t = _make_trainer("i2v", tmp_path, [item])
    assert t.build_batch_extra([item]) == {}


def test_build_batch_extra_noop_for_t2v(tmp_path):
    item = _still_item(tmp_path)
    t = _make_trainer("t2v", tmp_path, [item])
    assert t.build_batch_extra([item]) == {}
