"""qwen_image2 image-to-VAE path, from a real file to packed tokens (LANE-132 Task 3c).

Spec review 2.01 (``.agent/workdir/lane-104/LANE-132-carried-2.01.md``), carried
into BUILD as binding. Upstream's contract (``pipeline_qwenimage21.py``
653-662 at ``6256aa7666``): an image that is not RGBA is ``convert("RGBA")``-ed
(an RGB source gains an opaque alpha), normalized to [-1, 1], and given a frame
axis -- the VAE (``in_channels: 4``) receives ``[B, 4, 1, H, W]``. The alpha of
an RGBA source reaches the VAE as it is; white compositing is for the text
encoder's vision copy only (upstream 266-271).

Two REAL files, a 512x512 RGB JPEG and a 512x512 RGBA PNG
(``fixtures/qwen_image2/``), go through BOTH latent paths the trainer has:

* pre-caching -- ``PipelineCachingMixin._pre_cache_latents``;
* a training-time cache miss -- ``PipelineDataMixin._get_batch`` (the decode
  the train loop runs) into ``LatentManager.encode_and_cache_batch``.

What the VAE receives is observed by wrapping the real vendored VAE's
``encode`` (the call still runs); nothing on the path is stubbed.
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib

import numpy as np
import pytest
import structlog
import torch
from PIL import Image

from app.engine.components.latents import LatentManager
from app.engine.models.registry import registry

_FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "qwen_image2"
_spec = importlib.util.spec_from_file_location("qwen_image2_tiny", _FIX / "tiny.py")
tiny = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tiny)

RGB_JPG = _FIX / "rgb_512.jpg"
RGBA_PNG = _FIX / "rgba_512.png"


class _EncodeSpy:
    """Records every tensor the VAE is asked to encode, then encodes it."""

    def __init__(self, vae) -> None:
        self.inputs: list[torch.Tensor] = []
        self._orig = vae.encode
        vae.encode = self

    def __call__(self, x, *args, **kwargs):
        assert x.dim() == 5, f"VAE input rank: {x.dim()} != 5"
        assert x.shape[1] == 4, f"VAE input channels: {x.shape[1]} != 4"
        self.inputs.append(x.detach().clone())
        return self._orig(x, *args, **kwargs)


def _item(path: pathlib.Path, cache_dir: pathlib.Path) -> dict:
    return {
        "id": path.stem, "path": str(path), "caption": "c", "prefix": "",
        "dropout_rate": 0.0, "cache_dir": str(cache_dir),
        "target_w": 512, "target_h": 512,
    }


def _trainer(tmp_path, vae):
    from app.engine.models.families.qwen_image2.driver import QwenImage2Driver
    from app.engine.models.families.qwen_image2.trainer import QwenImage2Trainer

    registry.initialize()
    defn = registry.get_definition("qwen-image-2.1")
    t = object.__new__(QwenImage2Trainer)
    t.definition = defn
    t.device = torch.device("cpu")
    t.config = {"cache_latents": True}
    t.logger = structlog.get_logger("test_qwen_image2_latents")
    t.driver = QwenImage2Driver(defn, t.device)
    t.latent_manager = LatentManager(vae, device=t.device, arch_params=defn.architecture_params)
    cache = tmp_path / ".cache" / "qwen-image-2.1" / "v1" / "latents" / "original" / "512"
    t.inventory = [_item(RGB_JPG, cache), _item(RGBA_PNG, cache)]
    return t


def _alpha(path: pathlib.Path) -> torch.Tensor:
    """The source's own alpha in [-1, 1] (255 for a file that has none)."""
    im = Image.open(path)
    if im.mode != "RGBA":
        return torch.ones(512, 512)
    a = torch.from_numpy(np.asarray(im.getchannel("A"), dtype=np.float32))
    return a / 127.5 - 1.0


def _assert_vae_input(x: torch.Tensor, path: pathlib.Path) -> None:
    assert tuple(x.shape) == (1, 4, 1, 512, 512)
    assert x.min().item() >= -1.0 and x.max().item() <= 1.0
    alpha = x[0, 3, 0]
    assert torch.allclose(alpha, _alpha(path), atol=1e-6), f"{path.name}: alpha not preserved"
    if path == RGB_JPG:
        assert torch.all(alpha == 1.0)
    else:
        # A fully transparent region keeps its own colour for the VAE: it is
        # never composited over white (that is the TE vision copy's treatment).
        blue = x[0, 2, 0, :256, 256:]
        assert blue.mean().item() > 0.5 and x[0, 0, 0, :256, 256:].mean().item() < -0.5


def _assert_packs(latent: torch.Tensor) -> None:
    from app.engine.models.families.qwen_image2 import utils
    from app.engine.models.families.qwen_image2.vendor.pipeline_qwenimage21 import (
        QwenImage21PipelineHelpers,
    )

    assert tuple(latent.shape[-3:]) == (64, 32, 32)
    lat = latent.reshape(1, 64, 32, 32)
    packed = utils.pack_latents(lat)
    assert tuple(packed.shape) == (1, 1024, 64)
    oracle = QwenImage21PipelineHelpers._pack_latents(lat.unsqueeze(2), 1, 64, 32, 32)
    assert torch.equal(packed, oracle)


@pytest.mark.parametrize("path", [RGB_JPG, RGBA_PNG], ids=["rgb_jpeg", "rgba_png"])
def test_precache_feeds_the_vae_four_channels_and_a_frame_axis(tmp_path, path):
    vae = tiny.tiny_vae()
    spy = _EncodeSpy(vae)
    t = _trainer(tmp_path, vae)
    t.inventory = [i for i in t.inventory if i["path"] == str(path)]
    t._latent_cache_missing = 1

    asyncio.run(t._pre_cache_latents())

    assert len(spy.inputs) == 1
    _assert_vae_input(spy.inputs[0], path)
    item = t.inventory[0]
    cached = t.latent_manager.load_cached_latents(
        [item["id"]], [item["cache_dir"]], source_paths=[item["path"]], device=t.device,
    )
    assert cached is not None, "the pre-cache wrote no latent"
    _assert_packs(cached)


@pytest.mark.parametrize("path", [RGB_JPG, RGBA_PNG], ids=["rgb_jpeg", "rgba_png"])
def test_training_time_cache_miss_feeds_the_vae_the_same(tmp_path, path):
    vae = tiny.tiny_vae()
    spy = _EncodeSpy(vae)
    t = _trainer(tmp_path, vae)
    item = next(i for i in t.inventory if i["path"] == str(path))

    batch = t._get_batch([item])
    latents = t.latent_manager.encode_and_cache_batch(
        batch["images"], ids=batch["ids"], cache_dirs=None, source_paths=batch["paths"],
    )

    assert len(spy.inputs) == 1
    _assert_vae_input(spy.inputs[0], path)
    _assert_packs(latents)


def test_other_families_still_decode_rgb():
    """The RGBA branch is qwen_image2's alone: the shared default stays RGB,
    so every other family's VAE input is unchanged (their own tests pin it)."""
    from app.engine.core.pipeline import GenericTrainingPipeline
    from app.engine.models.families.qwen_image.trainer import QwenImageTrainer
    from app.engine.models.families.qwen_image2.trainer import QwenImage2Trainer

    assert GenericTrainingPipeline.VAE_IMAGE_MODE == "RGB"
    assert QwenImageTrainer.VAE_IMAGE_MODE == "RGB"
    assert QwenImage2Trainer.VAE_IMAGE_MODE == "RGBA"
