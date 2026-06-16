"""LTX-2 audio data path: pre-cache + batch collation (no GPU, no weights).

These two hooks are what make ``batch["audio_clean"]`` exist — without them the
driver's joint forward always falls into the video-only branch and the audio
LoRA receives zero gradient (the "trains but no audio" bug). Tests use a fake
driver/latent-manager and monkeypatch the PyAV decode so they run on CPU with no
model and no media files.

Covered:
- ``_pre_cache_aux`` encodes + caches one audio latent per video clip WITH
  audio, skips stills and audio-less clips, is idempotent (re-run skips cached),
  and offloads the audio VAE afterwards.
- ``build_batch_extra`` stacks cached latents into ``audio_clean`` + a presence
  ``audio_mask``; absent items get a zero latent + mask 0; an all-absent batch
  (or audio-off run) yields ``{}`` so the forward stays video-only.
"""

from __future__ import annotations

import structlog
import torch
from safetensors.torch import load_file, save_file

import app.engine.models.families.ltx2.audio_io as audio_io
from app.engine.models.families.ltx2.driver import Ltx2Driver
from app.engine.models.families.ltx2.trainer import Ltx2Trainer

L = 10  # audio latent length used throughout


class _FakeAudioVae:
    def __init__(self) -> None:
        self.offloaded_to = None

    def to(self, dev):
        self.offloaded_to = dev
        return self


class _FakeDriver:
    def __init__(self, *, train_audio=True, with_vae=True) -> None:
        self.train_audio = train_audio
        self.audio_vae = _FakeAudioVae() if with_vae else None
        self.audio_sampling_rate = 16000
        self.frame_rate = 24.0
        self.encode_calls = 0

    def encode_audio_clean(self, waveform, sample_rate):
        self.encode_calls += 1
        return torch.ones(1, L, 128)  # [B, L, 128]


class _FakeLM:
    """Deterministic, content-free cache filenames keyed on the item id."""

    @staticmethod
    def latent_filename(img_id, source_path, extra_key=""):
        return f"{img_id}.safetensors"


def _trainer(*, train_audio=True, with_vae=True, cache_latents=True):
    t = object.__new__(Ltx2Trainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.config = {"cache_latents": cache_latents}
    t.driver = _FakeDriver(train_audio=train_audio, with_vae=with_vae)
    t.latent_manager = _FakeLM()
    t.inventory = []
    return t


def _video_item(tmp_path, ident, *, has_audio=True):
    return {
        "id": ident,
        "path": f"/clips/{ident}{'' if has_audio else '_noaudio'}.mkv",
        "cache_dir": str(tmp_path / ident),
        "is_video": True,
        "target_frames": 25,
        "target_fps": 24.0,
        "trim_start_s": 0.0,
        "trim_end_s": 1.0,
        "_has_audio": has_audio,
    }


def _patch_decode(monkeypatch):
    """Fake load_audio_waveform: a clip "has audio" unless its path says noaudio."""

    def fake(path, *, trim_start_s, duration_s, target_sr):
        if "noaudio" in path:
            return None
        return torch.zeros(1, int(duration_s * target_sr)), target_sr

    monkeypatch.setattr(audio_io, "load_audio_waveform", fake)


# ── _pre_cache_aux ────────────────────────────────────────────────────────


def test_precache_writes_one_latent_per_audio_clip(tmp_path, monkeypatch):
    _patch_decode(monkeypatch)
    t = _trainer()
    t.inventory = [
        _video_item(tmp_path, "clipA"),
        _video_item(tmp_path, "clipB"),
        _video_item(tmp_path, "clipC", has_audio=False),  # no audio stream
        {"id": "img1", "path": "/img/a.png", "cache_dir": str(tmp_path / "img1"),
         "is_video": False},  # still → skipped
    ]

    t._pre_cache_aux()

    a = tmp_path / "clipA" / "audio" / "clipA.safetensors"
    b = tmp_path / "clipB" / "audio" / "clipB.safetensors"
    c = tmp_path / "clipC" / "audio" / "clipC.safetensors"
    assert a.exists() and b.exists()
    assert not c.exists()  # audio-less clip not cached
    assert not (tmp_path / "img1" / "audio").exists()  # still skipped
    assert load_file(str(a))["audio_latents"].shape == (L, 128)
    assert t.driver.encode_calls == 2


def test_precache_offloads_audio_vae(tmp_path, monkeypatch):
    _patch_decode(monkeypatch)
    t = _trainer()
    t.inventory = [_video_item(tmp_path, "clipA")]

    t._pre_cache_aux()

    assert t.driver.audio_vae.offloaded_to == "cpu"


def test_precache_is_idempotent(tmp_path, monkeypatch):
    _patch_decode(monkeypatch)
    t = _trainer()
    t.inventory = [_video_item(tmp_path, "clipA")]

    t._pre_cache_aux()
    assert t.driver.encode_calls == 1
    t._pre_cache_aux()  # cache file now exists → skip
    assert t.driver.encode_calls == 1


def test_precache_noop_when_audio_off(tmp_path, monkeypatch):
    _patch_decode(monkeypatch)
    t = _trainer(train_audio=False)
    t.inventory = [_video_item(tmp_path, "clipA")]

    t._pre_cache_aux()

    assert t.driver.encode_calls == 0
    assert not (tmp_path / "clipA" / "audio").exists()


# ── build_batch_extra ──────────────────────────────────────────────────────


def _seed_audio_cache(t, item, tensor):
    adir = t._audio_cache_dir(item["cache_dir"])
    import os

    os.makedirs(adir, exist_ok=True)
    fname = t.latent_manager.latent_filename(item["id"], item["path"])
    save_file({"audio_latents": tensor}, str(__import__("pathlib").Path(adir) / fname))


def test_build_batch_extra_stacks_and_masks(tmp_path):
    t = _trainer()
    a = _video_item(tmp_path, "clipA")
    b = _video_item(tmp_path, "clipB")  # left uncached → absent
    c = {"id": "img1", "path": "/img/a.png", "cache_dir": str(tmp_path / "img1"),
         "is_video": False}
    _seed_audio_cache(t, a, torch.full((L, 128), 3.0))

    extra = t.build_batch_extra([a, b, c])

    assert set(extra) == {"audio_clean", "audio_mask"}
    assert extra["audio_clean"].shape == (3, L, 128)
    assert torch.equal(extra["audio_mask"], torch.tensor([1.0, 0.0, 0.0]))
    # Present item keeps its values; absent items are zeros.
    assert torch.equal(extra["audio_clean"][0], torch.full((L, 128), 3.0))
    assert torch.count_nonzero(extra["audio_clean"][1]) == 0
    assert torch.count_nonzero(extra["audio_clean"][2]) == 0


def test_build_batch_extra_empty_when_no_audio_in_batch(tmp_path):
    t = _trainer()
    a = _video_item(tmp_path, "clipA")  # nothing cached
    assert t.build_batch_extra([a]) == {}


def test_build_batch_extra_empty_when_audio_off(tmp_path):
    t = _trainer(train_audio=False)
    a = _video_item(tmp_path, "clipA")
    _seed_audio_cache(t, a, torch.ones(L, 128))
    assert t.build_batch_extra([a]) == {}


def test_build_batch_extra_treats_shape_mismatch_as_absent(tmp_path):
    t = _trainer()
    a = _video_item(tmp_path, "clipA")
    b = _video_item(tmp_path, "clipB")
    _seed_audio_cache(t, a, torch.ones(L, 128))
    _seed_audio_cache(t, b, torch.ones(L + 3, 128))  # wrong length → treated absent

    extra = t.build_batch_extra([a, b])

    assert extra["audio_clean"].shape == (2, L, 128)
    assert torch.equal(extra["audio_mask"], torch.tensor([1.0, 0.0]))


# ── encode_audio_clean device co-location ──────────────────────────────────


class _RecVae:
    """Audio VAE fake that records device moves and returns a fixed latent."""

    def __init__(self) -> None:
        self.moved_to: list = []
        self.dtype = torch.float32
        self.latents_mean = None
        self.latents_std = None

    def to(self, dev):
        self.moved_to.append(dev)
        return self

    def encode(self, mel):  # extract_audio_latents handles a raw-tensor return
        b = mel.shape[0]
        return torch.zeros(b, 8, L, 16)  # → pack_audio_latents → [B, L, 128]


def test_encode_audio_clean_colocates_vae_with_input():
    """Regression: the audio VAE must be moved to the mel's device before encode.

    The generic orchestration only relocates the VIDEO VAE, so without an
    explicit move the CPU-resident audio VAE meets a CUDA mel → "Input type
    (CUDABFloat16Type) and weight type (CPUBFloat16Type) should be the same".
    Asserted device-agnostically (CPU in CI) by checking the VAE was moved to
    ``driver.device``.
    """
    drv = object.__new__(Ltx2Driver)
    drv.device = torch.device("cpu")
    drv.audio_sampling_rate = 16000
    drv._audio_mel = None
    drv.audio_vae = _RecVae()

    out = drv.encode_audio_clean(torch.zeros(1, 2, 1600), 16000)  # [B, C=2, N]

    assert drv.audio_vae.moved_to, "audio VAE was never co-located with the input"
    assert torch.device(drv.audio_vae.moved_to[-1]) == drv.device
    assert out.shape[-1] == 128  # packed audio latent feature dim
