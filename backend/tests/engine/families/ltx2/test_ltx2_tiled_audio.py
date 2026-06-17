"""LTX-2 audio pre-cache with Axis-A tiled windows (no GPU, no weights).

Each tiled window is a separate inventory item with a distinct trim window, so
_pre_cache_aux must encode + cache one audio latent PER WINDOW under a distinct
trim-keyed filename — none collapsed, none reused. This is a CHARACTERIZATION
test: _pre_cache_aux already keys on video_trim_extra_key, so no audio-path code
changes for tiling. A failure means tiling broke window distinctness (revisit
Task 5) or the audio path does not key on the trim window as assumed.
"""

from __future__ import annotations

import os

import structlog
import torch

import app.engine.models.families.ltx2.audio_io as audio_io
from app.engine.models.families.ltx2.trainer import Ltx2Trainer

L = 10


class _FakeAudioVae:
    def to(self, dev):
        return self


class _FakeDriver:
    def __init__(self):
        self.train_audio = True
        self.audio_vae = _FakeAudioVae()
        self.audio_sampling_rate = 16000
        self.frame_rate = 24.0
        self.encode_calls = 0

    def encode_audio_clean(self, waveform, sample_rate):
        self.encode_calls += 1
        return torch.ones(1, L, 128)


class _FakeLM:
    """Trim-aware filename so distinct windows produce distinct files."""

    @staticmethod
    def latent_filename(img_id, source_path, extra_key=""):
        return f"{img_id}__{extra_key}.safetensors"


def _trainer():
    t = object.__new__(Ltx2Trainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.config = {"cache_latents": True}
    t.driver = _FakeDriver()
    t.latent_manager = _FakeLM()
    t.inventory = []
    return t


def _window_item(tmp_path, ts, te):
    return {
        "id": "clip.mkv",
        "path": "/clips/clip.mkv",
        "cache_dir": str(tmp_path / "cache"),
        "is_video": True,
        "target_frames": 25,
        "target_fps": 24.0,
        "trim_start_s": ts,
        "trim_end_s": te,
    }


def _patch_decode(monkeypatch):
    def fake(path, *, trim_start_s, duration_s, target_sr):
        return torch.zeros(1, target_sr), target_sr

    monkeypatch.setattr(audio_io, "load_audio_waveform", fake)


def test_three_windows_cache_three_distinct_audio_latents(tmp_path, monkeypatch):
    _patch_decode(monkeypatch)
    t = _trainer()
    t.inventory = [
        _window_item(tmp_path, 0.0, 1.0),
        _window_item(tmp_path, 1.0, 2.0),
        _window_item(tmp_path, 2.0, 3.0),
    ]
    t._pre_cache_aux()
    assert t.driver.encode_calls == 3
    adir = t._audio_cache_dir(str(tmp_path / "cache"))
    files = sorted(os.listdir(adir))
    assert files == [
        "clip.mkv__t0.0-1.0.safetensors",
        "clip.mkv__t1.0-2.0.safetensors",
        "clip.mkv__t2.0-3.0.safetensors",
    ]


def test_rerun_is_idempotent_per_window(tmp_path, monkeypatch):
    _patch_decode(monkeypatch)
    t = _trainer()
    t.inventory = [_window_item(tmp_path, 0.0, 1.0), _window_item(tmp_path, 1.0, 2.0)]
    t._pre_cache_aux()
    assert t.driver.encode_calls == 2
    t._pre_cache_aux()  # all cached now
    assert t.driver.encode_calls == 2  # no new encodes


def test_audio_precache_noop_when_audio_disabled(tmp_path, monkeypatch):
    # WAN-like: train_audio False → _pre_cache_aux early-returns, 0 encodes,
    # no audio cache dir created.
    _patch_decode(monkeypatch)
    t = _trainer()
    t.driver.train_audio = False
    t.inventory = [_window_item(tmp_path, 0.0, 1.0), _window_item(tmp_path, 1.0, 2.0)]
    t._pre_cache_aux()
    assert t.driver.encode_calls == 0
    assert not os.path.exists(t._audio_cache_dir(str(tmp_path / "cache")))
