"""ACE-Step 1.5 audio data-pipeline unit tests.

Exercises the additive audio branch in ``PipelineDataMixin`` directly
(``_append_audio_item``) plus the standalone ``components/audio.py`` helpers
it depends on — no dataset API server, no real audio files beyond a tiny
synthetic WAV written to a tmp dir.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import soundfile as sf
import torch

from app.engine.components.audio import AudioClipLoader, round_duration_bucket
from app.engine.core.pipeline.pipeline_data import AUDIO_DUMMY_DIM, PipelineDataMixin


class _FakePipeline(PipelineDataMixin):
    """Bare host object carrying only the attrs ``_append_audio_item`` reads."""

    def __init__(self):
        self._audio_target_sample_rate = 48000
        self._audio_target_channels = 2
        self._audio_latent_hz = 25.0
        self._audio_duration_cap = 30.0


# ── round_duration_bucket ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "duration,cap,step,expected",
    [
        (3.2, 30.0, 5.0, 5.0),
        (0.0, 30.0, 5.0, 5.0),
        (5.0, 30.0, 5.0, 5.0),
        (5.01, 30.0, 5.0, 10.0),
        (40.0, 30.0, 5.0, 30.0),  # capped
        (-1.0, 30.0, 5.0, 5.0),  # negative clamps to 0 then buckets to step
    ],
)
def test_round_duration_bucket(duration, cap, step, expected):
    assert round_duration_bucket(duration, cap, step) == expected


# ── AudioClipLoader ───────────────────────────────────────────────────────


def _write_wav(tmp_path, samples: int, sample_rate: int, channels: int = 1) -> str:
    path = os.path.join(str(tmp_path), "clip.wav")
    data = (np.random.randn(samples, channels) * 0.1).astype(np.float32)
    sf.write(path, data, sample_rate)
    return path


def test_audio_clip_loader_pads_short_clip(tmp_path):
    path = _write_wav(tmp_path, samples=200, sample_rate=800, channels=1)
    clip = AudioClipLoader().load_clip(
        path, target_sample_rate=800, target_channels=2, window_s=1.0
    )
    assert clip.shape == (2, 800)
    assert clip.dtype == torch.float32
    # Tail beyond the source's 200 samples must be silence.
    assert torch.all(clip[:, 200:] == 0.0)


def test_audio_clip_loader_trims_long_clip(tmp_path):
    path = _write_wav(tmp_path, samples=5000, sample_rate=1000, channels=2)
    clip = AudioClipLoader().load_clip(
        path, target_sample_rate=1000, target_channels=2, window_s=1.0
    )
    assert clip.shape == (2, 1000)
    assert clip.abs().max().item() <= 1.0


def test_audio_clip_loader_resamples_and_upmixes_mono(tmp_path):
    path = _write_wav(tmp_path, samples=800, sample_rate=800, channels=1)
    clip = AudioClipLoader().load_clip(
        path, target_sample_rate=1600, target_channels=2, window_s=0.5
    )
    # window_s=0.5 @ 1600Hz -> 800 samples, both channels identical (mono upmix).
    assert clip.shape == (2, 800)
    assert torch.allclose(clip[0], clip[1])


def test_audio_clip_loader_downmixes_stereo_to_mono(tmp_path):
    path = _write_wav(tmp_path, samples=800, sample_rate=800, channels=2)
    clip = AudioClipLoader().load_clip(
        path, target_sample_rate=800, target_channels=1, window_s=1.0
    )
    assert clip.shape == (1, 800)


# ── _append_audio_item ───────────────────────────────────────────────────


def test_append_audio_item_builds_dummy_spatial_item(tmp_path):
    pipeline = _FakePipeline()
    inventory: list[dict] = []
    meta = {"is_audio": True, "duration_s": 12.0, "sample_rate": 44100, "channels": 1}
    pipeline._append_audio_item(
        inventory,
        img_path=str(tmp_path / "song.wav"),
        img_rel="song.wav",
        caption="a happy song",
        meta=meta,
        lyrics_content="la la la",
        ds_path=str(tmp_path),
        model_name="ace-step-1.5",
        ds_version="1.0.0",
        prefix="",
        repeats=2,
        ds_config={"caption_dropout_rate": 0.1},
        ds_use_captions=True,
        ds_use_model_aware=True,
    )
    assert len(inventory) == 2  # repeats=2
    item = inventory[0]
    assert item["is_audio"] is True
    assert item["is_video"] is False
    assert item["target_w"] == AUDIO_DUMMY_DIM
    assert item["target_h"] == AUDIO_DUMMY_DIM
    assert item["duration_s"] == 12.0
    assert item["window_s"] == 15.0  # round_duration_bucket(12.0, 30.0, 5.0)
    assert item["target_frames"] == round(15.0 * 25.0)
    assert item["lyrics_content"] == "la la la"
    assert item["source_sample_rate"] == 44100
    assert item["source_channels"] == 1
    assert item["has_masked"] is False
    assert "48000Hz-15s" in item["cache_dir"]


def test_append_audio_item_defaults_sample_rate_channels_when_missing():
    pipeline = _FakePipeline()
    inventory: list[dict] = []
    meta = {"is_audio": True, "duration_s": 3.0}  # no sample_rate/channels
    pipeline._append_audio_item(
        inventory,
        img_path="x.wav",
        img_rel="x.wav",
        caption="",
        meta=meta,
        lyrics_content="",
        ds_path="/ds",
        model_name="m",
        ds_version="1.0.0",
        prefix="",
        repeats=1,
        ds_config={},
        ds_use_captions=True,
        ds_use_model_aware=True,
    )
    item = inventory[0]
    assert item["source_sample_rate"] == 48000  # falls back to family default
    assert item["source_channels"] == 2
