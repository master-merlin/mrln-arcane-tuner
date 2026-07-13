"""ACE-Step 1.5 audio data-pipeline unit tests.

Exercises the additive audio branch in ``PipelineDataMixin`` directly
(``_append_audio_item``), the ``_pre_cache_latents`` audio branch in
``PipelineCachingMixin`` (fake items + a real ``LatentManager`` over a tiny
real ``AutoencoderOobleck``, ordered the way ``run_trainer`` orders it:
``prepare_data`` sets the ``_audio_*`` attrs, THEN pre-caching runs), plus
the standalone ``components/audio.py`` helpers — no dataset API server, no
real audio files beyond tiny synthetic WAVs written to tmp dirs.
"""

from __future__ import annotations

import asyncio
import os

import numpy as np
import pytest
import soundfile as sf
import structlog
import torch

from app.engine.components.audio import AudioClipLoader, round_duration_bucket
from app.engine.core.pipeline.pipeline_caching import PipelineCachingMixin
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


# ── _pre_cache_latents audio branch ──────────────────────────────────────


class _CachingHost(PipelineCachingMixin, PipelineDataMixin):
    """Combined host: PipelineDataMixin supplies ``_append_audio_item`` (item
    building) and PipelineCachingMixin supplies ``_pre_cache_latents`` — the
    same MRO shape the real ``GenericTrainingPipeline`` gives a trainer."""


def _make_caching_host(tmp_path, *, sample_rate=800, channels=2, latent_hz=100.0):
    """Host with the ``_audio_*`` attrs prepare_data() would set BEFORE
    ``_validate_latent_cache``/``_pre_cache_latents`` run (run_trainer order),
    plus a REAL LatentManager over a tiny REAL AutoencoderOobleck."""
    from diffusers import AutoencoderOobleck

    from app.engine.components.latents import LatentManager

    h = _CachingHost()
    h.logger = structlog.get_logger("test")
    h.device = torch.device("cpu")
    h.config = {"cache_latents": True, "duration_s": 30.0}
    # prepare_data()-set audio contract attrs (see pipeline_data.py).
    h._audio_target_sample_rate = sample_rate
    h._audio_target_channels = channels
    h._audio_latent_hz = latent_hz
    h._audio_duration_cap = 30.0
    vae = AutoencoderOobleck(
        encoder_hidden_size=8,
        downsampling_ratios=[2, 4],
        channel_multiples=[1, 2],
        decoder_channels=8,
        decoder_input_channels=8,
        audio_channels=channels,
        sampling_rate=sample_rate,
    )
    vae.eval()
    h.latent_manager = LatentManager(vae, device="cpu")
    return h


def _audio_inventory_item(host, tmp_path, *, duration_source_s=0.6):
    """One REAL audio inventory item: a synthetic WAV on disk, built through
    the SAME ``_append_audio_item`` path the real prepare_data() uses."""
    wav_path = os.path.join(str(tmp_path), "song.wav")
    sr = host._audio_target_sample_rate
    sf.write(
        wav_path,
        (np.random.randn(int(duration_source_s * sr), 1) * 0.1).astype(np.float32),
        sr,
    )
    inventory: list[dict] = []
    host._append_audio_item(
        inventory,
        img_path=wav_path,
        img_rel="song.wav",
        caption="a song",
        meta={"is_audio": True, "duration_s": duration_source_s, "sample_rate": sr, "channels": 1},
        lyrics_content="",
        ds_path=str(tmp_path),
        model_name="ace-step-1.5",
        ds_version="1.0.0",
        prefix="",
        repeats=1,
        ds_config={},
        ds_use_captions=True,
        ds_use_model_aware=True,
    )
    return inventory


def test_pre_cache_latents_audio_branch_end_to_end(tmp_path):
    """The audio branch of ``_pre_cache_latents`` (pipeline_caching.py):
    decodes via AudioClipLoader (not PIL, not VideoFrameLoader), encodes
    through the real VAE, and writes the latent under the SAME
    content-addressed filename inside the ``{sr}Hz-{window_s}s`` cache dir
    that the train loop later looks up (extra_key "" for audio)."""
    host = _make_caching_host(tmp_path)
    inventory = _audio_inventory_item(host, tmp_path)
    host.inventory = inventory
    item = inventory[0]

    # Pin the cache-KEY format before running: dir leaf is the duration-window
    # res_str analogue of video's WxHxNfFPS.
    assert os.path.basename(item["cache_dir"]) == "800Hz-5s"
    assert f"{host._audio_target_sample_rate}Hz-{item['window_s']:g}s" == "800Hz-5s"

    host._validate_latent_cache()
    assert host._latent_cache_missing == 1  # cold cache

    asyncio.run(host._pre_cache_latents())

    # The file exists under the content-addressed name (extra_key "" — audio
    # has no trim window) and the train loop's own lookup finds it.
    fname = host.latent_manager.latent_filename(item["id"], item["path"], "")
    assert os.path.isfile(os.path.join(item["cache_dir"], fname))
    loaded = host.latent_manager.load_cached_latents(
        [item["id"]], [item["cache_dir"]], source_paths=[item["path"]]
    )
    assert loaded is not None
    # [B, D, T_lat]: window 5s @ 800Hz = 4000 samples / downsample 8 = 500.
    assert loaded.shape == (1, 4, 500)

    # Re-validate: coverage is now complete (a second pre-cache is a no-op).
    host._validate_latent_cache()
    assert host._latent_cache_missing == 0


def test_pre_cache_latents_audio_never_touches_pil_or_video_loader(
    tmp_path, monkeypatch
):
    """Audio items must route through AudioClipLoader ONLY — Image.open on a
    WAV was the exact bug class the video branch's test pins for .mkv."""
    host = _make_caching_host(tmp_path)
    host.inventory = _audio_inventory_item(host, tmp_path)
    host._latent_cache_missing = 1

    import app.engine.core.pipeline.pipeline_caching as cmod

    def _boom(*a, **k):
        raise AssertionError("Image.open called on an audio clip")

    monkeypatch.setattr(cmod.Image, "open", _boom)

    import app.engine.components.video as vmod

    class _BoomLoader:
        def load_clip(self, *a, **k):
            raise AssertionError("VideoFrameLoader called on an audio clip")

    monkeypatch.setattr(vmod, "VideoFrameLoader", _BoomLoader)

    asyncio.run(host._pre_cache_latents())  # must not raise


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
