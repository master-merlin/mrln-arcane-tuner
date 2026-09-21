"""An H3 job's front half and one training step, through the REAL seams.

Why this module exists (LANE-92 VERIFY round 2): four defects in two review
rounds were all "a setting or an input kind the trainer accepts and then
handles wrongly", and the test that claimed stills train had been handed
fabricated 5-D latents — it stepped around the seam that breaks. Everything a
test in this lane claims about an input goes through here instead:

    real files on disk (mp4 with/without a soundtrack, a PNG)
    -> ``prepare_data``              (dataset API faked at the HTTP client only)
    -> ``_validate_latent_cache`` -> ``_pre_cache_latents``
       -> ``LatentManager`` -> ``H3PixelAdaptedVAE`` -> a tiny REAL
          ``diffusers.AutoencoderKLMiniMaxH3`` (random weights, the real
          ``_encode`` and its rank/chunking rules)
    -> ``_pre_cache_text_embeddings`` (stub Qwen3-VL, the real disk cache)
    -> ``_pre_cache_aux``            (real ``load_audio_waveform`` on the mp4)
    -> ``_iter_training_batches`` -> ``_get_batch`` -> ``load_cached_latents``
    -> ``encode_text`` -> the hooks in the order of ``pipeline_train.py:547-584``
       on the tiny REAL ``MiniMaxH3Transformer3DModel``.

Stubs sit BELOW the seams only: the text encoder (constant hidden states) and
the audio VAE (a time-local projection that asserts its input rank).
"""

from __future__ import annotations

import asyncio
import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import torch

from app.engine.models.families.minimax_h3.pixel_adapter import H3PixelAdaptedVAE
from app.engine.models.families.minimax_h3.trainer import MiniMaxH3Trainer
from app.engine.models.registry import ModelRegistry
from app.engine.tests.h3_text_stubs import StubProcessor, StubQwen3VL
from app.engine.tests.test_minimax_h3_clock import _PaddingAudioVAE

H3 = "minimax-h3-t2va"
SR = 32000
SIDE = 64
FPS = 24.0


def definition(def_id: str = H3):
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    return ModelRegistry._definitions[def_id]


def tiny_real_vae() -> H3PixelAdaptedVAE:
    """The INSTALLED visual VAE class at a toy width — its real ``_encode``
    (17-frame chunks, the 1-frame branch, the 5-D rank it indexes) — wrapped
    once in the pixel adapter, as the loader does."""
    from diffusers import AutoencoderKLMiniMaxH3

    torch.manual_seed(0)
    inner = AutoencoderKLMiniMaxH3(
        block_out_channels=(8, 8, 8, 8, 8, 8),
        layers_per_block=1,
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
    ).eval()
    return H3PixelAdaptedVAE(inner)


def write_clip(
    path, frames: int, *, soundtrack: bool, fps: float = FPS, sound_from_s: float = 0.0,
    caption: str | None = None, side: int = SIDE,
) -> dict:
    """A real mp4 (H.264, AAC when ``soundtrack``) and its /pairs row. The tone
    starts at ``sound_from_s`` — silence before it — so a window of the clip
    can be told from another by its audio."""
    from app.engine.components.video import VideoFrameLoader

    g = torch.Generator().manual_seed(frames)
    video = (torch.rand(3, frames, side, side, generator=g) * 2.0 - 1.0).float()
    audio = None
    if soundtrack:
        n = int(math.ceil(frames / fps * SR))
        tone = torch.sin(torch.arange(n, dtype=torch.float32) * 0.05) * 0.5
        tone[: int(sound_from_s * SR)] = 0.0
        audio = (torch.stack([tone, tone]), SR)
    VideoFrameLoader().encode_video(video, audio, fps, str(path))
    return {
        "media_file": path.name,
        "caption_content": f"a clip of {frames} frames" if caption is None else caption,
        "metadata": {
            "width": side, "height": side, "is_video": True, "enabled": True,
            "fps": fps, "duration_s": frames / fps,
        },
    }


def write_still(path) -> dict:
    from PIL import Image

    Image.new("RGB", (SIDE, SIDE), (120, 30, 200)).save(path)
    return {
        "media_file": path.name,
        "caption_content": "a still",
        "metadata": {"width": SIDE, "height": SIDE, "is_video": False, "enabled": True},
    }


def fake_api(monkeypatch, ds_path, pairs: list[dict], kind: str = "standard") -> None:
    info = {"path": str(ds_path), "version": "1.0.0", "kind": kind}

    async def _get(_self, url, *args, **kwargs):
        body = pairs if url.endswith("/pairs") else info
        return SimpleNamespace(status_code=200, json=lambda: json.loads(json.dumps(body)))

    monkeypatch.setattr(httpx.AsyncClient, "get", _get)


def base_config(**config) -> dict:
    return {
        "resolutions": [SIDE],
        "datasets": [{"dataset_name": "ds"}],
        "num_frames": 22,
        "train_batch_size": 1,
        "sample_prompts": [],
        **config,
    }


def shell(config: dict) -> MiniMaxH3Trainer:
    """A trainer without the heavy base ``__init__``; ``_setup_family`` is the
    REAL one and runs here, exactly where the orchestrator runs it: before any
    component exists."""
    t = object.__new__(MiniMaxH3Trainer)
    t.device = torch.device("cpu")
    t.definition = definition()
    t.logger = MagicMock()
    t._log_writer = None
    t.text_cache = {}
    t._te_unloaded = False
    t.autocast_dtype = torch.float32
    t.config = config
    t._setup_family()
    return t


def load_components(t: MiniMaxH3Trainer, build_tiny_transformer) -> None:
    t.components = {
        "vae": tiny_real_vae(),
        "audio_vae": _PaddingAudioVAE(),
        "tokenizer": StubProcessor(),
        "text_encoder": StubQwen3VL(hidden_size=16),
    }
    t._assign_components()
    t._tiny_transformer = build_tiny_transformer().eval()


def run_front_half(t: MiniMaxH3Trainer) -> None:
    """``run_trainer.py``'s order up to the first step."""
    asyncio.run(t.prepare_data())
    t._validate_latent_cache()
    asyncio.run(t._pre_cache_latents())
    t._pre_cache_aux()
    t._pre_cache_text_embeddings()
    t._offload_text_encoders()
    t.driver.assign_components({**t.components, "transformer": t._tiny_transformer})


def events(t: MiniMaxH3Trainer, name: str, level: str = "info") -> list[dict]:
    calls = getattr(t.logger, level).call_args_list
    return [c.kwargs for c in calls if c.args and c.args[0] == name]


def train_step(t: MiniMaxH3Trainer, items: list[dict], grad_accum: int = 1, seed: int = 11):
    """One micro-step, line for line what ``pipeline_train.py:422-584`` does
    for a video family with a warm cache. Returns ``(loss, pred, target, batch)``."""
    batch = t._get_batch(items, decode_pixels=False)
    extra_keys = batch.get("extra_keys")
    ek = {"extra_keys": extra_keys} if extra_keys else {}
    latents = t.latent_manager.load_cached_latents(
        batch["ids"], batch["cache_dirs"], source_paths=batch["paths"],
        device=t.device, dtype=t.autocast_dtype, **ek,
    )
    assert latents is not None, f"latent cache miss for {batch['ids']} after the pre-cache"
    text_emb = t.encode_text(batch["captions"], t.autocast_dtype, batch=batch)
    torch.manual_seed(seed)
    noise = torch.randn_like(latents)
    prepared_latents = t.prepare_latents_for_training(latents)
    prepared_noise = t.prepare_noise_for_training(noise)
    timesteps = t.sample_timesteps(prepared_latents.shape[0], latents)
    noisy_input = t.add_noise(prepared_latents, prepared_noise, timesteps)
    pred = t.forward_pass(noisy_input, timesteps, text_emb, batch)
    target = t.compute_target(prepared_latents, prepared_noise, timesteps)
    loss = t._compute_step_loss(pred, target, timesteps, batch, grad_accum)
    return loss, pred, target, batch
