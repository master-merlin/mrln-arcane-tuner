"""MiniMax-H3 trainer — the text-embedding lifecycle (plan row 2.1).

Production caller (DECISION-68 (a)): ``run_trainer.py:159`` calls
``_pre_cache_text_embeddings`` (a base NO-OP) and then ``_offload_text_encoders``;
the base ``encode_text`` (``pipeline_base.py``) returns ``None`` once the driver
reports no encoders. So the three trainer overrides this file pins are what
keeps a job from training on ``None`` embeddings after the 63 GB Qwen3-VL is
released. Stub encoder, no weights; the seams under test (the base offload,
the base caption-hint builder, the disk cache) are REAL.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.models.families.minimax_h3.driver import MiniMaxH3Driver
from app.engine.models.families.minimax_h3.trainer import MiniMaxH3Trainer
from app.engine.models.registry import ModelRegistry
from app.engine.tests.h3_text_stubs import StubProcessor, StubQwen3VL


def _definition(def_id: str = "minimax-h3-t2va"):
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    return ModelRegistry._definitions[def_id]


def _trainer(tmp_path, *, sample_prompt: str = "a [triggerword] clip") -> MiniMaxH3Trainer:
    """A real trainer + real driver + stub encoder, no heavy ``__init__``
    (the ``test_boogu_image_trainer.py::_trainer_shell`` shape)."""
    t = object.__new__(MiniMaxH3Trainer)
    t.device = torch.device("cpu")
    t.definition = _definition()
    t.driver = MiniMaxH3Driver(t.definition, t.device)
    t.logger = MagicMock()
    t._log_writer = None
    t.text_cache = {}
    t._te_unloaded = False
    t.config = {
        "cache_text_embeddings": True,
        "te_quantization": "none",
        "global_triggerword": "sks",
        "sample_prompts": [{"prompt": sample_prompt}],
        "datasets": [],
    }
    # One training clip: the base `_build_caption_hints` composes
    # "sks, a cat" (trigger + caption) and "" (dropout) from it.
    t.inventory = [{"id": "1", "path": str(tmp_path / "clip.mp4"), "caption": "a cat"}]
    t._resolve_te_cache_dirs = lambda: [str(tmp_path / "te")]
    t.components = {
        "tokenizer": StubProcessor(),
        "text_encoder": StubQwen3VL(hidden_size=8),
        "vae": object(),
        "audio_vae": object(),
    }
    t._assign_components()
    return t


def _encoder_calls(t: MiniMaxH3Trainer) -> int:
    return t.components["text_encoder"].calls


def test_te_cache_serves_embeddings_after_release(tmp_path):
    t = _trainer(tmp_path)
    te = t.components["text_encoder"]

    # ── warm while the encoder is resident ──
    t._pre_cache_text_embeddings()
    warmed = dict(t.text_cache)
    calls_after_warm = te.calls

    # ── release every encoder reference through the REAL base offload ──
    t._offload_text_encoders()
    assert t.driver.get_text_encoders() == {}
    assert "text_encoder" not in t.components
    assert t.driver.text_encoder is None
    assert t.text_cache, "cache empty after release"
    assert {"sks, a cat", "", "a sks clip"} <= set(warmed), sorted(warmed)
    assert calls_after_warm > 0

    # ── serve: a training caption, the empty prompt, a sample prompt ──
    out = t.encode_text(["sks, a cat", "", "a sks clip"], torch.float32)
    assert out is not None, "encode_text returned None with no encoders"
    assert isinstance(out, TextEncoderOutput)
    assert te.calls == calls_after_warm, "the released encoder was called"
    emb, mask = out.embeddings, out.attention_mask
    assert emb.shape[0] == 3 and mask.shape[0] == 3
    for i, cap in enumerate(["sks, a cat", "", "a sks clip"]):
        w_emb, w_mask = warmed[cap]
        n = int(w_mask.sum())
        assert int(mask[i].sum()) == n
        assert torch.equal(emb[i, :n], w_emb[:n].to(emb.dtype))
        assert torch.all(emb[i, n:] == 0)

    # ── a caption never warmed raises, naming it — never a silent reload ──
    with pytest.raises(RuntimeError, match="never-warmed caption"):
        t.encode_text(["never-warmed caption"], torch.float32)
    assert te.calls == calls_after_warm


def test_pre_cache_round_trips_through_the_disk_cache(tmp_path):
    """A second trainer on the same dataset serves from disk with ZERO
    encoder calls; the disk path carries the driver's cache scope (tap +
    tokenizer), so a different tap is a miss, never a stale hit."""
    first = _trainer(tmp_path)
    first._pre_cache_text_embeddings()
    n_first = _encoder_calls(first)
    assert n_first > 0

    second = _trainer(tmp_path)
    second._pre_cache_text_embeddings()
    assert _encoder_calls(second) == 0, "disk cache not consulted"
    assert set(second.text_cache) == set(first.text_cache)
    for cap, (emb, mask) in first.text_cache.items():
        emb2, mask2 = second.text_cache[cap]
        assert torch.equal(emb, emb2) and torch.equal(mask, mask2)

    third = _trainer(tmp_path)
    third.definition = third.definition.model_copy(
        update={
            "architecture_params": {
                **third.definition.architecture_params,
                "te.hidden_state_tap_index": 49,
            }
        }
    )
    third.driver.definition = third.definition
    third._pre_cache_text_embeddings()
    assert _encoder_calls(third) == n_first, "a different tap reused layer-50 embeddings"
    assert torch.all(third.text_cache["sks, a cat"][0] == 49.0)


def test_caching_off_is_refused_loudly(tmp_path):
    """H3 cannot keep the 63 GB encoder resident beside the DiT, so a config
    that asks for live per-step encoding is refused up front, not at step 1."""
    t = _trainer(tmp_path)
    t.config["cache_text_embeddings"] = False
    with pytest.raises(ValueError, match="cache_text_embeddings"):
        t._pre_cache_text_embeddings()


def test_transformer_is_materialised_only_after_the_release(tmp_path):
    """The loader defers the 62 GB DiT out of Phase A; the trainer materialises
    it through the loader's single-spec path ONLY once the encoder is gone
    (asserted on the driver and the component set), logging
    ``text_encoder released`` before ``transformer loaded``."""
    from app.engine.models.families.minimax_h3.loader import MiniMaxH3Loader

    t = _trainer(tmp_path)
    loader = MiniMaxH3Loader(t.device, defer_transformer=True)
    manifest = {s.key for s in loader.get_component_manifest(t.definition)}
    assert "transformer" not in manifest
    assert {"tokenizer", "text_encoder", "vae", "audio_vae"} == manifest
    eager = {s.key for s in MiniMaxH3Loader(t.device).get_component_manifest(t.definition)}
    assert "transformer" in eager

    dit = torch.nn.Linear(2, 2)
    seen: list = []

    def _single(spec, definition, root_path, dtype, target_device):
        seen.append((spec.key, spec.subfolder, dtype, target_device))
        return dit

    loader._load_single_spec = _single
    loader._root_path = str(tmp_path)
    t.loader = loader

    with pytest.raises(RuntimeError, match="text encoder still resident"):
        t._materialise_transformer()
    assert seen == []

    t._pre_cache_text_embeddings()
    t._offload_text_encoders()
    t._materialise_transformer()

    assert seen == [("transformer", "transformer", torch.bfloat16, "cpu")]
    assert t.components["transformer"] is dit
    # The base pipeline's primary key (`_move_component_to_gpu("unet")`, the
    # PEFT wrap, the saver) — aliased, or the DiT never reaches the GPU.
    assert t.components["unet"] is dit
    assert t.driver.get_primary_model() is dit
    assert t.transformer is dit

    events = [c.args[0] for c in t.logger.info.call_args_list]
    assert "text_encoder released" in events and "transformer loaded" in events
    assert events.index("text_encoder released") < events.index("transformer loaded")
    released = next(c for c in t.logger.info.call_args_list if c.args[0] == "text_encoder released")
    assert released.kwargs["weight_bytes"] == 1024 * 4
    assert "host_peak_bytes" in released.kwargs and "cuda_peak_bytes" in released.kwargs


# ── Row 2.5: trainer setup + lifecycle ───────────────────────────────────

_SLIDING_MSG = (
    "minimax_h3: temporal_coverage='sliding' is not supported in this release "
    "(its latent window assumes (F-1)/t+1 frames; H3 chunks 17n+5 -> 5n+2) — "
    "use 'first' or 'tiled'"
)


def _setup_shell(tmp_path, **config) -> MiniMaxH3Trainer:
    t = _trainer(tmp_path)
    t.config.update(config)
    return t


def test_setup_family_no_longer_raises(tmp_path):
    t = _setup_shell(tmp_path)
    assert t._setup_family() is None


def test_setup_family_wires_driver_and_loader(tmp_path):
    from app.engine.core.video_contract import resolve_video_profile
    from app.engine.models.families.minimax_h3.loader import MiniMaxH3Loader

    t = _setup_shell(tmp_path, train_audio=True)
    t._setup_family()
    assert isinstance(t.driver, MiniMaxH3Driver)
    assert isinstance(t.loader, MiniMaxH3Loader) and t.loader.defer_transformer is True
    assert t.driver.settings is not None, "driver.settings is None"
    assert t.driver.settings.train_audio is True
    assert t._resolve_train_audio() is True
    assert t._resolve_loading_dtype() is torch.bfloat16
    # First step of the row (round 15 MAJOR 15.02): a VIDEO family with a
    # soundtrack, not an audio-primary family.
    assert t.is_video_family is True
    assert resolve_video_profile(t.definition).has_audio is True
    assert t.is_audio_family is False


def test_setup_family_refuses_sliding_before_any_weight_loads(tmp_path, monkeypatch):
    from app.engine.models.families.minimax_h3 import loader as loader_mod

    constructed: list[tuple] = []
    real_init = loader_mod.MiniMaxH3Loader.__init__

    def _recording_init(self, device, **kwargs):
        constructed.append((device, kwargs))
        real_init(self, device, **kwargs)

    monkeypatch.setattr(loader_mod.MiniMaxH3Loader, "__init__", _recording_init)

    t = _setup_shell(tmp_path, temporal_coverage="sliding")
    with pytest.raises(ValueError) as exc:
        t._setup_family()
    msg = str(exc.value)
    assert msg == _SLIDING_MSG
    assert "sliding" in msg and "'first'" in msg and "'tiled'" in msg
    assert constructed == [], "the loader was built before the refusal"

    # Positive control: the two supported coverages reach the loader.
    for coverage in ("first", "tiled"):
        _setup_shell(tmp_path, temporal_coverage=coverage)._setup_family()
    assert len(constructed) == 2 and all(kw == {"defer_transformer": True} for _, kw in constructed)


class _StubAudioVAE(torch.nn.Module):
    """A mono audio VAE by shape: 800-sample hop, 32 latent channels, the
    posterior mean = per-hop channel-wise projection of the waveform."""

    hop = 800

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            latents_mean=[0.1] * 32, latents_std=[2.0] * 32, sampling_rate=32000
        )
        self.proj = torch.nn.Parameter(torch.linspace(-1, 1, 32).view(32, 1), requires_grad=False)

    def encode(self, sample: torch.Tensor):
        b, one, n = sample.shape
        assert one == 1
        frames = sample.reshape(b, 1, n // self.hop, self.hop).abs().mean(dim=-1)  # (b, 1, T)
        mean = self.proj * frames  # (b, 32, T)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: mean))

    def decode(self, latents: torch.Tensor):  # pragma: no cover - not decoded here
        raise AssertionError("never decoded in the trainer tests")


def _audio_shell(tmp_path) -> tuple[MiniMaxH3Trainer, dict]:
    from app.engine.components.latents import LatentManager
    from app.engine.models.families.minimax_h3.pixel_adapter import H3PixelAdaptedVAE
    from app.engine.tests.h3_text_stubs import StubVisualVAE

    t = _setup_shell(tmp_path, train_audio=True, cache_latents=True)
    t._setup_family()
    t.components = {
        "vae": H3PixelAdaptedVAE(StubVisualVAE()),
        "audio_vae": _StubAudioVAE(),
        "tokenizer": StubProcessor(),
        "text_encoder": StubQwen3VL(hidden_size=8),
    }
    t._assign_components()
    t.latent_manager = LatentManager(t.components["vae"], device=t.device)
    clip = str(_STEREO_CLICK_WAV)
    item = {
        "id": "click",
        "path": clip,
        "caption": "a click",
        "cache_dir": str(tmp_path / "ds" / ".cache" / "m" / "1.0.0" / "latents" / "original" / "64x64x24f24.0-h3x"),
        "is_video": True,
        "target_frames": 24,
        "target_fps": 24.0,
        "trim_start_s": 0.0,
        "prefix": "",
    }
    still = {"id": "still", "path": clip, "caption": "a still", "cache_dir": item["cache_dir"], "is_video": False, "prefix": ""}
    t.inventory = [item, still]
    return t, item


_STEREO_CLICK_WAV = Path(__file__).resolve().parent / "fixtures" / "h3_stereo_click.wav"


def test_pre_cache_aux_writes_audio_latents_via_the_family_module(tmp_path):
    from safetensors.torch import load_file

    from app.engine.components.audio_io import load_audio_waveform
    from app.engine.models.families.minimax_h3 import audio_latents as al

    t, item = _audio_shell(tmp_path)
    t._pre_cache_aux()

    adir = t._audio_cache_dir(item["cache_dir"])
    assert os.path.basename(os.path.dirname(adir)) == "audio"
    files = os.listdir(adir)
    assert len(files) == 1, files  # the still wrote nothing
    cached = load_file(os.path.join(adir, files[0]))["audio_latents"]

    wav, sr = load_audio_waveform(item["path"], trim_start_s=0.0, duration_s=1.0, target_sr=32000)
    assert wav.shape == (2, 32000) and sr == 32000
    expected = al.encode_stereo(t.components["audio_vae"], wav.unsqueeze(0))[0]
    assert cached.shape == (2, 32, 40)
    assert torch.allclose(cached, expected), "the cached latent did not come from audio_latents.encode_stereo"

    # A second pass is a no-op (content-addressed skip), and the version
    # segment moves with the audio VAE identity.
    before = os.path.getmtime(os.path.join(adir, files[0]))
    t._pre_cache_aux()
    assert os.path.getmtime(os.path.join(adir, files[0])) == before
    t.components["audio_vae"].config.latents_std = [3.0] * 32
    assert t._audio_cache_dir(item["cache_dir"]) != adir


class _ResidencyAudioVAE(_StubAudioVAE):
    """Tracks WHERE its weights live the way a real module does: it starts
    resident wherever Phase A left it (`host`, i.e. NOT the trainer device —
    `run_trainer.py` moves only `vae` to the GPU), moves on `to()`, and its
    `encode` refuses a waveform on any other device — the exact failure
    GATE-1 hit ("Input type torch.cuda.FloatTensor and weight type
    torch.FloatTensor")."""

    def __init__(self) -> None:
        super().__init__()
        self.where = "host"
        self.events: list[str] = []

    def to(self, *args, **kwargs):  # noqa: D102 - nn.Module signature
        dev = args[0] if args else kwargs.get("device")
        self.where = str(dev)
        self.events.append(f"to:{dev}")
        return self

    def encode(self, sample: torch.Tensor):
        if self.where != str(sample.device):
            raise RuntimeError(
                f"Input type ({sample.device}) and weight type ({self.where}) should be the same"
            )
        self.events.append("encode")
        return super().encode(sample)


def test_pre_cache_aux_moves_the_audio_vae_to_the_trainer_device_first(tmp_path):
    """GATE-1 finding (plan row 2.12): the orchestrator's VAE phase moves
    `vae` to the GPU and nothing moved `audio_vae`, so every clip failed to
    encode and the run refused. The precache must bring the audio VAE to the
    trainer device itself, encode there, and send it back to the CPU."""
    t, item = _audio_shell(tmp_path)
    t.components["audio_vae"] = _ResidencyAudioVAE()
    t._assign_components()
    t._pre_cache_aux()
    vae = t.components["audio_vae"]
    assert vae.events[:2] == [f"to:{t.device}", "encode"], vae.events
    assert vae.events[-1] == "to:cpu", vae.events
    assert os.listdir(t._audio_cache_dir(item["cache_dir"])), "no audio latent was written"


def test_build_batch_extra_carries_audio_latents(tmp_path):
    t, item = _audio_shell(tmp_path)
    t._pre_cache_aux()
    still = t.inventory[1]
    extra = t.build_batch_extra([item, still])
    assert "audio_clean" in extra, "no audio_clean key: the cached audio latents never reached the batch"
    assert extra["audio_clean"].shape == (2, 2, 32, 40)
    assert extra["audio_mask"].tolist() == [1.0, 0.0]
    assert "audio_latents" not in item, "inventory items must not retain tensors"


# ── Row 3.2 on the trainer: audio off keeps the rows packed end to end ──────


def test_train_audio_off_still_caches_and_packs_audio(tmp_path):
    """`train_audio=false` on the TRAINER: the audio cache is still written
    and the cached rows still reach the batch (H3 is single-stream; only
    `audio_loss_weight` goes to 0.0)."""
    t, item = _audio_shell(tmp_path)
    t.config["train_audio"] = False
    t._setup_family()  # rebuilds the driver: re-wire the stub components to it
    t._assign_components()
    assert t.driver.settings.train_audio is False
    t._pre_cache_aux()
    adir = t._audio_cache_dir(item["cache_dir"])
    assert os.path.isdir(adir) and len(os.listdir(adir)) == 1, (
        "audio_rows == 0; H3 is single-stream — train_audio=false skipped the audio cache"
    )
    extra = t.build_batch_extra([item])
    assert "audio_clean" in extra, "audio_rows == 0; H3 is single-stream — the trainer dropped the rows"
    assert extra["audio_clean"].shape == (1, 2, 32, 40)


def test_step0_banner_is_logged_at_setup(tmp_path):
    t = _setup_shell(tmp_path, train_audio=True, seed=77)
    t._setup_family()
    lines = [c.args[0] for c in t.logger.info.call_args_list if c.args and isinstance(c.args[0], str)]
    banners = [line for line in lines if line.startswith("h3_settings ")]
    assert len(banners) == 1, f"expected ONE step-0 banner, got {banners}"
    assert banners[0] == t.driver.step0_banner(u_seed=77)
    assert "train_audio=true (config)" in banners[0] and banners[0].endswith("u_seed=77")


def test_latent_manager_vae_is_the_pixel_adapter(tmp_path, monkeypatch):
    """The loader's post-load hook is the ONE wrap site; `prepare_data`
    builds the LatentManager on `components["vae"]`, so the manager must see
    the adapter — through the real single-spec load path, weights stubbed."""
    import httpx

    from app.engine.models.families.minimax_h3.pixel_adapter import H3PixelAdaptedVAE
    from app.engine.tests.h3_text_stubs import StubVisualVAE
    from app.engine.tests.test_minimax_h3_cache_integration import _dataset, _fake_api

    t = _setup_shell(tmp_path, resolutions=[64], datasets=[{"dataset_name": "ds"}], cache_latents=True)
    t._setup_family()
    spec = {s.key: s for s in t.loader.get_component_manifest(t.definition)}["vae"]
    monkeypatch.setattr(t.loader, "_resolve_component_path", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(t.loader, "_load_component", lambda *a, **k: StubVisualVAE())
    vae = t.loader._load_single_spec(spec, t.definition, str(tmp_path), torch.bfloat16, "cpu")
    t.components = {"vae": vae}
    t._assign_components()
    _fake_api(monkeypatch, _dataset(tmp_path))
    assert httpx.AsyncClient.get is not None
    asyncio.run(t.prepare_data())
    assert isinstance(t.latent_manager.vae, H3PixelAdaptedVAE)
    assert t.latent_manager.vae is t.components["vae"]


def test_update_primary_model_keeps_driver_in_sync(tmp_path):
    t = _setup_shell(tmp_path)
    t._setup_family()
    loaded = torch.nn.Linear(2, 2)
    t.components = {"transformer": loaded, "unet": loaded}
    t._assign_components()
    wrapped = torch.nn.Linear(2, 2)
    t._update_primary_model(wrapped)
    # The driver first: its forward runs whatever it holds.
    assert t.driver.transformer is wrapped, "the driver still holds the unwrapped model"
    assert t._get_primary_model() is wrapped
    assert t.transformer is wrapped
    assert t.components["unet"] is wrapped and t.components["transformer"] is wrapped


def test_text_encoders_empty_after_release(tmp_path):
    t = _trainer(tmp_path)
    te = t.components["text_encoder"]
    assert t._get_text_encoders() == {"text_encoder": te}
    t._pre_cache_text_embeddings()
    t._offload_text_encoders()
    assert t._get_text_encoders() == {}


def test_ref2va_materialises_the_ref_checkpoint(tmp_path):
    from app.engine.models.families.minimax_h3.loader import MiniMaxH3Loader

    loader = MiniMaxH3Loader(torch.device("cpu"), defer_transformer=True)
    seen: list = []
    loader._load_single_spec = lambda spec, *a: seen.append(spec.subfolder)
    loader._root_path = str(tmp_path)
    loader.load_transformer(_definition("minimax-h3-ref2va"), torch.bfloat16)
    assert seen == ["transformer_ref"]


# ── Production loss integration (plan row 2.6, ASTRA MAJOR-6) ────────────────


def _text_output(lengths: list[int], dim: int = 16) -> TextEncoderOutput:
    g = torch.Generator().manual_seed(7)
    L = max(lengths)
    emb = torch.randn(len(lengths), L, dim, generator=g)
    mask = torch.zeros(len(lengths), L, dtype=torch.long)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1
        emb[i, n:] = 0
    return TextEncoderOutput(embeddings=emb, attention_mask=mask)


def _loss_shell(tmp_path, build_tiny_transformer, **config) -> MiniMaxH3Trainer:
    """The real trainer setup on the tiny diffusers arch, audio ON."""
    t = _setup_shell(tmp_path, **{"train_audio": True, "audio_loss_weight": 0.1, **config})
    t._setup_family()
    t.driver.assign_components({"transformer": build_tiny_transformer().eval()})
    # The production state at step time: the caption-dropout entry "" is always
    # pre-cached, and the definition's cfg_augment.scale (4.0) reads it as the
    # uncond row. The refusal on a MISSING "" stays pinned by its own test.
    t.text_cache[""] = (torch.randn(2, 16, generator=torch.Generator().manual_seed(5)), torch.ones(2, dtype=torch.long))
    return t


def _run_step_hooks(t: MiniMaxH3Trainer, grad_accum: int, seed: int = 11):
    """Call the family hooks in the order of ``pipeline_train.py:547-584``
    and return ``(loss, pred, target, batch)``."""
    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(2, 24, 2, 6, 4, generator=g)
    noise = torch.randn(2, 24, 2, 6, 4, generator=g)
    batch = {
        "audio_clean": torch.randn(2, 2, 32, 3, generator=g),
        "audio_mask": torch.ones(2),
    }
    torch.manual_seed(seed)
    prepared_latents = t.prepare_latents_for_training(latents)
    prepared_noise = t.prepare_noise_for_training(noise)
    timesteps = t.sample_timesteps(prepared_latents.shape[0], latents)
    noisy_input = t.add_noise(prepared_latents, prepared_noise, timesteps)
    with torch.no_grad():
        pred = t.forward_pass(noisy_input, timesteps, _text_output([5, 8]), batch)
    target = t.compute_target(prepared_latents, prepared_noise, timesteps)
    loss = t._compute_step_loss(pred, target, timesteps, batch, grad_accum)
    return loss, pred, target, batch


def _expected_components(t: MiniMaxH3Trainer, pred, target, batch):
    """The three numbers recomputed from the same tensors, outside the trainer."""
    video_pred = pred[0] if isinstance(pred, tuple) else pred
    loss_video = torch.nn.functional.mse_loss(video_pred.float(), target.float())
    loss_audio = torch.nn.functional.mse_loss(batch["audio_pred"].float(), batch["audio_target"].float())
    return loss_video, loss_audio, loss_video + 0.1 * loss_audio


def test_step_loss_routes_through_driver(tmp_path, build_tiny_transformer):
    from app.engine.core.pipeline.pipeline_base import PipelineBaseMixin

    assert MiniMaxH3Trainer._compute_step_loss is not PipelineBaseMixin._compute_step_loss, (
        "the base MSE would run on H3's (video, audio) pair"
    )
    t = _loss_shell(tmp_path, build_tiny_transformer)
    loss, pred, target, batch = _run_step_hooks(t, grad_accum=1)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss_video, loss_audio, expected = _expected_components(t, pred, target, batch)
    assert loss_audio > 0, "the audio term is zero: no audio reached the loss"
    assert torch.allclose(loss, expected, atol=1e-6), (
        f"loss {float(loss):.6f} != loss_video + 0.1 * loss_audio {float(expected):.6f}"
    )
    # The audio stream was NOISED on its own clock before the forward
    # (report-back 2.4): x_t != x_0, and the target is the inverted velocity.
    audio_noisy = batch["audio_noisy"]
    assert audio_noisy.shape == batch["audio_clean"].shape
    assert not torch.allclose(audio_noisy, batch["audio_clean"]), "audio was fed clean"
    assert torch.allclose(batch["audio_target"], batch["audio_clean"] - batch["audio_noise"])


def test_train_audio_off_noises_the_packed_audio_and_costs_nothing(tmp_path, build_tiny_transformer):
    """Audio off on the production hooks: the rows are still noised on the
    audio clock and forwarded (an audio velocity comes back), and the step
    loss is exactly the video term."""
    t = _loss_shell(tmp_path, build_tiny_transformer, train_audio=False)
    assert t.driver.settings.audio_loss_weight == 0.0
    loss, pred, target, batch = _run_step_hooks(t, grad_accum=1)
    assert pred[1] is not None, "audio_rows == 0; H3 is single-stream"
    assert "audio_noisy" in batch and not torch.allclose(batch["audio_noisy"], batch["audio_clean"]), "audio was fed clean"
    loss_video = torch.nn.functional.mse_loss(pred[0].float(), target.float())
    assert torch.allclose(loss, loss_video, atol=1e-6)


def test_trainer_serves_the_uncond_row_when_cfg_augment_is_on(tmp_path, build_tiny_transformer):
    """Row 3.1's owed wiring: with `cfg_augment_scale > 1` the trainer puts the
    EMPTY-prompt embedding (always pre-cached — the caption-dropout entry)
    into `batch["text_embeddings_uncond"]` before the driver forward; at
    scale 1.0 no row is served and no uncond forward runs."""
    t = _loss_shell(tmp_path, build_tiny_transformer, cfg_augment_scale=4.0)
    t.text_cache[""] = (torch.randn(2, 16), torch.ones(2, dtype=torch.long))
    _loss, _pred, _target, batch = _run_step_hooks(t, grad_accum=1)
    unc = batch["text_embeddings_uncond"]
    assert torch.allclose(unc.embeddings[0].cpu(), t.text_cache[""][0]), "the uncond row is not the cached empty prompt"
    assert int(unc.attention_mask.sum()) == 2
    assert "video_pred_uncond" in batch, "the uncond forward never ran"

    off = _loss_shell(tmp_path, build_tiny_transformer, cfg_augment_scale=1.0)
    off.text_cache[""] = t.text_cache[""]
    _loss, _pred, _target, batch_off = _run_step_hooks(off, grad_accum=1)
    assert "text_embeddings_uncond" not in batch_off and "video_pred_uncond" not in batch_off


def test_cfg_augment_without_a_cached_empty_prompt_refuses_by_name(tmp_path, build_tiny_transformer):
    """The definition default (4.0) with NO `""` entry in the text cache: the
    step refuses loudly instead of training un-augmented."""
    t = _loss_shell(tmp_path, build_tiny_transformer)
    assert float(t.settings.cfg_augment_scale) == 4.0, "the definition default no longer reaches the trainer"
    del t.text_cache[""]
    with pytest.raises(RuntimeError, match="not pre-cached: ''"):
        _run_step_hooks(t, grad_accum=1)


def test_step_loss_scales_by_grad_accum(tmp_path, build_tiny_transformer):
    t = _loss_shell(tmp_path, build_tiny_transformer)
    loss1, pred, target, batch = _run_step_hooks(t, grad_accum=1)
    loss4 = t._compute_step_loss(pred, target, torch.tensor([0.7, 0.2]), batch, 4)
    assert torch.allclose(loss4 * 4, loss1, atol=1e-6), (
        f"grad_accum=4 loss {float(loss4):.6f} is not loss/4 ({float(loss1) / 4:.6f})"
    )


# ── Production sampling path (plan row 2.10; round 13 MAJOR 13.01, DECISION-67 (a)) ──
#
# `pipeline_base.py:_create_sampler` returns None, so `pipeline_optimization.py`
# sets `self.sampler = None` and the loop never samples — 28 families override
# it, this one did not. The test drives the REAL `train()` coroutine (the
# `tests/engine/test_nan_window_skip.py` shape: every hook the loop touches
# stubbed to trivial CPU tensors, the DB patched away) with the family's own
# `_create_sampler`, and counts the `generate_samples` calls.


class _LoopShell(MiniMaxH3Trainer):
    """MiniMaxH3Trainer with only the loop's data/forward/loss hooks stubbed;
    `_create_sampler` is the family's REAL override."""

    def __init__(self, tmp_path):  # noqa: D107 - never calls the heavy base __init__
        import time

        self.logger = MagicMock()
        self.definition = _definition()
        self.config = {
            "max_train_steps": 2,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "cache_latents": False,
            "save_every_n_steps": 0,
            "sample_before_training": False,
            "sample_every_n_steps": 2,
            "sample_prompts": [{"prompt": "a [triggerword] clip"}],
            "noise_offset": 0.0,
        }
        self.inventory = [{"id": 0}]
        self.device = torch.device("cpu")
        self.autocast_dtype = torch.float32
        self.use_amp = False
        self.scaler = MagicMock()
        self.scaler.is_enabled.return_value = False
        self.optimizer = MagicMock()
        self.optimizer.param_groups = [{"lr": 1e-4}]
        self.lr_scheduler = MagicMock()
        self.ema_handler = None
        self.global_step = 0
        self._log_writer = None
        self._aug_h_flip = False
        self._aug_v_flip = False
        self.checkpoint_manager = MagicMock()
        self.checkpoint_manager.output_dir = str(tmp_path)
        self.logger_component = MagicMock()
        self.logger_component.last_step_time = time.time()
        self.latent_manager = MagicMock()
        self.latent_manager.encode_and_cache_batch.return_value = torch.randn(1, 4)
        self.driver = MiniMaxH3Driver(self.definition, self.device)
        self.transformer = torch.nn.Linear(4, 4)
        self.components = {"unet": self.transformer, "transformer": self.transformer, "vae": object()}
        self._model = self.transformer

    def _get_primary_model(self):
        return self._model

    def _iter_training_batches(self, batch_size):
        while True:
            yield [{"id": 0}]

    def _get_batch(self, batch_items, decode_pixels=True):
        return {"images": torch.randn(1, 3, 8, 8), "ids": [0], "paths": ["dummy.mp4"], "captions": ["a cat"]}

    def _load_control_latents(self, batch):
        pass

    def _attach_conditioning(self, batch, latents):
        pass

    def build_batch_extra(self, items):
        return {}

    def encode_text(self, captions, dtype, batch=None):
        return torch.zeros(1, 4)

    def prepare_latents_for_training(self, latents):
        return latents

    def prepare_noise_for_training(self, noise):
        return noise

    def sample_timesteps(self, batch_size, latents=None):
        return torch.zeros(batch_size)

    def add_noise(self, latents, noise, timesteps):
        return latents

    def forward_pass(self, noisy_input, timesteps, text_embeddings, batch):
        return self._model(noisy_input)

    def compute_target(self, latents, noise, timesteps):
        return torch.zeros_like(latents)

    def _compute_step_loss(self, pred, target, timesteps, batch, grad_accum):
        return (pred - target).pow(2).mean()

    def _build_trainable_components(self):
        return {}

    def get_te_cache(self):
        return None

    def _build_cache_manifest(self):
        return None


def test_production_sampling_path_invokes_the_family_sampler(tmp_path):
    from unittest.mock import patch

    from app.engine.models.families.minimax_h3.sampler import MiniMaxH3Sampler

    t = _LoopShell(tmp_path)
    # What `pipeline_optimization.py` does last: `self.sampler = self._create_sampler()`.
    t.sampler = t._create_sampler()
    assert isinstance(t.sampler, MiniMaxH3Sampler), f"trainer.sampler is {t.sampler!r}"
    calls: list[tuple] = []
    t.sampler.generate_samples = lambda step, final=False: calls.append((step, final)) or []
    with patch("app.core.db.DatabaseEngine.get_instance", side_effect=RuntimeError("no db in test")):
        asyncio.run(t.train())
    assert calls == [(1, False)], f"generate_samples calls: {calls} (expected exactly once, at the configured step)"


def test_create_sampler_is_off_when_sampling_is_off(tmp_path):
    t = _LoopShell(tmp_path)
    t.config["sample_every_n_steps"] = 0
    assert t._create_sampler() is None


def test_component_losses_logged(tmp_path, build_tiny_transformer):
    t = _loss_shell(tmp_path, build_tiny_transformer)
    loss, pred, target, batch = _run_step_hooks(t, grad_accum=4)
    loss_video, loss_audio, unscaled = _expected_components(t, pred, target, batch)
    lines = [c for c in t.logger.info.call_args_list if c.args and c.args[0] == "h3_step_loss"]
    assert lines, "no h3_step_loss line was logged"
    kw = lines[-1].kwargs
    assert set(kw) >= {"loss", "loss_video", "loss_audio"}, f"line keys {sorted(kw)}"
    # UNSCALED numbers (the /grad_accum is the optimiser's business, not the log's).
    assert abs(kw["loss_video"] - float(loss_video)) < 1e-6
    assert abs(kw["loss_audio"] - float(loss_audio)) < 1e-6
    assert abs(kw["loss"] - float(unscaled)) < 1e-6, (
        f"logged loss {kw['loss']:.6f} != unscaled {float(unscaled):.6f} (scaled would be {float(loss):.6f})"
    )


# ── Override ledger (plan § Override ledger; row 2.14; DECISION-68 (a)) ─────
#
# Every base hook with a default is owned by a row or `not needed` in the
# plan's ledger. These literals are the ledger's three sets AFTER the set-C
# entries were resolved by their owning rows (2.1: `_offload_text_encoders`
# -> A, `get_te_cache` / `set_te_cache` -> B; 2.5: `_resolve_loading_dtype`
# -> A; 1.2: `init_scheduler` -> B, the driver's trivial `None` baseline). An
# override the ledger calls `not needed`, or a set-A row nobody implemented,
# fails BY NAME below.

LEDGER_BASE_HOOKS = frozenset({
    # pipeline_base.py
    "_reraise_resolver_failure", "is_video_family", "is_audio_family", "_driver_hook_override",
    "init_scheduler", "get_lora_targets", "get_lora_exclude_modules", "get_te_lora_targets",
    "encode_text", "forward_pass", "compute_target", "sample_timesteps", "add_noise",
    "prepare_noise_for_training", "compute_loss_weight", "_compute_step_loss", "build_batch_extra",
    "prepare_latents_for_training", "_attach_conditioning", "on_epoch_end", "_create_sampler",
    "get_te_cache", "set_te_cache", "_apply_run_seed", "setup", "_setup_family",
    "_resolve_loading_dtype", "_assign_components", "_get_primary_model", "_get_text_encoders",
    "_freeze_all", "_apply_quantization", "_update_primary_model",
    # pipeline_caching.py
    "_pre_cache_text_embeddings", "_pre_cache_aux", "_build_caption_hints",
    "_expand_wildcards_for_precache", "_resolve_te_cache_dirs", "_validate_latent_cache",
    "_pre_cache_latents", "_build_cache_manifest",
})
LEDGER_LTX2_ONLY_HOOKS = frozenset({
    "_resolve_train_audio", "_sample_prompt_texts", "_offload_text_encoders", "_slice_te_output",
    "_audio_cache_dir", "_audio_cache_version",
})
LEDGER_HOOKS = LEDGER_BASE_HOOKS | LEDGER_LTX2_ONLY_HOOKS

# Set A — required trainer overrides (the ledger's literal + the resolved set-C entries).
LEDGER_SET_A = frozenset({
    "_audio_cache_dir", "_audio_cache_version", "_compute_step_loss", "_create_sampler",
    "_get_primary_model", "_get_text_encoders", "_pre_cache_aux", "_pre_cache_text_embeddings",
    "_resolve_train_audio", "_sample_prompt_texts", "_setup_family", "_update_primary_model",
    "add_noise", "build_batch_extra", "compute_target", "encode_text", "forward_pass",
    "sample_timesteps",
    "_offload_text_encoders",   # set C -> A (row 2.1: Qwen3-VL released whole, unconditionally)
    "_resolve_loading_dtype",   # set C -> A (row 2.5: bf16 from the definition)
})
# Set B — inherited base delegations, NEVER overridden on the trainer.
LEDGER_SET_B = frozenset({
    "compute_loss_weight", "get_lora_exclude_modules", "get_lora_targets",
    "prepare_latents_for_training", "prepare_noise_for_training",
    "get_te_cache", "set_te_cache",   # set C -> B (row 2.1: the driver keeps the trivial baseline)
    "init_scheduler",                 # set C -> B (row 1.2: driver `None`, base default runs)
})


def test_ledger_hook_names_exist_on_the_code():
    """The ledger names are the CODE's names: every base hook is an attribute of
    the base pipeline, every ltx2-only hook an override on ``Ltx2Trainer``.
    A renamed or deleted hook makes the ledger (and this literal) stale by name."""
    from app.engine.core.pipeline import GenericTrainingPipeline
    from app.engine.models.families.ltx2.trainer import Ltx2Trainer

    missing_base = sorted(n for n in LEDGER_BASE_HOOKS if not hasattr(GenericTrainingPipeline, n))
    missing_ltx2 = sorted(n for n in LEDGER_LTX2_ONLY_HOOKS if n not in Ltx2Trainer.__dict__)
    assert not missing_base, f"ledger base hooks not on the base pipeline: {missing_base}"
    assert not missing_ltx2, f"ledger ltx2-only hooks not overridden on Ltx2Trainer: {missing_ltx2}"
    assert not (LEDGER_SET_A & LEDGER_SET_B)
    assert LEDGER_SET_A <= LEDGER_HOOKS and LEDGER_SET_B <= LEDGER_HOOKS


def test_minimax_h3_trainer_overrides_match_the_ledger():
    """(i) ``MiniMaxH3Trainer.__dict__ ∩ hooks`` equals set A exactly — an
    unlisted override and an unimplemented set-A row each fail by name;
    (ii) every set-B name is ABSENT from the trainer's own dict."""
    overridden = set(MiniMaxH3Trainer.__dict__) & LEDGER_HOOKS
    unlisted = sorted(overridden - LEDGER_SET_A)
    unimplemented = sorted(LEDGER_SET_A - overridden)
    assert not unlisted, f"MiniMaxH3Trainer overrides hooks the ledger does not own (set A): {unlisted}"
    assert not unimplemented, f"set-A hooks the ledger owns but MiniMaxH3Trainer does not override: {unimplemented}"
    present_b = sorted(LEDGER_SET_B & set(MiniMaxH3Trainer.__dict__))
    assert not present_b, f"set-B delegations must stay inherited, but the trainer overrides: {present_b}"


class _ProbeDriver(MiniMaxH3Driver):
    """A stub driver: each set-B target answers with a marker, so the base's
    route to the driver is observed at its OUTPUT."""

    def __init__(self, definition, device):
        super().__init__(definition, device)
        self.calls: list[str] = []

    def prepare_latents(self, latents):
        self.calls.append("prepare_latents")
        return latents * 2

    def prepare_noise(self, noise):
        self.calls.append("prepare_noise")
        return noise + 1

    def get_lora_targets(self):
        self.calls.append("get_lora_targets")
        return ["probe.target"]

    def get_lora_exclude_modules(self):
        self.calls.append("get_lora_exclude_modules")
        return ["probe.exclude"]


def test_set_b_delegations_reach_the_driver_through_the_base(tmp_path):
    """The base still routes each set-B hook to the driver method the ledger
    names (`pipeline_base.py` `get_lora_targets` / `get_lora_exclude_modules` /
    `prepare_noise_for_training` -> `driver.prepare_noise` /
    `prepare_latents_for_training` -> `driver.prepare_latents`); the CLOBBER
    hooks `get_te_cache` / `set_te_cache` / `init_scheduler` run the base
    default because the H3 driver keeps the trivial baseline; and
    `compute_loss_weight` is the base's uniform ``None`` (H3's weighting lives
    in `driver.compute_loss`, row 2.4)."""
    t = _trainer(tmp_path)
    t.driver = _ProbeDriver(t.definition, t.device)
    x = torch.ones(2, 3)
    assert torch.equal(t.prepare_latents_for_training(x), x * 2)
    assert torch.equal(t.prepare_noise_for_training(x), x + 1)
    assert t.get_lora_targets() == ["probe.target"]
    assert t.get_lora_exclude_modules() == ["probe.exclude"]
    assert t.driver.calls == ["prepare_latents", "prepare_noise", "get_lora_targets", "get_lora_exclude_modules"]
    # CLOBBER hooks: the real driver's baseline is trivial -> the base default answers.
    t.driver = MiniMaxH3Driver(t.definition, t.device)
    t.text_cache = {"sks, a cat": object()}
    assert t.get_te_cache() == {"te": t.text_cache}
    t.set_te_cache({"te": {"restored": 1}})
    assert t.text_cache == {"restored": 1}
    assert t.init_scheduler() is None
    assert t.compute_loss_weight(torch.tensor([0.5, 0.25])) is None
