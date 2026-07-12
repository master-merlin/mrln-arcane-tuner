"""LTX-2 per-prompt still previews (Phase 3 — wire ``SamplePromptConfig.num_frames``).

Contract: a sample prompt's ``num_frames`` overrides the run-level
``sample_num_frames``. ``None`` = run default; ``1`` = still image. The still
flows through the SAME video noise/rope/fps plumbing (``F=1`` satisfies the
``8n+1`` rule and packs to a single latent frame), and a 1-frame LTX-2 sample
keeps the existing no-audio path.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from app.engine.core.video_contract import frame_predicate
from app.engine.models.families.ltx2.driver import Ltx2Driver
from app.engine.models.families.ltx2.sampler import Ltx2Sampler


def _noise_sampler(config: dict) -> tuple[Ltx2Sampler, dict]:
    """Sampler whose driver captures the pre-pack latent shape."""
    drv = object.__new__(Ltx2Driver)
    captured: dict = {}

    def _prep(latents):
        captured["shape"] = tuple(latents.shape)
        return latents

    drv.prepare_latents = _prep
    definition = SimpleNamespace(
        architecture_params={
            "video.vae_spatial": 32,
            "video.vae_temporal": 8,
            "transformer.in_channels": 128,
        }
    )
    pipe = SimpleNamespace(
        driver=drv, definition=definition, config=config, device=torch.device("cpu")
    )
    s = object.__new__(Ltx2Sampler)
    s.pipeline = pipe
    s.config = config
    s.device = pipe.device
    return s, captured


def test_frame_rule_accepts_a_single_still():
    assert frame_predicate("8n+1")(1) is True


def test_num_frames_none_uses_run_default():
    s, captured = _noise_sampler({"sample_num_frames": 25})
    gen = torch.Generator().manual_seed(0)
    s._create_initial_noise(1024, 1024, gen)
    # 25 frames → (25-1)//8+1 = 4 latent frames.
    assert captured["shape"][2] == 4


def test_per_prompt_num_frames_1_is_still():
    s, captured = _noise_sampler({"sample_num_frames": 25})
    s._active_prompt_cfg = {"num_frames": 1}
    gen = torch.Generator().manual_seed(0)
    s._create_initial_noise(1024, 1024, gen)
    # (1-1)//8+1 = 1 latent frame — a still.
    assert captured["shape"][2] == 1


def test_per_prompt_num_frames_snaps_to_frame_rule():
    s, captured = _noise_sampler({"sample_num_frames": 25})
    s._active_prompt_cfg = {"num_frames": 30}  # not 8n+1 → snaps down to 25
    gen = torch.Generator().manual_seed(0)
    s._create_initial_noise(1024, 1024, gen)
    assert captured["shape"][2] == 4  # (25-1)//8+1


# ── Audio gating: a 1-frame sample keeps the no-audio path ─────────────────


class _RecTransformer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def parameters(self):
        yield torch.zeros(1, dtype=torch.float32)

    def __call__(self, **kw):
        self.calls.append(kw)
        return (kw["hidden_states"], kw["audio_hidden_states"])


def _denoise_sampler(*, latent_frames: int) -> Ltx2Sampler:
    """Audio-TRAINED driver whose latent grid has ``latent_frames`` frames."""
    drv = object.__new__(Ltx2Driver)
    drv.transformer = _RecTransformer()
    drv.audio_in_channels = 128
    drv.caption_channels = 3840
    drv.frame_rate = 24.0
    drv.audio_sampling_rate = 16000
    drv._latent_shape = (latent_frames, 4, 5)
    drv.train_audio = True

    class _AudioVae:
        config = type(
            "C", (), {"sample_rate": 16000, "mel_hop_length": 160, "mel_bins": 64}
        )()
        temporal_compression_ratio = 4

    class _Vocoder:
        config = type("C", (), {"output_sampling_rate": 24000})()

    drv.audio_vae = _AudioVae()
    drv.vocoder = _Vocoder()

    pipe = SimpleNamespace(
        driver=drv,
        transformer=drv.transformer,
        vae=object(),
        config={"sample_num_frames": 25},
        device=torch.device("cpu"),
    )
    s = object.__new__(Ltx2Sampler)
    s.pipeline = pipe
    s.config = pipe.config
    s.device = pipe.device
    return s


def _te():
    from app.engine.core.text_encoding import TextEncoderOutput

    return TextEncoderOutput(
        embeddings=torch.zeros(1, 11, 3840),
        attention_mask=None,
        pooled=torch.ones(1, 11, 3840),
    )


def test_still_sample_takes_no_audio_path():
    """F=1 latent grid on an audio-trained run → isolated dummy audio, no stash."""
    s = _denoise_sampler(latent_frames=1)
    noise = torch.zeros(1, 7, 128)
    out = s.denoise(noise, _te(), num_steps=2, guidance_scale=1.0, seed=0)
    call = s.pipeline.transformer.calls[-1]
    assert call["isolate_modalities"] is True
    assert s.pipeline.driver._last_audio_latents is None
    assert out.shape == noise.shape


def test_clip_sample_still_runs_joint_audio():
    """Guard the still-gate did not disable audio for real clips (F>1)."""
    s = _denoise_sampler(latent_frames=4)
    noise = torch.zeros(1, 7, 128)
    s.denoise(noise, _te(), num_steps=2, guidance_scale=1.0, seed=0)
    call = s.pipeline.transformer.calls[-1]
    assert call["isolate_modalities"] is False
    assert s.pipeline.driver._last_audio_latents is not None
