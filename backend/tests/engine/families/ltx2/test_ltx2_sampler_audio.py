"""LTX-2 sampler audio path (no GPU, no weights).

Pins the joint audio+video SAMPLING contract that makes a generated clip carry
sound: the denoise loop co-integrates a packed audio-latent stream with
``isolate_modalities=False`` and stashes the result, and ``_decode_audio``
turns it into a waveform via denormalize → unpack → audio VAE → vocoder, tagged
at the vocoder's output sample rate. Video-only runs keep the isolated dummy
audio token. Uses a recording-fake transformer + fake audio VAE/vocoder.
"""

from __future__ import annotations

import torch

from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.models.families.ltx2.driver import Ltx2Driver
from app.engine.models.families.ltx2.sampler import Ltx2Sampler


class _RecTransformer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def parameters(self):
        yield torch.zeros(1, dtype=torch.float32)

    def __call__(self, **kw):
        self.calls.append(kw)
        # Velocity stand-ins: echo video + audio inputs back as (video, audio).
        return (kw["hidden_states"], kw["audio_hidden_states"])


class _FakeAudioVae:
    def __init__(self) -> None:
        self.config = type(
            "C", (), {"sample_rate": 16000, "mel_hop_length": 160, "mel_bins": 64}
        )()
        self.temporal_compression_ratio = 4
        self.mel_compression_ratio = 4
        self.latents_mean = torch.zeros(128)
        self.latents_std = torch.ones(128)
        self.dtype = torch.float32
        self.decoded_shape: tuple | None = None

    def decode(self, lat, return_dict=False):
        self.decoded_shape = tuple(lat.shape)  # expect [B, 8, L, 16]
        return (torch.zeros(lat.shape[0], 64, 50),)  # fake mel


class _FakeVocoder:
    def __init__(self) -> None:
        self.config = type("C", (), {"output_sampling_rate": 24000})()

    def __call__(self, mel):
        return torch.zeros(mel.shape[0], 1, 1000)  # [B, C, N]


def _sampler(*, train_audio: bool) -> Ltx2Sampler:
    drv = object.__new__(Ltx2Driver)
    drv.transformer = _RecTransformer()
    drv.audio_in_channels = 128
    drv.caption_channels = 3840
    drv.use_prompt_embeddings = True  # LTX-2.3 default; 2.5 sets it False
    drv.audio_cross_attention_dim = 2048
    drv.frame_rate = 24.0
    drv.audio_sampling_rate = 16000
    drv._latent_shape = (3, 4, 5)
    drv.train_audio = train_audio
    drv.audio_vae = _FakeAudioVae() if train_audio else None
    drv.vocoder = _FakeVocoder() if train_audio else None

    pipe = type("P", (), {})()
    pipe.driver = drv
    pipe.transformer = drv.transformer
    pipe.vae = object()
    pipe.config = {"sample_num_frames": 25}
    pipe.device = torch.device("cpu")
    pipe.components = {}

    s = object.__new__(Ltx2Sampler)
    s.pipeline = pipe
    s.config = pipe.config
    s.device = pipe.device
    return s


def _te():
    return TextEncoderOutput(
        embeddings=torch.zeros(1, 11, 3840),
        attention_mask=None,
        pooled=torch.ones(1, 11, 3840),  # audio text emb
    )


# ── audio length ────────────────────────────────────────────────────────


def test_audio_num_frames_follows_clip_duration():
    s = _sampler(train_audio=True)
    # 25 frames @ 24fps ≈ 1.0417s; 16000/160/4 = 25 latents/s → round(26.04) = 26.
    assert s._audio_num_frames() == 26


# ── denoise contract ──────────────────────────────────────────────────────


def test_denoise_audio_on_runs_joint_and_stashes_latents():
    s = _sampler(train_audio=True)
    noise = torch.zeros(1, 7, 128)  # packed video latents

    out = s.denoise(noise, _te(), num_steps=3, guidance_scale=1.0, seed=0)

    call = s.pipeline.transformer.calls[-1]
    assert call["isolate_modalities"] is False
    assert call["audio_hidden_states"].shape == (1, 26, 128)
    assert call["audio_num_frames"] == 26
    assert torch.equal(call["audio_encoder_hidden_states"], torch.ones(1, 11, 3840))
    # Final audio latents stashed for the decode phase.
    assert s.pipeline.driver._last_audio_latents.shape == (1, 26, 128)
    assert out.shape == noise.shape  # video latents returned


def test_denoise_video_only_keeps_isolated_dummy_audio():
    s = _sampler(train_audio=False)
    noise = torch.zeros(1, 7, 128)

    s.denoise(noise, _te(), num_steps=2, guidance_scale=1.0, seed=0)

    call = s.pipeline.transformer.calls[-1]
    assert call["isolate_modalities"] is True
    assert call["audio_hidden_states"].shape == (1, 1, 128)  # single zero token
    assert call["audio_num_frames"] == 1
    assert s.pipeline.driver._last_audio_latents is None


# ── decode contract ───────────────────────────────────────────────────────


def test_decode_audio_runs_vae_vocoder_and_tags_output_rate():
    s = _sampler(train_audio=True)
    drv = s.pipeline.driver
    drv._last_audio_latents = torch.zeros(1, 26, 128)

    result = s._decode_audio(drv)

    assert result is not None
    wav, sr = result
    assert sr == 24000  # the VOCODER's output rate, not the 16k mel domain
    assert wav.shape == (1, 1000)  # [B, C, N] → [C, N]
    # Unpacked to the spectrogram-latent layout the audio VAE decodes: [B, 8, L, 16].
    assert drv.audio_vae.decoded_shape == (1, 8, 26, 16)


def test_decode_audio_none_without_latents():
    s = _sampler(train_audio=True)
    s.pipeline.driver._last_audio_latents = None
    assert s._decode_audio(s.pipeline.driver) is None


def test_decode_latents_degrades_to_silent_on_audio_error(monkeypatch):
    s = _sampler(train_audio=True)
    monkeypatch.setattr(s, "_decode_video", lambda vae, driver, latents: torch.zeros(3, 2, 4, 4))

    def _boom(driver):
        raise RuntimeError("vocoder boom")

    monkeypatch.setattr(s, "_decode_audio", _boom)

    art = s.decode_latents(torch.zeros(1, 7, 128))

    assert art.audio is None  # silent fallback
    assert art.frames.shape == (3, 2, 4, 4)  # video sample preserved


def test_decode_latents_audio_error_is_logged_visibly_and_counted(monkeypatch):
    """The silent-clip degradation must be surfaced (warning) + occurrence-counted."""
    import app.engine.models.families.ltx2.sampler as sampler_mod

    class _RecLogger:
        def __init__(self):
            self.warnings = []

        def warning(self, event, **kw):
            self.warnings.append((event, kw))

        def info(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

    rec = _RecLogger()
    monkeypatch.setattr(sampler_mod, "logger", rec)

    s = _sampler(train_audio=True)
    monkeypatch.setattr(s, "_decode_video", lambda vae, driver, latents: torch.zeros(3, 2, 4, 4))
    monkeypatch.setattr(
        s, "_decode_audio", lambda driver: (_ for _ in ()).throw(RuntimeError("vocoder boom"))
    )

    s.decode_latents(torch.zeros(1, 7, 128))
    s.decode_latents(torch.zeros(1, 7, 128))

    fails = [kw for ev, kw in rec.warnings if ev == "ltx2_audio_decode_failed"]
    assert len(fails) == 2
    assert fails[0]["occurrence"] == 1 and fails[1]["occurrence"] == 2


# ── classifier-free guidance (CFG) ────────────────────────────────────────


class _EmbVelTransformer:
    """Velocity = uniform field equal to ``encoder_hidden_states.mean()``.

    Makes the velocity depend on the PROMPT, so a conditional vs unconditional
    embedding produce different velocities — which lets us pin the CFG combine
    ``v = v_uncond + scale * (v_cond - v_uncond)`` to an exact value.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def parameters(self):
        yield torch.zeros(1, dtype=torch.float32)

    def __call__(self, **kw):
        self.calls.append(kw)
        h = kw["hidden_states"]
        m = kw["encoder_hidden_states"].float().mean()
        v_video = (torch.zeros_like(h, dtype=torch.float32) + m).to(h.dtype)
        a = kw["audio_hidden_states"]
        v_audio = torch.zeros_like(a, dtype=torch.float32).to(a.dtype)
        return (v_video, v_audio)


def _te_const(val: float, ch: int = 8) -> TextEncoderOutput:
    return TextEncoderOutput(
        embeddings=torch.full((1, 4, ch), float(val)),
        attention_mask=torch.ones(1, 4),
        pooled=torch.full((1, 4, ch), float(val)),
    )


def _cfg_sampler() -> Ltx2Sampler:
    """Video-only sampler whose uncond (negative) embedding is a 0.25 field."""
    drv = object.__new__(Ltx2Driver)
    drv.transformer = _EmbVelTransformer()
    drv.audio_in_channels = 128
    drv.caption_channels = 8
    drv.use_prompt_embeddings = True  # tiny stand-in dims; the flag is 2.3's
    drv.audio_cross_attention_dim = 8
    drv.frame_rate = 24.0
    drv.audio_sampling_rate = 16000
    drv._latent_shape = (3, 4, 5)
    drv.train_audio = False
    drv.audio_vae = None
    drv.vocoder = None

    pipe = type("P", (), {})()
    pipe.driver = drv
    pipe.transformer = drv.transformer
    pipe.vae = object()
    pipe.config = {"sample_num_frames": 25}
    pipe.device = torch.device("cpu")
    pipe.components = {}
    # The sampler's encode_prompt(neg) routes through the trainer's cache; here
    # the uncond embedding is a constant 0.25 field (distinct from the 1.0 cond).
    pipe.encode_text = lambda caps, dtype: _te_const(0.25)

    s = object.__new__(Ltx2Sampler)
    s.pipeline = pipe
    s.config = pipe.config
    s.device = pipe.device
    return s


def test_denoise_no_cfg_is_single_conditional_forward():
    """guidance_scale<=1: one forward/step, velocity = the conditional only."""
    s = _cfg_sampler()
    noise = torch.zeros(1, 7, 8)

    out = s.denoise(noise, _te_const(1.0), num_steps=2, guidance_scale=1.0, seed=0)

    assert len(s.pipeline.transformer.calls) == 2  # one forward per step
    # x0=0, Σdt=-1, constant v=cond.mean()=1.0 → endpoint all -1.0.
    assert torch.allclose(out, torch.full_like(out, -1.0), atol=1e-5)


def test_denoise_cfg_runs_uncond_and_combines_velocity():
    """guidance_scale>1: cond+uncond forwards, combined v = u + s*(c-u)."""
    s = _cfg_sampler()
    noise = torch.zeros(1, 7, 8)

    out = s.denoise(noise, _te_const(1.0), num_steps=2, guidance_scale=3.0, seed=0)

    # Two forwards per step (conditional + unconditional).
    assert len(s.pipeline.transformer.calls) == 4
    means = [
        c["encoder_hidden_states"].float().mean().item()
        for c in s.pipeline.transformer.calls
    ]
    assert any(abs(m - 1.0) < 1e-6 for m in means)   # conditional used
    assert any(abs(m - 0.25) < 1e-6 for m in means)  # unconditional used
    # v = 0.25 + 3*(1.0-0.25) = 2.5 (constant) → endpoint all -2.5.
    assert torch.allclose(out, torch.full_like(out, -2.5), atol=1e-5)
