"""FAM-2: LTX-2 sampler emits per-step ``Sampling {i}/{N}`` status (audit P1b).

Every image family streams ``Sampling {i}/{N}`` (1-based, once per denoise
step) through ``self._log_writer.status(...)``; the LTX-2 video Euler loop
emitted nothing. The string format must stay byte-identical to the image
families' (e.g. ``krea2/sampler.py``) — the UI consumes it via
job_log.jsonl → LogTailer.

Stubs mirror ``test_ltx2_sampler_audio.py`` (video-only path, real
``Ltx2Driver`` methods via ``object.__new__``).
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
        return (kw["hidden_states"], kw["audio_hidden_states"])


class _StatusRecorder:
    """JobLogWriter stand-in recording every status(...) call."""

    def __init__(self) -> None:
        self.statuses: list[str] = []

    def status(self, message: str) -> None:
        self.statuses.append(message)


def _sampler() -> Ltx2Sampler:
    drv = object.__new__(Ltx2Driver)
    drv.transformer = _RecTransformer()
    drv.audio_in_channels = 128
    drv.caption_channels = 3840
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

    s = object.__new__(Ltx2Sampler)
    s.pipeline = pipe
    s.config = pipe.config
    s.device = pipe.device
    return s


def _te() -> TextEncoderOutput:
    return TextEncoderOutput(
        embeddings=torch.zeros(1, 11, 3840),
        attention_mask=None,
        pooled=torch.ones(1, 11, 3840),
    )


def test_ltx2_denoise_emits_per_step_sampling_status():
    s = _sampler()
    lw = _StatusRecorder()
    s._log_writer = lw
    noise = torch.zeros(1, 7, 128)

    s.denoise(noise, _te(), num_steps=3, guidance_scale=1.0, seed=0)

    assert lw.statuses == ["Sampling 1/3", "Sampling 2/3", "Sampling 3/3"]


def test_ltx2_denoise_without_log_writer_is_safe():
    s = _sampler()
    noise = torch.zeros(1, 7, 128)

    out = s.denoise(noise, _te(), num_steps=2, guidance_scale=1.0, seed=0)

    assert out.shape == noise.shape
