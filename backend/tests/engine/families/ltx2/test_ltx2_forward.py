"""LTX-2 forward/sampler reconciliation contract (no GPU).

Pins the driver's call into the real ``LTX2VideoTransformer3DModel`` joint
audio+video API: required audio inputs, RAW ``[0,1000]`` timestep + ``sigma``,
RoPE ``num_frames/height/width/fps``, ``isolate_modalities`` for the video-only
path, and the ``(video, audio)`` tuple return.  Also covers the ``encode_text``
stack axis and the single-image ``_latent_shape``.
"""

from __future__ import annotations

import torch

from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.models.families.ltx2.driver import Ltx2Driver


class _RecTransformer:
    """Records forward kwargs; returns a (video, audio) tuple."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return (kw["hidden_states"], kw["audio_hidden_states"])


def _video_driver() -> Ltx2Driver:
    drv = object.__new__(Ltx2Driver)
    drv.transformer = _RecTransformer()
    drv.audio_in_channels = 128
    drv.caption_channels = 3840
    drv.frame_rate = 24.0
    drv._latent_shape = (3, 4, 5)
    return drv


def test_forward_pass_matches_joint_transformer_contract():
    drv = _video_driver()
    noisy = torch.zeros(2, 7, 128)  # packed [B, tokens, in_channels]
    video_emb = torch.zeros(2, 11, 3840)
    te = TextEncoderOutput(embeddings=video_emb, attention_mask=None, pooled=None)
    timesteps = torch.tensor([500.0, 250.0])

    out = drv.forward_pass(noisy, timesteps, te, {})
    call = drv.transformer.calls[0]

    # The frame_rate kwarg never existed on the real model — it's fps.
    assert "frame_rate" not in call
    # Timestep is RAW [0,1000] (NOT ÷1000), and sigma mirrors it.
    assert torch.equal(call["timestep"], timesteps)
    assert torch.equal(call["sigma"], timesteps)
    # Required audio stream (video-only → single zero token).
    assert call["audio_hidden_states"].shape == (2, 1, 128)
    assert call["audio_encoder_hidden_states"].shape == (2, 1, 3840)
    assert torch.count_nonzero(call["audio_hidden_states"]) == 0
    # Video conditioning passed through verbatim (3840-dim → caption_projection).
    assert torch.equal(call["encoder_hidden_states"], video_emb)
    # RoPE coords from the latent grid + fps.
    assert (call["num_frames"], call["height"], call["width"]) == (3, 4, 5)
    assert call["fps"] == 24.0
    # Video-only isolation so the dummy audio cannot leak into the video grad.
    assert call["isolate_modalities"] is True
    # Only the video prediction is returned.
    assert torch.equal(out, noisy)


def test_forward_pass_requires_latent_shape():
    drv = _video_driver()
    drv._latent_shape = None
    import pytest

    with pytest.raises(RuntimeError, match="prepare_latents"):
        drv.forward_pass(
            torch.zeros(1, 4, 128),
            torch.tensor([100.0]),
            TextEncoderOutput(embeddings=torch.zeros(1, 5, 3840)),
            {},
        )


def test_prepare_latents_records_grid_for_video_and_still():
    drv = object.__new__(Ltx2Driver)
    drv.patch_size = 1
    drv.patch_size_t = 1

    drv.prepare_latents(torch.zeros(2, 128, 9, 16, 16))  # 5D video
    assert drv._latent_shape == (9, 16, 16)

    drv.prepare_latents(torch.zeros(2, 128, 16, 16))  # 4D still → F=1
    assert drv._latent_shape == (1, 16, 16)


def test_encode_text_stacks_layers_on_last_axis():
    """The connector expects (B, L, caption_channels, num_layers) — num_layers LAST."""

    class _FakeTE:
        def __call__(self, input_ids, attention_mask, output_hidden_states, use_cache):
            b, length = input_ids.shape
            out = type("O", (), {})()
            out.hidden_states = tuple(torch.randn(b, length, 3840) for _ in range(3))
            return out

    class _FakeTok:
        def __call__(self, captions, return_tensors, padding, truncation, max_length):
            b = len(captions)
            return {
                "input_ids": torch.zeros(b, max_length, dtype=torch.long),
                "attention_mask": torch.ones(b, max_length, dtype=torch.long),
            }

    class _RecConnectors:
        def __init__(self) -> None:
            self.shape: tuple | None = None

        def __call__(self, text_encoder_hidden_states, attention_mask):
            self.shape = tuple(text_encoder_hidden_states.shape)
            b, length = attention_mask.shape
            return (
                torch.zeros(b, length, 3840),
                torch.zeros(b, length, 3840),
                torch.ones(b, length),
            )

    drv = object.__new__(Ltx2Driver)
    drv.text_encoder = _FakeTE()
    drv.tokenizer = _FakeTok()
    drv.connectors = _RecConnectors()
    drv.te_max_length = 8
    drv.device = torch.device("cpu")

    out = drv.encode_text(["hi", "yo"], torch.float32)

    # [B=2, L=8, caption_channels=3840, num_layers=3] — stacked on dim=-1.
    assert drv.connectors.shape == (2, 8, 3840, 3)
    assert out.embeddings.shape == (2, 8, 3840)
    assert out.pooled.shape == (2, 8, 3840)  # audio text emb in the pooled slot
