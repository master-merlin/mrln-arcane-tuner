"""WAN 2.2 i2v preview 36-channel pad — ``Wan22Sampler.denoise`` must not crash.

``Wan22Sampler.denoise`` fully OVERRIDES the shared ``WanVideoSamplerBase.denoise``
(dual-expert boundary switch) and feeds the raw 16-channel noise latent straight
to the expert. ``wan22_i2v_a14b.yaml`` pins ``transformer.in_channels: 36`` for
BOTH experts (the A14B i2v checkpoint's ``patch_embedding`` is
``[noisy16, mask4, cond16]``), so every in-training preview on a wan22-i2v run
(step-0 baseline and every interval) crashed: "expected input ... to have 36
channels, but got 16" — the classic "sibling override not mirrored" bug (the
base gained a ``pad_to_36`` guard after a wan2.1-i2v GPU-UAT crash, but the
wan22 override was never updated to match).

Fix: ``WanVideoSamplerBase._maybe_pad_i2v()`` is the single source of truth for
the zero-pad decision (a no-op at 16ch); both the base ``denoise`` and
``Wan22Sampler._forward`` route through it.
"""

from __future__ import annotations

import torch
from diffusers import WanTransformer3DModel

from app.engine.models.families.wan22.sampler import Wan22Sampler


def _tiny_model(
    in_channels: int, out_channels: int, seed: int = 0
) -> WanTransformer3DModel:
    """Tiny WanTransformer3DModel (fp32, CPU), text_dim=16 — mirrors the
    bernini_r sampler test's ``_tiny_model`` pattern
    (test_bernini_r_sampler.py:33-51).
    """
    torch.manual_seed(seed)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=in_channels,
        out_channels=out_channels,
        text_dim=16,
        freq_dim=64,
        ffn_dim=64,
        num_layers=2,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        rope_max_seq_len=64,
    )
    return model.to(torch.float32).eval()


class _Driver:
    """Minimal driver stand-in: a single 36-in-channel expert, no router.

    ``Wan22Sampler.denoise`` resolves ``high``/``low`` via
    ``getattr(driver, "transformer_high", None) or driver.get_primary_model()``
    — with neither attribute present both experts fall back to the same tiny
    model, matching the brief's "tiny 36-in-channel WanTransformer3DModel"
    single-model setup.
    """

    def __init__(self, model: WanTransformer3DModel) -> None:
        self._m = model

    def get_primary_model(self) -> WanTransformer3DModel:
        return self._m


class _Defn:
    """Minimal i2v definition stand-in (mode + pinned 36-in-channel expert)."""

    architecture_params = {"transformer.in_channels": 36, "mode": "i2v"}


class _Pipeline:
    """Minimal trainer stand-in the sampler binds to."""

    def __init__(self, model: WanTransformer3DModel, emb: torch.Tensor) -> None:
        self.config: dict = {}
        self.device = torch.device("cpu")
        self.autocast_dtype = torch.bfloat16
        self.driver = _Driver(model)
        self.definition = _Defn()
        self._emb = emb

    # encode_prompt() (inherited from the base) routes the empty-negative
    # through here when CFG is on; unused at guidance_scale=1.0 below.
    def encode_text(self, caps, dtype):
        return self._emb


def test_wan22_i2v_denoise_pads_16ch_noise_to_36_without_crashing():
    """The historical crash: an i2v A14B run pins 36-in-channel experts but the
    preview noise latent is created 16-channel (the shared 16ch noise builder);
    ``Wan22Sampler.denoise`` must zero-pad before the ``patch_embedding``, not
    crash with a channel-count mismatch.
    """
    model = _tiny_model(in_channels=36, out_channels=16)
    emb = torch.randn(1, 5, 16)
    sampler = Wan22Sampler(_Pipeline(model, emb))

    noise = torch.randn(1, 16, 1, 8, 8)
    latents = sampler.denoise(noise, emb, num_steps=2, guidance_scale=1.0, seed=0)

    # Prediction lives in the same 16-channel noise space as the input latent —
    # the pad is purely a transformer-input concern, never a target-space change.
    assert latents.shape == noise.shape


def test_wan22_t2v_denoise_stays_16ch_no_pad():
    """T2V (16-in-channel experts) must be byte-identical to before: the pad
    helper is a no-op, so the transformer is fed the RAW 16-channel input.
    """
    model = _tiny_model(in_channels=16, out_channels=16)
    seen: list[torch.Tensor] = []
    real_forward = model.forward

    def _recording_forward(hidden_states, *args, **kwargs):
        seen.append(hidden_states)
        return real_forward(hidden_states, *args, **kwargs)

    model.forward = _recording_forward  # type: ignore[method-assign]

    emb = torch.randn(1, 5, 16)
    sampler = Wan22Sampler(_Pipeline(model, emb))

    noise = torch.randn(1, 16, 1, 8, 8)
    sampler.denoise(noise, emb, num_steps=2, guidance_scale=1.0, seed=0)

    assert seen, "expert forward was never called"
    assert all(h.shape[1] == 16 for h in seen), (
        "t2v input must stay 16-channel (no pad)"
    )
