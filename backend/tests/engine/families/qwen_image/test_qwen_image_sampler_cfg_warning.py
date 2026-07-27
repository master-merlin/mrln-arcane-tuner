"""qwen_image sampler — warn-once when guidance_scale/negative prompt are
configured but silently ignored (W5.T10).

QwenImageSampler.denoise() always runs a single unconditional forward
(``guidance=None``, matching the model's ``guidance_embeds: false`` config) —
there is no true-CFG second pass. A ``guidance_scale > 1`` or a configured
``sample_negative_prompt`` therefore has NO effect on the sample, which used
to fail silently. These tests pin the warn-once guard (mirrors
boogu_image/sampler.py's ``_warn_negative_prompt_ignored_once`` pattern)
without changing the actual sampling output.
"""

from __future__ import annotations

import types

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.qwen_image.sampler import QwenImageSampler


class _CaptureTransformer(torch.nn.Module):
    """Returns a correctly-shaped patch sequence; ignores input content."""

    def __init__(self, out_channels: int, patch_size: int = 2) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(
            patch_size=patch_size,
            out_channels=out_channels,
        )
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        *,
        hidden_states,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        timestep,
        guidance,
        img_shapes,
        return_dict,
    ):
        b, seq, _ = hidden_states.shape
        pdim = self.config.out_channels * self.config.patch_size**2
        return (torch.zeros(b, seq, pdim),)


def _make_sampler(config: dict) -> QwenImageSampler:
    pipeline = types.SimpleNamespace(
        config=config,
        device=torch.device("cpu"),
        definition=ModelDefinition(
            id="qwen-image",
            family="qwen_image",
            name="Qwen-Image",
            defaults={},
            components={},
        ),
        transformer=_CaptureTransformer(out_channels=16),
    )
    sampler = QwenImageSampler(pipeline)
    # Normally set by _create_initial_noise(); bypass it for this unit test.
    lat_h, lat_w = 4, 4
    sampler._lat_h = lat_h
    sampler._lat_w = lat_w
    sampler._vae_sf = 8
    sampler._sample_height = lat_h * 8
    sampler._sample_width = lat_w * 8
    return sampler


def _noise(lat_h: int = 4, lat_w: int = 4, channels: int = 16) -> torch.Tensor:
    """Packed latent shape: [B, (H/2)*(W/2), C*4]."""
    return torch.randn(1, (lat_h // 2) * (lat_w // 2), channels * 4)


def _prompt_embedding() -> dict:
    return {"embeds": torch.randn(1, 8, 16), "mask": torch.ones(1, 8, dtype=torch.long)}


def test_denoise_warns_once_when_guidance_scale_above_one(monkeypatch):
    from app.engine.models.families.qwen_image import sampler as sampler_module

    sampler = _make_sampler({})
    warnings: list[dict] = []
    monkeypatch.setattr(
        sampler_module.logger,
        "warning",
        lambda event, **kw: warnings.append({"event": event, **kw}),
    )

    sampler.denoise(
        _noise(), _prompt_embedding(), num_steps=2, guidance_scale=4.0, seed=0
    )
    sampler.denoise(
        _noise(), _prompt_embedding(), num_steps=2, guidance_scale=4.0, seed=0
    )

    assert len(warnings) == 1  # once per instance, not once per call
    assert "guidance_scale" in warnings[0]["event"]
    assert warnings[0]["guidance_scale"] == 4.0


def test_denoise_warns_once_when_negative_prompt_configured(monkeypatch):
    from app.engine.models.families.qwen_image import sampler as sampler_module

    sampler = _make_sampler({"sample_negative_prompt": "blurry, low quality"})
    warnings: list[dict] = []
    monkeypatch.setattr(
        sampler_module.logger,
        "warning",
        lambda event, **kw: warnings.append({"event": event, **kw}),
    )

    sampler.denoise(
        _noise(), _prompt_embedding(), num_steps=2, guidance_scale=1.0, seed=0
    )

    assert len(warnings) == 1


def test_denoise_does_not_warn_with_default_config(monkeypatch):
    """guidance_scale == 1.0 and no negative prompt -> no warning at all."""
    from app.engine.models.families.qwen_image import sampler as sampler_module

    sampler = _make_sampler({})
    warnings: list[dict] = []
    monkeypatch.setattr(
        sampler_module.logger,
        "warning",
        lambda event, **kw: warnings.append({"event": event, **kw}),
    )

    sampler.denoise(
        _noise(), _prompt_embedding(), num_steps=2, guidance_scale=1.0, seed=0
    )

    assert warnings == []


def test_denoise_output_shape_unaffected_by_guidance_scale():
    """The warning is purely observational — sampling output (still a single
    unconditional forward) must be byte-identical in shape regardless of the
    (ignored) guidance_scale value."""
    sampler_a = _make_sampler({})
    sampler_b = _make_sampler({})
    torch.manual_seed(0)
    noise = _noise()
    embed = _prompt_embedding()

    out_a = sampler_a.denoise(
        noise.clone(), embed, num_steps=2, guidance_scale=1.0, seed=0
    )
    out_b = sampler_b.denoise(
        noise.clone(), embed, num_steps=2, guidance_scale=7.5, seed=0
    )

    assert out_a["latents"].shape == out_b["latents"].shape
