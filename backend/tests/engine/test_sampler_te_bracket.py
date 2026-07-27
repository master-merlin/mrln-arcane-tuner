"""Tests for W5.T1: sampling-phase TE bracket gated on ``needs_live_te``.

The base ``GenericSamplingPipeline._sample_single`` used to unconditionally
round-trip the text encoder(s) to GPU and back around every ``encode_prompt``
call — even for the default cache-serving families where ``encode_prompt``
never touches the TE (``cache_text_embeddings=True``, prompts pre-warmed via
``_get_cached_text_embeddings``). That was an 8-16 GB PCIe copy per prompt
per sampling round, redundant with each family's cache-miss self-bracket
(e.g. ``qwen_image/trainer.py``). Only samplers whose ``encode_prompt`` calls
the driver's encoder directly every round (``microsoft_lens``, ``ideogram4``)
need the bracket — gated here by the ``needs_live_te`` class flag.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from app.engine.core.sampling import GenericSamplingPipeline


class _RecordingSampler(GenericSamplingPipeline):
    """Minimal concrete sampler recording TE/VAE GPU-bracket calls.

    ``_ensure_on_gpu``/``_offload_to_cpu`` are overridden as recording stubs
    so the test doesn't need real ``torch.nn.Module`` components — it only
    needs to observe which component-name lists get bracketed.
    """

    def __init__(self, pipeline) -> None:
        super().__init__(pipeline)
        self.ensure_on_gpu_calls: list[list[str]] = []
        self.offload_calls: list[list[str]] = []

    def _ensure_on_gpu(self, names: list[str]) -> list[str]:
        self.ensure_on_gpu_calls.append(list(names))
        return list(names)

    def _offload_to_cpu(self, names: list[str]) -> None:
        self.offload_calls.append(list(names))

    def encode_prompt(self, prompt):
        return {"prompt": prompt}

    def denoise(self, noise, prompt_embedding, num_steps, guidance_scale, seed):
        return noise

    def decode_latents(self, latents):
        return object()

    def _create_initial_noise(self, width, height, generator):
        return torch.zeros(1)


def _make_pipeline():
    driver = SimpleNamespace(get_text_encoders=lambda: {"text_encoder": object()})
    return SimpleNamespace(config={}, device=torch.device("cpu"), driver=driver)


def test_base_default_needs_live_te_is_false():
    assert GenericSamplingPipeline.needs_live_te is False


def test_cache_serving_sampler_skips_te_bracket():
    """``needs_live_te=False``: no GPU round-trip for the TE.

    The VAE decode-phase bracket (Phase 3) is unconditional and always
    fires — only the TE bracket (Phase 1) is gated.
    """
    sampler = _RecordingSampler(_make_pipeline())
    sampler.needs_live_te = False

    sampler._sample_single({"prompt": "a cat", "seed": 1}, step=0)

    assert sampler.ensure_on_gpu_calls == [["vae"]]
    assert sampler.offload_calls == [["vae"]]


def test_live_te_sampler_runs_te_bracket():
    """``needs_live_te=True``: TE still brackets GPU on/off per prompt."""
    sampler = _RecordingSampler(_make_pipeline())
    sampler.needs_live_te = True

    sampler._sample_single({"prompt": "a cat", "seed": 1}, step=0)

    assert sampler.ensure_on_gpu_calls == [["text_encoder"], ["vae"]]
    assert sampler.offload_calls == [["text_encoder"], ["vae"]]


def test_microsoft_lens_sampler_declares_needs_live_te():
    from app.engine.models.families.microsoft_lens.sampler import (
        MicrosoftLensSampler,
    )

    assert MicrosoftLensSampler.needs_live_te is True


def test_ideogram4_sampler_declares_needs_live_te():
    from app.engine.models.families.ideogram4.sampler import IdeogramV4Sampler

    assert IdeogramV4Sampler.needs_live_te is True
