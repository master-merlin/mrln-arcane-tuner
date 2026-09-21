"""The shared sampler treats "no prompts" the same however the config says it.

LANE-92 VERIFY 3.02: ``sample_prompts: null`` raised ``TypeError`` inside
``generate_samples``; ``pipeline_train`` caught it and warned ``sampling
failed`` at the baseline, every cadence step and the final round of an
otherwise valid job. Family-agnostic: the concrete sampler below has no model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from app.engine.core.sampling import GenericSamplingPipeline


class _NoModelSampler(GenericSamplingPipeline):
    def encode_prompt(self, prompt):  # pragma: no cover - never reached without prompts
        raise AssertionError("no prompt may be encoded when the job has none")

    def denoise(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("unreachable")

    def decode_latents(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("unreachable")

    def _create_initial_noise(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("unreachable")


def _sampler(config: dict) -> _NoModelSampler:
    return _NoModelSampler(SimpleNamespace(config=config, device=torch.device("cpu")))


@pytest.mark.parametrize("config", [{}, {"sample_prompts": []}, {"sample_prompts": None}], ids=["absent", "empty", "null"])
def test_a_job_without_prompts_samples_nothing_and_raises_nothing(config):
    sampler = _sampler(config)
    assert sampler._get_sample_prompts() == []
    assert sampler.generate_samples(step=-1) == []
    assert sampler.generate_samples(step=9, final=True) == []


def test_prompts_still_come_through_as_dicts():
    class _Model:
        def model_dump(self):
            return {"prompt": "from a model"}

    sampler = _sampler({"sample_prompts": [{"prompt": "a dict"}, _Model(), "not a prompt"]})
    assert sampler._get_sample_prompts() == [{"prompt": "a dict"}, {"prompt": "from a model"}]
