"""WAN 2.2 per-prompt still previews — inherits the shared sampler_base noise
builder (Phase 3). Confirms the ``num_frames`` override reaches wan22, which
subclasses ``WanVideoSamplerBase`` (it overrides ``denoise``, not the noise
builder).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from app.engine.models.families.wan22.sampler import Wan22Sampler


def _sampler(config: dict) -> Wan22Sampler:
    pipe = SimpleNamespace(config=config, device=torch.device("cpu"))
    s = object.__new__(Wan22Sampler)
    s.pipeline = pipe
    s.config = config
    s.device = pipe.device
    return s


def test_num_frames_none_uses_run_default():
    s = _sampler({"sample_num_frames": 17})
    gen = torch.Generator().manual_seed(0)
    noise = s._create_initial_noise(512, 512, gen)
    assert noise.shape[2] == 5  # (17-1)//4+1


def test_per_prompt_num_frames_1_is_still():
    s = _sampler({"sample_num_frames": 17})
    s._active_prompt_cfg = {"num_frames": 1}
    gen = torch.Generator().manual_seed(0)
    noise = s._create_initial_noise(512, 512, gen)
    assert noise.shape[2] == 1
