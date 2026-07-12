"""WAN per-prompt still previews (Phase 3 — wire ``SamplePromptConfig.num_frames``).

Covers the shared ``wan_shared/sampler_base.py`` noise builder (wan21 + wan22
inherit it). A prompt's ``num_frames`` overrides ``sample_num_frames``: ``None``
= run default; ``1`` = still (one latent frame, ``4n+1`` rule satisfied).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from app.engine.core.video_contract import frame_predicate
from app.engine.models.families.wan21.sampler import Wan21Sampler


def _sampler(config: dict) -> Wan21Sampler:
    pipe = SimpleNamespace(config=config, device=torch.device("cpu"))
    s = object.__new__(Wan21Sampler)
    s.pipeline = pipe
    s.config = config
    s.device = pipe.device
    return s


def test_frame_rule_accepts_a_single_still():
    assert frame_predicate("4n+1")(1) is True


def test_num_frames_none_uses_run_default():
    s = _sampler({"sample_num_frames": 17})
    gen = torch.Generator().manual_seed(0)
    noise = s._create_initial_noise(512, 512, gen)
    # 17 frames → (17-1)//4+1 = 5 latent frames.
    assert noise.shape[2] == 5


def test_per_prompt_num_frames_1_is_still():
    s = _sampler({"sample_num_frames": 17})
    s._active_prompt_cfg = {"num_frames": 1}
    gen = torch.Generator().manual_seed(0)
    noise = s._create_initial_noise(512, 512, gen)
    assert noise.shape[2] == 1  # (1-1)//4+1


def test_per_prompt_num_frames_snaps_to_frame_rule():
    s = _sampler({"sample_num_frames": 17})
    s._active_prompt_cfg = {"num_frames": 30}  # not 4n+1 → snaps down to 29
    gen = torch.Generator().manual_seed(0)
    noise = s._create_initial_noise(512, 512, gen)
    assert noise.shape[2] == 8  # (29-1)//4+1
