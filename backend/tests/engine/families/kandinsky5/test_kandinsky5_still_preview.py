"""Kandinsky 5.0 per-prompt still previews (Phase 3 — wire
``SamplePromptConfig.num_frames``). ``None`` = run default; ``1`` = still (one
latent frame, ``4n+1`` rule satisfied). Channels-last 5D noise.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from app.engine.core.video_contract import frame_predicate
from app.engine.models.families.kandinsky5.sampler import Kandinsky5Sampler


def _sampler(config: dict) -> Kandinsky5Sampler:
    driver = SimpleNamespace(vae_spatial=8, in_visual_dim=4)
    pipe = SimpleNamespace(config=config, device=torch.device("cpu"), driver=driver)
    s = object.__new__(Kandinsky5Sampler)
    s.pipeline = pipe
    s.config = config
    s.device = pipe.device
    return s


def test_frame_rule_accepts_a_single_still():
    assert frame_predicate("4n+1")(1) is True


def test_num_frames_none_uses_run_default():
    s = _sampler({"sample_num_frames": 17})
    gen = torch.Generator().manual_seed(0)
    noise = s._create_initial_noise(768, 512, gen)
    # channels-last [1, F_lat, H/8, W/8, C]; 17 → (17-1)//4+1 = 5.
    assert noise.shape[1] == 5


def test_per_prompt_num_frames_1_is_still():
    s = _sampler({"sample_num_frames": 17})
    s._active_prompt_cfg = {"num_frames": 1}
    gen = torch.Generator().manual_seed(0)
    noise = s._create_initial_noise(768, 512, gen)
    assert noise.shape[1] == 1  # (1-1)//4+1
