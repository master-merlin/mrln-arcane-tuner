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


# ── i2v-checkpoint previews (GPU UAT 2026-07-14) ─────────────────────────────


class _Stub36ChTransformer(torch.nn.Module):
    """A14B i2v stand-in: 36-in-channel patch_embedding, 16-channel output."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(in_channels=36)
        self.seen_channels: list[int] = []
        self._param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states=None, timestep=None, encoder_hidden_states=None,
                encoder_hidden_states_image=None, return_dict=False):
        self.seen_channels.append(hidden_states.shape[1])
        if hidden_states.shape[1] != 36:
            raise RuntimeError(
                f"expected input to have 36 channels, but got "
                f"{hidden_states.shape[1]} channels instead"
            )
        return (torch.zeros_like(hidden_states[:, :16]),)


def test_denoise_zero_pads_for_i2v_checkpoints():
    """An A14B i2v checkpoint (in_channels=36) must receive the zero-padded
    [noisy16, mask4, cond16] input in previews — the plain 16-channel t2v
    path crashed step-0 AND final sampling on wan2.1-i2v-14b-480p (GPU UAT
    2026-07-14: 'expected input ... to have 36 channels, but got 16'). Zero
    mask+cond = 'no pinned frame', the same semantics the training-side F=1
    still guard uses (driver_base.build_still_t2v_input)."""
    s = _sampler({})
    tf = _Stub36ChTransformer()
    s.pipeline.driver = SimpleNamespace(get_primary_model=lambda: tf)
    s.pipeline.autocast_dtype = torch.float32

    noise = torch.randn(1, 16, 3, 4, 4)
    latents = s.denoise(noise, torch.zeros(1, 5, 8), num_steps=2,
                        guidance_scale=1.0, seed=42)

    assert latents.shape == noise.shape
    assert tf.seen_channels and all(c == 36 for c in tf.seen_channels)


def test_denoise_keeps_16_channels_for_t2v_checkpoints():
    """T2V checkpoints (in_channels=16) keep the byte-identical plain path."""
    s = _sampler({})

    class _Stub16(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(in_channels=16)
            self.seen_channels = []
            self._param = torch.nn.Parameter(torch.zeros(1))

        def forward(self, hidden_states=None, timestep=None,
                    encoder_hidden_states=None, encoder_hidden_states_image=None,
                    return_dict=False):
            self.seen_channels.append(hidden_states.shape[1])
            return (torch.zeros_like(hidden_states),)

    tf = _Stub16()
    s.pipeline.driver = SimpleNamespace(get_primary_model=lambda: tf)
    s.pipeline.autocast_dtype = torch.float32

    noise = torch.randn(1, 16, 3, 4, 4)
    s.denoise(noise, torch.zeros(1, 5, 8), num_steps=2, guidance_scale=1.0, seed=42)
    assert tf.seen_channels and all(c == 16 for c in tf.seen_channels)
