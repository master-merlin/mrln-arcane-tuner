"""Regression: the WAN sampler must run its forward under autocast.

GPU smoke (wan21_t2v / wan22_t2v) crashed at sample time with
``Input type (float) and bias type (struct c10::BFloat16) should be the same``.
Root cause: under mixed-precision training the WAN transformer is a MIXED-dtype
module — most weights are bf16 but precision-sensitive params
(``scale_shift_table``, ``time_embedder``, norms) stay fp32. The sampler used
``next(parameters()).dtype`` (whichever param is first → fp32 ``scale_shift_table``)
and cast every input to that single dtype, so fp32 inputs hit the bf16
``patch_embedding`` and crashed. No single manual cast can satisfy a mixed
module; the fix runs the forward under ``torch.autocast`` (the same regime as
training, per-op casting) while the Euler trajectory stays fp32 in
``euler_integrate`` (so the no-autocast-collapse contract still holds).

These tests reproduce the mixed-dtype hazard on CPU (CPU autocast supports
bf16) and pin that the sampler tolerates it.
"""

import torch

from app.engine.models.families.wan21.sampler import Wan21Sampler


class _MixedDtypeFakeWan(torch.nn.Module):
    """Reproduces the real WAN mixed-dtype module.

    An fp32 parameter is registered FIRST (so ``next(parameters()).dtype`` is
    fp32, like ``scale_shift_table``) plus a bf16 compute layer (like
    ``patch_embedding``). Feeding fp32 inputs WITHOUT autocast crashes on the
    bf16 layer — the exact GPU bug.
    """

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        # fp32, registered FIRST → next(parameters()).dtype == torch.float32
        self.scale_shift_table = torch.nn.Parameter(torch.zeros(2, channels))
        # bf16 compute layer (the crashing patch_embedding analogue)
        self.proj = torch.nn.Linear(channels, channels).to(torch.bfloat16)

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image=None,
        return_dict=False,
    ):
        # [B, C, F, H, W] → channels-last for the Linear → back
        h = self.proj(hidden_states.movedim(1, -1)).movedim(-1, 1)
        return (h,)


class _DriverStub:
    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model

    def get_primary_model(self) -> torch.nn.Module:
        return self._model


class _PipelineFull:
    """Trainer stub exposing what the FULL denoise path reads."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.config = {"sample_num_frames": 5}
        self.device = torch.device("cpu")
        self.autocast_dtype = torch.bfloat16
        self.driver = _DriverStub(model)


def test_mixed_dtype_fake_crashes_without_autocast():
    """Sanity: the fake genuinely reproduces the bug — fp32 input on a bf16
    layer with NO autocast raises. Proves the denoise test below is not vacuous.
    """
    fake = _MixedDtypeFakeWan(channels=16)
    x = torch.randn(1, 16, 2, 4, 4, dtype=torch.float32)
    raised = False
    try:
        fake(hidden_states=x, timestep=torch.zeros(1), encoder_hidden_states=None)
    except RuntimeError:
        raised = True
    assert raised, "fp32 input on a bf16 layer must crash without autocast"


def test_wan_sampler_denoise_handles_mixed_dtype_model():
    """The WAN sampler runs the forward under autocast, so a mixed bf16/fp32
    transformer (the real WAN mixed-precision state) samples without the
    'Input type float vs bias BFloat16' crash; the trajectory stays fp32.
    """
    fake = _MixedDtypeFakeWan(channels=16)
    sampler = Wan21Sampler(_PipelineFull(fake))

    noise = torch.randn(1, 16, 2, 4, 4, dtype=torch.float32)
    text = torch.zeros(1, 8, 16)  # unused by the fake; shape-valid stand-in

    out = sampler.denoise(noise, text, num_steps=3, guidance_scale=1.0, seed=0)

    assert out.dtype == torch.float32, "trajectory must stay fp32 (no collapse)"
    assert torch.isfinite(out).all(), "denoise output must be finite"
    assert out.shape == noise.shape


def test_wan22_dual_expert_denoise_handles_mixed_dtype_model():
    """WAN 2.2 overrides ``denoise`` for boundary-based expert switching — a
    SEPARATE code path that had the same manual-cast bug. With boundary 0.5 and
    a 1→0 sigma schedule, BOTH experts (each mixed-dtype) run in one denoise;
    autocast must keep it from crashing while the trajectory stays fp32.
    """
    from app.engine.models.families.wan22.sampler import Wan22Sampler

    high = _MixedDtypeFakeWan(channels=16)
    low = _MixedDtypeFakeWan(channels=16)

    class _Wan22Driver:
        boundary = 0.5
        transformer_high = high
        transformer_low = low

        def get_primary_model(self):
            return high

    class _Wan22Pipeline:
        def __init__(self):
            self.config = {"resolutions": [480], "sample_num_frames": 5}
            self.device = torch.device("cpu")
            self.autocast_dtype = torch.bfloat16
            self.driver = _Wan22Driver()

    sampler = Wan22Sampler(_Wan22Pipeline())
    noise = torch.randn(1, 16, 2, 4, 4, dtype=torch.float32)
    text = torch.zeros(1, 8, 16)

    out = sampler.denoise(noise, text, num_steps=4, guidance_scale=1.0, seed=0)

    assert out.dtype == torch.float32, "trajectory must stay fp32 (no collapse)"
    assert torch.isfinite(out).all(), "dual-expert denoise output must be finite"
    assert out.shape == noise.shape
