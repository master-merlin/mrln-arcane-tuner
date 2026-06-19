"""Regression: WAN must condition the transformer on the RAW [0,1000] timestep.

GPU smoke (wan21_t2v / wan22_t2v) produced a PURE-NOISE sample at step 0 — i.e.
with the freshly-initialized (zero-contribution) LoRA, so the *frozen base*
WAN model was failing to denoise at all.

Root cause: the diffusers WAN transformer's time embedder consumes the RAW
FlowMatchEuler timestep in ``[0, 1000]`` — ``WanTimeTextImageEmbedding`` uses
``Timesteps(downscale_freq_shift=0)`` (a sinusoidal embedding, no internal
``/1000``), and ``WanPipeline`` feeds it ``scheduler.timesteps`` which the
``FlowMatchEulerDiscreteScheduler`` defines as ``sigmas * num_train_timesteps``
(``[0, 1000]``). The shared WAN driver instead divided by 1000 in BOTH
``add_noise`` (correct — the lerp needs ``sigma ∈ [0,1]``) AND ``forward_pass``
(WRONG — it fed the transformer ``[0,1]``). The frozen, non-LoRA'd time embedder
then read every step as ``t ≈ 0`` ("already clean") → predicted ≈ 0 velocity →
the noise was never removed → pure-noise samples. LTX-2 already passes the raw
timestep (and samples correctly); the WAN shared driver never got that fix.

These tests pin that the WAN ``forward_pass`` and both WAN samplers condition the
transformer on the raw ``[0,1000]`` timestep (``add_noise``'s ``/1000`` lerp,
which is correct, is covered by ``assert_flowmatch_timestep_contract``).
"""

import torch

from app.engine.models.families.wan_shared.driver_base import WanDriverBase
from app.engine.models.families.wan_shared.sampler_base import WanVideoSamplerBase
from app.engine.strategies.sigma_schedule import shifted_sigmas


class _RecordingWan(torch.nn.Module):
    """Fake WAN transformer that records every ``timestep`` it is called with."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_timesteps: list[torch.Tensor] = []

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image=None,
        return_dict=False,
    ):
        self.seen_timesteps.append(timestep.detach().clone().float())
        # Velocity-shaped output (zeros → euler trajectory stays finite).
        return (torch.zeros_like(hidden_states),)


class _Drv(WanDriverBase):
    """Concrete WAN driver stub wired to a recording transformer."""

    def __init__(self, transformer: torch.nn.Module, *, is_i2v: bool = False) -> None:
        # Bypass the real __init__ (needs a definition + device); set only what
        # forward_pass reads.
        self.is_i2v = is_i2v
        self.transformer = transformer
        self.BATCH_FIRST_FRAME_LATENT = WanDriverBase.BATCH_FIRST_FRAME_LATENT
        self.BATCH_IMAGE_EMBED = WanDriverBase.BATCH_IMAGE_EMBED

    def get_saver(self):  # pragma: no cover - unused
        return None


# ── Training forward ──────────────────────────────────────────────────────


class TestForwardPassTimestepScale:
    def test_t2v_forward_feeds_raw_1000_scale_timestep(self):
        """``forward_pass`` must pass the transformer the RAW [0,1000] timestep,
        not ``timesteps / 1000``.
        """
        fake = _RecordingWan()
        drv = _Drv(fake, is_i2v=False)

        timesteps = torch.tensor([500.0])  # mid of the [0,1000] FlowMatchEuler range
        noisy = torch.randn(1, 16, 3, 8, 8)
        text = torch.zeros(1, 4, 16)

        drv.forward_pass(noisy, timesteps, text, {})

        seen = fake.seen_timesteps[0]
        assert torch.allclose(seen, timesteps, atol=1e-4), (
            f"WAN forward_pass must feed the transformer the raw [0,1000] "
            f"timestep; got {seen.tolist()} for input {timesteps.tolist()} "
            f"(a 1000× under-scale → frozen time embedder reads t≈0 → noise)."
        )

    def test_i2v_forward_feeds_raw_1000_scale_timestep(self):
        """The I2V path (36-ch conditioned input) must also feed raw [0,1000]."""
        fake = _RecordingWan()
        drv = _Drv(fake, is_i2v=True)

        timesteps = torch.tensor([750.0])
        noisy = torch.randn(1, 16, 3, 8, 8)
        text = torch.zeros(1, 4, 16)
        batch = {drv.BATCH_FIRST_FRAME_LATENT: torch.randn(1, 16, 1, 8, 8)}

        drv.forward_pass(noisy, timesteps, text, batch)

        seen = fake.seen_timesteps[0]
        assert torch.allclose(seen, timesteps, atol=1e-4), (
            f"WAN i2v forward_pass must feed raw [0,1000]; got {seen.tolist()}."
        )


# ── Samplers ───────────────────────────────────────────────────────────────


class _DriverStub:
    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model

    def get_primary_model(self) -> torch.nn.Module:
        return self._model


class _Pipeline:
    def __init__(self, model: torch.nn.Module) -> None:
        self.config = {"sample_num_frames": 5}
        self.device = torch.device("cpu")
        self.autocast_dtype = torch.bfloat16
        self.driver = _DriverStub(model)


class TestSamplerTimestepScale:
    def test_wan_sampler_feeds_raw_1000_scale_timesteps(self):
        """``WanVideoSamplerBase.denoise`` must condition the transformer on
        ``sigma * 1000`` (the [0,1000] FlowMatchEuler scale), matching training.
        """
        fake = _RecordingWan()
        sampler = WanVideoSamplerBase(_Pipeline(fake))

        num_steps = 4
        noise = torch.randn(1, 16, 2, 4, 4)
        text = torch.zeros(1, 8, 16)
        sampler.denoise(noise, text, num_steps=num_steps, guidance_scale=1.0, seed=0)

        seen = torch.cat(fake.seen_timesteps)  # one scalar per step
        expected = shifted_sigmas(num_steps, sampler.shift)[:-1] * 1000.0
        assert torch.allclose(seen, expected, atol=1e-3), (
            f"WAN sampler must feed timestep = sigma*1000; got {seen.tolist()} "
            f"expected {expected.tolist()}. Feeding the bare sigma ([0,1]) makes "
            f"the frozen base model read t≈0 → pure-noise samples."
        )
        # Sanity: the high-noise step is near the full scale, not near 0.
        assert seen.max().item() > 100.0


class TestWan22SamplerTimestepScale:
    def test_wan22_dual_expert_feeds_raw_1000_scale_timesteps(self):
        """WAN 2.2 overrides ``denoise`` (a separate code path) — it must also
        feed ``sigma * 1000``. Boundary routing still compares the [0,1] sigma
        fraction, so both experts receive the scaled timestep.
        """
        from app.engine.models.families.wan22.sampler import Wan22Sampler

        high = _RecordingWan()
        low = _RecordingWan()

        class _Wan22Driver:
            # Real T2V boundary. With the shift=3.0 schedule the evaluated sigmas
            # are ~[1.0, 0.9, 0.75, 0.5], so 0.875 routes the first two steps to
            # high and the rest to low → BOTH experts run.
            boundary = 0.875
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
        num_steps = 4
        noise = torch.randn(1, 16, 2, 4, 4)
        text = torch.zeros(1, 8, 16)
        sampler.denoise(noise, text, num_steps=num_steps, guidance_scale=1.0, seed=0)

        seen = torch.cat(high.seen_timesteps + low.seen_timesteps)
        # Every timestep handed to either expert must be on the [0,1000] scale.
        assert seen.max().item() > 100.0, (
            f"WAN 2.2 sampler must feed timestep = sigma*1000; got {seen.tolist()}."
        )
        # And both experts actually ran (boundary 0.5 on a 1→0 schedule).
        assert high.seen_timesteps, "high-noise expert never ran"
        assert low.seen_timesteps, "low-noise expert never ran"
