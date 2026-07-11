"""Vendor smoke tests for the boogu_image family (Task 1 — vendor drop only).

No loader/driver/trainer/sampler yet — this pins three contracts against the
vendored upstream code (github.com/boogu-project/Boogu-Image @
ac9e40c1350fd60c502137a678ad1001d51e2ae7, Apache-2.0) so later tasks can build
on stable ground:

  1. Import smoke: every vendored module imports cleanly with NO ``boogu``
     package installed, and the flash_attn/triton fallback paths are the ones
     actually exercised in this env. flash_attn is genuinely not importable
     here; triton IS importable as a bare module (torch-bundled, 3.7.1) but
     lacks discoverable distribution metadata, so the vendored
     ``is_triton_available()`` still reports False and the block_lumina2
     "cuda" gate is never true anyway -- both facts are pinned below.
  2. Tiny transformer instantiate + forward on CPU: a divisibility-respecting
     tiny config, fed a LIST of two different-resolution [C, H, W] samples
     (BooguImageTransformer2DModel's real I/O contract — variable resolution
     per sample), asserting list-out/list-in shape parity + finite values +
     eager (non-flash) attention processors.
  3. Scheduler math: the shipped config (do_shift, dynamic_time_shift=False,
     time_shift_version="v1", seq_len=4096) derives a static mu of 1.15
     exactly (lin(4096) with the default (256, 0.5) -> (4096, 1.15) anchors
     evaluates to y2 at x=x2); ``step()`` matches hand-computed fp32 Euler
     including the trailing synthetic 1.0 timestep; an all-zero model_output
     walk leaves the sample bit-identical.

Tiny transformer config (divisibility-respecting, deliberately un-real):
  hidden_size=16, num_attention_heads=2, num_kv_heads=1 (2 % 1 == 0),
  head_dim=8, axes_dim_rope=(2, 2, 4) (sums to head_dim=8), axes_lens all 64
  (comfortably above the tiny token counts used here), num_layers=2 with
  num_double_stream_layers=1 (1 double-stream + 1 single-stream block),
  num_refiner_layers=1, multiple_of=8, in_channels=4, patch_size=2,
  instruction_feat_dim=8 (reduce_type="mean" so a plain Tensor instruction
  input is used as-is), timestep_scale=1000 (matches the real checkpoint:
  callers pass [0, 1) sigmas, the transformer scales internally).
"""

from __future__ import annotations

import importlib.util

import torch

BOOGU_VENDOR = "app.engine.models.families.boogu_image.vendor"

TINY_AXES_DIM_ROPE = (2, 2, 4)
TINY_AXES_LENS = (64, 64, 64)
TINY_HIDDEN_SIZE = 16
TINY_NUM_ATTENTION_HEADS = 2
TINY_NUM_KV_HEADS = 1
TINY_IN_CHANNELS = 4
TINY_PATCH_SIZE = 2
TINY_INSTRUCTION_FEAT_DIM = 8


def _build_tiny_transformer():
    from app.engine.models.families.boogu_image.vendor.models.transformers.transformer_boogu import (
        BooguImageTransformer2DModel,
    )

    model = BooguImageTransformer2DModel(
        patch_size=TINY_PATCH_SIZE,
        in_channels=TINY_IN_CHANNELS,
        out_channels=None,
        hidden_size=TINY_HIDDEN_SIZE,
        num_layers=2,
        num_double_stream_layers=1,
        num_refiner_layers=1,
        num_attention_heads=TINY_NUM_ATTENTION_HEADS,
        num_kv_heads=TINY_NUM_KV_HEADS,
        multiple_of=8,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        axes_dim_rope=TINY_AXES_DIM_ROPE,
        axes_lens=TINY_AXES_LENS,
        instruction_feature_configs=dict(
            instruction_feat_dim=TINY_INSTRUCTION_FEAT_DIM,
            reduce_type="mean",
            num_instruction_feat_layers=1,
        ),
        prompt_tuning_configs=dict(use_prompt_tuning=False),
        timestep_scale=1000.0,
    )
    model.eval()
    return model


class TestImportSmoke:
    """Every vendored module imports cleanly; flash_attn/triton fallback paths are live."""

    def test_flash_attn_not_importable_in_this_env(self):
        # Pins the premise of this whole test module: if flash_attn ever becomes
        # importable in this venv, the "eager path is under test" claims below
        # need re-verification (flash_attn is a soft dep upstream, no install here).
        assert importlib.util.find_spec("flash_attn") is None

    def test_triton_is_importable_but_reports_unavailable(self):
        # Unlike flash_attn, a `triton` module IS importable in this venv
        # (3.7.1, bundled as a torch/Windows build dependency) -- but it has
        # no discoverable distribution metadata under that name, so
        # importlib.metadata.version("triton") raises PackageNotFoundError
        # and the vendored is_triton_available() (see next test) still
        # correctly reports False. Pinned here so a future venv change that
        # fixes triton's metadata doesn't silently flip which attention/norm
        # path the tests below exercise without anyone noticing.
        import importlib.metadata

        assert importlib.util.find_spec("triton") is not None
        try:
            importlib.metadata.version("triton")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise AssertionError(
                "triton now has discoverable package metadata in this venv -- "
                "is_triton_available() may no longer be False; re-verify the "
                "fallback-path assumptions in this test module"
            )

    def test_import_utils_reports_both_unavailable(self):
        from app.engine.models.families.boogu_image.vendor.utils.import_utils import (
            is_flash_attn_available,
            is_triton_available,
        )

        assert is_flash_attn_available() is False
        # False regardless of raw triton importability (see
        # test_triton_is_importable_but_reports_unavailable) because
        # is_triton_available() gates on importlib.metadata, not find_spec.
        assert is_triton_available() is False

    def test_import_teacache_util(self):
        from app.engine.models.families.boogu_image.vendor.utils.teacache_util import (
            TeaCacheParams,
        )

        params = TeaCacheParams()
        assert params.accumulated_rel_l1_distance == 0

    def test_import_cache_functions(self):
        from app.engine.models.families.boogu_image.vendor.cache_functions import (
            cal_type,
        )

        assert callable(cal_type)

    def test_import_taylorseer_utils(self):
        from app.engine.models.families.boogu_image.vendor.taylorseer_utils import (
            derivative_approximation,
            taylor_cache_init,
            taylor_formula,
        )

        assert callable(derivative_approximation)
        assert callable(taylor_cache_init)
        assert callable(taylor_formula)

    def test_import_attention_processor(self):
        from app.engine.models.families.boogu_image.vendor.models import (
            attention_processor,
        )

        assert hasattr(attention_processor, "BooguImageAttnProcessor")
        assert hasattr(attention_processor, "BooguImageDoubleStreamSelfAttnProcessor")
        # Flash variants still import (guarded classes), they just can't be
        # instantiated without flash_attn (see test_flash_processor_raises_without_flash_attn).
        assert hasattr(attention_processor, "BooguImageAttnProcessorFlash2Varlen")

    def test_flash_processor_raises_without_flash_attn(self):
        from app.engine.models.families.boogu_image.vendor.models.attention_processor import (
            BooguImageAttnProcessorFlash2Varlen,
        )

        try:
            BooguImageAttnProcessorFlash2Varlen()
        except ImportError:
            pass
        else:
            raise AssertionError(
                "expected ImportError instantiating the flash processor without flash_attn"
            )

    def test_import_embeddings(self):
        from app.engine.models.families.boogu_image.vendor.models.embeddings import (
            TimestepEmbedding,
            apply_rotary_emb,
        )

        assert callable(apply_rotary_emb)
        assert TimestepEmbedding is not None

    def test_import_rope(self):
        from app.engine.models.families.boogu_image.vendor.models.transformers.rope import (
            BooguImageDoubleStreamRotaryPosEmbed,
            BooguImagePromptTuningRotaryPosEmbed,
        )

        assert BooguImageDoubleStreamRotaryPosEmbed is not None
        assert BooguImagePromptTuningRotaryPosEmbed is not None

    def test_import_block_lumina2_uses_stock_rmsnorm_fallback(self):
        # is_triton_available() is False (and even if triton were installed,
        # the upstream gate also requires "cuda" in os.getenv("device", "cpu"),
        # which is never true here) -> the module-level RMSNorm binding must
        # be torch.nn's, not the vendored triton kernel.
        from app.engine.models.families.boogu_image.vendor.models.transformers import (
            block_lumina2,
        )

        assert block_lumina2.RMSNorm is torch.nn.RMSNorm

    def test_import_components(self):
        from app.engine.models.families.boogu_image.vendor.models.transformers.components import (
            swiglu,
        )

        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        out = swiglu(x, y)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_import_transformer_boogu(self):
        from app.engine.models.families.boogu_image.vendor.models.transformers.transformer_boogu import (
            BooguImageTransformer2DModel,
            PromptEmbedding,
        )

        assert BooguImageTransformer2DModel is not None
        assert PromptEmbedding is not None

    def test_import_scheduler(self):
        from app.engine.models.families.boogu_image.vendor.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
            FlowMatchEulerDiscreteScheduler,
        )

        assert FlowMatchEulerDiscreteScheduler is not None

    def test_import_lora_conversion(self):
        from app.engine.models.families.boogu_image.vendor.lora_conversion import (
            _convert_non_diffusers_lumina2_lora_to_diffusers,
        )

        assert callable(_convert_non_diffusers_lumina2_lora_to_diffusers)


class TestTinyTransformerForward:
    """Tiny CPU instantiate + forward pins the variable-resolution list I/O contract."""

    def test_builds_with_tiny_divisibility_respecting_config(self):
        model = _build_tiny_transformer()
        assert model.config.hidden_size == TINY_HIDDEN_SIZE
        assert (
            model.config.hidden_size // model.config.num_attention_heads
            == sum(TINY_AXES_DIM_ROPE)
        )
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_rejects_mismatched_axes_dim_rope(self):
        from app.engine.models.families.boogu_image.vendor.models.transformers.transformer_boogu import (
            BooguImageTransformer2DModel,
        )

        try:
            BooguImageTransformer2DModel(
                patch_size=2,
                in_channels=4,
                hidden_size=16,
                num_layers=1,
                num_double_stream_layers=0,
                num_refiner_layers=1,
                num_attention_heads=2,
                num_kv_heads=1,
                multiple_of=8,
                norm_eps=1e-5,
                axes_dim_rope=(1, 1, 1),  # sums to 3, != head_dim 8
                axes_lens=TINY_AXES_LENS,
                instruction_feature_configs=dict(
                    instruction_feat_dim=8, reduce_type="mean"
                ),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for mismatched axes_dim_rope")

    def test_forward_variable_resolution_list_io(self):
        from app.engine.models.families.boogu_image.vendor.models.transformers.rope import (
            BooguImageDoubleStreamRotaryPosEmbed,
        )

        model = _build_tiny_transformer()

        # Two DIFFERENT resolutions in the same batch -- pins that the model
        # really treats hidden_states as a list of independent [C, H, W]
        # samples, not a padded/stacked tensor.
        hidden_states = [
            torch.randn(TINY_IN_CHANNELS, 4, 4),  # 2x2 = 4 patch tokens
            torch.randn(TINY_IN_CHANNELS, 4, 6),  # 2x3 = 6 patch tokens
        ]
        timestep = torch.rand(2)  # [0, 1) sigmas, per verified upstream facts
        instruction_hidden_states = torch.randn(2, 3, TINY_INSTRUCTION_FEAT_DIM)
        instruction_attention_mask = torch.ones(2, 3, dtype=torch.bool)

        freqs_cis = BooguImageDoubleStreamRotaryPosEmbed.get_freqs_cis(
            TINY_AXES_DIM_ROPE, TINY_AXES_LENS, theta=10000
        )

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                timestep=timestep,
                instruction_hidden_states=instruction_hidden_states,
                freqs_cis=freqs_cis,
                instruction_attention_mask=instruction_attention_mask,
                ref_image_hidden_states=None,
                return_dict=False,
            )

        assert isinstance(output, list)
        assert len(output) == 2
        for sample_in, sample_out in zip(hidden_states, output):
            assert sample_out.shape == sample_in.shape
            assert torch.isfinite(sample_out).all()

    def test_forward_uses_eager_non_flash_attention_processors(self):
        from app.engine.models.families.boogu_image.vendor.models.attention_processor import (
            BooguImageAttnProcessor,
            BooguImageAttnProcessorFlash2Varlen,
            BooguImageDoubleStreamSelfAttnProcessor,
            BooguImageDoubleStreamSelfAttnProcessorFlash2Varlen,
        )

        model = _build_tiny_transformer()

        # Single-stream / refiner blocks share BooguImageTransformerBlock ->
        # attn.processor. Double-stream blocks additionally carry a
        # double_stream self-attn processor on img_self_attn (same class) and
        # a dedicated dual-stream processor called directly on
        # img_instruct_attn.processor.
        single_stream_block = model.single_stream_layers[0]
        assert isinstance(single_stream_block.attn.processor, BooguImageAttnProcessor)
        assert not isinstance(
            single_stream_block.attn.processor, BooguImageAttnProcessorFlash2Varlen
        )

        double_stream_block = model.double_stream_layers[0]
        assert isinstance(
            double_stream_block.img_self_attn.processor, BooguImageAttnProcessor
        )
        assert isinstance(
            double_stream_block.img_instruct_attn.processor,
            BooguImageDoubleStreamSelfAttnProcessor,
        )
        assert not isinstance(
            double_stream_block.img_instruct_attn.processor,
            BooguImageDoubleStreamSelfAttnProcessorFlash2Varlen,
        )


class TestSchedulerMath:
    """Pins the custom (non-interchangeable) scheduler's shipped-config math."""

    def _make_scheduler(self):
        from app.engine.models.families.boogu_image.vendor.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
            FlowMatchEulerDiscreteScheduler,
        )

        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            do_shift=True,
            dynamic_time_shift=False,
            time_shift_version="v1",
            seq_len=4096,
        )

    def test_set_timesteps_signature_contract(self):
        sched = self._make_scheduler()
        # Exactly this call shape must work (no mu=/sigmas= kwargs, per the
        # verified upstream facts -- this is NOT a stock FlowMatchEuler).
        sched.set_timesteps(
            num_inference_steps=4, device="cpu", timesteps=None, num_tokens=None
        )
        assert sched.timesteps.shape == (4,)
        # Trailing synthetic 1.0 step appended internally.
        assert sched._timesteps.shape == (5,)
        assert sched._timesteps[-1].item() == 1.0

    def test_static_mu_is_1_15(self):
        from app.engine.models.families.boogu_image.vendor.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
            FlowMatchEulerDiscreteScheduler,
        )

        # lin(x1=256, y1=0.5, x2=4096, y2=1.15) evaluated at x=seq_len=4096
        # equals y2 exactly by construction (x2 is the anchor point) -- this
        # is the "static mu = 1.15" fact from the recon notes.
        lin = FlowMatchEulerDiscreteScheduler._get_lin_function(y1=0.5, y2=1.15)
        mu = lin(4096)
        assert abs(mu - 1.15) < 1e-9

        # Cross-check against the actual shift applied by set_timesteps: at
        # t=0 the v1 time-shift is a no-op (t=0 is a fixed point), so probe
        # via the private helper directly with a mid-range value instead.
        import numpy as np

        shifted = FlowMatchEulerDiscreteScheduler._time_shift_v1(
            np.array([0.5], dtype=np.float32), mu=mu, sigma=1.0
        )
        expected = 1.0 - (
            np.exp(mu) / (np.exp(mu) + np.power(1.0 / 0.5 - 1.0, 1.0))
        )
        assert abs(float(shifted[0]) - float(expected)) < 1e-5

    def test_step_matches_hand_computed_euler_including_trailing_step(self):
        sched = self._make_scheduler()
        sched.set_timesteps(num_inference_steps=4, device="cpu", num_tokens=None)

        sample = torch.randn(1, 2, 2, 2)
        model_output = torch.randn(1, 2, 2, 2)

        for step_index, t in enumerate(sched.timesteps):
            t_cur = sched._timesteps[step_index]
            t_next = sched._timesteps[step_index + 1]
            expected = (sample.to(torch.float32) + (t_next - t_cur) * model_output).to(
                model_output.dtype
            )

            out = sched.step(model_output, t, sample, return_dict=False)
            sample = out[0]

            assert torch.allclose(sample, expected, atol=1e-6)

        # Final step_index must have walked past the trailing synthetic 1.0.
        assert sched.step_index == len(sched.timesteps)

    def test_zero_model_output_walk_leaves_sample_unchanged(self):
        sched = self._make_scheduler()
        sched.set_timesteps(num_inference_steps=6, device="cpu", num_tokens=None)

        sample = torch.randn(1, 3, 4, 4)
        original = sample.clone()
        model_output = torch.zeros_like(sample)

        for t in sched.timesteps:
            out = sched.step(model_output, t, sample, return_dict=False)
            sample = out[0]

        assert torch.equal(sample, original)
