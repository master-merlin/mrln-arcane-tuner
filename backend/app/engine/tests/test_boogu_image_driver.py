"""boogu_image driver tests (Task 4) — the real training-side model logic.

Mirrors the krea2/kandinsky5 driver-test style: a tiny REAL vendored
transformer + a mocked ``ModelDefinition``, no GPU, no network.

Covers the load-bearing correctness contracts from task-4-brief.md:

1. Time convention (DERIVED from the vendored scheduler + upstream
   pipeline_boogu.py — see task-4-report.md for the file:line evidence):
   ``x_t = (1-t)*noise + t*x0`` (t=0 noise, t=1 clean), target = ``x0 - noise``.
2. Timestep scale: raw ``[0, 1)`` reaches the transformer, no ``/1000``/``*1000``
   anywhere in the driver.
3. Perfect-velocity round-trip: the REAL vendored scheduler loop, driven by
   an oracle "transformer" that always returns the true (t-independent)
   velocity, must land exactly on x0 — pins sign + loop direction in one shot.
4. List-of-tensors I/O adapter: batched ``[B,C,H,W]`` <-> list-of-``[C,H,W]``,
   per-sample identity preserved, equal-resolution fast path exercised on a
   real tiny transformer.
5. freqs_cis / rope wiring: driver's construction matches a direct
   upstream-style ``BooguImageDoubleStreamRotaryPosEmbed.get_freqs_cis`` call.
6. Forward signature: kwargs reaching ``transformer.forward`` match the
   documented contract exactly.
7. LoRA targeting: curated list passes through verbatim, empty list fails
   loudly (no silent attention-only fallback), PEFT wraps the tiny model
   correctly (processor-owned linears, GQA widths, to_out.0).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.boogu_image.driver import BooguImageDriver
from app.engine.models.families.boogu_image.vendor.models.transformers.rope import (
    BooguImageDoubleStreamRotaryPosEmbed,
)
from app.engine.models.families.boogu_image.vendor.models.transformers.transformer_boogu import (
    BooguImageTransformer2DModel,
)
from app.engine.models.families.boogu_image.vendor.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
    FlowMatchEulerDiscreteScheduler,
)

# Same divisibility-respecting tiny config as test_boogu_image_vendor.py /
# test_boogu_image_definitions.py / test_boogu_image_loader.py.
TINY_AXES_DIM_ROPE = (2, 2, 4)
TINY_AXES_LENS = (64, 64, 64)
TINY_HIDDEN_SIZE = 16
TINY_IN_CHANNELS = 4
TINY_INSTRUCTION_FEAT_DIM = 8


def _tiny_transformer_config(**overrides) -> dict:
    cfg = dict(
        patch_size=2,
        in_channels=TINY_IN_CHANNELS,
        out_channels=None,
        hidden_size=TINY_HIDDEN_SIZE,
        num_layers=2,
        num_double_stream_layers=1,
        num_refiner_layers=1,
        num_attention_heads=2,
        num_kv_heads=1,
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
    cfg.update(overrides)
    return cfg


def _tiny_transformer(**overrides) -> BooguImageTransformer2DModel:
    model = BooguImageTransformer2DModel(**_tiny_transformer_config(**overrides))
    model.eval()
    return model


def _definition(lora_targets: list[str] | None = ()) -> MagicMock:
    d = MagicMock(spec=ModelDefinition)
    d.family = "boogu_image"
    d.id = "boogu-image-test"
    d.lora_targetable_modules = list(lora_targets) if lora_targets else lora_targets
    d.architecture_params = {}
    return d


def _driver(lora_targets: list[str] | None = ()) -> BooguImageDriver:
    return BooguImageDriver(_definition(lora_targets), torch.device("cpu"))


def _fake_text(batch: int, length: int = 3, dim: int = TINY_INSTRUCTION_FEAT_DIM):
    embeds = torch.randn(batch, length, dim)
    mask = torch.ones(batch, length, dtype=torch.bool)
    return (embeds, mask)


# ── Contract 1 + 3: time convention / perfect-velocity round-trip ─────────


class TestTimeConventionRoundTrip:
    """Pins sign + loop direction using the REAL vendored scheduler."""

    def _make_scheduler(self) -> FlowMatchEulerDiscreteScheduler:
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            do_shift=True,
            dynamic_time_shift=False,
            time_shift_version="v1",
            seq_len=4096,
        )

    def test_add_noise_endpoints(self):
        """t=0 -> pure noise, t=1 -> clean latents."""
        drv = _driver()
        x0 = torch.randn(2, 4, 4, 4)
        noise = torch.randn(2, 4, 4, 4)

        at_0 = drv.add_noise(x0, noise, torch.zeros(2))
        at_1 = drv.add_noise(x0, noise, torch.ones(2))

        assert torch.allclose(at_0, noise, atol=1e-6)
        assert torch.allclose(at_1, x0, atol=1e-6)

    def test_compute_target_is_data_minus_noise(self):
        drv = _driver()
        x0 = torch.randn(2, 4, 4, 4)
        noise = torch.randn(2, 4, 4, 4)
        # t-independent — probe a couple of different t to prove it.
        for t in (torch.zeros(2), torch.full((2,), 0.37), torch.ones(2)):
            target = drv.compute_target(x0, noise, t)
            assert torch.allclose(target, x0 - noise, atol=1e-6)

    def test_perfect_velocity_round_trip_lands_on_x0(self):
        """Full REAL scheduler loop, oracle "transformer" = true velocity.

        Starting from the EXACT starting noise and integrating the true
        (t-independent) velocity ``x0 - noise`` through the real vendored
        ``scheduler.step()`` must land exactly on x0 (Euler integration of a
        CONSTANT velocity field is exact regardless of step spacing, since
        consecutive Δt's over the full walk telescope to
        ``t_end - t_start == 1.0 - 0.0 == 1``). This single test catches sign
        flips, inverted loops, off-by-one timestep walks, and double-scaling.
        """
        drv = _driver()
        sched = self._make_scheduler()
        sched.set_timesteps(num_inference_steps=7, device="cpu", num_tokens=None)

        torch.manual_seed(0)
        x0 = torch.randn(1, 2, 3, 3, dtype=torch.float64)
        noise = torch.randn(1, 2, 3, 3, dtype=torch.float64)

        # Oracle: the TRUE, t-independent velocity under our derived
        # convention == the driver's own compute_target.
        oracle_velocity = drv.compute_target(x0, noise, torch.tensor([0.0]))
        assert torch.equal(oracle_velocity, x0 - noise)

        sample = noise.clone().to(torch.float32)
        for t in sched.timesteps:
            model_output = oracle_velocity.to(torch.float32)
            sample = sched.step(model_output, t, sample, return_dict=False)[0]

        assert torch.allclose(sample.to(torch.float64), x0, atol=1e-4)
        # Sanity: the walk actually stepped past the trailing synthetic 1.0.
        assert sched.step_index == len(sched.timesteps)

    def test_perfect_velocity_round_trip_is_step_count_invariant(self):
        """The exact-landing property must hold for ANY step count (proves
        it's not a coincidence of one particular schedule)."""
        drv = _driver()
        torch.manual_seed(1)
        x0 = torch.randn(1, 2, 2, 2)
        noise = torch.randn(1, 2, 2, 2)
        oracle_velocity = drv.compute_target(x0, noise, torch.tensor([0.0]))

        for n_steps in (1, 3, 12):
            sched = self._make_scheduler()
            sched.set_timesteps(num_inference_steps=n_steps, device="cpu", num_tokens=None)
            sample = noise.clone()
            for t in sched.timesteps:
                sample = sched.step(oracle_velocity, t, sample, return_dict=False)[0]
            assert torch.allclose(sample, x0, atol=1e-4), f"failed at n_steps={n_steps}"


# ── Contract 2: timestep scale (raw [0,1), no /1000 or *1000) ─────────────


class TestTimestepScale:
    def test_sample_timesteps_returns_raw_0_1(self):
        drv = _driver()
        config = {"timestep_sampling": "uniform"}
        t = drv.sample_timesteps(8, torch.device("cpu"), config)
        assert t.shape == (8,)
        assert torch.all(t >= 0.0) and torch.all(t <= 1.0)

    def test_forward_pass_receives_raw_unscaled_timestep(self):
        """Spy on the transformer call — captured timestep must equal the
        raw input exactly (no /1000, no *1000)."""
        drv = _driver()
        captured = {}

        def fake_transformer(**kwargs):
            captured.update(kwargs)
            return [h.clone() for h in kwargs["hidden_states"]]

        fake_transformer.config = MagicMock(
            axes_dim_rope=TINY_AXES_DIM_ROPE, axes_lens=TINY_AXES_LENS,
        )
        drv.model = fake_transformer

        noisy = torch.randn(2, TINY_IN_CHANNELS, 4, 4)
        raw_t = torch.tensor([0.123, 0.876])
        drv.forward_pass(noisy, raw_t, _fake_text(2), {})

        assert torch.equal(captured["timestep"], raw_t.to(noisy.dtype))

    def test_add_noise_and_compute_target_never_divide_or_multiply_by_1000(self):
        """The [0,1)-domain t must be used AS-IS — an accidental /1000 or
        *1000 would collapse add_noise toward one endpoint for any realistic
        t in [0,1]."""
        drv = _driver()
        x0 = torch.ones(1, 1, 2, 2)
        noise = torch.zeros(1, 1, 2, 2)
        t = torch.tensor([0.5])
        noisy = drv.add_noise(x0, noise, t)
        # (1-0.5)*0 + 0.5*1 = 0.5 -- a /1000 would give ~0 (all noise-ish),
        # a *1000 would clamp/blow up.
        assert torch.allclose(noisy, torch.full_like(noisy, 0.5))


# ── Contract 4: list-tensor I/O adapter ────────────────────────────────────


class TestListTensorAdapter:
    def test_forward_pass_equal_resolution_fast_path_on_real_tiny_model(self):
        model = _tiny_transformer()
        drv = _driver()
        drv.model = model

        noisy = torch.randn(2, TINY_IN_CHANNELS, 4, 4)
        t = torch.rand(2)
        with torch.no_grad():
            out = drv.forward_pass(noisy, t, _fake_text(2), {})

        assert out.shape == noisy.shape
        assert torch.isfinite(out).all()

    def test_forward_pass_list_adapter_preserves_per_sample_identity(self):
        """Explicit tensor->list->model->list->stack round trip: each output
        sample must correspond to its OWN input sample, not a shuffled one.

        Uses an identity-like stand-in "model" that tags each list entry so
        cross-sample mixing would be caught (real model's own per-sample
        list handling is already proven correct upstream; this test proves
        OUR adapter code doesn't reorder/mix samples).
        """
        drv = _driver()

        def tagging_model(**kwargs):
            # Return each sample scaled by its own batch index + 1, so a
            # mis-ordered re-stack is detectable.
            hs = kwargs["hidden_states"]
            assert isinstance(hs, list)
            return [h * (i + 1) for i, h in enumerate(hs)]

        tagging_model.config = MagicMock(
            axes_dim_rope=TINY_AXES_DIM_ROPE, axes_lens=TINY_AXES_LENS,
        )
        drv.model = tagging_model

        sample0 = torch.ones(TINY_IN_CHANNELS, 4, 4)
        sample1 = torch.full((TINY_IN_CHANNELS, 4, 4), 2.0)
        noisy = torch.stack([sample0, sample1], dim=0)
        t = torch.tensor([0.1, 0.9])

        out = drv.forward_pass(noisy, t, _fake_text(2), {})

        assert out.shape == noisy.shape
        assert torch.allclose(out[0], sample0 * 1)
        assert torch.allclose(out[1], sample1 * 2)

    def test_forward_pass_converts_batched_tensor_to_list_before_model_call(self):
        drv = _driver()
        captured = {}

        def fake_transformer(**kwargs):
            captured.update(kwargs)
            return [h.clone() for h in kwargs["hidden_states"]]

        fake_transformer.config = MagicMock(
            axes_dim_rope=TINY_AXES_DIM_ROPE, axes_lens=TINY_AXES_LENS,
        )
        drv.model = fake_transformer

        noisy = torch.randn(3, TINY_IN_CHANNELS, 4, 4)
        drv.forward_pass(noisy, torch.rand(3), _fake_text(3), {})

        hs = captured["hidden_states"]
        assert isinstance(hs, list)
        assert len(hs) == 3
        for i, sample in enumerate(hs):
            assert sample.shape == (TINY_IN_CHANNELS, 4, 4)
            assert torch.equal(sample, noisy[i])


# ── Contract 5: freqs_cis / rope wiring ────────────────────────────────────


class TestFreqsCisWiring:
    def test_matches_direct_upstream_style_invocation(self):
        drv = _driver()
        model = _tiny_transformer()
        drv.model = model

        built = drv._build_freqs_cis(model)
        expected = BooguImageDoubleStreamRotaryPosEmbed.get_freqs_cis(
            TINY_AXES_DIM_ROPE, TINY_AXES_LENS, theta=10000,
        )

        assert len(built) == len(expected) == 3
        for b, e in zip(built, expected):
            assert torch.equal(b, e)

    def test_reads_axes_config_from_model_not_hardcoded(self):
        """A model with DIFFERENT axes_lens must produce a differently-shaped
        table -- proves the driver reads the model's own config, not a
        hardcoded real-checkpoint constant."""
        drv = _driver()
        model = _tiny_transformer(axes_lens=(8, 8, 8))
        built = drv._build_freqs_cis(model)
        for t in built:
            assert t.shape[0] == 8

    def test_cache_invalidates_on_axes_change(self):
        drv = _driver()
        model_a = _tiny_transformer()
        built_a = drv._build_freqs_cis(model_a)
        model_b = _tiny_transformer(axes_lens=(8, 8, 8))
        built_b = drv._build_freqs_cis(model_b)
        assert built_a[0].shape[0] != built_b[0].shape[0]


# ── Contract 6: forward signature ──────────────────────────────────────────


class TestForwardSignature:
    def test_kwargs_reaching_transformer_match_documented_contract(self):
        drv = _driver()
        captured = {}

        def fake_transformer(**kwargs):
            captured.update(kwargs)
            return [h.clone() for h in kwargs["hidden_states"]]

        fake_transformer.config = MagicMock(
            axes_dim_rope=TINY_AXES_DIM_ROPE, axes_lens=TINY_AXES_LENS,
        )
        drv.model = fake_transformer

        noisy = torch.randn(1, TINY_IN_CHANNELS, 4, 4)
        embeds, mask = _fake_text(1)
        drv.forward_pass(noisy, torch.rand(1), (embeds, mask), {})

        assert set(captured.keys()) == {
            "hidden_states", "timestep", "instruction_hidden_states",
            "freqs_cis", "instruction_attention_mask",
            "ref_image_hidden_states", "return_dict",
        }
        assert captured["instruction_hidden_states"] is embeds
        assert captured["instruction_attention_mask"] is mask
        assert captured["ref_image_hidden_states"] is None
        assert captured["return_dict"] is False

    def test_missing_attention_mask_raises(self):
        drv = _driver()
        drv.model = MagicMock()
        with pytest.raises(ValueError, match="instruction_attention_mask"):
            drv.forward_pass(
                torch.randn(1, TINY_IN_CHANNELS, 4, 4),
                torch.rand(1),
                torch.randn(1, 3, TINY_INSTRUCTION_FEAT_DIM),  # bare tensor, no mask
                {},
            )

    def test_bare_tensor_text_embeddings_treated_as_embeddings_only(self):
        """Non-tuple text_embeddings (e.g. a mistaken direct TextEncoderOutput
        pass-through) must fail via the explicit mask guard, not silently."""
        drv = _driver()
        drv.model = MagicMock()
        with pytest.raises(ValueError):
            drv.forward_pass(
                torch.randn(1, TINY_IN_CHANNELS, 4, 4),
                torch.rand(1),
                object(),  # not a tuple, not usable
                {},
            )


# ── Contract 7: LoRA targeting ─────────────────────────────────────────────


def _attn_ff_suffixes(container) -> list[str]:
    def _is_modulation_linear(name: str) -> bool:
        parts = name.split(".")
        return len(parts) >= 2 and parts[-1] == "linear" and "norm" in parts[-2]

    return sorted(
        name
        for name, mod in container.named_modules()
        if isinstance(mod, torch.nn.Linear) and name and not _is_modulation_linear(name)
    )


def _tiny_expanded_targets(model: BooguImageTransformer2DModel) -> list[str]:
    """Mirrors test_boogu_image_definitions.py's derivation, scoped to the
    TINY model's block counts (1 double-stream + 1 single-stream + 1 each
    refiner)."""
    targets: list[str] = []
    for suf in _attn_ff_suffixes(model.double_stream_layers[0]):
        targets.append(f"double_stream_layers.0.{suf}")
    for suf in _attn_ff_suffixes(model.single_stream_layers[0]):
        targets.append(f"single_stream_layers.0.{suf}")
    for container_name in ("noise_refiner", "ref_image_refiner", "context_refiner"):
        for suf in _attn_ff_suffixes(model.single_stream_layers[0]):
            targets.append(f"{container_name}.0.{suf}")
    return targets


class TestLoraTargeting:
    def test_verbatim_pass_through_no_reexpansion(self):
        full_paths = [
            "double_stream_layers.0.img_instruct_attn.processor.img_to_q",
            "single_stream_layers.3.attn.to_k",
        ]
        drv = _driver(lora_targets=full_paths)
        assert drv.get_lora_targets() == full_paths

    def test_empty_curated_list_raises_loudly(self):
        drv = _driver(lora_targets=[])
        with pytest.raises(RuntimeError, match="curated"):
            drv.get_lora_targets()

    def test_none_curated_list_raises_loudly(self):
        drv = _driver(lora_targets=None)
        with pytest.raises(RuntimeError, match="curated"):
            drv.get_lora_targets()

    def test_no_attention_only_fallback_exists(self):
        """Regression pin: Task 2's dead-code attention-only fallback (a
        literal ``return ["attn.to_q", "attn.to_k", "attn.to_v",
        "attn.to_out.0"]``) must be gone entirely, not just unreachable.
        Checks the actual fallback LIST LITERAL, not incidental docstring
        prose that explains why there's no fallback."""
        import inspect

        source = inspect.getsource(BooguImageDriver.get_lora_targets)
        assert '["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"]' not in source

    def test_peft_wraps_tiny_model_with_expected_count_and_names(self):
        from peft import LoraConfig, get_peft_model

        model = _tiny_transformer()
        targets = _tiny_expanded_targets(model)
        drv = _driver(lora_targets=targets)

        peft_model = get_peft_model(
            model, LoraConfig(r=4, lora_alpha=4, target_modules=drv.get_lora_targets()),
        )

        lora_modules = {
            name: module
            for name, module in peft_model.named_modules()
            if hasattr(module, "lora_A") and getattr(module, "lora_A", None)
        }
        assert len(lora_modules) == len(targets), (
            f"expected {len(targets)} wrapped modules, got {len(lora_modules)}: "
            f"missing={set(f'base_model.model.{t}' for t in targets) - set(lora_modules)}"
        )

        names = set(lora_modules)
        assert any(n.endswith("img_instruct_attn.processor.img_to_q") for n in names)
        assert any(n.endswith("img_instruct_attn.processor.img_to_k") for n in names)
        assert any(n.endswith("img_instruct_attn.processor.instruct_to_v") for n in names)
        assert any(n.endswith("img_instruct_attn.processor.instruct_out") for n in names)
        assert any(n.endswith("double_stream_layers.0.img_instruct_attn.to_out.0") for n in names)

    def test_peft_preserves_gqa_asymmetric_widths(self):
        """to_k/to_v (kv_dim, narrower) must stay narrower than to_q/to_out
        (query_dim) after PEFT wrapping -- a symmetrized wrap would silently
        break inference-time LoRA merge for GQA blocks."""
        from peft import LoraConfig, get_peft_model

        model = _tiny_transformer()
        targets = _tiny_expanded_targets(model)
        drv = _driver(lora_targets=targets)
        peft_model = get_peft_model(
            model, LoraConfig(r=2, lora_alpha=2, target_modules=drv.get_lora_targets()),
        )

        def _base_out_features(dotted: str) -> int:
            mod = dict(peft_model.named_modules())[f"base_model.model.{dotted}"]
            return mod.base_layer.out_features

        # Single-stream block 0: to_q/to_out.0 vs to_k/to_v out widths.
        q_width = _base_out_features("single_stream_layers.0.attn.to_q")
        out_width = _base_out_features("single_stream_layers.0.attn.to_out.0")
        k_width = _base_out_features("single_stream_layers.0.attn.to_k")
        v_width = _base_out_features("single_stream_layers.0.attn.to_v")

        assert q_width == out_width == TINY_HIDDEN_SIZE  # 2 heads * 8 head_dim
        assert k_width == v_width == 8  # 1 kv_head * 8 head_dim (GQA narrower)
        assert k_width < q_width


# ── init_scheduler / assign_components / resolve_loading_dtype ────────────


class TestSchedulerAndComponentWiring:
    def test_init_scheduler_returns_loader_provided_instance_not_a_fresh_default(self):
        drv = _driver()
        sentinel_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            do_shift=True,
            dynamic_time_shift=False,
            time_shift_version="v1",
            seq_len=1234,  # distinctive, would never come from a fresh default
        )
        drv.assign_components({"scheduler": sentinel_scheduler})

        result = drv.init_scheduler()

        assert result is sentinel_scheduler
        assert drv.scheduler is sentinel_scheduler
        assert result.config.seq_len == 1234

    def test_init_scheduler_without_assign_components_raises(self):
        drv = _driver()
        with pytest.raises(RuntimeError, match="scheduler"):
            drv.init_scheduler()

    def test_assign_components_wires_all_five(self):
        drv = _driver()
        components = {
            "unet": MagicMock(name="unet"),
            "vae": MagicMock(name="vae"),
            "text_encoder": MagicMock(name="text_encoder"),
            "processor": MagicMock(name="processor"),
            "scheduler": MagicMock(name="scheduler"),
        }
        drv.assign_components(components)

        assert drv.model is components["unet"]
        assert drv.vae is components["vae"]
        assert drv.text_encoder is components["text_encoder"]
        assert drv.processor is components["processor"]
        assert drv.scheduler is components["scheduler"]
        assert drv.get_components() is components
        assert drv.get_primary_model() is components["unet"]
        assert drv.get_text_encoders() == {"text_encoder": components["text_encoder"]}

    def test_resolve_loading_dtype_is_bf16(self):
        assert _driver().resolve_loading_dtype() is torch.bfloat16

    def test_get_te_lora_targets_empty(self):
        assert _driver().get_te_lora_targets() == []
