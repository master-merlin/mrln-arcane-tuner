"""Tests for the prx_pixel family (pixel-space Photoroom PRXPixel, NO VAE).

TDD order (mirrors test_prx_family.py):
  Task 3: family registration + definition loading
  Task 4: loader manifest (NO VAE spec; Qwen3VLTextModel TE)
  Task 5: driver — x0-objective contract (scaled noise, clean-pixel target,
          normalized-t forward, prx_shared LoRA targets on the pixel
          config variant: in_channels=3 + bottleneck img_in +
          resolution_embeds=True)
  Task 6: trainer override trio (encode_text tuple, _update_primary_model
          driver sync, transformer property) + pixel passthrough wiring +
          TE disk-cache layout
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.engine.core.definitions import ModelDefinition


@pytest.fixture(autouse=True)
def _restore_model_registry():
    """Snapshot + restore ``ModelRegistry`` class state around every test.

    Registration tests mutate the registry's class-level discovery caches
    inline (resetting ``_discovered`` / ``_families`` / ``_definitions`` to
    force a re-scan). Left unrestored those mutations leak into later tests
    in the session (same pattern as test_prx_family.py).
    """
    from app.engine.models.registry import ModelRegistry

    saved = {
        "_families": dict(ModelRegistry._families),
        "_definitions": dict(ModelRegistry._definitions),
        "_paths": dict(ModelRegistry._paths),
        "_discovered": ModelRegistry._discovered,
        "_definitions_loaded": ModelRegistry._definitions_loaded,
    }
    try:
        yield
    finally:
        ModelRegistry._families = saved["_families"]
        ModelRegistry._definitions = saved["_definitions"]
        ModelRegistry._paths = saved["_paths"]
        ModelRegistry._discovered = saved["_discovered"]
        ModelRegistry._definitions_loaded = saved["_definitions_loaded"]


def _make_pixel_definition(**kwargs) -> MagicMock:
    """Build a mock prx_pixel ModelDefinition for loader/driver tests."""
    definition = MagicMock(spec=ModelDefinition)
    definition.family = "prx_pixel"
    definition.id = kwargs.get("id", "prx-pixel-test")
    definition.components = {}
    definition.architecture_params = kwargs.get("architecture_params", {})
    definition.lora_targetable_modules = kwargs.get("lora_targetable_modules", [])
    definition.defaults = kwargs.get("defaults", {})
    return definition


# ── Task 3: Family Registration ──────────────────────────────────────────────


def test_family_registered():
    """prx_pixel family must register with the pixel_transformer archetype."""
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()

    fam = ModelRegistry._families.get("prx_pixel")
    assert fam is not None, "prx_pixel family not registered"
    assert fam.archetype == "pixel_transformer", (
        f"expected archetype='pixel_transformer', got {fam.archetype!r}"
    )


def test_definition_loaded():
    """prx-pixel-t2i definition must load from its YAML file."""
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()

    fam_defs = {
        d.id: d
        for d in ModelRegistry._definitions.values()
        if d.family == "prx_pixel"
    }
    assert "prx-pixel-t2i" in fam_defs, (
        f"missing prx-pixel-t2i definition; found: {set(fam_defs)}"
    )

    base = fam_defs["prx-pixel-t2i"]
    # Canonical checkpoint repo (verified model_index.json)
    assert base.components["repo"].path == "huggingface:Photoroom/prxpixel-t2i", (
        f"wrong repo path: {base.components['repo'].path!r}"
    )
    # Standard T2I — no paired control inputs
    assert base.control_inputs == 0
    # Native 1024 default resolution (default_sample_size=1024)
    assert base.defaults.get("resolution") == 1024
    # Pipeline __call__ defaults
    assert base.defaults.get("guidance_scale") == 4.0
    assert base.defaults.get("num_inference_steps") == 28

    # Verified transformer config facts (checkpoint transformer/config.json,
    # fetched 2026-07-08 — the PIXEL variant differs from class defaults).
    arch = base.architecture_params
    assert arch.get("transformer.in_channels") == 3, "pixel space: RGB in"
    assert arch.get("transformer.patch_size") == 16
    assert arch.get("transformer.hidden_size") == 3584
    assert arch.get("transformer.depth") == 24
    assert arch.get("transformer.num_heads") == 28
    assert arch.get("transformer.context_in_dim") == 2048
    assert arch.get("transformer.bottleneck_size") == 768
    assert arch.get("transformer.resolution_embeds") is True
    assert arch.get("transformer.time_factor") == 1000.0
    # Pipeline registered config (model_index.json)
    assert arch.get("pipeline.noise_scale") == 2.0
    assert arch.get("pipeline.prompt_max_tokens") == 256
    # NO VAE — a vae.* section would resurrect latent-space assumptions.
    assert not any(k.startswith("vae.") for k in arch), (
        "prx_pixel is pixel-space — no vae.* architecture params allowed"
    )
    # Scheduler facts (checkpoint scheduler_config.json): static shift 3.0.
    assert arch.get("scheduler.num_train_timesteps") == 1000
    assert arch.get("scheduler.shift") == 3.0
    # TE facts (checkpoint text_encoder/config.json — Qwen3VLTextModel)
    assert arch.get("te.type") == "qwen3_vl_text"
    assert arch.get("te.hidden_size") == 2048
    assert arch.get("te.num_hidden_layers") == 28
    # prompt_max_tokens drives tokenization, NOT tokenizer.model_max_length
    # (the Qwen tokenizer's own model_max_length is far larger than 256).
    assert arch.get("te.max_length") == 256


# ── Task 4: Loader Manifest ──────────────────────────────────────────────────


def test_manifest_components_no_vae():
    """PRXPixelLoader manifest declares tokenizer + TE + transformer and
    NOTHING else — a vae spec would make GenericComponentLoader try to
    download/load a VAE that does not exist in the checkpoint."""
    import torch

    from app.engine.models.families.prx_pixel.loader import PRXPixelLoader

    loader = PRXPixelLoader(torch.device("cpu"))
    definition = _make_pixel_definition()
    specs = loader.get_component_manifest(definition)

    keys = {s.key for s in specs}
    assert keys == {"tokenizer", "text_encoder", "unet"}, (
        f"manifest must be exactly tokenizer/text_encoder/unet, got {keys}"
    )

    spec_map = {s.key: s for s in specs}

    # Tokenizer: AutoTokenizer (Qwen2TokenizerFast resolves via tokenizer.json
    # — mirrors what PRXPixelPipeline.from_pretrained materializes).
    assert "AutoTokenizer" in spec_map["tokenizer"].hf_class, (
        f"tokenizer hf_class wrong: {spec_map['tokenizer'].hf_class}"
    )
    assert spec_map["tokenizer"].is_torch_model is False

    # Text encoder: the checkpoint's model_index.json declares
    # ["transformers", "Qwen3VLTextModel"] — top-level transformers export.
    assert spec_map["text_encoder"].hf_class == "transformers.Qwen3VLTextModel", (
        f"text_encoder hf_class wrong: {spec_map['text_encoder'].hf_class}"
    )
    assert spec_map["text_encoder"].subfolder == "text_encoder"

    # Transformer mapped to "unet" (repo convention), diffusers-native class
    assert "PRXTransformer2DModel" in spec_map["unet"].hf_class, (
        f"unet hf_class wrong: {spec_map['unet'].hf_class}"
    )
    assert spec_map["unet"].subfolder == "transformer"


def test_manifest_classes_are_importable():
    """Every hf_class in the manifest resolves through the generic loader's
    importlib seam (Qwen3VLTextModel must be a real top-level export)."""
    import torch

    from app.engine.core.pipeline.loader_base import GenericComponentLoader
    from app.engine.models.families.prx_pixel.loader import PRXPixelLoader

    loader = PRXPixelLoader(torch.device("cpu"))
    for spec in loader.get_component_manifest(_make_pixel_definition()):
        cls = GenericComponentLoader._import_class(spec.hf_class)
        assert cls is not None, f"{spec.hf_class} not importable"


def test_loader_dtype_policy_is_generic():
    """Dtype policy inherits the generic path (bf16 via driver), no per-spec
    overrides — identical policy to the latent prx sibling."""
    import torch

    from app.engine.core.pipeline.loader_base import GenericComponentLoader
    from app.engine.models.families.prx_pixel.loader import PRXPixelLoader

    assert PRXPixelLoader._resolve_dtype is GenericComponentLoader._resolve_dtype, (
        "PRXPixelLoader must inherit the generic dtype policy"
    )

    loader = PRXPixelLoader(torch.device("cpu"))
    for spec in loader.get_component_manifest(_make_pixel_definition()):
        assert spec.dtype_override is None, (
            f"{spec.key} must not force a dtype override"
        )


# ── Task 5: Driver — x0 objective + scaled noise + prx_shared targets ────────

# Tiny PIXEL-variant transformer config for CPU tests: the checkpoint's
# distinguishing flags ON (in_channels=3, bottleneck img_in,
# resolution_embeds). sum(axes_dim) must equal head_dim (32 / 2 = 16).
_TINY_CFG = dict(
    in_channels=3,
    patch_size=2,
    context_in_dim=8,
    hidden_size=32,
    num_heads=2,
    depth=1,
    axes_dim=[8, 8],
    bottleneck_size=16,
    resolution_embeds=True,
)

# Arch params matching the real definition's driver-relevant keys.
_ARCH = {
    "te.max_length": 256,
    "scheduler.num_train_timesteps": 1000,
    "pipeline.noise_scale": 2.0,
    "pipeline.velocity_t_floor": 0.05,
}


def _build_tiny_model():
    import torch  # noqa: PLC0415

    from diffusers.models.transformers.transformer_prx import (
        PRXTransformer2DModel,
    )

    torch.manual_seed(0)
    return PRXTransformer2DModel(**_TINY_CFG).eval()


def _make_driver(model=None, arch=None):
    import torch  # noqa: PLC0415

    from app.engine.models.families.prx_pixel.driver import PRXPixelDriver

    definition = _make_pixel_definition(
        architecture_params=dict(_ARCH) if arch is None else arch,
    )
    drv = PRXPixelDriver(definition, torch.device("cpu"))
    drv.assign_components(
        {"unet": model, "text_encoder": None, "tokenizer": None},
    )
    return drv


def test_tiny_pixel_variant_builds_bottleneck_and_resolution_embedder():
    """The tiny test model must actually exercise the pixel-variant paths:
    Sequential (two-layer bottleneck) img_in + a resolution embedder."""
    import torch  # noqa: PLC0415

    model = _build_tiny_model()
    assert isinstance(model.img_in, torch.nn.Sequential), (
        "bottleneck_size must produce a two-layer img_in"
    )
    assert model.resolution_embedder is not None


def test_shared_targets_match_pixel_variant_no_top_level_sweep():
    """prx_shared's fused-projection patterns still match exactly one Linear
    per block on the PIXEL config variant — and the bottleneck img_in.0 /
    img_in.1 Linears are NOT swept in (exclusion-free contract holds)."""
    import torch.nn as nn  # noqa: PLC0415

    from app.engine.models.families.prx_shared import matching_linear_modules

    model = _build_tiny_model()

    # The bottleneck img_in exposes top-level Linears img_in.0 / img_in.1 —
    # they must exist (variant sanity) and match NO target pattern.
    linear_names = [
        name for name, m in model.named_modules() if isinstance(m, nn.Linear)
    ]
    assert "img_in.0" in linear_names and "img_in.1" in linear_names

    matches = matching_linear_modules(model)
    for pattern, names in matches.items():
        assert len(names) == _TINY_CFG["depth"], (
            f"{pattern!r} must match exactly one Linear per block, got {names}"
        )
        for name in names:
            assert name.startswith("blocks."), (
                f"{pattern!r} matched non-block module {name!r} — "
                "top-level collision (would need exclude_modules)"
            )


def test_driver_lora_targets_use_shared_list_and_need_no_excludes():
    """Driver defaults delegate to prx_shared; no exclude_modules needed."""
    from app.engine.models.families.prx_shared import get_prx_lora_targets

    drv = _make_driver(_build_tiny_model())
    assert drv.get_lora_targets() == get_prx_lora_targets()
    assert drv.get_lora_exclude_modules() is None


def test_driver_forward_normalizes_timesteps_exactly_once():
    """forward_pass receives raw [0,1000] and hands t/1000 to the transformer.

    PRX convention (shared adapter): the ÷1000 happens BEFORE the forward,
    exactly once — the model's time_factor=1000 re-scales internally. Any
    extra scaling silently produces pure-noise LoRAs (flow-match
    timestep-scale gotcha). Identical seam to the latent sibling; pinned
    here on the pixel variant.
    """
    import torch  # noqa: PLC0415

    model = _build_tiny_model()
    drv = _make_driver(model)

    captured: dict = {}
    original_forward = model.forward

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return original_forward(*args, **kwargs)

    model.forward = _spy

    B, C, H, W = 1, 3, 8, 8
    noisy = torch.randn(B, C, H, W)
    emb = torch.randn(B, 5, 8)
    mask = torch.ones(B, 5, dtype=torch.bool)

    with torch.no_grad():
        pred = drv.forward_pass(
            noisy_input=noisy,
            timesteps=torch.tensor([500.0]),
            text_embeddings=(emb, mask),
            batch={},
        )

    ts = captured["timestep"]
    assert torch.allclose(ts.float(), torch.tensor([0.5])), (
        f"transformer must receive t/1000 = 0.5, got {ts}"
    )
    # The bool text mask must reach the transformer (PRX consumes it).
    assert captured["attention_mask"] is mask
    # Unpacked PIXELS go straight in — patchify happens INSIDE the model.
    assert captured["hidden_states"] is noisy
    assert pred.shape == (B, C, H, W), f"unexpected shape: {pred.shape}"
    assert pred.isfinite().all(), "output contains NaN or inf"
    assert pred.float().std() > 0, "output is degenerate (zero std)"


def test_driver_compute_target_is_clean_pixels_x0():
    """x0 OBJECTIVE: the target is the CLEAN image, not noise - latents.

    PRXPixel is trained with x-prediction (pipeline docstring + the per-step
    x0→velocity conversion in PRXPixelPipeline.__call__). MSE(x0_pred, x0)
    is therefore MSE(pred, compute_target(...)) with target == latents.
    """
    import torch  # noqa: PLC0415

    drv = _make_driver(None)
    latents = torch.randn(2, 3, 8, 8)
    noise = torch.randn(2, 3, 8, 8)
    target = drv.compute_target(latents, noise, torch.tensor([500.0, 250.0]))
    assert torch.equal(target, latents), (
        "prx_pixel target must be the clean pixels (x0), got something else"
    )
    assert not torch.equal(target, noise - latents), (
        "velocity target leaked in — this is an x0-prediction model"
    )


def test_driver_prepare_noise_scales_by_noise_scale():
    """TRAINING NOISE ×2.0: prepare_noise applies the pipeline's noise_scale
    so the loop's linear interpolation becomes x_t = (1-t)·x0 + t·(2ε) —
    matching sampling, which starts from randn×2.0."""
    import torch  # noqa: PLC0415

    drv = _make_driver(None)
    noise = torch.randn(2, 3, 8, 8)
    assert torch.allclose(drv.prepare_noise(noise), noise * 2.0)

    # The scale is definition-driven (pipeline.noise_scale), not hardcoded.
    drv15 = _make_driver(None, arch={"pipeline.noise_scale": 1.5})
    assert torch.allclose(drv15.prepare_noise(noise), noise * 1.5)


def test_training_interpolation_matches_flow_recipe():
    """End-to-end training-noise contract through the REAL loop seams:
    driver.prepare_noise → NoiseInterpolation('linear').add_noise yields
    x_t = (1-t)·x0 + t·(noise·2.0) with t on the [0,1000] scale."""
    import torch  # noqa: PLC0415

    from app.engine.strategies.noise_interpolation import NoiseInterpolation

    drv = _make_driver(None)
    torch.manual_seed(1)
    x0 = torch.randn(2, 3, 8, 8)
    eps = torch.randn(2, 3, 8, 8)
    t = torch.tensor([250.0, 750.0])

    prepared = drv.prepare_noise(eps)
    x_t = NoiseInterpolation("linear").add_noise(x0, prepared, t)

    t01 = (t / 1000.0).view(-1, 1, 1, 1)
    expected = (1.0 - t01) * x0 + t01 * (eps * 2.0)
    assert torch.allclose(x_t, expected, atol=1e-6)


def test_euler_step_with_perfect_x0_oracle_moves_toward_x0():
    """Objective contract (a): converting a PERFECT x0 prediction to
    velocity via the pipeline formula and taking one FlowMatchEuler step
    must move the trajectory strictly toward x0 (and reach x0 exactly at
    the final step where sigma_next = 0)."""
    import torch  # noqa: PLC0415

    from diffusers import FlowMatchEulerDiscreteScheduler

    from app.engine.models.families.prx_pixel.driver import x0_to_velocity

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=3.0,
    )
    scheduler.set_timesteps(4)
    scheduler.set_begin_index(0)

    torch.manual_seed(2)
    x0 = torch.randn(1, 3, 8, 8)
    latents = torch.randn(1, 3, 8, 8) * 2.0  # randn × noise_scale start
    start_dist = (latents - x0).norm()

    for t in scheduler.timesteps:
        before = (latents - x0).norm()
        velocity = x0_to_velocity(latents, x0, t)
        latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
        after = (latents - x0).norm()
        assert after < before, (
            f"step at t={float(t)} moved AWAY from x0 ({before} -> {after})"
        )

    # NOTE: exact terminal reconstruction is intentionally NOT asserted for
    # this schedule — the shift-3 sigmas end below the 0.05 t-floor (last
    # sigma ≈ 0.009), so the clamped division under-corrects the very last
    # step. The honest contract is strict monotone contraction (above) plus
    # near-convergence:
    assert (latents - x0).norm() < 0.02 * start_dist

    # Exactness DOES hold whenever sigma ≥ t_floor at the final step: with a
    # 1-step schedule (sigma 1.0 → 0.0) the perfect oracle lands on x0.
    scheduler.set_timesteps(1)
    scheduler.set_begin_index(0)
    latents1 = torch.randn(1, 3, 8, 8) * 2.0
    t = scheduler.timesteps[0]
    velocity = x0_to_velocity(latents1, x0, t)
    out = scheduler.step(velocity, t, latents1, return_dict=False)[0]
    assert torch.allclose(out, x0, atol=1e-4), (
        "perfect-x0 oracle must reconstruct x0 in one full-sigma step"
    )


def test_x0_to_velocity_applies_t_floor():
    """The pipeline clamps t/1000 at 0.05 before dividing — without the
    floor the final low-t steps explode the velocity."""
    import torch  # noqa: PLC0415

    from app.engine.models.families.prx_pixel.driver import x0_to_velocity

    latents = torch.full((1, 3, 2, 2), 2.0)
    x0 = torch.zeros(1, 3, 2, 2)

    # t = 10 → t/1000 = 0.01 → clamped to 0.05 → v = 2.0 / 0.05 = 40.
    v = x0_to_velocity(latents, x0, torch.tensor(10.0))
    assert torch.allclose(v, torch.full_like(v, 40.0)), (
        f"t-floor 0.05 not applied: got {v.flatten()[0]}"
    )

    # Above the floor the raw t is used: t = 500 → v = 2.0 / 0.5 = 4.
    v_mid = x0_to_velocity(latents, x0, torch.tensor(500.0))
    assert torch.allclose(v_mid, torch.full_like(v_mid, 4.0))


def test_no_double_scaling_forward_receives_normalized_t_only():
    """Objective contract (c): the transformer sees t/1000 exactly once —
    running forward at two different raw t values yields different outputs
    (a double ÷1000 would collapse the embedding range to near-zero and
    make them indistinguishable)."""
    import torch  # noqa: PLC0415

    model = _build_tiny_model()
    drv = _make_driver(model)

    torch.manual_seed(3)
    noisy = torch.randn(1, 3, 8, 8)
    emb = torch.randn(1, 5, 8)
    mask = torch.ones(1, 5, dtype=torch.bool)

    captured: list[float] = []
    original_forward = model.forward

    def _spy(*args, **kwargs):
        captured.append(float(kwargs["timestep"].flatten()[0]))
        return original_forward(*args, **kwargs)

    model.forward = _spy

    with torch.no_grad():
        drv.forward_pass(noisy, torch.tensor([1000.0]), (emb, mask), {})
        drv.forward_pass(noisy, torch.tensor([50.0]), (emb, mask), {})

    assert captured == pytest.approx([1.0, 0.05]), (
        f"expected normalized [1.0, 0.05], got {captured} — "
        "÷1000 must happen exactly once"
    )


def _stub_tokenizer(full_len: int = 256, record: list | None = None):
    """Stub tokenizer satisfying the PRXPixel tokenize contract.

    model_max_length is deliberately HUGE (the real Qwen2TokenizerFast's is
    far larger than 256) — the driver must tokenize to prompt_max_tokens
    (te.max_length == 256), never to tokenizer.model_max_length.
    """
    import torch  # noqa: PLC0415

    tok = MagicMock()
    tok.model_max_length = 131072

    def _fake_tokenize(texts, **kwargs):
        if record is not None:
            record.append({"texts": list(texts), **kwargs})
        n = len(texts)
        mask = torch.ones(n, full_len, dtype=torch.long)
        mask[:, -10:] = 0  # last 10 positions are padding
        return {
            "input_ids": torch.zeros(n, full_len, dtype=torch.long),
            "attention_mask": mask,
        }

    tok.side_effect = _fake_tokenize
    return tok


def _stub_te(hidden: "object", record: list | None = None):
    """Stub Qwen3VLTextModel returning dict-style last_hidden_state."""
    import torch  # noqa: PLC0415

    te = MagicMock()

    def _fake_te(**kwargs):
        if record is not None:
            record.append(kwargs)
        return {"last_hidden_state": hidden}

    te.side_effect = _fake_te
    te.parameters = lambda: iter([torch.zeros(1)])
    return te


def test_driver_encode_text_replicates_pixel_pipeline():
    """encode_text mirrors PRXPixelPipeline._tokenize_prompts +
    _encode_prompt_standard:

    1. LIGHT cleaning only (_basic_clean: ftfy + HTML unescape — NO
       DeepFloyd lowercasing, unlike the latent sibling);
    2. tokenize padding='max_length', max_length=prompt_max_tokens (256,
       NOT tokenizer.model_max_length), truncation=True;
    3. TE forward WITH output_hidden_states=True → ['last_hidden_state'];
    4. boolean attention mask returned alongside — NO zero-masking, NO
       slicing.
    """
    import torch  # noqa: PLC0415

    from app.engine.models.families.prx_pixel.driver import PRXPixelDriver

    B, D, seq = 2, 16, 256
    tokenize_calls: list[dict] = []
    te_calls: list[dict] = []

    definition = _make_pixel_definition(architecture_params=dict(_ARCH))
    drv = PRXPixelDriver(definition, torch.device("cpu"))

    torch.manual_seed(3)
    hidden = torch.randn(B, seq, D)
    drv.tokenizer = _stub_tokenizer(seq, tokenize_calls)
    drv.text_encoder = _stub_te(hidden, te_calls)

    out = drv.encode_text(["A Photo of a CAT!", "Ein  HUND"], torch.float32)

    # 1. basic_clean only: case and inner whitespace PRESERVED (the
    #    DeepFloyd clean_text path would lowercase + collapse whitespace).
    assert tokenize_calls[0]["texts"] == ["A Photo of a CAT!", "Ein  HUND"], (
        "pixel pipeline uses _basic_clean — text must not be lowercased"
    )

    # 2. Tokenizer semantics: 256 budget, NOT tokenizer.model_max_length.
    tk = tokenize_calls[0]
    assert tk["padding"] == "max_length"
    assert tk["truncation"] is True
    assert tk["max_length"] == 256
    assert tk["return_tensors"] == "pt"

    # 3. TE called with output_hidden_states=True and the BOOL mask
    assert te_calls[0]["output_hidden_states"] is True
    assert te_calls[0]["attention_mask"].dtype == torch.bool

    # 4. Un-sliced, un-masked last_hidden_state + bool mask
    assert out.embeddings.shape == (B, seq, D)
    assert torch.allclose(out.embeddings, hidden)
    assert out.attention_mask is not None
    assert out.attention_mask.dtype == torch.bool
    assert out.attention_mask.shape == (B, seq)
    assert out.attention_mask[:, -10:].sum() == 0


def test_basic_clean_matches_pixel_pipeline_module_clean():
    """The shared helper's clean_text=False path (TextPreprocessor
    .basic_clean) must be behavior-identical to pipeline_prx_pixel's
    module-level _basic_clean — that equivalence is what lets prx_pixel
    reuse encode_prx_text without a fork."""
    from diffusers.pipelines.prx.pipeline_prx import TextPreprocessor
    from diffusers.pipelines.prx.pipeline_prx_pixel import _basic_clean

    pre = TextPreprocessor()
    samples = [
        "A Photo of a CAT!",
        "  padded  &amp; entity ",
        "MiXeD CaSe &lt;tag&gt;",
    ]
    for s in samples:
        assert pre.basic_clean(s) == _basic_clean(s)


def test_driver_basic_contracts():
    """Scheduler None (flow match), bf16 loading, no TE LoRA, no VAE slot."""
    import torch  # noqa: PLC0415

    drv = _make_driver(None)
    assert drv.init_scheduler() is None
    assert drv.resolve_loading_dtype() == torch.bfloat16
    assert drv.get_te_lora_targets() == []
    # Pixel space: the driver must not carry a VAE reference.
    assert getattr(drv, "vae", None) is None


def test_driver_layer_manifest_single_block_stack():
    """Layer manifest exposes the single 24-block (here 1-block) stack."""
    model = _build_tiny_model()
    drv = _make_driver(model)
    manifest = drv.get_layer_manifest()

    assert len(manifest.transformer_blocks) == _TINY_CFG["depth"]
    assert manifest.transformer_blocks[0].name == "blocks.0"

    topo = drv.get_block_topology()
    assert len(topo) == 1
    assert topo[0]["attr_path"] == "blocks"
    assert topo[0]["count"] == _TINY_CFG["depth"]


# ── Task 6: Trainer override trio + pixel passthrough + TE cache ─────────────


def _build_real_trainer_shell():
    """Minimal PRXPixelTrainer shell with a REAL driver + tiny transformer.

    Binds the real trainer methods (encode seam) without calling setup().
    """
    import torch  # noqa: PLC0415

    from app.engine.models.families.prx_pixel.driver import PRXPixelDriver
    from app.engine.models.families.prx_pixel.trainer import PRXPixelTrainer

    definition = _make_pixel_definition(architecture_params=dict(_ARCH))

    trainer = MagicMock(spec=PRXPixelTrainer)
    trainer.device = torch.device("cpu")
    trainer.definition = definition
    trainer.config = {"cache_text_embeddings": False}
    trainer.text_cache = {}
    trainer.logger = MagicMock()

    drv = PRXPixelDriver(definition, torch.device("cpu"))
    tiny_model = _build_tiny_model()
    torch.manual_seed(5)
    hidden = torch.randn(1, 256, 8)  # D == tiny context_in_dim
    drv.assign_components({
        "unet": tiny_model,
        "text_encoder": _stub_te(hidden),
        "tokenizer": _stub_tokenizer(256),
    })
    trainer.driver = drv

    trainer.encode_text = lambda captions, dtype, batch=None: (
        PRXPixelTrainer.encode_text(trainer, captions, dtype, batch)
    )
    trainer._encode_text_direct = lambda captions, dtype: (
        PRXPixelTrainer._encode_text_direct(trainer, captions, dtype)
    )
    trainer._get_cached_text_embeddings = lambda captions, dtype: (
        PRXPixelTrainer._get_cached_text_embeddings(trainer, captions, dtype)
    )
    return trainer


def test_trainer_encode_to_forward_real_seam():
    """C1/C2: trainer.encode_text returns a (emb, mask) TUPLE consumable by
    driver.forward_pass — the whole encode→forward round trip produces a
    finite [B, 3, H, W] x0 prediction."""
    import torch  # noqa: PLC0415

    trainer = _build_real_trainer_shell()

    text_emb = trainer.encode_text(["a prx pixel test caption"], torch.float32)
    assert isinstance(text_emb, tuple) and len(text_emb) == 2, (
        f"encode_text must return a 2-tuple, got {type(text_emb)}"
    )
    emb, mask = text_emb
    assert emb.ndim == 3, f"embeddings must be 3-D [B,L,D], got {emb.ndim}-D"
    assert emb.shape[1] == 256, f"seq len must be 256, got {emb.shape[1]}"
    assert mask.ndim == 2 and mask.shape[1] == 256
    assert mask.dtype == torch.bool

    B, C, H, W = 1, 3, 8, 8
    with torch.no_grad():
        pred = trainer.driver.forward_pass(
            noisy_input=torch.randn(B, C, H, W),
            timesteps=torch.tensor([500.0]),
            text_embeddings=text_emb,
            batch={},
        )
    assert pred.shape == (B, C, H, W), f"unexpected pred shape: {pred.shape}"
    assert pred.isfinite().all(), "forward_pass output contains NaN/inf"


def test_trainer_cached_encode_returns_batched_tuple():
    """Cached path stacks per-caption entries back to ([B,L,D], [B,L])."""
    import torch  # noqa: PLC0415

    trainer = _build_real_trainer_shell()
    trainer.config = {"cache_text_embeddings": True}
    trainer.text_encoder = trainer.driver.text_encoder

    out = trainer.encode_text(["cap one", "cap two"], torch.float32)
    assert isinstance(out, tuple) and len(out) == 2
    emb, mask = out
    assert emb.shape[0] == 2 and emb.ndim == 3
    assert mask.shape[0] == 2 and mask.ndim == 2
    assert set(trainer.text_cache) == {"cap one", "cap two"}
    # Cache entries are (emb [L,D], mask [L]) CPU tuples
    cached_emb, cached_mask = trainer.text_cache["cap one"]
    assert cached_emb.ndim == 2 and cached_mask.ndim == 1


def test_trainer_peft_model_sync():
    """C3/C4: _update_primary_model syncs driver.model, components, and the
    read-only ``transformer`` property resolves to the wrapped model."""
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    from app.engine.models.families.prx_pixel.driver import PRXPixelDriver
    from app.engine.models.families.prx_pixel.trainer import PRXPixelTrainer

    definition = _make_pixel_definition()
    trainer = MagicMock(spec=PRXPixelTrainer)
    trainer.device = torch.device("cpu")
    trainer.definition = definition

    drv = PRXPixelDriver(definition, torch.device("cpu"))
    tiny_model = _build_tiny_model()
    drv.assign_components({
        "unet": tiny_model,
        "text_encoder": None,
        "tokenizer": None,
    })
    trainer.driver = drv
    trainer.components = {"unet": tiny_model}
    trainer.model = tiny_model

    class _FakePEFT(nn.Module):
        pass

    peft_wrapped = _FakePEFT()
    PRXPixelTrainer._update_primary_model(trainer, peft_wrapped)

    assert trainer.driver.model is peft_wrapped, (
        "C3: driver.model was NOT updated after _update_primary_model"
    )
    assert trainer.components.get("unet") is peft_wrapped
    assert trainer.model is peft_wrapped

    transformer_val = PRXPixelTrainer.transformer.fget(trainer)
    assert transformer_val is peft_wrapped, (
        "C4: trainer.transformer property must resolve to the wrapped model"
    )


def test_trainer_configure_managers_installs_pixel_passthrough():
    """Pixel-space wiring: after _configure_managers the latent manager must
    be the shared PixelPassthroughLatentManager (the default LatentManager
    would raise on encode_and_cache_batch with vae=None)."""
    from unittest.mock import patch  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from app.engine.components.pixel_latents import (
        PixelPassthroughLatentManager,
    )
    from app.engine.models.families.prx_pixel.trainer import PRXPixelTrainer

    trainer = MagicMock(spec=PRXPixelTrainer)
    trainer.device = torch.device("cpu")
    trainer.definition = _make_pixel_definition()
    trainer.config = {"output_dir": "outputs", "lora_name": "t"}
    trainer.components = {}
    trainer.logger = MagicMock()
    trainer.driver = MagicMock()
    trainer.latent_manager = None

    with patch(
        "app.engine.core.pipeline.pipeline_optimization.CheckpointManager",
        return_value=MagicMock(),
    ):
        PRXPixelTrainer._configure_managers(trainer, max_train_steps=10)

    assert isinstance(trainer.latent_manager, PixelPassthroughLatentManager), (
        "_configure_managers must install the shared pixel passthrough"
    )


def test_trainer_latent_cache_hooks_are_noops():
    """No latent cache in pixel space: _validate_latent_cache just zeroes the
    missing counter and the async _pre_cache_latents does nothing."""
    import asyncio  # noqa: PLC0415

    from app.engine.models.families.prx_pixel.trainer import PRXPixelTrainer

    trainer = MagicMock(spec=PRXPixelTrainer)
    PRXPixelTrainer._validate_latent_cache(trainer)
    assert trainer._latent_cache_missing == 0

    asyncio.run(PRXPixelTrainer._pre_cache_latents(trainer))


def test_trainer_te_disk_cache_layout(tmp_path):
    """TE disk cache lands in embeddings/{te_quant}/te1|te2 (emb + mask)."""
    import torch  # noqa: PLC0415

    from app.engine.components.text_embeddings import TextEmbeddingCache
    from app.engine.models.families.prx_pixel.trainer import PRXPixelTrainer

    trainer = MagicMock(spec=PRXPixelTrainer)
    trainer.config = {
        "cache_text_embeddings": True,
        "te_quantization": "none",
        "sample_prompts": [],
    }
    trainer.device = torch.device("cpu")
    trainer.text_cache = {}
    trainer.logger = MagicMock()
    trainer._log_writer = None
    trainer.text_encoder = MagicMock()
    trainer._build_caption_hints.return_value = {"a prx pixel caption": "hint0"}
    trainer._resolve_te_cache_dirs.return_value = [str(tmp_path)]
    trainer._resolve_loading_dtype.return_value = torch.float32
    trainer._sample_prompt_texts.side_effect = lambda: (
        PRXPixelTrainer._sample_prompt_texts(trainer)
    )

    def _fake_encode(captions, dtype):
        b = len(captions)
        return (
            torch.zeros(b, 256, 16),
            torch.ones(b, 256, dtype=torch.bool),
        )

    trainer._encode_text_direct = _fake_encode

    PRXPixelTrainer._pre_cache_text_embeddings(trainer)

    te1 = tmp_path / "embeddings" / "none" / "te1"
    te2 = tmp_path / "embeddings" / "none" / "te2"
    assert te1.is_dir(), "te1 (embeddings) cache dir missing"
    assert te2.is_dir(), "te2 (attention mask) cache dir missing"

    emb = TextEmbeddingCache.load("a prx pixel caption", str(te1), "hint0")
    mask = TextEmbeddingCache.load("a prx pixel caption", str(te2), "hint0")
    assert emb is not None and emb.shape == (256, 16)
    assert mask is not None and mask.shape == (256,)
    # In-memory cache holds the (emb, mask) tuple
    assert "a prx pixel caption" in trainer.text_cache
    cached_emb, cached_mask = trainer.text_cache["a prx pixel caption"]
    assert cached_emb.shape == (256, 16)
    assert cached_mask.shape == (256,)


def test_trainer_warms_sample_and_negative_prompts():
    """Pre-cache also warms expanded sample + negative prompts so the TE can
    stay offloaded during sampling (krea2 VRAM-spike lesson)."""
    import torch  # noqa: PLC0415

    from app.engine.models.families.prx_pixel.trainer import PRXPixelTrainer

    trainer = MagicMock(spec=PRXPixelTrainer)
    trainer.config = {
        "cache_text_embeddings": True,
        "te_quantization": "none",
        "sample_prompts": [{"prompt": "a red car"}],
        "sample_negative_prompt": "blurry",
    }
    trainer.device = torch.device("cpu")
    trainer.text_cache = {}
    trainer.logger = MagicMock()
    trainer._log_writer = None
    trainer.text_encoder = MagicMock()
    trainer._build_caption_hints.return_value = {"a dataset caption": "hint0"}
    trainer._resolve_te_cache_dirs.return_value = []
    trainer._resolve_loading_dtype.return_value = torch.float32
    trainer._sample_prompt_texts.side_effect = lambda: (
        PRXPixelTrainer._sample_prompt_texts(trainer)
    )

    def _fake_encode(captions, dtype):
        b = len(captions)
        return (
            torch.zeros(b, 256, 16),
            torch.ones(b, 256, dtype=torch.bool),
        )

    trainer._encode_text_direct = _fake_encode

    PRXPixelTrainer._pre_cache_text_embeddings(trainer)

    assert "a red car" in trainer.text_cache
    assert "blurry" in trainer.text_cache
