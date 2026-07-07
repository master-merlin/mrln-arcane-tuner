"""Tests for the ovis_image family (diffusers-0.39-native Ovis-Image).

TDD order (mirrors test_krea2_family.py):
  Task 1: family registration + definition loading
  Task 2: loader manifest (component specs + dtype policy)
  Task 3: driver (LoRA targets, forward_pass timestep scale, compute_target,
          encode_text replicating OvisImagePipeline.encode_prompt)
  Task 4: trainer override trio (encode_text tuple, _update_primary_model
          driver sync, transformer property) + TE disk-cache layout
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
    in the session (same pattern as test_krea2_family.py).
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


def _make_ovis_definition(**kwargs) -> MagicMock:
    """Build a mock Ovis-Image ModelDefinition for loader/driver tests."""
    definition = MagicMock(spec=ModelDefinition)
    definition.family = "ovis_image"
    definition.id = kwargs.get("id", "ovis-image-test")
    definition.components = {}
    definition.architecture_params = kwargs.get("architecture_params", {})
    definition.lora_targetable_modules = kwargs.get("lora_targetable_modules", [])
    return definition


# ── Task 1: Family Registration ──────────────────────────────────────────────


def test_family_registered():
    """ovis_image family must appear in ModelRegistry with the correct archetype."""
    from app.engine.models.registry import ModelRegistry

    # Reset discovery state so this test is hermetic
    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()

    fam = ModelRegistry._families.get("ovis_image")
    assert fam is not None, "ovis_image family not registered"
    assert fam.archetype == "latent_diffusion", (
        f"expected archetype='latent_diffusion', got {fam.archetype!r}"
    )


def test_definition_loaded():
    """ovis-image-base definition must load from its YAML file."""
    from app.engine.models.registry import ModelRegistry

    # Full reset so definitions are re-scanned
    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()

    fam_defs = {
        d.id: d
        for d in ModelRegistry._definitions.values()
        if d.family == "ovis_image"
    }
    assert "ovis-image-base" in fam_defs, (
        f"missing ovis-image-base definition; found: {set(fam_defs)}"
    )

    base = fam_defs["ovis-image-base"]
    # Canonical checkpoint repo (verified fact — see plan)
    assert base.components["repo"].path == "huggingface:AIDC-AI/Ovis-Image-7B", (
        f"wrong repo path: {base.components['repo'].path!r}"
    )
    # Standard T2I — no paired control inputs
    assert base.control_inputs == 0

    # Verified transformer config facts (from the checkpoint's config.json,
    # identical to the diffusers 0.39 OvisImageTransformer2DModel defaults).
    arch = base.architecture_params
    assert arch.get("transformer.num_layers") == 6
    assert arch.get("transformer.num_single_layers") == 27
    assert arch.get("transformer.num_attention_heads") == 24
    assert arch.get("transformer.attention_head_dim") == 128
    assert arch.get("transformer.joint_attention_dim") == 2048
    assert arch.get("transformer.in_channels") == 64
    assert arch.get("transformer.patch_size") == 1
    # Latent space is 16-channel (packed 2x2 -> 64 transformer channels)
    assert arch.get("vae.latent_channels") == 16
    # Scheduler facts from the checkpoint's scheduler_config.json
    assert arch.get("scheduler.use_dynamic_shifting") is True
    assert arch.get("scheduler.base_shift") == 0.5
    assert arch.get("scheduler.max_shift") == 1.15
    assert arch.get("scheduler.base_image_seq_len") == 256
    assert arch.get("scheduler.max_image_seq_len") == 4096
    # TE prompt-template facts from OvisImagePipeline
    assert arch.get("te.max_sequence_length") == 256
    assert arch.get("te.user_prompt_begin_id") == 28


# ── Task 2: Loader Manifest ──────────────────────────────────────────────────


def test_manifest_components():
    """OvisImageLoader manifest declares all four diffusers-native components."""
    import torch

    from app.engine.models.families.ovis_image.loader import OvisImageLoader

    loader = OvisImageLoader(torch.device("cpu"))
    definition = _make_ovis_definition()
    specs = loader.get_component_manifest(definition)

    keys = {s.key for s in specs}
    assert {"tokenizer", "text_encoder", "vae", "unet"} <= keys, (
        f"missing required manifest keys; got {keys}"
    )

    spec_map = {s.key: s for s in specs}

    # Tokenizer: AutoTokenizer (fast Qwen2TokenizerFast resolves via tokenizer.json)
    assert "AutoTokenizer" in spec_map["tokenizer"].hf_class, (
        f"tokenizer hf_class wrong: {spec_map['tokenizer'].hf_class}"
    )
    assert spec_map["tokenizer"].is_torch_model is False

    # Text encoder: plain transformers.Qwen3Model (text-only, hidden 2048)
    assert spec_map["text_encoder"].hf_class == "transformers.Qwen3Model", (
        f"text_encoder hf_class wrong: {spec_map['text_encoder'].hf_class}"
    )
    assert spec_map["text_encoder"].subfolder == "text_encoder"

    # VAE: standard diffusers AutoencoderKL
    assert spec_map["vae"].hf_class == "diffusers.AutoencoderKL", (
        f"vae hf_class wrong: {spec_map['vae'].hf_class}"
    )

    # Transformer mapped to "unet" (repo convention), diffusers-native class
    assert "OvisImageTransformer2DModel" in spec_map["unet"].hf_class, (
        f"unet hf_class wrong: {spec_map['unet'].hf_class}"
    )
    assert spec_map["unet"].subfolder == "transformer"


def test_loader_dtype_policy_matches_zimage():
    """Dtype policy is identical to zimage's.

    Neither loader overrides the generic ``_resolve_dtype`` (the effective
    bf16 loading dtype comes from ``driver.resolve_loading_dtype()``, passed
    explicitly as ``torch_dtype``), and no component carries a
    ``dtype_override``.
    """
    import torch

    from app.engine.core.pipeline.loader_base import GenericComponentLoader
    from app.engine.models.families.ovis_image.loader import OvisImageLoader
    from app.engine.models.families.zimage.loader import ZImageLoader

    assert (
        OvisImageLoader._resolve_dtype is GenericComponentLoader._resolve_dtype
    ), "OvisImageLoader must inherit the generic dtype policy (like zimage)"
    assert ZImageLoader._resolve_dtype is GenericComponentLoader._resolve_dtype

    loader = OvisImageLoader(torch.device("cpu"))
    definition = _make_ovis_definition()
    for spec in loader.get_component_manifest(definition):
        assert spec.dtype_override is None, (
            f"{spec.key} must not force a dtype override"
        )


# ── Task 3: Driver ───────────────────────────────────────────────────────────

# Tiny transformer config for CPU tests: axes_dims_rope must sum to
# attention_head_dim; joint_attention_dim is the raw text-embedding width.
_TINY_CFG = dict(
    patch_size=1,
    in_channels=64,
    out_channels=64,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=16,
    axes_dims_rope=(2, 4, 2),
)


def _build_tiny_model():
    import torch  # noqa: PLC0415

    from diffusers.models.transformers.transformer_ovis_image import (
        OvisImageTransformer2DModel,
    )

    torch.manual_seed(0)
    return OvisImageTransformer2DModel(**_TINY_CFG).eval()


def _make_driver(model=None, arch=None):
    import torch  # noqa: PLC0415

    from app.engine.models.families.ovis_image.driver import OvisImageDriver

    definition = _make_ovis_definition(architecture_params=arch or {})
    drv = OvisImageDriver(definition, torch.device("cpu"))
    drv.assign_components(
        {"unet": model, "vae": None, "text_encoder": None, "tokenizer": None},
    )
    return drv


def test_driver_lora_targets_cover_double_and_single_blocks():
    """Every default LoRA target pattern matches ≥1 module in BOTH streams.

    Derived from the real diffusers OvisImageTransformer2DModel module tree
    (tiny 1+1-layer instance): double blocks carry joint attention
    (to_q/k/v/out + add_*_proj/to_add_out) and two swiglu FeedForwards;
    single blocks carry pre-only attention + proj_mlp/proj_out.
    """
    import torch  # noqa: PLC0415

    model = _build_tiny_model()
    drv = _make_driver(model)
    targets = drv.get_lora_targets()

    linear_names = {
        n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    }
    for t in targets:
        assert any(n == t or n.endswith("." + t) for n in linear_names), (
            f"LoRA target {t!r} matches no Linear module"
        )

    # Double-stream coverage
    double = {t for t in targets if any(
        n.startswith("transformer_blocks.") and (n == t or n.endswith("." + t))
        for n in linear_names
    )}
    for expected in (
        "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
        "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj",
        "ff_context.net.2",
    ):
        assert expected in double, f"double-block target {expected!r} missing"

    # Single-stream coverage
    single = {t for t in targets if any(
        n.startswith("single_transformer_blocks.")
        and (n == t or n.endswith("." + t))
        for n in linear_names
    )}
    for expected in ("attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp", "proj_out"):
        assert expected in single, f"single-block target {expected!r} missing"


def test_driver_excludes_top_level_proj_out():
    """The FINAL projection (top-level ``proj_out``) must be excluded.

    ``proj_out`` as a PEFT suffix target also matches the model's top-level
    output projection; the driver returns the regex-string exclusion
    ``"proj_out"`` (fullmatch — hits ONLY the top-level module, keeping
    ``single_transformer_blocks.N.proj_out`` targetable).
    """
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from peft.tuners.lora.layer import LoraLayer  # noqa: PLC0415

    model = _build_tiny_model()
    drv = _make_driver(model)

    assert drv.get_lora_exclude_modules() == "proj_out"

    peft_model = get_peft_model(
        _build_tiny_model(),
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=drv.get_lora_targets(),
            exclude_modules=drv.get_lora_exclude_modules(),
        ),
    )
    wrapped = {
        n.replace("base_model.model.", "")
        for n, m in peft_model.named_modules()
        if isinstance(m, LoraLayer)
    }
    assert "proj_out" not in wrapped, "top-level proj_out must NOT be wrapped"
    assert "single_transformer_blocks.0.proj_out" in wrapped, (
        "single-block proj_out must stay wrapped"
    )


def test_driver_forward_divides_timesteps_by_1000_exactly_once():
    """forward_pass receives raw [0,1000] and hands t/1000 to the transformer.

    The transformer multiplies by 1000 internally, so any extra scaling
    silently produces pure-noise LoRAs (flow-match timestep-scale gotcha).
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

    B, C, H, W = 1, 16, 4, 4
    noisy = torch.randn(B, C, H, W)
    emb = torch.randn(B, 5, 16)
    mask = torch.ones(B, 5, dtype=torch.long)

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
    assert pred.shape == (B, C, H, W), f"unexpected shape: {pred.shape}"
    assert pred.isfinite().all(), "output contains NaN or inf"
    assert pred.float().std() > 0, "output is degenerate (zero std)"

    # txt_ids per OvisImagePipeline._prepare_prompt_embeds: cols 1 AND 2
    # both carry arange(text_seq_len); col 0 stays zero.
    txt_ids = captured["txt_ids"]
    assert txt_ids.shape == (5, 3)
    expected = torch.arange(5, dtype=txt_ids.dtype)
    assert torch.equal(txt_ids[:, 0], torch.zeros(5, dtype=txt_ids.dtype))
    assert torch.equal(txt_ids[:, 1], expected)
    assert torch.equal(txt_ids[:, 2], expected)


def test_driver_compute_target_is_flow_match():
    """compute_target = noise - latents (standard flow-match velocity)."""
    import torch  # noqa: PLC0415

    drv = _make_driver(None)
    latents = torch.randn(2, 16, 4, 4)
    noise = torch.randn(2, 16, 4, 4)
    target = drv.compute_target(latents, noise, torch.tensor([500.0, 250.0]))
    assert torch.equal(target, noise - latents)


def test_driver_basic_contracts():
    """Scheduler None (flow match), bf16 loading, no TE LoRA."""
    import torch  # noqa: PLC0415

    drv = _make_driver(None)
    assert drv.init_scheduler() is None
    assert drv.resolve_loading_dtype() == torch.bfloat16
    assert drv.get_te_lora_targets() == []


def test_driver_encode_text_replicates_pipeline_encode_prompt():
    """encode_text mirrors OvisImagePipeline._get_ovis_prompt_embeds:

    1. chat template applied with the pipeline system prompt prefix,
       ``add_generation_prompt=True`` and ``enable_thinking=False``;
    2. tokenized with padding='max_length',
       max_length = max_sequence_length + user_prompt_begin_id,
       ``add_special_tokens=False``;
    3. TE forward WITHOUT output_hidden_states → last_hidden_state;
    4. embeddings zero-masked (× attention_mask) then sliced
       ``[:, user_prompt_begin_id:, :]`` — mask sliced identically.
    """
    import torch  # noqa: PLC0415

    from app.engine.models.families.ovis_image.driver import (
        OVIS_SYSTEM_PROMPT,
        OvisImageDriver,
    )

    definition = _make_ovis_definition(
        architecture_params={
            "te.max_sequence_length": 256,
            "te.user_prompt_begin_id": 28,
        },
    )
    drv = OvisImageDriver(definition, torch.device("cpu"))

    B, D = 2, 16
    full_len = 256 + 28
    template_calls: list[dict] = []
    tokenize_calls: list[dict] = []

    tok = MagicMock()

    def _fake_template(message, **kwargs):
        template_calls.append({"message": message, **kwargs})
        return message[0]["content"]

    tok.apply_chat_template.side_effect = _fake_template

    def _fake_tokenize(texts, **kwargs):
        tokenize_calls.append(kwargs)
        n = len(texts)
        out = MagicMock()
        out.input_ids = torch.zeros(n, full_len, dtype=torch.long)
        # Last 10 positions are padding
        mask = torch.ones(n, full_len, dtype=torch.long)
        mask[:, -10:] = 0
        out.attention_mask = mask
        return out

    tok.side_effect = _fake_tokenize

    te = MagicMock()
    torch.manual_seed(3)
    hidden = torch.randn(B, full_len, D)

    def _fake_te(**kwargs):
        out = MagicMock()
        out.last_hidden_state = hidden
        return out

    te.side_effect = _fake_te
    te.parameters = lambda: iter([torch.zeros(1)])

    drv.tokenizer = tok
    drv.text_encoder = te

    out = drv.encode_text(["a fox", "a cat"], torch.float32)

    # 1. Chat template semantics
    assert len(template_calls) == 2
    for call, cap in zip(template_calls, ["a fox", "a cat"]):
        msg = call["message"]
        assert msg[0]["role"] == "user"
        assert msg[0]["content"] == OVIS_SYSTEM_PROMPT + cap, (
            "prompt must be prefixed with the pipeline system prompt"
        )
        assert call["add_generation_prompt"] is True
        assert call["enable_thinking"] is False
        assert call["tokenize"] is False

    # 2. Tokenizer semantics
    tk = tokenize_calls[0]
    assert tk["padding"] == "max_length"
    assert tk["truncation"] is True
    assert tk["max_length"] == full_len
    assert tk["add_special_tokens"] is False

    # 3+4. Masked, sliced last_hidden_state
    mask_full = torch.ones(B, full_len)
    mask_full[:, -10:] = 0
    expected = (hidden * mask_full[..., None])[:, 28:, :]
    assert out.embeddings.shape == (B, 256, D)
    assert torch.allclose(out.embeddings, expected)
    assert out.attention_mask is not None
    assert out.attention_mask.shape == (B, 256)
    assert out.attention_mask[:, -10:].sum() == 0
    # Zero-masked tail must actually be zero in the embeddings
    assert out.embeddings[:, -10:, :].abs().sum() == 0
