"""Tests for the dreamlite family (ByteDance DreamLite, diffusers-0.39-native).

TDD order (mirrors test_ovis_image_family.py / test_krea2_family.py):
  Task 1: family registration + BOTH definitions (base + mobile) loading
  Task 2: loader manifest (component specs + revision plumbing + dtype policy)
  Task 4: driver (LoRA targets vs a tiny DreamLiteUNetModel, forward_pass
          UNET signature contract, raw-timestep contract, compute_target)
  Task 5: trainer override trio (encode_text tuple, _update_primary_model
          driver sync, transformer property) + TE disk-cache layout

Verified checkpoint facts (unet/config.json, revision="diffusers"):
  block_out_channels (256, 512, 896); attention_head_dim (4, 8, 14) â†’
  head_dim 64 at EVERY level; num_kv_heads=1 (MQA) â†’ to_k/to_v
  out_features = 64; transformer_layers_per_block (1, 2, 4);
  cross_attention_dim 2304; encoder_hid_dim_type "text_proj_rms" (2048â†’2304);
  addition_embed_type "time" (time_ids [w, h]); ff_mult 3; qk_norm rms_norm.
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


def _make_dreamlite_definition(**kwargs) -> MagicMock:
    """Build a mock DreamLite ModelDefinition for loader/driver tests."""
    definition = MagicMock(spec=ModelDefinition)
    definition.family = "dreamlite"
    definition.id = kwargs.get("id", "dreamlite-test")
    definition.components = {}
    definition.architecture_params = kwargs.get("architecture_params", {})
    definition.lora_targetable_modules = kwargs.get(
        "lora_targetable_modules", [],
    )
    return definition


# â”€â”€ Task 1: Family Registration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_family_registered():
    """dreamlite family must appear in ModelRegistry with the correct archetype."""
    from app.engine.models.registry import ModelRegistry

    # Reset discovery state so this test is hermetic
    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()

    fam = ModelRegistry._families.get("dreamlite")
    assert fam is not None, "dreamlite family not registered"
    assert fam.archetype == "latent_diffusion", (
        f"expected archetype='latent_diffusion', got {fam.archetype!r}"
    )


def test_both_definitions_loaded():
    """dreamlite-base AND dreamlite-mobile definitions must load from YAML."""
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
        if d.family == "dreamlite"
    }
    assert "dreamlite-base" in fam_defs, (
        f"missing dreamlite-base definition; found: {set(fam_defs)}"
    )
    assert "dreamlite-mobile" in fam_defs, (
        f"missing dreamlite-mobile definition; found: {set(fam_defs)}"
    )

    base = fam_defs["dreamlite-base"]
    mobile = fam_defs["dreamlite-mobile"]

    # Canonical checkpoints â€” BOTH need the "diffusers" revision (@revision
    # suffix understood by ModelPathResolver since the dreamlite family).
    assert base.components["repo"].path == (
        "huggingface:carlofkl/DreamLite-base@diffusers"
    ), f"wrong base repo path: {base.components['repo'].path!r}"
    assert mobile.components["repo"].path == (
        "huggingface:carlofkl/DreamLite-mobile@diffusers"
    ), f"wrong mobile repo path: {mobile.components['repo'].path!r}"

    # Standard T2I â€” the diptych "[Edit]" mode is out of scope.
    assert base.control_inputs == 0
    assert mobile.control_inputs == 0

    # CFG split: base is the CFG (30-step) checkpoint, mobile is the
    # CFG-distilled (4-step) checkpoint â€” the krea2 Raw/Turbo convention.
    assert base.defaults.get("is_distilled") is False
    assert mobile.defaults.get("is_distilled") is True
    assert base.defaults.get("guidance_scale") == 3.5
    assert mobile.defaults.get("guidance_scale") == 0
    assert mobile.defaults.get("num_inference_steps") == 4

    # Verified unet config facts (checkpoint unet/config.json).
    for defn in (base, mobile):
        arch = defn.architecture_params
        assert arch.get("unet._class_name") == "DreamLiteUNetModel"
        assert arch.get("unet.block_out_channels") == [256, 512, 896]
        assert arch.get("unet.attention_head_dim") == [4, 8, 14]
        assert arch.get("unet.transformer_layers_per_block") == [1, 2, 4]
        assert arch.get("unet.layers_per_block") == 2
        assert arch.get("unet.cross_attention_dim") == 2304
        assert arch.get("unet.encoder_hid_dim") == 2048
        assert arch.get("unet.encoder_hid_dim_type") == "text_proj_rms"
        assert arch.get("unet.addition_embed_type") == "time"
        assert arch.get("unet.num_kv_heads") == 1
        assert arch.get("unet.qk_norm") == "rms_norm"
        assert arch.get("unet.ff_mult") == 3
        assert arch.get("unet.in_channels") == 4
        assert arch.get("unet.sample_size") == 128
        # VAE: AutoencoderTiny (taesdxl) â€” 4 latent channels, 8Ã— spatial.
        assert arch.get("vae._class_name") == "AutoencoderTiny"
        assert arch.get("vae.latent_channels") == 4
        assert arch.get("vae.vae_scale_factor") == 8
        assert arch.get("vae.scaling_factor") == 1.0
        assert arch.get("vae.shift_factor") == 0.0
        # Scheduler facts (checkpoint scheduler_config.json).
        assert arch.get("scheduler.use_dynamic_shifting") is True
        assert arch.get("scheduler.base_shift") == 0.5
        assert arch.get("scheduler.max_shift") == 1.15
        assert arch.get("scheduler.base_image_seq_len") == 256
        assert arch.get("scheduler.max_image_seq_len") == 4096
        # DreamLitePipeline prompt-template contract.
        assert arch.get("te.max_sequence_length") == 200
        assert arch.get("te.drop_idx") == 34


# â”€â”€ Task 2: Loader Manifest â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_manifest_components():
    """DreamLiteLoader manifest declares all four diffusers-native components.

    The checkpoint stores its primary model under ``unet/`` (it IS a U-Net â€”
    ``model_index.json`` names the component "unet"), unlike the DiT
    families' ``transformer/`` subfolder. All four components are
    diffusers-0.39 / transformers-4.57-native: the TE config is saved by
    transformers 4.57.3 (verified), so no krea2-style rope translation is
    needed and the plain manifest path suffices.
    """
    import torch  # noqa: PLC0415

    from app.engine.models.families.dreamlite.loader import DreamLiteLoader

    loader = DreamLiteLoader(torch.device("cpu"))
    definition = _make_dreamlite_definition()
    specs = loader.get_component_manifest(definition)

    keys = {s.key for s in specs}
    assert {"tokenizer", "text_encoder", "vae", "unet"} <= keys, (
        f"missing required manifest keys; got {keys}"
    )

    spec_map = {s.key: s for s in specs}

    # Tokenizer: AutoTokenizer (fast Qwen2TokenizerFast via tokenizer.json)
    assert "AutoTokenizer" in spec_map["tokenizer"].hf_class
    assert spec_map["tokenizer"].is_torch_model is False

    # Text encoder: Qwen3-VL base model (no LM head; hidden_states[-1] is
    # identical to the pipeline's Qwen3VLForConditionalGeneration tap).
    assert spec_map["text_encoder"].hf_class == "transformers.Qwen3VLModel"
    assert spec_map["text_encoder"].subfolder == "text_encoder"

    # VAE: AutoencoderTiny (taesdxl â€” NO latent_dist, encode returns .latents)
    assert spec_map["vae"].hf_class == "diffusers.AutoencoderTiny"
    assert spec_map["vae"].subfolder == "vae"

    # Primary model: DreamLiteUNetModel under the checkpoint's unet/ subfolder
    assert spec_map["unet"].hf_class == "diffusers.DreamLiteUNetModel"
    assert spec_map["unet"].subfolder == "unet"


def test_loader_dtype_policy_is_generic():
    """No dtype overrides â€” bf16 comes from driver.resolve_loading_dtype."""
    import torch  # noqa: PLC0415

    from app.engine.core.pipeline.loader_base import GenericComponentLoader
    from app.engine.models.families.dreamlite.loader import DreamLiteLoader

    assert (
        DreamLiteLoader._resolve_dtype is GenericComponentLoader._resolve_dtype
    ), "DreamLiteLoader must inherit the generic dtype policy"

    loader = DreamLiteLoader(torch.device("cpu"))
    definition = _make_dreamlite_definition()
    for spec in loader.get_component_manifest(definition):
        assert spec.dtype_override is None, (
            f"{spec.key} must not force a dtype override"
        )


# â”€â”€ Task 4: Driver â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Tiny UNet config for CPU tests. Structurally faithful to the checkpoint:
# per-level attention_head_dim (== NUM HEADS per the UNet naming bug),
# MQA (num_kv_heads=1) + rms_norm qk_norm, text_proj_rms encoder projection
# (TE width 12 â†’ cross-attn 16), "time" addition embedding consuming
# time_ids, ff_mult 3, sep-convs, linear projections, and the checkpoint's
# per-level transformer_layers_per_block (1, 2, 4).
_TINY_UNET_CFG = dict(
    in_channels=4,
    out_channels=4,
    block_out_channels=(8, 16, 32),
    layers_per_block=1,
    transformer_layers_per_block=(1, 2, 4),
    attention_head_dim=4,
    cross_attention_dim=16,
    norm_num_groups=8,
    use_linear_projection=True,
    encoder_hid_dim=12,
    encoder_hid_dim_type="text_proj_rms",
    addition_embed_type="time",
    addition_time_embed_dim=8,
    projection_class_embeddings_input_dim=16,
    num_kv_heads=1,
    qk_norm="rms_norm",
    ff_mult=3,
    use_sep_conv=True,
)

_TINY_TEXT_DIM = 12  # == encoder_hid_dim of the tiny model


def _build_tiny_unet():
    import torch  # noqa: PLC0415

    from diffusers.models.unets.unet_dreamlite import DreamLiteUNetModel

    torch.manual_seed(0)
    return DreamLiteUNetModel(**_TINY_UNET_CFG).eval()


def _make_driver(model=None, arch=None):
    import torch  # noqa: PLC0415

    from app.engine.models.families.dreamlite.driver import DreamLiteDriver

    definition = _make_dreamlite_definition(architecture_params=arch or {})
    drv = DreamLiteDriver(definition, torch.device("cpu"))
    drv.assign_components(
        {"unet": model, "vae": None, "text_encoder": None, "tokenizer": None},
    )
    return drv


def test_driver_lora_targets_match_unet_module_tree():
    """Every default LoRA target pattern matches â‰¥1 Linear in the tiny UNet;
    self-attention (attn1) exists ONLY where the checkpoint has it."""
    import torch  # noqa: PLC0415

    model = _build_tiny_unet()
    drv = _make_driver(model)
    targets = drv.get_lora_targets()

    linear_names = {
        n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    }
    for t in targets:
        assert any(n == t or n.endswith("." + t) for n in linear_names), (
            f"LoRA target {t!r} matches no Linear module"
        )

    # attn1 topology: NO self-attention in down_blocks.0/1 and up_blocks.1
    # (the "NoSelfAttn" blocks); attn1 lives in down_blocks.2, mid_block,
    # and up_blocks.0. up_blocks.2 has no attention at all.
    assert not any(
        n.startswith("down_blocks.0") and ".attn1." in n for n in linear_names
    ), "down_blocks.0 must NOT have self-attention"
    assert not any(
        n.startswith("down_blocks.1") and ".attn1." in n for n in linear_names
    )
    assert not any(
        n.startswith("up_blocks.1") and ".attn1." in n for n in linear_names
    )
    assert any(
        n.startswith("down_blocks.2") and ".attn1." in n for n in linear_names
    ), "down_blocks.2 must have self-attention"
    assert any(
        n.startswith("mid_block") and ".attn1." in n for n in linear_names
    ), "mid_block must have self-attention"
    assert any(
        n.startswith("up_blocks.0") and ".attn1." in n for n in linear_names
    )
    assert not any(n.startswith("up_blocks.2") for n in linear_names if ".attn" in n)

    # Cross-attention everywhere there IS attention
    for prefix in ("down_blocks.0", "down_blocks.1", "down_blocks.2",
                   "mid_block", "up_blocks.0", "up_blocks.1"):
        assert any(
            n.startswith(prefix) and ".attn2." in n for n in linear_names
        ), f"{prefix} must have cross-attention"


def test_driver_lora_targets_pin_real_checkpoint_module_count():
    """Meta-instantiate the REAL checkpoint unet config and pin the LoRA
    surface: 312 target modules â†’ 624 keys; MQA to_k/to_v out_features = 64
    at EVERY level (head_dim 64, num_kv_heads 1)."""
    import torch  # noqa: PLC0415

    from diffusers.models.unets.unet_dreamlite import DreamLiteUNetModel

    real_cfg = dict(
        sample_size=128,
        in_channels=4,
        out_channels=4,
        block_out_channels=(256, 512, 896),
        attention_head_dim=(4, 8, 14),
        cross_attention_dim=2304,
        layers_per_block=2,
        transformer_layers_per_block=(1, 2, 4),
        use_linear_projection=True,
        encoder_hid_dim=2048,
        encoder_hid_dim_type="text_proj_rms",
        addition_embed_type="time",
        addition_time_embed_dim=256,
        projection_class_embeddings_input_dim=512,
        num_kv_heads=1,
        qk_norm="rms_norm",
        ff_mult=3,
        use_sep_conv=True,
        norm_num_groups=32,
    )
    with torch.device("meta"):
        model = DreamLiteUNetModel(**real_cfg)

    drv = _make_driver(None)
    targets = drv.get_lora_targets()

    linear = {
        n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    }
    matched = {
        n for n in linear
        if any(n == t or n.endswith("." + t) for t in targets)
    }
    assert len(matched) == 312, (
        f"expected 312 LoRA target modules on the real config, got {len(matched)}"
    )

    # MQA pin: every to_k / to_v projection is 64-wide (head_dim * kv_heads).
    for n in matched:
        if n.endswith(("to_k", "to_v")):
            assert linear[n].out_features == 64, (
                f"{n}: expected MQA width 64, got {linear[n].out_features}"
            )

    # Param count sanity â€” 0.39 B (the vram_estimator entry's provenance).
    n_params = sum(p.numel() for p in model.parameters())
    assert 380e6 < n_params < 400e6, f"unexpected param count {n_params:,}"


def test_driver_forward_matches_pipeline_unet_call():
    """forward_pass mirrors DreamLitePipeline's UNet invocation EXACTLY:

    1. model input = cat([latents, zeros_like(latents)], dim=3) â€” the
       generate-mode width concat (zero image-conditioning half);
    2. timestep passed RAW on the [0, 1000] scale (``t.expand(B).to(dtype)``
       â€” the UNet's sinusoidal time_proj consumes raw timesteps; NO /1000);
    3. encoder_attention_mask forwarded;
    4. added_cond_kwargs = {"time_ids": [[w_px, h_px]] * B} (pixel dims =
       latent dims Ã— vae_scale_factor);
    5. prediction sliced back to the latent width (``[..., :W]``).
    """
    import torch  # noqa: PLC0415

    model = _build_tiny_unet()
    drv = _make_driver(model, arch={"vae.vae_scale_factor": 8})

    captured: dict = {}
    original_forward = model.forward

    def _spy(sample, *args, **kwargs):
        captured["sample"] = sample
        captured.update(kwargs)
        return original_forward(sample, *args, **kwargs)

    model.forward = _spy

    B, C, H, W = 2, 4, 8, 8
    noisy = torch.randn(B, C, H, W)
    emb = torch.randn(B, 7, _TINY_TEXT_DIM)
    mask = torch.ones(B, 7, dtype=torch.long)
    mask[:, -2:] = 0

    with torch.no_grad():
        pred = drv.forward_pass(
            noisy_input=noisy,
            timesteps=torch.tensor([500.0, 250.0]),
            text_embeddings=(emb, mask),
            batch={},
        )

    # 1. Width-doubled input, right half zeros
    sample = captured["sample"]
    assert sample.shape == (B, C, H, 2 * W), f"bad model input: {sample.shape}"
    assert torch.equal(sample[..., :W], noisy)
    assert sample[..., W:].abs().sum() == 0, "conditioning half must be zeros"

    # 2. RAW timesteps â€” identical values, latent dtype (NO /1000!)
    ts = captured["timestep"]
    assert torch.allclose(ts.float(), torch.tensor([500.0, 250.0])), (
        f"UNet must receive RAW [0,1000] timesteps, got {ts}"
    )
    assert ts.dtype == noisy.dtype

    # 3. Mask forwarded as-is
    assert torch.equal(captured["encoder_attention_mask"], mask)

    # 4. time_ids = pixel (w, h) per batch row
    time_ids = captured["added_cond_kwargs"]["time_ids"]
    assert time_ids.shape == (B, 2)
    assert torch.allclose(
        time_ids.float(),
        torch.tensor([[64.0, 64.0], [64.0, 64.0]]),
    ), f"time_ids must carry pixel (w, h) = latent*8, got {time_ids}"

    # 5. Output sliced back to latent width
    assert pred.shape == (B, C, H, W), f"unexpected pred shape: {pred.shape}"
    assert pred.isfinite().all(), "output contains NaN or inf"
    assert pred.float().std() > 0, "output is degenerate (zero std)"


def test_driver_compute_target_is_flow_match():
    """compute_target = noise - latents (standard flow-match velocity)."""
    import torch  # noqa: PLC0415

    drv = _make_driver(None)
    latents = torch.randn(2, 4, 8, 8)
    noise = torch.randn(2, 4, 8, 8)
    target = drv.compute_target(latents, noise, torch.tensor([500.0, 250.0]))
    assert torch.equal(target, noise - latents)


def test_driver_basic_contracts():
    """Scheduler None (flow match), bf16 loading, no TE LoRA, no excludes."""
    import torch  # noqa: PLC0415

    drv = _make_driver(None)
    assert drv.init_scheduler() is None
    assert drv.resolve_loading_dtype() == torch.bfloat16
    assert drv.get_te_lora_targets() == []


def test_driver_encode_text_replicates_pipeline_encode_prompt():
    """encode_text mirrors DreamLitePipeline.encode_prompt (generate mode):

    1. the caption is inserted into the pinned chat template;
    2. tokenized with max_length = max_sequence_length + drop_idx,
       padding + truncation;
    3. TE forward WITH output_hidden_states â†’ hidden_states[-1];
    4. per-sequence mask-select, drop the first drop_idx (34) template
       tokens, re-pad with zeros; fresh 0/1 mask.

    Deviation for cacheability (mask-equivalent): embeddings/mask are
    right-padded to the FIXED te.max_sequence_length instead of the batch
    max â€” padded positions carry mask 0 and zero embeddings, exactly the
    pipeline's padding convention.
    """
    import torch  # noqa: PLC0415

    from app.engine.models.families.dreamlite.driver import (
        DREAMLITE_PROMPT_TEMPLATE,
        DreamLiteDriver,
    )

    # The pinned upstream template (pipeline_dreamlite.py:219-224)
    assert DREAMLITE_PROMPT_TEMPLATE.startswith("<|im_start|>system\n")
    assert "{}" in DREAMLITE_PROMPT_TEMPLATE
    assert DREAMLITE_PROMPT_TEMPLATE.endswith("<|im_start|>assistant\n")

    max_seq, drop_idx = 20, 34
    definition = _make_dreamlite_definition(
        architecture_params={
            "te.max_sequence_length": max_seq,
            "te.drop_idx": drop_idx,
        },
    )
    drv = DreamLiteDriver(definition, torch.device("cpu"))

    D = 12
    full_len = max_seq + drop_idx  # 54
    valid = {0: full_len, 1: drop_idx + 5}  # cap 1: only 5 user tokens
    tokenize_calls: list[dict] = []

    tok = MagicMock()

    def _fake_tokenize(text=None, **kwargs):
        tokenize_calls.append({"texts": text, **kwargs})
        n = len(text)
        out = MagicMock()
        out.input_ids = torch.zeros(n, full_len, dtype=torch.long)
        mask = torch.zeros(n, full_len, dtype=torch.long)
        for i in range(n):
            mask[i, : valid[i]] = 1
        out.attention_mask = mask
        out.to = lambda *_a, **_k: out
        return out

    tok.side_effect = _fake_tokenize

    te = MagicMock()
    torch.manual_seed(3)
    hidden = torch.randn(2, full_len, D)

    def _fake_te(**kwargs):
        assert kwargs.get("output_hidden_states") is True, (
            "TE must be called with output_hidden_states=True"
        )
        out = MagicMock()
        out.hidden_states = [torch.zeros_like(hidden), hidden]
        return out

    te.side_effect = _fake_te
    te.parameters = lambda: iter([torch.zeros(1)])

    drv.tokenizer = tok
    drv.text_encoder = te

    out = drv.encode_text(["a fox", "cat"], torch.float32)

    # 1+2. Template + tokenizer semantics
    tk = tokenize_calls[0]
    assert tk["texts"] == [
        DREAMLITE_PROMPT_TEMPLATE.format("a fox"),
        DREAMLITE_PROMPT_TEMPLATE.format("cat"),
    ]
    assert tk["max_length"] == full_len
    assert tk["truncation"] is True
    assert tk["padding"] is True

    # 3+4. hidden_states[-1], mask-select, drop 34, re-pad to max_seq
    assert out.embeddings.shape == (2, max_seq, D)
    # cap 0: full-length valid â†’ 20 user tokens survive the drop
    assert torch.allclose(out.embeddings[0], hidden[0, drop_idx:, :])
    # cap 1: 5 valid user tokens, rest zero-padded
    assert torch.allclose(out.embeddings[1, :5], hidden[1, drop_idx:drop_idx + 5, :])
    assert out.embeddings[1, 5:].abs().sum() == 0

    mask = out.attention_mask
    assert mask.shape == (2, max_seq)
    assert mask[0].sum() == max_seq
    assert mask[1].sum() == 5
    assert mask[1, :5].sum() == 5


# â”€â”€ Task 5: Trainer override trio + [Generate] prefix + TE cache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_TE_ARCH = {
    "te.max_sequence_length": 20,
    "te.drop_idx": 34,
    "vae.vae_scale_factor": 8,
}


def _stub_tokenizer_and_te(D: int = 12, full_len: int = 54):
    """Stub tokenizer + TE satisfying the DreamLite prompt-embed contract."""
    import torch  # noqa: PLC0415

    tok = MagicMock()

    def _fake_tokenize(text=None, **kwargs):
        n = len(text)
        out = MagicMock()
        out.input_ids = torch.zeros(n, full_len, dtype=torch.long)
        out.attention_mask = torch.ones(n, full_len, dtype=torch.long)
        out.to = lambda *_a, **_k: out
        return out

    tok.side_effect = _fake_tokenize

    te = MagicMock()

    def _fake_te(**kwargs):
        b, seq = kwargs["input_ids"].shape
        out = MagicMock()
        out.hidden_states = [torch.zeros(b, seq, D), torch.randn(b, seq, D)]
        return out

    te.side_effect = _fake_te
    te.parameters = lambda: iter([torch.zeros(1)])
    return tok, te


def _build_real_trainer_shell():
    """Minimal DreamLiteTrainer shell with a REAL driver + tiny UNet.

    Binds the real trainer methods (encode seam) without calling setup().
    """
    import torch  # noqa: PLC0415

    from app.engine.models.families.dreamlite.driver import DreamLiteDriver
    from app.engine.models.families.dreamlite.trainer import DreamLiteTrainer

    definition = _make_dreamlite_definition(architecture_params=dict(_TE_ARCH))

    trainer = MagicMock(spec=DreamLiteTrainer)
    trainer.device = torch.device("cpu")
    trainer.definition = definition
    trainer.config = {"cache_text_embeddings": False}
    trainer.text_cache = {}
    trainer.logger = MagicMock()

    drv = DreamLiteDriver(definition, torch.device("cpu"))
    tiny_model = _build_tiny_unet()
    tok, te = _stub_tokenizer_and_te(D=_TINY_TEXT_DIM)
    drv.assign_components({
        "unet": tiny_model,
        "vae": None,
        "text_encoder": te,
        "tokenizer": tok,
    })
    trainer.driver = drv

    trainer.encode_text = lambda captions, dtype, batch=None: (
        DreamLiteTrainer.encode_text(trainer, captions, dtype, batch)
    )
    trainer.encode_uncond_text = lambda texts, dtype: (
        DreamLiteTrainer.encode_uncond_text(trainer, texts, dtype)
    )
    trainer._positive_key = DreamLiteTrainer._positive_key
    trainer._encode_keys = lambda keys, dtype: (
        DreamLiteTrainer._encode_keys(trainer, keys, dtype)
    )
    trainer._encode_text_direct = lambda texts, dtype: (
        DreamLiteTrainer._encode_text_direct(trainer, texts, dtype)
    )
    trainer._get_cached_text_embeddings = lambda keys, dtype: (
        DreamLiteTrainer._get_cached_text_embeddings(trainer, keys, dtype)
    )
    return trainer


def test_trainer_encode_to_forward_real_seam():
    """C1/C2: trainer.encode_text returns a (emb, mask) TUPLE consumable by
    driver.forward_pass â€” the whole encodeâ†’forward round trip produces a
    finite [B, C, H, W] prediction."""
    import torch  # noqa: PLC0415

    trainer = _build_real_trainer_shell()

    text_emb = trainer.encode_text(["a dreamlite test caption"], torch.float32)
    assert isinstance(text_emb, tuple) and len(text_emb) == 2, (
        f"encode_text must return a 2-tuple, got {type(text_emb)}"
    )
    emb, mask = text_emb
    assert emb.ndim == 3, f"embeddings must be 3-D [B,L,D], got {emb.ndim}-D"
    assert emb.shape[1] == 20, f"fixed seq len must be 20, got {emb.shape[1]}"
    assert mask.ndim == 2 and mask.shape[1] == 20

    B, C, H, W = 1, 4, 8, 8
    with torch.no_grad():
        pred = trainer.driver.forward_pass(
            noisy_input=torch.randn(B, C, H, W),
            timesteps=torch.tensor([500.0]),
            text_embeddings=text_emb,
            batch={},
        )
    assert pred.shape == (B, C, H, W), f"unexpected pred shape: {pred.shape}"
    assert pred.isfinite().all(), "forward_pass output contains NaN/inf"


def test_trainer_encode_applies_generate_prefix_negatives_stay_raw():
    """The pipeline's __call__ encodes ``[negative, "[Generate]: "+prompt]``:
    trainer.encode_text (captions / sample positives) must prefix, while
    encode_uncond_text (CFG negatives) must NOT. Cache keys carry the
    transformation so a caption and an identical negative never collide."""
    import torch  # noqa: PLC0415

    from app.engine.models.families.dreamlite.driver import (
        DREAMLITE_PROMPT_TEMPLATE,
    )

    trainer = _build_real_trainer_shell()
    trainer.config = {"cache_text_embeddings": True}
    trainer.text_encoder = trainer.driver.text_encoder

    tok = trainer.driver.tokenizer

    trainer.encode_text(["blurry"], torch.float32)
    templated_pos = tok.call_args.kwargs["text"]
    assert templated_pos == [
        DREAMLITE_PROMPT_TEMPLATE.format("[Generate]: blurry"),
    ], "positive captions must carry the [Generate] prefix inside the template"

    trainer.encode_uncond_text(["blurry"], torch.float32)
    templated_neg = tok.call_args.kwargs["text"]
    assert templated_neg == [DREAMLITE_PROMPT_TEMPLATE.format("blurry")], (
        "negatives must NOT carry the [Generate] prefix"
    )

    # Distinct cache keys â€” no positive/negative collision.
    assert "[Generate]: blurry" in trainer.text_cache
    assert "blurry" in trainer.text_cache
    assert len(trainer.text_cache) == 2


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
    assert set(trainer.text_cache) == {
        "[Generate]: cap one", "[Generate]: cap two",
    }
    # Cache entries are (emb [L,D], mask [L]) CPU tuples
    cached_emb, cached_mask = trainer.text_cache["[Generate]: cap one"]
    assert cached_emb.ndim == 2 and cached_mask.ndim == 1


def test_trainer_peft_model_sync():
    """C3/C4: _update_primary_model syncs driver.model, components, and the
    read-only ``transformer`` property resolves to the wrapped model."""
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    from app.engine.models.families.dreamlite.driver import DreamLiteDriver
    from app.engine.models.families.dreamlite.trainer import DreamLiteTrainer

    definition = _make_dreamlite_definition()
    trainer = MagicMock(spec=DreamLiteTrainer)
    trainer.device = torch.device("cpu")
    trainer.definition = definition

    drv = DreamLiteDriver(definition, torch.device("cpu"))
    tiny_model = _build_tiny_unet()
    drv.assign_components({
        "unet": tiny_model, "vae": None,
        "text_encoder": None, "tokenizer": None,
    })
    trainer.driver = drv
    trainer.components = {"unet": tiny_model}
    trainer.model = tiny_model

    class _FakePEFT(nn.Module):
        pass

    peft_wrapped = _FakePEFT()
    DreamLiteTrainer._update_primary_model(trainer, peft_wrapped)

    assert trainer.driver.model is peft_wrapped, (
        "C3: driver.model was NOT updated after _update_primary_model"
    )
    assert trainer.components.get("unet") is peft_wrapped
    assert trainer.model is peft_wrapped

    transformer_val = DreamLiteTrainer.transformer.fget(trainer)
    assert transformer_val is peft_wrapped, (
        "C4: trainer.transformer property must resolve to the wrapped model"
    )


def test_trainer_te_disk_cache_layout(tmp_path):
    """TE disk cache lands in embeddings/{te_quant}/te1|te2 (emb + mask);
    dataset captions are warmed under their PREFIXED cache key."""
    import torch  # noqa: PLC0415

    from app.engine.components.text_embeddings import TextEmbeddingCache
    from app.engine.models.families.dreamlite.trainer import DreamLiteTrainer

    trainer = MagicMock(spec=DreamLiteTrainer)
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
    trainer._build_caption_hints.return_value = {"a dreamlite caption": "hint0"}
    trainer._resolve_te_cache_dirs.return_value = [str(tmp_path)]
    trainer._resolve_loading_dtype.return_value = torch.float32
    trainer._sample_prompt_texts.side_effect = lambda: (
        DreamLiteTrainer._sample_prompt_texts(trainer)
    )
    trainer._positive_key = DreamLiteTrainer._positive_key

    def _fake_encode(texts, dtype):
        b = len(texts)
        return torch.zeros(b, 20, 12), torch.ones(b, 20, dtype=torch.long)

    trainer._encode_text_direct = _fake_encode

    DreamLiteTrainer._pre_cache_text_embeddings(trainer)

    te1 = tmp_path / "embeddings" / "none" / "te1"
    te2 = tmp_path / "embeddings" / "none" / "te2"
    assert te1.is_dir(), "te1 (embeddings) cache dir missing"
    assert te2.is_dir(), "te2 (attention mask) cache dir missing"

    key = "[Generate]: a dreamlite caption"
    emb = TextEmbeddingCache.load(key, str(te1), "hint0")
    mask = TextEmbeddingCache.load(key, str(te2), "hint0")
    assert emb is not None and emb.shape == (20, 12)
    assert mask is not None and mask.shape == (20,)
    assert key in trainer.text_cache
    cached_emb, cached_mask = trainer.text_cache[key]
    assert cached_emb.shape == (20, 12)
    assert cached_mask.shape == (20,)


def test_trainer_warms_sample_and_negative_prompts():
    """Pre-cache also warms expanded sample prompts (PREFIXED) and the
    negative prompt (RAW) so the TE can stay offloaded during sampling
    (krea2 VRAM-spike lesson)."""
    import torch  # noqa: PLC0415

    from app.engine.models.families.dreamlite.trainer import DreamLiteTrainer

    trainer = MagicMock(spec=DreamLiteTrainer)
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
        DreamLiteTrainer._sample_prompt_texts(trainer)
    )
    trainer._positive_key = DreamLiteTrainer._positive_key

    def _fake_encode(texts, dtype):
        b = len(texts)
        return torch.zeros(b, 20, 12), torch.ones(b, 20, dtype=torch.long)

    trainer._encode_text_direct = _fake_encode

    DreamLiteTrainer._pre_cache_text_embeddings(trainer)

    assert "[Generate]: a red car" in trainer.text_cache, (
        "sample prompts must be warmed under their PREFIXED key"
    )
    assert "blurry" in trainer.text_cache, (
        "the negative prompt must be warmed RAW (un-prefixed)"
    )
    assert "[Generate]: a dataset caption" in trainer.text_cache


def test_base_and_mobile_architecture_params_identical():
    """Portability requirement: base and mobile share ONE architecture.

    Verified against the hub: unet/vae/text_encoder/scheduler configs are
    byte-identical between DreamLite-base and DreamLite-mobile (only the
    weights differ). The YAMLs must therefore carry identical
    ``architecture_params`` so a LoRA trained on one definition loads onto
    the other.
    """
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()

    fam_defs = {
        d.id: d
        for d in ModelRegistry._definitions.values()
        if d.family == "dreamlite"
    }
    base = fam_defs["dreamlite-base"]
    mobile = fam_defs["dreamlite-mobile"]
    assert base.architecture_params == mobile.architecture_params, (
        "architecture_params must be byte-identical across base and mobile"
    )
