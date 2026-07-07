"""Tests for the prx family (diffusers-0.39-native Photoroom PRX, latent).

TDD order (mirrors test_ovis_image_family.py):
  Task 1: family registration + definition loading
  Task 2: loader manifest (component specs + dtype policy)
  Task 3: prx_shared LoRA targets verified against a tiny real
          PRXTransformer2DModel (FUSED projections — no to_q/to_k/to_v)
  Task 4: driver (normalized-timestep contract, compute_target,
          encode_text replicating PRXPipeline.encode_prompt)
  Task 5: trainer override trio (encode_text tuple, _update_primary_model
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
    in the session (same pattern as test_ovis_image_family.py).
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


def _make_prx_definition(**kwargs) -> MagicMock:
    """Build a mock PRX ModelDefinition for loader/driver tests."""
    definition = MagicMock(spec=ModelDefinition)
    definition.family = "prx"
    definition.id = kwargs.get("id", "prx-test")
    definition.components = {}
    definition.architecture_params = kwargs.get("architecture_params", {})
    definition.lora_targetable_modules = kwargs.get("lora_targetable_modules", [])
    definition.defaults = kwargs.get("defaults", {})
    return definition


# ── Task 1: Family Registration ──────────────────────────────────────────────


def test_family_registered():
    """prx family must appear in ModelRegistry with the correct archetype."""
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()

    fam = ModelRegistry._families.get("prx")
    assert fam is not None, "prx family not registered"
    assert fam.archetype == "latent_diffusion", (
        f"expected archetype='latent_diffusion', got {fam.archetype!r}"
    )


def test_prx_shared_is_not_a_family():
    """prx_shared is a shared module, NOT a registrable family."""
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()

    assert "prx_shared" not in ModelRegistry._families


def test_definition_loaded():
    """prx-sft definition must load from its YAML file."""
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()

    fam_defs = {
        d.id: d
        for d in ModelRegistry._definitions.values()
        if d.family == "prx"
    }
    assert "prx-sft" in fam_defs, (
        f"missing prx-sft definition; found: {set(fam_defs)}"
    )

    base = fam_defs["prx-sft"]
    # Canonical checkpoint repo (verified fact — see plan)
    assert base.components["repo"].path == "huggingface:Photoroom/prx-512-t2i-sft", (
        f"wrong repo path: {base.components['repo'].path!r}"
    )
    # Standard T2I — no paired control inputs
    assert base.control_inputs == 0
    # Native 512 default resolution (default_sample_size=512)
    assert base.defaults.get("resolution") == 512

    # Verified transformer config facts (checkpoint transformer/config.json,
    # identical to the diffusers 0.39 PRXTransformer2DModel defaults).
    arch = base.architecture_params
    assert arch.get("transformer.depth") == 16
    assert arch.get("transformer.hidden_size") == 1792
    assert arch.get("transformer.num_heads") == 28
    assert arch.get("transformer.context_in_dim") == 2304
    assert arch.get("transformer.in_channels") == 16
    assert arch.get("transformer.patch_size") == 2
    assert arch.get("transformer.time_factor") == 1000.0
    # 16-channel Flux-style AutoencoderKL (8x)
    assert arch.get("vae.latent_channels") == 16
    assert arch.get("vae.vae_scale_factor") == 8
    # Scheduler facts (checkpoint scheduler_config.json): static shift 3.0,
    # NO dynamic shifting — the sampler must not compute mu.
    assert arch.get("scheduler.shift") == 3.0
    assert arch.get("scheduler.use_dynamic_shifting") is False
    # TE facts (checkpoint text_encoder/config.json — T5GemmaEncoder)
    assert arch.get("te.hidden_size") == 2304
    assert arch.get("te.num_hidden_layers") == 26
    assert arch.get("te.max_length") == 256


# ── Task 2: Loader Manifest ──────────────────────────────────────────────────


def test_manifest_components():
    """PRXLoader manifest declares all four components with correct classes."""
    import torch

    from app.engine.models.families.prx.loader import PRXLoader

    loader = PRXLoader(torch.device("cpu"))
    definition = _make_prx_definition()
    specs = loader.get_component_manifest(definition)

    keys = {s.key for s in specs}
    assert {"tokenizer", "text_encoder", "vae", "unet"} <= keys, (
        f"missing required manifest keys; got {keys}"
    )

    spec_map = {s.key: s for s in specs}

    # Tokenizer: AutoTokenizer (GemmaTokenizerFast resolves via tokenizer.json)
    assert "AutoTokenizer" in spec_map["tokenizer"].hf_class, (
        f"tokenizer hf_class wrong: {spec_map['tokenizer'].hf_class}"
    )
    assert spec_map["tokenizer"].is_torch_model is False

    # Text encoder: T5GemmaEncoder is NOT exported at transformers top level
    # (model_index.json even declares library "prx") — the manifest must use
    # the full module path so importlib can resolve it.
    assert spec_map["text_encoder"].hf_class == (
        "transformers.models.t5gemma.modeling_t5gemma.T5GemmaEncoder"
    ), f"text_encoder hf_class wrong: {spec_map['text_encoder'].hf_class}"
    assert spec_map["text_encoder"].subfolder == "text_encoder"

    # VAE: standard diffusers AutoencoderKL (Flux-style, 16ch)
    assert spec_map["vae"].hf_class == "diffusers.AutoencoderKL", (
        f"vae hf_class wrong: {spec_map['vae'].hf_class}"
    )

    # Transformer mapped to "unet" (repo convention), diffusers-native class
    assert "PRXTransformer2DModel" in spec_map["unet"].hf_class, (
        f"unet hf_class wrong: {spec_map['unet'].hf_class}"
    )
    assert spec_map["unet"].subfolder == "transformer"


def test_manifest_classes_are_importable():
    """Every hf_class in the manifest resolves through the generic loader's
    importlib seam (the T5GemmaEncoder full-path quirk is the reason this
    test exists)."""
    import torch

    from app.engine.core.pipeline.loader_base import GenericComponentLoader
    from app.engine.models.families.prx.loader import PRXLoader

    loader = PRXLoader(torch.device("cpu"))
    for spec in loader.get_component_manifest(_make_prx_definition()):
        cls = GenericComponentLoader._import_class(spec.hf_class)
        assert cls is not None, f"{spec.hf_class} not importable"


def test_loader_dtype_policy_matches_zimage():
    """Dtype policy is identical to zimage's (generic bf16 via driver)."""
    import torch

    from app.engine.core.pipeline.loader_base import GenericComponentLoader
    from app.engine.models.families.prx.loader import PRXLoader
    from app.engine.models.families.zimage.loader import ZImageLoader

    assert PRXLoader._resolve_dtype is GenericComponentLoader._resolve_dtype, (
        "PRXLoader must inherit the generic dtype policy (like zimage)"
    )
    assert ZImageLoader._resolve_dtype is GenericComponentLoader._resolve_dtype

    loader = PRXLoader(torch.device("cpu"))
    definition = _make_prx_definition()
    for spec in loader.get_component_manifest(definition):
        assert spec.dtype_override is None, (
            f"{spec.key} must not force a dtype override"
        )


# ── Task 3: prx_shared LoRA targets ──────────────────────────────────────────

# Tiny transformer config for CPU tests: sum(axes_dim) must equal
# head_dim = hidden_size / num_heads (32 / 2 = 16).
_TINY_CFG = dict(
    in_channels=4,
    patch_size=2,
    context_in_dim=8,
    hidden_size=32,
    num_heads=2,
    depth=1,
    axes_dim=[8, 8],
)


def _build_tiny_model():
    import torch  # noqa: PLC0415

    from diffusers.models.transformers.transformer_prx import (
        PRXTransformer2DModel,
    )

    torch.manual_seed(0)
    return PRXTransformer2DModel(**_TINY_CFG).eval()


def _make_driver(model=None, arch=None):
    import torch  # noqa: PLC0415

    from app.engine.models.families.prx.driver import PRXDriver

    definition = _make_prx_definition(architecture_params=arch or {})
    drv = PRXDriver(definition, torch.device("cpu"))
    drv.assign_components(
        {"unet": model, "vae": None, "text_encoder": None, "tokenizer": None},
    )
    return drv


def test_shared_lora_targets_are_fused_projections():
    """The shared target list is the FUSED-projection set — no to_q/to_k/to_v
    (PRX has none), no modulation.lin (zero-init adaLN stays frozen)."""
    from app.engine.models.families.prx_shared import (
        PRX_BLOCK_LORA_TARGETS,
        PRX_TARGETS_PER_BLOCK,
    )

    assert PRX_TARGETS_PER_BLOCK == 6
    assert set(PRX_BLOCK_LORA_TARGETS) == {
        "attention.img_qkv_proj",
        "attention.txt_kv_proj",
        "attention.to_out.0",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    for forbidden in ("to_q", "to_k", "to_v", "modulation.lin"):
        assert forbidden not in PRX_BLOCK_LORA_TARGETS


def test_shared_targets_match_tiny_model_and_fused_shapes():
    """Every pattern matches exactly one Linear per block on a real tiny
    PRXTransformer2DModel; fused shapes hold (img_qkv 3×hidden, txt_kv
    2×hidden); NO top-level module is swept in (exclusion-free contract)."""
    import torch  # noqa: PLC0415

    from app.engine.models.families.prx_shared import matching_linear_modules

    model = _build_tiny_model()
    hidden = _TINY_CFG["hidden_size"]

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

    modules = dict(model.named_modules())
    qkv = modules["blocks.0.attention.img_qkv_proj"]
    assert isinstance(qkv, torch.nn.Linear)
    assert qkv.out_features == 3 * hidden, "img_qkv_proj must be fused 3×hidden"
    kv = modules["blocks.0.attention.txt_kv_proj"]
    assert kv.out_features == 2 * hidden, "txt_kv_proj must be fused 2×hidden"
    out = modules["blocks.0.attention.to_out.0"]
    assert out.out_features == hidden


def test_driver_lora_targets_use_shared_list_and_need_no_excludes():
    """Driver defaults delegate to prx_shared; no exclude_modules needed
    (unlike ovis_image's top-level proj_out collision)."""
    from app.engine.models.families.prx_shared import get_prx_lora_targets

    drv = _make_driver(_build_tiny_model())
    assert drv.get_lora_targets() == get_prx_lora_targets()
    assert drv.get_lora_exclude_modules() is None


def test_driver_lora_targets_definition_override():
    """Definition-provided lora_targetable_modules win over the defaults."""
    import torch  # noqa: PLC0415

    from app.engine.models.families.prx.driver import PRXDriver

    definition = _make_prx_definition(
        lora_targetable_modules=["attention.img_qkv_proj"],
    )
    drv = PRXDriver(definition, torch.device("cpu"))
    assert drv.get_lora_targets() == ["attention.img_qkv_proj"]


# ── Task 4: Driver ───────────────────────────────────────────────────────────


def test_driver_forward_normalizes_timesteps_exactly_once():
    """forward_pass receives raw [0,1000] and hands t/1000 to the transformer.

    PRX convention: the ÷1000 happens in the driver BEFORE the forward (the
    model's time_factor=1000 re-scales internally). Any extra scaling
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

    B, C, H, W = 1, 4, 8, 8
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
    # Unpacked latents go straight in — patchify happens INSIDE the model.
    assert captured["hidden_states"] is noisy
    assert pred.shape == (B, C, H, W), f"unexpected shape: {pred.shape}"
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
    """Scheduler None (flow match), bf16 loading, no TE LoRA."""
    import torch  # noqa: PLC0415

    drv = _make_driver(None)
    assert drv.init_scheduler() is None
    assert drv.resolve_loading_dtype() == torch.bfloat16
    assert drv.get_te_lora_targets() == []


def _stub_tokenizer(full_len: int = 256, record: list | None = None):
    """Stub tokenizer satisfying the PRX tokenize contract."""
    import torch  # noqa: PLC0415

    tok = MagicMock()
    tok.model_max_length = full_len

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
    """Stub T5GemmaEncoder returning dict-style last_hidden_state."""
    import torch  # noqa: PLC0415

    te = MagicMock()

    def _fake_te(**kwargs):
        if record is not None:
            record.append(kwargs)
        return {"last_hidden_state": hidden}

    te.side_effect = _fake_te
    te.parameters = lambda: iter([torch.zeros(1)])
    return te


def test_driver_encode_text_replicates_pipeline_encode_prompt():
    """encode_text mirrors PRXPipeline._encode_prompt_standard:

    1. DeepFloyd-style cleaning (TextPreprocessor.clean_text) by default;
    2. tokenize padding='max_length', max_length=tokenizer.model_max_length
       (256), truncation=True;
    3. TE forward WITH output_hidden_states=True → ['last_hidden_state'];
    4. boolean attention mask returned alongside — NO zero-masking, NO
       slicing (unlike ovis_image).
    """
    import torch  # noqa: PLC0415

    from app.engine.models.families.prx.driver import PRXDriver

    B, D, full_len = 2, 16, 256
    tokenize_calls: list[dict] = []
    te_calls: list[dict] = []

    definition = _make_prx_definition(
        architecture_params={"te.max_length": 256},
    )
    drv = PRXDriver(definition, torch.device("cpu"))

    torch.manual_seed(3)
    hidden = torch.randn(B, full_len, D)
    drv.tokenizer = _stub_tokenizer(full_len, tokenize_calls)
    drv.text_encoder = _stub_te(hidden, te_calls)

    out = drv.encode_text(["A Photo of a CAT!", "Ein  HUND"], torch.float32)

    # 1. DeepFloyd cleaning applied (lowercased, whitespace collapsed)
    assert tokenize_calls[0]["texts"] == ["a photo of a cat!", "ein hund"]

    # 2. Tokenizer semantics
    tk = tokenize_calls[0]
    assert tk["padding"] == "max_length"
    assert tk["truncation"] is True
    assert tk["max_length"] == full_len
    assert tk["return_tensors"] == "pt"

    # 3. TE called with output_hidden_states=True and the BOOL mask
    assert te_calls[0]["output_hidden_states"] is True
    assert te_calls[0]["attention_mask"].dtype == torch.bool

    # 4. Un-sliced, un-masked last_hidden_state + bool mask
    assert out.embeddings.shape == (B, full_len, D)
    assert torch.allclose(out.embeddings, hidden)
    assert out.attention_mask is not None
    assert out.attention_mask.dtype == torch.bool
    assert out.attention_mask.shape == (B, full_len)
    assert out.attention_mask[:, -10:].sum() == 0


def test_driver_layer_manifest_single_block_stack():
    """Layer manifest exposes the single 16-block (here 1-block) stack."""
    model = _build_tiny_model()
    drv = _make_driver(model)
    manifest = drv.get_layer_manifest()

    assert len(manifest.transformer_blocks) == _TINY_CFG["depth"]
    assert manifest.transformer_blocks[0].name == "blocks.0"

    topo = drv.get_block_topology()
    assert len(topo) == 1
    assert topo[0]["attr_path"] == "blocks"
    assert topo[0]["count"] == _TINY_CFG["depth"]


# ── Task 5: Trainer override trio + TE cache ─────────────────────────────────


def _build_real_trainer_shell():
    """Minimal PRXTrainer shell with a REAL driver + tiny transformer.

    Binds the real trainer methods (encode seam) without calling setup().
    """
    import torch  # noqa: PLC0415

    from app.engine.models.families.prx.driver import PRXDriver
    from app.engine.models.families.prx.trainer import PRXTrainer

    definition = _make_prx_definition(
        architecture_params={"te.max_length": 256},
    )

    trainer = MagicMock(spec=PRXTrainer)
    trainer.device = torch.device("cpu")
    trainer.definition = definition
    trainer.config = {"cache_text_embeddings": False}
    trainer.text_cache = {}
    trainer.logger = MagicMock()

    drv = PRXDriver(definition, torch.device("cpu"))
    tiny_model = _build_tiny_model()
    torch.manual_seed(5)
    hidden = torch.randn(1, 256, 8)  # D == tiny context_in_dim
    drv.assign_components({
        "unet": tiny_model,
        "vae": None,
        "text_encoder": _stub_te(hidden),
        "tokenizer": _stub_tokenizer(256),
    })
    trainer.driver = drv

    trainer.encode_text = lambda captions, dtype, batch=None: (
        PRXTrainer.encode_text(trainer, captions, dtype, batch)
    )
    trainer._encode_text_direct = lambda captions, dtype: (
        PRXTrainer._encode_text_direct(trainer, captions, dtype)
    )
    trainer._get_cached_text_embeddings = lambda captions, dtype: (
        PRXTrainer._get_cached_text_embeddings(trainer, captions, dtype)
    )
    return trainer


def test_trainer_encode_to_forward_real_seam():
    """C1/C2: trainer.encode_text returns a (emb, mask) TUPLE consumable by
    driver.forward_pass — the whole encode→forward round trip produces a
    finite [B, C, H, W] prediction."""
    import torch  # noqa: PLC0415

    trainer = _build_real_trainer_shell()

    text_emb = trainer.encode_text(["a prx test caption"], torch.float32)
    assert isinstance(text_emb, tuple) and len(text_emb) == 2, (
        f"encode_text must return a 2-tuple, got {type(text_emb)}"
    )
    emb, mask = text_emb
    assert emb.ndim == 3, f"embeddings must be 3-D [B,L,D], got {emb.ndim}-D"
    assert emb.shape[1] == 256, f"seq len must be 256, got {emb.shape[1]}"
    assert mask.ndim == 2 and mask.shape[1] == 256
    assert mask.dtype == torch.bool

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

    from app.engine.models.families.prx.driver import PRXDriver
    from app.engine.models.families.prx.trainer import PRXTrainer

    definition = _make_prx_definition()
    trainer = MagicMock(spec=PRXTrainer)
    trainer.device = torch.device("cpu")
    trainer.definition = definition

    drv = PRXDriver(definition, torch.device("cpu"))
    tiny_model = _build_tiny_model()
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
    PRXTrainer._update_primary_model(trainer, peft_wrapped)

    assert trainer.driver.model is peft_wrapped, (
        "C3: driver.model was NOT updated after _update_primary_model"
    )
    assert trainer.components.get("unet") is peft_wrapped
    assert trainer.model is peft_wrapped

    transformer_val = PRXTrainer.transformer.fget(trainer)
    assert transformer_val is peft_wrapped, (
        "C4: trainer.transformer property must resolve to the wrapped model"
    )


def test_trainer_te_disk_cache_layout(tmp_path):
    """TE disk cache lands in embeddings/{te_quant}/te1|te2 (emb + mask)."""
    import torch  # noqa: PLC0415

    from app.engine.components.text_embeddings import TextEmbeddingCache
    from app.engine.models.families.prx.trainer import PRXTrainer

    trainer = MagicMock(spec=PRXTrainer)
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
    trainer._build_caption_hints.return_value = {"a prx caption": "hint0"}
    trainer._resolve_te_cache_dirs.return_value = [str(tmp_path)]
    trainer._resolve_loading_dtype.return_value = torch.float32
    trainer._sample_prompt_texts.side_effect = lambda: (
        PRXTrainer._sample_prompt_texts(trainer)
    )

    def _fake_encode(captions, dtype):
        b = len(captions)
        return (
            torch.zeros(b, 256, 16),
            torch.ones(b, 256, dtype=torch.bool),
        )

    trainer._encode_text_direct = _fake_encode

    PRXTrainer._pre_cache_text_embeddings(trainer)

    te1 = tmp_path / "embeddings" / "none" / "te1"
    te2 = tmp_path / "embeddings" / "none" / "te2"
    assert te1.is_dir(), "te1 (embeddings) cache dir missing"
    assert te2.is_dir(), "te2 (attention mask) cache dir missing"

    emb = TextEmbeddingCache.load("a prx caption", str(te1), "hint0")
    mask = TextEmbeddingCache.load("a prx caption", str(te2), "hint0")
    assert emb is not None and emb.shape == (256, 16)
    assert mask is not None and mask.shape == (256,)
    # In-memory cache holds the (emb, mask) tuple
    assert "a prx caption" in trainer.text_cache
    cached_emb, cached_mask = trainer.text_cache["a prx caption"]
    assert cached_emb.shape == (256, 16)
    assert cached_mask.shape == (256,)


def test_trainer_warms_sample_and_negative_prompts():
    """Pre-cache also warms expanded sample + negative prompts so the TE can
    stay offloaded during sampling (krea2 VRAM-spike lesson)."""
    import torch  # noqa: PLC0415

    from app.engine.models.families.prx.trainer import PRXTrainer

    trainer = MagicMock(spec=PRXTrainer)
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
        PRXTrainer._sample_prompt_texts(trainer)
    )

    def _fake_encode(captions, dtype):
        b = len(captions)
        return (
            torch.zeros(b, 256, 16),
            torch.ones(b, 256, dtype=torch.bool),
        )

    trainer._encode_text_direct = _fake_encode

    PRXTrainer._pre_cache_text_embeddings(trainer)

    assert "a red car" in trainer.text_cache
    assert "blurry" in trainer.text_cache
