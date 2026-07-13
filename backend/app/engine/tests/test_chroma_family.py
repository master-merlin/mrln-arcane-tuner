"""Tests for the chroma family (diffusers-0.39-native Chroma1-Base/HD).

TDD order (mirrors test_ovis_image_family.py):
  Task 1: family registration + definition loading (both chroma1-base and
          chroma1-hd)
  Task 2: loader manifest (component specs + dtype policy)
  Task 3: driver (LoRA targets, forward_pass timestep scale + attention-mask
          extension, compute_target, encode_text replicating
          ChromaPipeline._get_t5_prompt_embeds incl. the padding foot-gun)
  Task 4: trainer override trio (encode_text tuple, _update_primary_model
          driver sync) + TE disk-cache layout
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from app.engine.core.definitions import ModelDefinition


@pytest.fixture(autouse=True)
def _restore_model_registry():
    """Snapshot + restore ``ModelRegistry`` class state around every test."""
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


def _make_chroma_definition(**kwargs) -> MagicMock:
    """Build a mock Chroma ModelDefinition for loader/driver tests."""
    definition = MagicMock(spec=ModelDefinition)
    definition.family = "chroma"
    definition.id = kwargs.get("id", "chroma-test")
    definition.components = {}
    definition.architecture_params = kwargs.get("architecture_params", {})
    definition.lora_targetable_modules = kwargs.get("lora_targetable_modules", [])
    return definition


# ── Task 1: Family Registration ──────────────────────────────────────────────


def test_family_registered():
    """chroma family must appear in ModelRegistry with the correct archetype."""
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()

    fam = ModelRegistry._families.get("chroma")
    assert fam is not None, "chroma family not registered"
    assert fam.archetype == "latent_diffusion", (
        f"expected archetype='latent_diffusion', got {fam.archetype!r}"
    )


def _reload_definitions():
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()
    return ModelRegistry


def test_definition_loaded_chroma1_hd():
    """chroma1-hd definition must load from its YAML file with verified facts."""
    ModelRegistry = _reload_definitions()

    fam_defs = {
        d.id: d for d in ModelRegistry._definitions.values() if d.family == "chroma"
    }
    assert "chroma1-hd" in fam_defs, f"missing chroma1-hd; found: {set(fam_defs)}"

    hd = fam_defs["chroma1-hd"]
    assert hd.components["repo"].path == "huggingface:lodestones/Chroma1-HD"
    assert hd.control_inputs == 0

    arch = hd.architecture_params
    assert arch.get("transformer.num_layers") == 19
    assert arch.get("transformer.num_single_layers") == 38
    assert arch.get("transformer.num_attention_heads") == 24
    assert arch.get("transformer.attention_head_dim") == 128
    assert arch.get("transformer.joint_attention_dim") == 4096
    assert arch.get("transformer.in_channels") == 64
    assert arch.get("transformer.patch_size") == 1
    assert arch.get("vae.latent_channels") == 16
    # chroma1-hd: static shift, NOT dynamic (verified checkpoint fact).
    assert arch.get("scheduler.use_dynamic_shifting") is False
    assert arch.get("scheduler.shift") == 3.0
    assert arch.get("scheduler.use_beta_sigmas") is False
    assert arch.get("te.t5_max_length") == 512


def test_definition_loaded_chroma1_base():
    """chroma1-base definition must load with its own (sparser) scheduler."""
    ModelRegistry = _reload_definitions()

    fam_defs = {
        d.id: d for d in ModelRegistry._definitions.values() if d.family == "chroma"
    }
    assert "chroma1-base" in fam_defs, f"missing chroma1-base; found: {set(fam_defs)}"

    base = fam_defs["chroma1-base"]
    assert base.components["repo"].path == "huggingface:lodestones/Chroma1-Base"
    assert base.control_inputs == 0

    arch = base.architecture_params
    assert arch.get("transformer.num_layers") == 19
    assert arch.get("transformer.num_single_layers") == 38
    # chroma1-base: beta-sigma resampling, static shift == 1.0 (class default,
    # since the checkpoint's own scheduler_config.json only sets
    # num_train_timesteps + use_beta_sigmas).
    assert arch.get("scheduler.use_dynamic_shifting") is False
    assert arch.get("scheduler.use_beta_sigmas") is True
    assert arch.get("scheduler.shift") == 1.0


def test_both_definitions_share_identical_transformer_and_vae_facts():
    """Both checkpoints are the SAME architecture — only weights differ."""
    ModelRegistry = _reload_definitions()

    hd = ModelRegistry._definitions["chroma1-hd"]
    base = ModelRegistry._definitions["chroma1-base"]

    for key in (
        "transformer.num_layers", "transformer.num_single_layers",
        "transformer.attention_head_dim", "transformer.num_attention_heads",
        "transformer.joint_attention_dim", "transformer.axes_dims_rope",
        "vae.latent_channels", "vae.scaling_factor", "vae.shift_factor",
        "te.d_model", "te.num_layers",
    ):
        assert hd.architecture_params.get(key) == base.architecture_params.get(key), (
            f"{key} diverges between chroma1-hd and chroma1-base"
        )


# ── Task 2: Loader Manifest ──────────────────────────────────────────────────


def test_manifest_components():
    """ChromaLoader manifest declares all four diffusers-native components."""
    from app.engine.models.families.chroma.loader import ChromaLoader

    loader = ChromaLoader(torch.device("cpu"))
    definition = _make_chroma_definition()
    specs = loader.get_component_manifest(definition)

    keys = {s.key for s in specs}
    assert {"tokenizer", "text_encoder", "vae", "unet"} <= keys, (
        f"missing required manifest keys; got {keys}"
    )

    spec_map = {s.key: s for s in specs}

    # Tokenizer: SLOW T5Tokenizer (checkpoint ships no tokenizer.json)
    assert spec_map["tokenizer"].hf_class == "transformers.T5Tokenizer", (
        f"tokenizer hf_class wrong: {spec_map['tokenizer'].hf_class}"
    )
    assert spec_map["tokenizer"].is_torch_model is False

    # Text encoder: T5EncoderModel, the ONLY encoder (no CLIP anywhere)
    assert spec_map["text_encoder"].hf_class == "transformers.T5EncoderModel", (
        f"text_encoder hf_class wrong: {spec_map['text_encoder'].hf_class}"
    )
    assert spec_map["text_encoder"].subfolder == "text_encoder"
    assert "tokenizer_2" not in keys and "text_encoder_2" not in keys, (
        "chroma must have NO CLIP path anywhere"
    )

    # VAE: standard diffusers AutoencoderKL
    assert spec_map["vae"].hf_class == "diffusers.AutoencoderKL"

    # Transformer mapped to "unet", diffusers-native class
    assert spec_map["unet"].hf_class == "diffusers.ChromaTransformer2DModel", (
        f"unet hf_class wrong: {spec_map['unet'].hf_class}"
    )
    assert spec_map["unet"].subfolder == "transformer"


def test_loader_dtype_policy_matches_flux1():
    """Dtype policy is generic (no per-component override), like flux1's."""
    import torch as _torch

    from app.engine.core.pipeline.loader_base import GenericComponentLoader
    from app.engine.models.families.chroma.loader import ChromaLoader

    assert ChromaLoader._resolve_dtype is GenericComponentLoader._resolve_dtype

    loader = ChromaLoader(_torch.device("cpu"))
    definition = _make_chroma_definition()
    for spec in loader.get_component_manifest(definition):
        assert spec.dtype_override is None, (
            f"{spec.key} must not force a dtype override"
        )


# ── Task 3: Driver ───────────────────────────────────────────────────────────

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
    approximator_num_channels=8,
    approximator_hidden_dim=16,
    approximator_layers=1,
)


def _build_tiny_model():
    from diffusers.models.transformers.transformer_chroma import (
        ChromaTransformer2DModel,
    )

    torch.manual_seed(0)
    return ChromaTransformer2DModel(**_TINY_CFG).eval()


def _make_driver(model=None, arch=None):
    from app.engine.models.families.chroma.driver import ChromaDriver

    definition = _make_chroma_definition(architecture_params=arch or {})
    drv = ChromaDriver(definition, torch.device("cpu"))
    drv.assign_components(
        {"unet": model, "vae": None, "text_encoder": None, "tokenizer": None},
    )
    return drv


def test_driver_lora_targets_cover_double_and_single_blocks():
    """Every default LoRA target pattern matches >=1 module in BOTH streams
    (Chroma's blocks reuse FluxAttention/FluxAttnProcessor verbatim)."""
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

    double = {t for t in targets if any(
        n.startswith("transformer_blocks.") and (n == t or n.endswith("." + t))
        for n in linear_names
    )}
    for expected in (
        "to_q", "to_k", "to_v", "to_out.0",
        "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
        "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
    ):
        assert expected in double, f"double-block target {expected!r} missing"

    single = {t for t in targets if any(
        n.startswith("single_transformer_blocks.")
        and (n == t or n.endswith("." + t))
        for n in linear_names
    )}
    for expected in ("to_q", "to_k", "to_v", "proj_mlp", "proj_out"):
        assert expected in single, f"single-block target {expected!r} missing"


def test_driver_no_te_lora_and_no_exclude():
    """T5 stays frozen; no top-level proj_out exclusion (flux1 parity, NOT
    ovis_image's curated-exclude approach — see driver.py docstring)."""
    drv = _make_driver(None)
    assert drv.get_te_lora_targets() == []
    assert drv.get_lora_exclude_modules() is None, (
        "ChromaDriver must not override get_lora_exclude_modules (flux1 parity)"
    )


def test_driver_basic_contracts():
    """Scheduler None (flow match), bf16 loading."""
    drv = _make_driver(None)
    assert drv.init_scheduler() is None
    assert drv.resolve_loading_dtype() == torch.bfloat16


def test_driver_compute_target_is_flow_match_default():
    """compute_target = noise - latents (IModelDriver's own flow-match
    default; ChromaDriver deliberately does not override it)."""
    drv = _make_driver(None)
    latents = torch.randn(2, 64, 4, 4)
    noise = torch.randn(2, 64, 4, 4)
    target = drv.compute_target(latents, noise, torch.tensor([500.0, 250.0]))
    assert torch.equal(target, noise - latents)


def test_driver_forward_divides_timesteps_by_1000_exactly_once():
    """forward_pass receives raw [0,1000] and hands t/1000 to the transformer.

    The transformer multiplies by 1000 internally (transformer_chroma.py
    line 528), so any extra scaling silently produces pure-noise LoRAs
    (flow-match timestep-scale gotcha).
    """
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
    mask = torch.ones(B, 5)

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

    # txt_ids per pipeline_chroma.py line 324: ALL-ZERO (Flux convention,
    # NOT Ovis's arange trick).
    txt_ids = captured["txt_ids"]
    assert txt_ids.shape == (5, 3)
    assert torch.equal(txt_ids, torch.zeros(5, 3, dtype=txt_ids.dtype))

    # attention_mask extended with all-ones image tokens (packed len = 4).
    full_mask = captured["attention_mask"]
    assert full_mask.shape == (B, 5 + 4)
    assert torch.equal(full_mask[:, :5], mask)
    assert torch.equal(full_mask[:, 5:], torch.ones(B, 4))

    # guidance kwarg must NOT be passed at all — Chroma's transformer takes
    # no guidance input (unlike FluxTransformer2DModel).
    assert "guidance" not in captured


def test_driver_forward_handles_missing_mask():
    """A mask-less (plain tensor) text_embeddings input must not crash and
    must pass attention_mask=None through to the transformer."""
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

    with torch.no_grad():
        pred = drv.forward_pass(
            noisy_input=noisy,
            timesteps=torch.tensor([500.0]),
            text_embeddings=emb,
            batch={},
        )
    assert pred.isfinite().all()
    assert captured["attention_mask"] is None


def test_driver_encode_text_replicates_pipeline_padding_foot_gun():
    """encode_text mirrors ChromaPipeline._get_t5_prompt_embeds:

    1. tokenized padding='max_length', max_length=te_t5_max_length,
       truncation=True;
    2. T5 forward WITH the raw tokenizer attention_mask (Chroma diverges
       from FLUX here);
    3. the RETURNED mask is the MODIFIED one — `mask_indices <= seq_lengths`
       (note <=, not <): exactly ONE padding position past the real content
       survives unmasked. This is the documented Chroma quality foot-gun.
    """
    from app.engine.models.families.chroma.driver import ChromaDriver

    definition = _make_chroma_definition(
        architecture_params={"te.t5_max_length": 10},
    )
    drv = ChromaDriver(definition, torch.device("cpu"))

    B, D, L = 2, 16, 10
    real_lengths = [4, 7]  # caption 0 has 4 real tokens, caption 1 has 7

    tok = MagicMock()

    def _fake_tokenize(texts, **kwargs):
        n = len(texts)
        out = MagicMock()
        out.input_ids = torch.zeros(n, L, dtype=torch.long)
        mask = torch.zeros(n, L, dtype=torch.long)
        for i in range(n):
            mask[i, : real_lengths[i]] = 1
        out.attention_mask = mask
        return out

    tok.side_effect = _fake_tokenize
    drv.tokenizer = tok

    te = MagicMock()
    torch.manual_seed(2)
    hidden = torch.randn(B, L, D)
    seen_attention_masks = []

    def _fake_te(input_ids, output_hidden_states=None, attention_mask=None):
        seen_attention_masks.append(attention_mask.clone())
        return (hidden,)

    te.side_effect = _fake_te
    drv.text_encoder = te

    out = drv.encode_text(["a fox", "a longer caption here"], torch.float32)

    # 2. T5 forward received the RAW tokenizer mask (not the modified one).
    assert torch.equal(
        seen_attention_masks[0],
        torch.stack([
            torch.tensor([1] * 4 + [0] * 6),
            torch.tensor([1] * 7 + [0] * 3),
        ]),
    )

    # 3. Returned mask keeps exactly ONE extra position unmasked past the
    # real content (`<=`, not `<`): positions 0..real_len are 1, rest 0.
    assert out.embeddings.shape == (B, L, D)
    assert out.attention_mask.shape == (B, L)
    expected0 = torch.tensor([1.0] * (real_lengths[0] + 1) + [0.0] * (L - real_lengths[0] - 1))
    expected1 = torch.tensor([1.0] * (real_lengths[1] + 1) + [0.0] * (L - real_lengths[1] - 1))
    assert torch.allclose(out.attention_mask[0], expected0), (
        f"caption 0 mask should unmask exactly one padding token past real "
        f"content: got {out.attention_mask[0]}"
    )
    assert torch.allclose(out.attention_mask[1], expected1)
    assert torch.equal(out.embeddings, hidden)


# ── Task 4: Trainer override trio + TE cache ─────────────────────────────────


def test_trainer_update_primary_model_syncs_driver_transformer():
    """_update_primary_model must sync self.transformer, self.components,
    AND driver.transformer (flux1 pattern, since ChromaDriver stores its
    primary model on `.transformer`, not `.model` like ovis_image)."""
    import torch.nn as nn

    from app.engine.models.families.chroma.driver import ChromaDriver
    from app.engine.models.families.chroma.trainer import ChromaTrainer

    definition = _make_chroma_definition()
    trainer = MagicMock(spec=ChromaTrainer)
    trainer.driver = ChromaDriver(definition, torch.device("cpu"))
    trainer.components = {"unet": None}
    trainer.transformer = None

    class _FakePEFT(nn.Module):
        pass

    peft_wrapped = _FakePEFT()
    ChromaTrainer._update_primary_model(trainer, peft_wrapped)

    assert trainer.transformer is peft_wrapped
    assert trainer.components.get("unet") is peft_wrapped
    assert trainer.driver.transformer is peft_wrapped, (
        "driver.transformer was NOT synced after _update_primary_model"
    )


def _stub_tokenizer_and_te(D: int = 16, L: int = 8):
    tok = MagicMock()

    def _fake_tokenize(texts, **kwargs):
        n = len(texts)
        out = MagicMock()
        out.input_ids = torch.zeros(n, L, dtype=torch.long)
        out.attention_mask = torch.ones(n, L, dtype=torch.long)
        return out

    tok.side_effect = _fake_tokenize

    te = MagicMock()

    def _fake_te(input_ids, output_hidden_states=None, attention_mask=None):
        b, seq = input_ids.shape
        return (torch.randn(b, seq, D),)

    te.side_effect = _fake_te
    te.parameters = lambda: iter([torch.zeros(1)])
    return tok, te


def _build_real_trainer_shell():
    from app.engine.models.families.chroma.driver import ChromaDriver
    from app.engine.models.families.chroma.trainer import ChromaTrainer

    definition = _make_chroma_definition(
        architecture_params={"te.t5_max_length": 8},
    )

    trainer = MagicMock(spec=ChromaTrainer)
    trainer.device = torch.device("cpu")
    trainer.definition = definition
    trainer.config = {"cache_text_embeddings": False}
    trainer.text_cache = {}
    trainer.logger = MagicMock()

    drv = ChromaDriver(definition, torch.device("cpu"))
    tiny_model = _build_tiny_model()
    tok, te = _stub_tokenizer_and_te(D=16, L=8)
    drv.assign_components({
        "unet": tiny_model, "vae": None, "text_encoder": te, "tokenizer": tok,
    })
    trainer.driver = drv

    trainer.encode_text = lambda captions, dtype, batch=None: (
        ChromaTrainer.encode_text(trainer, captions, dtype, batch)
    )
    trainer._encode_text_direct = lambda captions, dtype: (
        ChromaTrainer._encode_text_direct(trainer, captions, dtype)
    )
    trainer._get_cached_text_embeddings = lambda captions, dtype: (
        ChromaTrainer._get_cached_text_embeddings(trainer, captions, dtype)
    )
    return trainer


def test_trainer_encode_to_forward_real_seam():
    """encode_text returns a (emb, mask) TUPLE consumable by
    driver.forward_pass — full encode->forward round trip is finite."""
    trainer = _build_real_trainer_shell()

    text_emb = trainer.encode_text(["a chroma test caption"], torch.float32)
    assert isinstance(text_emb, tuple) and len(text_emb) == 2
    emb, mask = text_emb
    assert emb.ndim == 3 and emb.shape[1] == 8
    assert mask.ndim == 2 and mask.shape[1] == 8

    B, C, H, W = 1, 16, 4, 4
    with torch.no_grad():
        pred = trainer.driver.forward_pass(
            noisy_input=torch.randn(B, C, H, W),
            timesteps=torch.tensor([500.0]),
            text_embeddings=text_emb,
            batch={},
        )
    assert pred.shape == (B, C, H, W)
    assert pred.isfinite().all()


def test_trainer_cached_encode_returns_batched_tuple():
    """Cached path stacks per-caption entries back to ([B,L,D], [B,L])."""
    trainer = _build_real_trainer_shell()
    trainer.config = {"cache_text_embeddings": True}
    trainer.text_encoder = trainer.driver.text_encoder

    out = trainer.encode_text(["cap one", "cap two"], torch.float32)
    assert isinstance(out, tuple) and len(out) == 2
    emb, mask = out
    assert emb.shape[0] == 2 and emb.ndim == 3
    assert mask.shape[0] == 2 and mask.ndim == 2
    assert set(trainer.text_cache) == {"cap one", "cap two"}
    cached_emb, cached_mask = trainer.text_cache["cap one"]
    assert cached_emb.ndim == 2 and cached_mask.ndim == 1


def test_trainer_te_disk_cache_layout(tmp_path):
    """TE disk cache lands in embeddings/{te_quant}/te1|te2 (emb + mask)."""
    from app.engine.components.text_embeddings import TextEmbeddingCache
    from app.engine.models.families.chroma.trainer import ChromaTrainer

    trainer = MagicMock(spec=ChromaTrainer)
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
    trainer._build_caption_hints.return_value = {"a chroma caption": "hint0"}
    trainer._resolve_te_cache_dirs.return_value = [str(tmp_path)]
    trainer._resolve_loading_dtype.return_value = torch.float32
    trainer._sample_prompt_texts.side_effect = lambda: (
        ChromaTrainer._sample_prompt_texts(trainer)
    )

    def _fake_encode(captions, dtype):
        b = len(captions)
        return torch.zeros(b, 8, 16), torch.ones(b, 8)

    trainer._encode_text_direct = _fake_encode

    ChromaTrainer._pre_cache_text_embeddings(trainer)

    te1 = tmp_path / "embeddings" / "none" / "te1"
    te2 = tmp_path / "embeddings" / "none" / "te2"
    assert te1.is_dir()
    assert te2.is_dir()

    emb = TextEmbeddingCache.load("a chroma caption", str(te1), "hint0")
    mask = TextEmbeddingCache.load("a chroma caption", str(te2), "hint0")
    assert emb is not None and emb.shape == (8, 16)
    assert mask is not None and mask.shape == (8,)
    cached_emb, cached_mask = trainer.text_cache["a chroma caption"]
    assert cached_emb.shape == (8, 16)
    assert cached_mask.shape == (8,)
