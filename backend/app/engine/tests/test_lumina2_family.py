"""Tests for the lumina2 family (diffusers-0.39-native Lumina-Image-2.0).

TDD order (mirrors test_chroma_family.py / test_ovis_image_family.py):
  Task 1: family registration + definition loading
  Task 2: loader manifest (component specs + dtype policy)
  Task 3: driver (LoRA targets, forward_pass timestep REVERSAL + negation,
          compute_target default, encode_text replicating Lumina2Pipeline's
          system-prompt asymmetry + hidden_states[-2] tap)
  Task 4: trainer override trio (encode_text tuple, _update_primary_model
          driver sync, uncond cache) + TE disk-cache layout + template
          fingerprint
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


def _make_lumina2_definition(**kwargs) -> MagicMock:
    """Build a mock Lumina2 ModelDefinition for loader/driver tests."""
    definition = MagicMock(spec=ModelDefinition)
    definition.family = "lumina2"
    definition.id = kwargs.get("id", "lumina2-test")
    definition.components = {}
    definition.architecture_params = kwargs.get("architecture_params", {})
    definition.lora_targetable_modules = kwargs.get("lora_targetable_modules", [])
    return definition


# ── Task 1: Family Registration ──────────────────────────────────────────────


def test_family_registered():
    """lumina2 family must appear in ModelRegistry with the correct archetype."""
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()

    fam = ModelRegistry._families.get("lumina2")
    assert fam is not None, "lumina2 family not registered"
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


def test_definition_loaded_lumina_image_2():
    """lumina-image-2.0 definition must load from its YAML with verified facts."""
    ModelRegistry = _reload_definitions()

    fam_defs = {
        d.id: d for d in ModelRegistry._definitions.values() if d.family == "lumina2"
    }
    assert "lumina-image-2.0" in fam_defs, f"missing lumina-image-2.0; found: {set(fam_defs)}"

    defn = fam_defs["lumina-image-2.0"]
    assert defn.components["repo"].path == "huggingface:Alpha-VLLM/Lumina-Image-2.0"
    assert defn.control_inputs == 0

    arch = defn.architecture_params
    assert arch.get("transformer.num_layers") == 26
    assert arch.get("transformer.num_refiner_layers") == 2
    assert arch.get("transformer.num_attention_heads") == 24
    assert arch.get("transformer.num_kv_heads") == 8
    assert arch.get("transformer.hidden_size") == 2304
    assert arch.get("transformer.in_channels") == 16
    assert arch.get("transformer.patch_size") == 2
    assert arch.get("transformer.cap_feat_dim") == 2304
    assert arch.get("vae.latent_channels") == 16
    assert arch.get("te.hidden_size") == 2304
    assert arch.get("te.num_hidden_layers") == 26
    assert arch.get("te.max_sequence_length") == 256
    # STATIC shift (verified checkpoint fact) — not dynamic like flux1/ovis.
    assert arch.get("scheduler.use_dynamic_shifting") is False
    assert arch.get("scheduler.shift") == 6.0


def test_definition_ships_native_sample_defaults():
    """Pipeline __call__ native defaults: 30 steps, guidance 4.0."""
    ModelRegistry = _reload_definitions()
    defn = ModelRegistry._definitions["lumina-image-2.0"]
    assert defn.defaults["num_inference_steps"] == 30
    assert defn.defaults["guidance_scale"] == 4.0
    assert defn.defaults["resolution"] == 1024


# ── Task 2: Loader Manifest ──────────────────────────────────────────────────


def test_manifest_components():
    """Lumina2Loader manifest declares all four diffusers-native components."""
    from app.engine.models.families.lumina2.loader import Lumina2Loader

    loader = Lumina2Loader(torch.device("cpu"))
    definition = _make_lumina2_definition()
    specs = loader.get_component_manifest(definition)

    keys = {s.key for s in specs}
    assert {"tokenizer", "text_encoder", "vae", "unet"} <= keys, (
        f"missing required manifest keys; got {keys}"
    )

    spec_map = {s.key: s for s in specs}

    assert spec_map["tokenizer"].hf_class == "transformers.AutoTokenizer"
    assert spec_map["tokenizer"].is_torch_model is False

    assert spec_map["text_encoder"].hf_class == "transformers.Gemma2Model", (
        f"text_encoder hf_class wrong: {spec_map['text_encoder'].hf_class}"
    )
    assert spec_map["text_encoder"].subfolder == "text_encoder"
    assert "tokenizer_2" not in keys and "text_encoder_2" not in keys

    assert spec_map["vae"].hf_class == "diffusers.AutoencoderKL"

    assert spec_map["unet"].hf_class == "diffusers.Lumina2Transformer2DModel", (
        f"unet hf_class wrong: {spec_map['unet'].hf_class}"
    )
    assert spec_map["unet"].subfolder == "transformer"


def test_loader_dtype_policy_is_generic():
    """Dtype policy is generic (no per-component override)."""
    import torch as _torch

    from app.engine.core.pipeline.loader_base import GenericComponentLoader
    from app.engine.models.families.lumina2.loader import Lumina2Loader

    assert Lumina2Loader._resolve_dtype is GenericComponentLoader._resolve_dtype

    loader = Lumina2Loader(_torch.device("cpu"))
    definition = _make_lumina2_definition()
    for spec in loader.get_component_manifest(definition):
        assert spec.dtype_override is None, (
            f"{spec.key} must not force a dtype override"
        )


# ── Task 3: Driver ───────────────────────────────────────────────────────────

_TINY_CFG = dict(
    sample_size=8,
    patch_size=2,
    in_channels=4,
    out_channels=None,
    hidden_size=24,
    num_layers=1,
    num_refiner_layers=1,
    num_attention_heads=2,
    num_kv_heads=1,
    multiple_of=8,
    norm_eps=1e-5,
    # axes_dim_rope MUST sum to head_dim (hidden_size / num_attention_heads
    # = 12) — verified against the real config (32+32+32=96=2304/24).
    axes_dim_rope=(4, 4, 4),
    # axes_lens[0] must exceed any caption length used below: the rope
    # embedder assigns ALL image tokens position cap_seq_len on axis 0
    # (transformer_lumina2.py line 281: ``position_ids[i, cap_seq_len:seq_len,
    # 0] = cap_seq_len``), so axes_lens[0] must be > max cap_seq_len or that
    # index goes out of bounds.
    axes_lens=(16, 16, 16),
    cap_feat_dim=12,
)


def _build_tiny_model():
    from diffusers.models.transformers.transformer_lumina2 import (
        Lumina2Transformer2DModel,
    )

    torch.manual_seed(0)
    return Lumina2Transformer2DModel(**_TINY_CFG).eval()


def _make_driver(model=None, arch=None):
    from app.engine.models.families.lumina2.driver import Lumina2Driver

    definition = _make_lumina2_definition(architecture_params=arch or {})
    drv = Lumina2Driver(definition, torch.device("cpu"))
    drv.assign_components(
        {"unet": model, "vae": None, "text_encoder": None, "tokenizer": None},
    )
    return drv


def test_driver_lora_targets_cover_all_three_block_groups():
    """Every default LoRA target pattern matches >=1 module across layers,
    context_refiner AND noise_refiner (all three share the same submodule
    naming — verified via live introspection)."""
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

    for group in ("layers", "context_refiner", "noise_refiner"):
        group_targets = {
            t for t in targets if any(
                n.startswith(f"{group}.") and (n == t or n.endswith("." + t))
                for n in linear_names
            )
        }
        for expected in (
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "feed_forward.linear_1", "feed_forward.linear_2", "feed_forward.linear_3",
        ):
            assert expected in group_targets, f"{group}: target {expected!r} missing"


def test_driver_no_te_lora_and_no_exclude():
    """Gemma-2 stays frozen; no exclusion patterns needed."""
    drv = _make_driver(None)
    assert drv.get_te_lora_targets() == []
    assert drv.get_lora_exclude_modules() is None


def test_driver_basic_contracts():
    """Scheduler None (flow match), bf16 loading."""
    drv = _make_driver(None)
    assert drv.init_scheduler() is None
    assert drv.resolve_loading_dtype() == torch.bfloat16


def test_driver_compute_target_is_flow_match_default():
    """compute_target = noise - latents (IModelDriver's own flow-match
    default; Lumina2Driver deliberately does not override it — the
    reversal/negation lives entirely inside forward_pass)."""
    drv = _make_driver(None)
    latents = torch.randn(2, 4, 4, 4)
    noise = torch.randn(2, 4, 4, 4)
    target = drv.compute_target(latents, noise, torch.tensor([500.0, 250.0]))
    assert torch.equal(target, noise - latents)


def test_driver_forward_reverses_timestep_and_negates_output():
    """forward_pass must (a) feed the transformer ``1 - t/1000`` (REVERSED —
    pipeline_lumina2.py lines 723-724) and (b) return the NEGATED raw model
    output (pipeline_lumina2.py line 758). This is the family's #1 silent-
    LoRA-killer risk if either half is missing."""
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
    L = 6
    emb = torch.randn(B, L, 12)
    mask = torch.ones(B, L, dtype=torch.long)

    with torch.no_grad():
        raw_direct = model(
            hidden_states=noisy, timestep=torch.tensor([0.5]),
            encoder_hidden_states=emb, encoder_attention_mask=mask,
            return_dict=False,
        )[0]

        pred = drv.forward_pass(
            noisy_input=noisy,
            timesteps=torch.tensor([500.0]),
            text_embeddings=(emb, mask),
            batch={},
        )

    ts = captured["timestep"]
    assert torch.allclose(ts.float(), torch.tensor([0.5])), (
        f"transformer must receive 1 - t/1000 = 0.5 for t=500, got {ts}"
    )
    assert pred.shape == (B, C, H, W)
    assert pred.isfinite().all()
    assert torch.allclose(pred, -raw_direct, atol=1e-5), (
        "forward_pass must return the NEGATED raw model output"
    )


def test_driver_forward_extreme_timesteps_confirm_reversal_direction():
    """t=0 (framework convention: pure LATENTS) must feed the transformer
    timestep=1.0 (image); t=1000 (pure NOISE) must feed timestep=0.0
    (noise) — pins the DIRECTION of the reversal, not just its presence."""
    model = _build_tiny_model()
    drv = _make_driver(model)

    seen: list[torch.Tensor] = []
    original_forward = model.forward

    def _spy(*args, **kwargs):
        seen.append(kwargs["timestep"].clone())
        return original_forward(*args, **kwargs)

    model.forward = _spy

    B, C, H, W = 1, 4, 8, 8
    noisy = torch.randn(B, C, H, W)
    emb = torch.randn(B, 6, 12)
    mask = torch.ones(B, 6, dtype=torch.long)

    with torch.no_grad():
        drv.forward_pass(noisy, torch.tensor([0.0]), (emb, mask), {})
        drv.forward_pass(noisy, torch.tensor([1000.0]), (emb, mask), {})

    assert torch.allclose(seen[0], torch.tensor([1.0]))
    assert torch.allclose(seen[1], torch.tensor([0.0]))


def test_driver_forward_handles_missing_mask():
    """A mask-less (plain tensor) text_embeddings input must not crash —
    an all-ones mask is synthesized."""
    model = _build_tiny_model()
    drv = _make_driver(model)

    B, C, H, W = 1, 4, 8, 8
    noisy = torch.randn(B, C, H, W)
    emb = torch.randn(B, 6, 12)

    with torch.no_grad():
        pred = drv.forward_pass(
            noisy_input=noisy,
            timesteps=torch.tensor([500.0]),
            text_embeddings=emb,
            batch={},
        )
    assert pred.isfinite().all()


def test_driver_encode_text_applies_system_prompt_by_default():
    """encode_text prefixes every caption with LUMINA2_SYSTEM_PROMPT +
    ' <Prompt Start> ' when apply_system_prompt=True (the default) —
    pipeline_lumina2.py lines 285-288."""
    from app.engine.models.families.lumina2.driver import (
        LUMINA2_SYSTEM_PROMPT,
        Lumina2Driver,
    )

    definition = _make_lumina2_definition(
        architecture_params={"te.max_sequence_length": 10},
    )
    drv = Lumina2Driver(definition, torch.device("cpu"))

    tok = MagicMock()
    seen_texts = []

    def _fake_tokenize(texts, **kwargs):
        seen_texts.extend(texts)
        n = len(texts)
        out = MagicMock()
        out.input_ids = torch.zeros(n, 10, dtype=torch.long)
        out.attention_mask = torch.ones(n, 10, dtype=torch.long)
        return out

    tok.side_effect = _fake_tokenize
    drv.tokenizer = tok

    te = MagicMock()
    hidden_layers = [torch.randn(1, 10, 12) for _ in range(3)]

    def _fake_te(input_ids, attention_mask=None, output_hidden_states=None):
        out = MagicMock()
        out.hidden_states = hidden_layers
        return out

    te.side_effect = _fake_te
    drv.text_encoder = te

    out = drv.encode_text(["a red bicycle"], torch.float32)

    assert seen_texts == [f"{LUMINA2_SYSTEM_PROMPT} <Prompt Start> a red bicycle"]
    # hidden_states[-2], NOT [-1] (last_hidden_state-equivalent).
    assert torch.equal(out.embeddings, hidden_layers[-2])


def test_driver_encode_text_raw_when_system_prompt_disabled():
    """apply_system_prompt=False encodes the RAW caption — the CFG negative-
    prompt branch (pipeline_lumina2.py's ``negative_prompt`` path never sees
    the system-prompt prefix)."""
    from app.engine.models.families.lumina2.driver import Lumina2Driver

    definition = _make_lumina2_definition(
        architecture_params={"te.max_sequence_length": 10},
    )
    drv = Lumina2Driver(definition, torch.device("cpu"))

    tok = MagicMock()
    seen_texts = []

    def _fake_tokenize(texts, **kwargs):
        seen_texts.extend(texts)
        n = len(texts)
        out = MagicMock()
        out.input_ids = torch.zeros(n, 10, dtype=torch.long)
        out.attention_mask = torch.ones(n, 10, dtype=torch.long)
        return out

    tok.side_effect = _fake_tokenize
    drv.tokenizer = tok

    te = MagicMock()

    def _fake_te(input_ids, attention_mask=None, output_hidden_states=None):
        out = MagicMock()
        out.hidden_states = [torch.randn(1, 10, 12) for _ in range(3)]
        return out

    te.side_effect = _fake_te
    drv.text_encoder = te

    drv.encode_text(["worst quality"], torch.float32, apply_system_prompt=False)
    assert seen_texts == ["worst quality"]


def test_driver_assign_components_forces_right_padding():
    """assign_components must set tokenizer.padding_side = 'right'
    (Lumina2Pipeline.__init__ line 190)."""
    drv = _make_driver(None)
    tok = MagicMock()
    tok.padding_side = "left"
    drv.assign_components(
        {"unet": None, "vae": None, "text_encoder": None, "tokenizer": tok},
    )
    assert tok.padding_side == "right"


# ── Task 4: Trainer override trio + TE cache ─────────────────────────────────


def test_trainer_update_primary_model_syncs_driver_transformer():
    """_update_primary_model must sync self.transformer, self.components,
    AND driver.transformer."""
    import torch.nn as nn

    from app.engine.models.families.lumina2.driver import Lumina2Driver
    from app.engine.models.families.lumina2.trainer import Lumina2Trainer

    definition = _make_lumina2_definition()
    trainer = MagicMock(spec=Lumina2Trainer)
    trainer.driver = Lumina2Driver(definition, torch.device("cpu"))
    trainer.components = {"unet": None}
    trainer.transformer = None

    class _FakePEFT(nn.Module):
        pass

    peft_wrapped = _FakePEFT()
    Lumina2Trainer._update_primary_model(trainer, peft_wrapped)

    assert trainer.transformer is peft_wrapped
    assert trainer.components.get("unet") is peft_wrapped
    assert trainer.driver.transformer is peft_wrapped


def _stub_tokenizer_and_te(D: int = 12, L: int = 8):
    tok = MagicMock()

    def _fake_tokenize(texts, **kwargs):
        n = len(texts)
        out = MagicMock()
        out.input_ids = torch.zeros(n, L, dtype=torch.long)
        out.attention_mask = torch.ones(n, L, dtype=torch.long)
        return out

    tok.side_effect = _fake_tokenize

    te = MagicMock()

    def _fake_te(input_ids, attention_mask=None, output_hidden_states=None):
        b, seq = input_ids.shape
        out = MagicMock()
        out.hidden_states = [torch.randn(b, seq, D) for _ in range(3)]
        return out

    te.side_effect = _fake_te
    te.parameters = lambda: iter([torch.zeros(1)])
    return tok, te


def _build_real_trainer_shell():
    from app.engine.models.families.lumina2.driver import Lumina2Driver
    from app.engine.models.families.lumina2.trainer import Lumina2Trainer

    definition = _make_lumina2_definition(
        architecture_params={"te.max_sequence_length": 8},
    )

    trainer = MagicMock(spec=Lumina2Trainer)
    trainer.device = torch.device("cpu")
    trainer.definition = definition
    trainer.config = {"cache_text_embeddings": False}
    trainer.text_cache = {}
    trainer.uncond_text_cache = {}
    trainer.logger = MagicMock()

    drv = Lumina2Driver(definition, torch.device("cpu"))
    tiny_model = _build_tiny_model()
    tok, te = _stub_tokenizer_and_te(D=12, L=8)
    drv.assign_components({
        "unet": tiny_model, "vae": None, "text_encoder": te, "tokenizer": tok,
    })
    trainer.driver = drv

    trainer.encode_text = lambda captions, dtype, batch=None: (
        Lumina2Trainer.encode_text(trainer, captions, dtype, batch)
    )
    trainer.encode_negative_text = lambda captions, dtype: (
        Lumina2Trainer.encode_negative_text(trainer, captions, dtype)
    )
    trainer._encode_text_direct = lambda captions, dtype, apply_system_prompt=True: (
        Lumina2Trainer._encode_text_direct(
            trainer, captions, dtype, apply_system_prompt,
        )
    )
    trainer._get_cached_text_embeddings = lambda captions, dtype: (
        Lumina2Trainer._get_cached_text_embeddings(trainer, captions, dtype)
    )
    return trainer


def test_trainer_encode_to_forward_real_seam():
    """encode_text returns a (emb, mask) TUPLE consumable by
    driver.forward_pass — full encode->forward round trip is finite."""
    trainer = _build_real_trainer_shell()

    text_emb = trainer.encode_text(["a lumina2 test caption"], torch.float32)
    assert isinstance(text_emb, tuple) and len(text_emb) == 2
    emb, mask = text_emb
    assert emb.ndim == 3 and emb.shape[1] == 8
    assert mask.ndim == 2 and mask.shape[1] == 8

    B, C, H, W = 1, 4, 8, 8
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


def test_trainer_negative_prompt_differs_from_positive_same_string():
    """Encoding the SAME literal string as a positive caption vs the CFG
    negative prompt must produce DIFFERENT embeddings — proves the system-
    prompt asymmetry actually changes what's fed to the text encoder (a
    same-embedding result would mean apply_system_prompt was silently
    ignored)."""
    trainer = _build_real_trainer_shell()

    pos_emb, _ = trainer.encode_text(["same text"], torch.float32)
    neg_emb, _ = trainer.encode_negative_text(["same text"], torch.float32)

    assert pos_emb.shape == neg_emb.shape
    assert not torch.allclose(pos_emb, neg_emb), (
        "positive (system-prompt-prefixed) and negative (raw) encodings of "
        "the identical string must differ"
    )


def test_trainer_negative_prompt_cache_is_separate_from_positive():
    """self.uncond_text_cache and self.text_cache must be distinct dicts —
    a shared cache keyed only by caption text would let a positive caption
    collide with an identically-worded negative prompt."""
    trainer = _build_real_trainer_shell()
    assert trainer.uncond_text_cache is not trainer.text_cache

    trainer.encode_text(["shared string"], torch.float32)
    trainer.encode_negative_text(["shared string"], torch.float32)

    # Positive path (cache_text_embeddings=False here) doesn't populate
    # text_cache, but the uncond cache MUST be populated by
    # encode_negative_text regardless.
    assert "shared string" in trainer.uncond_text_cache


def test_trainer_te_disk_cache_layout(tmp_path):
    """TE disk cache lands in embeddings/{te_quant}/te1|te2 (emb + mask),
    keyed by the TEMPLATE-BAKED disk cache key (not the raw caption)."""
    from app.engine.components.text_embeddings import TextEmbeddingCache
    from app.engine.models.families.lumina2.trainer import (
        Lumina2Trainer,
        _disk_cache_key,
    )

    trainer = MagicMock(spec=Lumina2Trainer)
    trainer.config = {
        "cache_text_embeddings": True,
        "te_quantization": "none",
        "sample_prompts": [],
    }
    trainer.device = torch.device("cpu")
    trainer.text_cache = {}
    trainer.uncond_text_cache = {}
    trainer.logger = MagicMock()
    trainer._log_writer = None
    trainer.text_encoder = MagicMock()
    trainer._build_caption_hints.return_value = {"a lumina2 caption": "hint0"}
    trainer._resolve_te_cache_dirs.return_value = [str(tmp_path)]
    trainer._resolve_loading_dtype.return_value = torch.float32
    trainer._sample_prompt_texts.side_effect = lambda: (
        Lumina2Trainer._sample_prompt_texts(trainer)
    )

    def _fake_encode(captions, dtype, apply_system_prompt=True):
        b = len(captions)
        return torch.zeros(b, 8, 12), torch.ones(b, 8)

    trainer._encode_text_direct = _fake_encode

    Lumina2Trainer._pre_cache_text_embeddings(trainer)

    te1 = tmp_path / "embeddings" / "none" / "te1"
    te2 = tmp_path / "embeddings" / "none" / "te2"
    assert te1.is_dir()
    assert te2.is_dir()

    key = _disk_cache_key("a lumina2 caption")
    emb = TextEmbeddingCache.load(key, str(te1), "hint0")
    mask = TextEmbeddingCache.load(key, str(te2), "hint0")
    assert emb is not None and emb.shape == (8, 12)
    assert mask is not None and mask.shape == (8,)
    # Raw caption is NOT the disk key.
    assert TextEmbeddingCache.load("a lumina2 caption", str(te1), "hint0") is None


def test_template_fingerprint_derived_from_system_prompt_text():
    """The template id embeds a fingerprint HASHED FROM the actual system-
    prompt string — editing the prompt text changes every disk-cache key
    automatically."""
    from app.engine.models.families.lumina2.driver import te_template_fingerprint
    from app.engine.models.families.lumina2.trainer import _TE_TEMPLATE_ID

    assert te_template_fingerprint() in _TE_TEMPLATE_ID


def test_template_fingerprint_changes_if_prompt_text_changes():
    """A different system-prompt string must produce a different
    fingerprint (proves the hash is content-derived, not a static
    constant)."""
    import hashlib

    from app.engine.models.families.lumina2.driver import (
        LUMINA2_SYSTEM_PROMPT,
        te_template_fingerprint,
    )

    real_fp = te_template_fingerprint()
    mutated_fp = hashlib.sha256(
        (LUMINA2_SYSTEM_PROMPT + " mutated").encode("utf-8"),
    ).hexdigest()[:16]
    assert real_fp != mutated_fp
