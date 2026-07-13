"""Tests for the omnigen2 family (vendored OmniGen2, edit-first).

Battery (mirrors test_lumina2_family.py / test_boogu_image_* patterns):
  1. Family registration + definition loading (verified checkpoint facts)
  2. Loader manifest (component specs, vendored scheduler, dtype policy)
  3. Driver: LoRA targets across all FOUR block groups, RAW-[0,1) timestep
     pass-through (no reversal here — the model's clock IS the native
     clock), inverted add_noise/compute_target pairing, control-latent
     ref-image wiring (List[List] adapter, output actually depends on the
     control), chat-template encode + hidden_states[-1] tap
  4. Trainer: override trio (encode tuple, _update_primary_model driver
     sync, init_scheduler delegation), ragged TE cache, disk-cache layout
     + template fingerprint
  5. Family dispatch: control_inputs > 0 -> OmniGen2EditTrainer
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


def _make_definition(**kwargs) -> MagicMock:
    definition = MagicMock(spec=ModelDefinition)
    definition.family = "omnigen2"
    definition.id = kwargs.get("id", "omnigen2-test")
    definition.components = {}
    definition.architecture_params = kwargs.get("architecture_params", {})
    definition.lora_targetable_modules = kwargs.get("lora_targetable_modules", [])
    definition.control_inputs = kwargs.get("control_inputs", 1)
    return definition


# ── Task 1: Family Registration ──────────────────────────────────────────────


def test_family_registered():
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()

    fam = ModelRegistry._families.get("omnigen2")
    assert fam is not None, "omnigen2 family not registered"
    assert fam.archetype == "latent_diffusion"


def _reload_definitions():
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()
    return ModelRegistry


def test_definition_loaded_omnigen2():
    """omnigen2 definition must load from its YAML with checkpoint facts."""
    ModelRegistry = _reload_definitions()

    fam_defs = {
        d.id: d for d in ModelRegistry._definitions.values() if d.family == "omnigen2"
    }
    assert "omnigen2" in fam_defs, f"missing omnigen2; found: {set(fam_defs)}"

    defn = fam_defs["omnigen2"]
    assert defn.components["repo"].path == "huggingface:OmniGen2/OmniGen2"
    # EDIT-FIRST: paired Target/Control consumption is the point.
    assert defn.control_inputs == 1

    arch = defn.architecture_params
    # transformer/config.json facts (fetched 2026-07-13).
    assert arch.get("transformer.hidden_size") == 2520
    assert arch.get("transformer.num_layers") == 32
    assert arch.get("transformer.num_refiner_layers") == 2
    assert arch.get("transformer.num_attention_heads") == 21
    assert arch.get("transformer.num_kv_heads") == 7
    assert arch.get("transformer.in_channels") == 16
    assert arch.get("transformer.patch_size") == 2
    assert arch.get("transformer.text_feat_dim") == 2048
    assert arch.get("transformer.timestep_scale") == 1000
    assert arch.get("transformer.axes_dim_rope") == [40, 40, 40]
    assert arch.get("transformer.axes_lens") == [1024, 1664, 1664]
    # scheduler/scheduler_config.json facts.
    assert arch.get("scheduler.dynamic_time_shift") is True
    assert arch.get("scheduler.num_train_timesteps") == 1000
    # FLUX.1-dev VAE verbatim.
    assert arch.get("vae.latent_channels") == 16
    assert arch.get("vae.scaling_factor") == 0.3611
    assert arch.get("vae.shift_factor") == 0.1159
    assert arch.get("te.hidden_size") == 2048
    assert arch.get("te.max_sequence_length") == 256


def test_definition_ships_edit_native_sample_defaults():
    """Edit-tuned defaults from upstream example_edit.sh (50 steps, text
    guidance 5.0, image guidance 2.0)."""
    ModelRegistry = _reload_definitions()
    defn = ModelRegistry._definitions["omnigen2"]
    assert defn.defaults["num_inference_steps"] == 50
    assert defn.defaults["guidance_scale"] == 5.0
    assert defn.defaults["image_guidance_scale"] == 2.0
    assert defn.defaults["resolution"] == 1024


def test_family_dispatches_edit_trainer_for_control_inputs():
    """control_inputs > 0 -> OmniGen2EditTrainer; == 0 -> OmniGen2Trainer."""
    from app.engine.models.families.omnigen2.family import OmniGen2Family
    from app.engine.models.families.omnigen2.trainer import OmniGen2Trainer
    from app.engine.models.families.omnigen2.trainer_edit import OmniGen2EditTrainer

    fam_edit = OmniGen2Family.__new__(OmniGen2Family)
    fam_edit.definition = _make_definition(control_inputs=1)
    assert fam_edit.get_trainer_class() is OmniGen2EditTrainer

    fam_t2i = OmniGen2Family.__new__(OmniGen2Family)
    fam_t2i.definition = _make_definition(control_inputs=0)
    assert fam_t2i.get_trainer_class() is OmniGen2Trainer


# ── Task 2: Loader Manifest ──────────────────────────────────────────────────


def test_manifest_components():
    from app.engine.models.families.omnigen2.loader import OmniGen2Loader

    loader = OmniGen2Loader(torch.device("cpu"))
    specs = loader.get_component_manifest(_make_definition())

    keys = {s.key for s in specs}
    assert {"text_encoder", "processor", "vae", "scheduler"} <= keys
    # Transformer is hand-loaded in load(), never via the manifest.
    assert "unet" not in keys and "transformer" not in keys

    spec_map = {s.key: s for s in specs}

    assert spec_map["text_encoder"].hf_class == (
        "transformers.Qwen2_5_VLForConditionalGeneration"
    )
    assert spec_map["text_encoder"].subfolder == "mllm"

    assert spec_map["processor"].hf_class == "transformers.Qwen2_5_VLProcessor"
    assert spec_map["processor"].is_torch_model is False

    assert spec_map["vae"].hf_class == "diffusers.AutoencoderKL"

    # VENDORED scheduler class — never the stock diffusers one of the same
    # name (opposite time direction).
    assert "vendor.schedulers" in spec_map["scheduler"].hf_class
    assert spec_map["scheduler"].hf_class.endswith("FlowMatchEulerDiscreteScheduler")
    assert spec_map["scheduler"].is_torch_model is False


def test_loader_dtype_policy_is_generic():
    from app.engine.core.pipeline.loader_base import GenericComponentLoader
    from app.engine.models.families.omnigen2.loader import OmniGen2Loader

    assert OmniGen2Loader._resolve_dtype is GenericComponentLoader._resolve_dtype

    loader = OmniGen2Loader(torch.device("cpu"))
    for spec in loader.get_component_manifest(_make_definition()):
        assert spec.dtype_override is None


# ── Task 3: Driver ───────────────────────────────────────────────────────────

# axes_dim_rope must sum to head_dim (24/2 = 12); axes_lens[0] must exceed
# text length + shifted image positions used below.
_TINY_CFG = dict(
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
    axes_dim_rope=(4, 4, 4),
    axes_lens=(64, 64, 64),
    text_feat_dim=12,
    timestep_scale=1000.0,
)


def _build_tiny_model():
    from app.engine.models.families.omnigen2.vendor.models.transformers.transformer_omnigen2 import (
        OmniGen2Transformer2DModel,
    )

    torch.manual_seed(0)
    model = OmniGen2Transformer2DModel(**_TINY_CFG).eval()
    # Upstream zero-inits the AdaLN gates (norm1.linear) and the output head
    # (norm_out.linear_1/2) — a FRESH model is therefore the exact-zero
    # function (modulated blocks collapse to identity, output proj is 0).
    # Real checkpoints overwrite these; for behavioral tests randomize them
    # so the forward is non-trivial.
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.numel() > 0 and not p.any():
                p.normal_(std=0.02)
    return model


def _make_driver(model=None, arch=None, scheduler=None):
    from app.engine.models.families.omnigen2.driver import OmniGen2Driver

    definition = _make_definition(architecture_params=arch or {})
    drv = OmniGen2Driver(definition, torch.device("cpu"))
    drv.assign_components({
        "unet": model,
        "vae": None,
        "text_encoder": None,
        "processor": None,
        "scheduler": scheduler,
    })
    return drv


def test_driver_lora_targets_cover_all_four_block_groups():
    """Every target pattern matches >=1 Linear across layers, noise_refiner,
    ref_image_refiner AND context_refiner (all four share the same
    OmniGen2TransformerBlock submodule names)."""
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

    for group in ("layers", "noise_refiner", "ref_image_refiner", "context_refiner"):
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
    drv = _make_driver(None)
    assert drv.get_te_lora_targets() == []
    assert drv.get_lora_exclude_modules() is None


def test_driver_basic_contracts():
    """bf16 loading; init_scheduler returns the LOADER-provided vendored
    instance and fails loudly without one (never a fresh/stock default)."""
    from app.engine.models.families.omnigen2.vendor.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    drv = _make_driver(None)
    assert drv.resolve_loading_dtype() == torch.bfloat16
    with pytest.raises(RuntimeError, match="scheduler"):
        drv.init_scheduler()

    vendored = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, dynamic_time_shift=True,
    )
    drv2 = _make_driver(None, scheduler=vendored)
    assert drv2.init_scheduler() is vendored


def test_driver_forward_passes_raw_timestep_unchanged():
    """forward_pass must feed the transformer the RAW [0, 1) native t — no
    reversal (the model's clock IS the native clock, unlike lumina2's
    driver-side flip) and no /1000 or *1000 (the transformer's own
    timestep_scale=1000 config multiplies internally)."""
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
    emb = torch.randn(B, 6, 12)
    mask = torch.ones(B, 6, dtype=torch.long)

    with torch.no_grad():
        pred = drv.forward_pass(
            noisy_input=noisy,
            timesteps=torch.tensor([0.37]),
            text_embeddings=(emb, mask),
            batch={},
        )

    assert torch.allclose(captured["timestep"].float(), torch.tensor([0.37])), (
        f"transformer must receive the raw native t=0.37, got {captured['timestep']}"
    )
    assert pred.shape == (B, C, H, W)
    assert pred.isfinite().all()


def test_driver_forward_requires_attention_mask():
    model = _build_tiny_model()
    drv = _make_driver(model)
    with pytest.raises(ValueError, match="attention_mask"):
        drv.forward_pass(
            noisy_input=torch.randn(1, 4, 8, 8),
            timesteps=torch.tensor([0.5]),
            text_embeddings=torch.randn(1, 6, 12),
            batch={},
        )


def test_driver_forward_control_latents_reach_ref_image_pathway():
    """batch['control_latents'] (per-slot [B, C, h, w]) must be adapted to
    the model's per-item List[List[Tensor[C, h, w]]] contract AND actually
    change the prediction (proves the control flows, not just plumbs)."""
    model = _build_tiny_model()
    drv = _make_driver(model)

    captured: dict = {}
    original_forward = model.forward

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return original_forward(*args, **kwargs)

    model.forward = _spy

    B, C, H, W = 2, 4, 8, 8
    torch.manual_seed(1)
    noisy = torch.randn(B, C, H, W)
    emb = torch.randn(B, 6, 12)
    mask = torch.ones(B, 6, dtype=torch.long)
    control = torch.randn(B, C, H, W)

    with torch.no_grad():
        pred_ctrl = drv.forward_pass(
            noisy_input=noisy,
            timesteps=torch.tensor([0.5, 0.5]),
            text_embeddings=(emb, mask),
            batch={"control_latents": [control]},
        )

    ref = captured["ref_image_hidden_states"]
    assert isinstance(ref, list) and len(ref) == B
    for i in range(B):
        assert isinstance(ref[i], list) and len(ref[i]) == 1  # one slot
        assert ref[i][0].shape == (C, H, W)
        assert torch.equal(ref[i][0], control[i].to(ref[i][0].dtype))

    with torch.no_grad():
        pred_no_ctrl = drv.forward_pass(
            noisy_input=noisy,
            timesteps=torch.tensor([0.5, 0.5]),
            text_embeddings=(emb, mask),
            batch={},
        )
    assert captured["ref_image_hidden_states"] is None
    assert not torch.allclose(pred_ctrl, pred_no_ctrl), (
        "control latents must actually influence the prediction"
    )


def test_driver_forward_t2i_no_control_finite():
    """Pure-T2I fallback (no control in batch) must run the zero-length
    ref-sequence path without crashing."""
    model = _build_tiny_model()
    drv = _make_driver(model)
    with torch.no_grad():
        pred = drv.forward_pass(
            noisy_input=torch.randn(1, 4, 8, 8),
            timesteps=torch.tensor([0.25]),
            text_embeddings=(torch.randn(1, 6, 12), torch.ones(1, 6, dtype=torch.long)),
            batch={},
        )
    assert pred.isfinite().all()


def _stub_processor_and_te(D: int = 12, L: int = 8):
    """Tokenizer/processor + TE stubs for encode_text tests."""
    tokenizer = MagicMock()
    templated: list = []
    tokenized: list = []

    def _apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
        templated.append(messages)
        return "|".join(m["content"] for m in messages)

    tokenizer.apply_chat_template = _apply_chat_template

    def _tokenize(texts, **kwargs):
        tokenized.extend(texts)
        n = len(texts)
        out = MagicMock()
        out.input_ids = torch.zeros(n, L, dtype=torch.long)
        out.attention_mask = torch.ones(n, L, dtype=torch.long)
        return out

    tokenizer.side_effect = _tokenize

    processor = MagicMock()
    processor.tokenizer = tokenizer

    te = MagicMock()
    hidden_layers = [torch.randn(1, L, D) for _ in range(3)]

    def _fake_te(input_ids, attention_mask=None, output_hidden_states=None):
        out = MagicMock()
        out.hidden_states = hidden_layers
        return out

    te.side_effect = _fake_te
    te.parameters = lambda: iter([torch.zeros(1)])
    return processor, te, templated, tokenized, hidden_layers


def test_driver_encode_text_chat_template_and_last_layer_tap():
    """encode_text must (a) wrap every caption in the [system, user] chat
    template with the verbatim system prompt, and (b) tap hidden_states[-1]
    (pipeline ~L324-328 / train.py last_hidden_state)."""
    from app.engine.models.families.omnigen2.driver import (
        OMNIGEN2_SYSTEM_PROMPT,
        OmniGen2Driver,
    )

    definition = _make_definition(
        architecture_params={"te.max_sequence_length": 8},
    )
    drv = OmniGen2Driver(definition, torch.device("cpu"))
    processor, te, templated, tokenized, hidden_layers = _stub_processor_and_te()
    drv.processor = processor
    drv.text_encoder = te

    out = drv.encode_text(["make the sky purple"], torch.float32)

    assert len(templated) == 1
    messages = templated[0]
    assert messages[0] == {"role": "system", "content": OMNIGEN2_SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "make the sky purple"}
    assert tokenized == [f"{OMNIGEN2_SYSTEM_PROMPT}|make the sky purple"]

    # hidden_states[-1] — the LAST layer, not [-2] (that's lumina2's tap).
    assert torch.equal(out.embeddings, hidden_layers[-1])
    assert out.attention_mask is not None


def test_driver_encode_text_negative_prompt_same_template():
    """The CFG negative ('' by default) goes through the SAME chat template
    — no lumina2-style raw/prefixed asymmetry (pipeline L413-418)."""
    from app.engine.models.families.omnigen2.driver import (
        OMNIGEN2_SYSTEM_PROMPT,
        OmniGen2Driver,
    )

    drv = OmniGen2Driver(
        _make_definition(architecture_params={"te.max_sequence_length": 8}),
        torch.device("cpu"),
    )
    processor, te, templated, tokenized, _ = _stub_processor_and_te()
    drv.processor = processor
    drv.text_encoder = te

    drv.encode_text([""], torch.float32)
    assert templated[0][0]["content"] == OMNIGEN2_SYSTEM_PROMPT
    assert templated[0][1] == {"role": "user", "content": ""}


# ── Task 4: Trainer override trio + TE cache ─────────────────────────────────


def test_trainer_update_primary_model_syncs_driver_transformer():
    import torch.nn as nn

    from app.engine.models.families.omnigen2.driver import OmniGen2Driver
    from app.engine.models.families.omnigen2.trainer import OmniGen2Trainer

    definition = _make_definition()
    trainer = MagicMock(spec=OmniGen2Trainer)
    trainer.driver = OmniGen2Driver(definition, torch.device("cpu"))
    trainer.components = {"unet": None}
    trainer.transformer = None

    class _FakePEFT(nn.Module):
        pass

    peft_wrapped = _FakePEFT()
    OmniGen2Trainer._update_primary_model(trainer, peft_wrapped)

    assert trainer.transformer is peft_wrapped
    assert trainer.components.get("unet") is peft_wrapped
    assert trainer.driver.transformer is peft_wrapped


def test_trainer_init_scheduler_delegates_to_driver():
    """The trainer-level init_scheduler must return the driver's (i.e. the
    loader-provided vendored) scheduler — the base hook's None would
    clobber components['scheduler'] (boogu clobber lesson)."""
    from app.engine.models.families.omnigen2.trainer import OmniGen2Trainer
    from app.engine.models.families.omnigen2.vendor.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    vendored = FlowMatchEulerDiscreteScheduler()
    trainer = MagicMock(spec=OmniGen2Trainer)
    trainer.driver = _make_driver(None, scheduler=vendored)
    assert OmniGen2Trainer.init_scheduler(trainer) is vendored


def test_trainer_convention_trio_delegates_to_driver():
    """add_noise / compute_target / sample_timesteps must delegate to the
    driver's INVERTED convention (leaving the base mixin defaults would
    silently train a pure-noise LoRA — wrong sign AND wrong scale)."""
    from app.engine.models.families.omnigen2.trainer import OmniGen2Trainer

    trainer = MagicMock(spec=OmniGen2Trainer)
    trainer.driver = _make_driver(None)
    trainer.device = torch.device("cpu")
    trainer.config = {}
    trainer.global_step = 0
    trainer.max_train_steps = 100

    latents = torch.randn(2, 4, 4, 4)
    noise = torch.randn(2, 4, 4, 4)
    t = torch.tensor([0.25, 0.75])

    noisy = OmniGen2Trainer.add_noise(trainer, latents, noise, t)
    expected = (1.0 - t.view(-1, 1, 1, 1)) * noise + t.view(-1, 1, 1, 1) * latents
    assert torch.allclose(noisy, expected)

    target = OmniGen2Trainer.compute_target(trainer, latents, noise, t)
    assert torch.equal(target, latents - noise)

    ts = OmniGen2Trainer.sample_timesteps(trainer, 512)
    assert ts.shape == (512,)
    assert ((ts >= 0) & (ts <= 1)).all(), "native timesteps must be raw [0, 1]"


def _build_real_trainer_shell():
    from app.engine.models.families.omnigen2.driver import OmniGen2Driver
    from app.engine.models.families.omnigen2.trainer import OmniGen2Trainer

    definition = _make_definition(
        architecture_params={"te.max_sequence_length": 8},
    )

    trainer = MagicMock(spec=OmniGen2Trainer)
    trainer.device = torch.device("cpu")
    trainer.definition = definition
    trainer.config = {"cache_text_embeddings": False}
    trainer.text_cache = {}
    trainer.logger = MagicMock()

    drv = OmniGen2Driver(definition, torch.device("cpu"))
    tiny_model = _build_tiny_model()
    processor, te, _, _, _ = _stub_processor_and_te(D=12, L=8)
    drv.assign_components({
        "unet": tiny_model, "vae": None, "text_encoder": te,
        "processor": processor, "scheduler": None,
    })
    trainer.driver = drv

    trainer.encode_text = lambda captions, dtype, batch=None: (
        OmniGen2Trainer.encode_text(trainer, captions, dtype, batch)
    )
    trainer._encode_text_direct = lambda captions, dtype: (
        OmniGen2Trainer._encode_text_direct(trainer, captions, dtype)
    )
    trainer._get_cached_text_embeddings = lambda captions, dtype: (
        OmniGen2Trainer._get_cached_text_embeddings(trainer, captions, dtype)
    )
    trainer._trim_entry = OmniGen2Trainer._trim_entry
    return trainer


def test_trainer_encode_to_forward_real_seam():
    """encode_text returns an (emb, mask) TUPLE consumable by
    driver.forward_pass — full encode->forward round trip is finite."""
    trainer = _build_real_trainer_shell()

    text_emb = trainer.encode_text(["an omnigen2 test caption"], torch.float32)
    assert isinstance(text_emb, tuple) and len(text_emb) == 2
    emb, mask = text_emb
    assert emb.ndim == 3
    assert mask.ndim == 2

    B, C, H, W = 1, 4, 8, 8
    with torch.no_grad():
        pred = trainer.driver.forward_pass(
            noisy_input=torch.randn(B, C, H, W),
            timesteps=torch.tensor([0.5]),
            text_embeddings=text_emb,
            batch={},
        )
    assert pred.shape == (B, C, H, W)
    assert pred.isfinite().all()


def test_trainer_cached_encode_ragged_entries_reassemble_padded():
    """padding='longest' -> variable-length cache entries; reassembly must
    pad embeddings AND masks to the batch max (a plain torch.stack over
    ragged entries would crash — boogu Finding-1 class)."""
    from app.engine.models.families.omnigen2.trainer import OmniGen2Trainer

    trainer = _build_real_trainer_shell()
    trainer.config = {"cache_text_embeddings": True}
    trainer.text_encoder = trainer.driver.text_encoder

    # Pre-seed RAGGED entries (different true lengths).
    trainer.text_cache = {
        "short": (torch.randn(3, 12), torch.ones(3, dtype=torch.long)),
        "a much longer caption": (torch.randn(7, 12), torch.ones(7, dtype=torch.long)),
    }

    emb, mask = OmniGen2Trainer._get_cached_text_embeddings(
        trainer, ["short", "a much longer caption"], torch.float32,
    )
    assert emb.shape == (2, 7, 12)
    assert mask.shape == (2, 7)
    assert int(mask[0].sum()) == 3 and int(mask[1].sum()) == 7
    # Padding rows must be zeros.
    assert torch.equal(emb[0, 3:], torch.zeros(4, 12))


def test_trainer_te_disk_cache_layout(tmp_path):
    """TE disk cache lands in embeddings/{te_quant}/te1|te2, keyed by the
    TEMPLATE-BAKED key (not the raw caption)."""
    from app.engine.components.text_embeddings import TextEmbeddingCache
    from app.engine.models.families.omnigen2.trainer import (
        OmniGen2Trainer,
        _disk_cache_key,
    )

    trainer = MagicMock(spec=OmniGen2Trainer)
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
    trainer._build_caption_hints.return_value = {"an omnigen2 caption": "hint0"}
    trainer._resolve_te_cache_dirs.return_value = [str(tmp_path)]
    trainer._resolve_loading_dtype.return_value = torch.float32
    trainer._sample_prompt_texts.side_effect = lambda: (
        OmniGen2Trainer._sample_prompt_texts(trainer)
    )
    trainer._trim_entry = OmniGen2Trainer._trim_entry

    def _fake_encode(captions, dtype):
        b = len(captions)
        return torch.zeros(b, 8, 12), torch.ones(b, 8)

    trainer._encode_text_direct = _fake_encode

    OmniGen2Trainer._pre_cache_text_embeddings(trainer)

    te1 = tmp_path / "embeddings" / "none" / "te1"
    te2 = tmp_path / "embeddings" / "none" / "te2"
    assert te1.is_dir() and te2.is_dir()

    key = _disk_cache_key("an omnigen2 caption")
    emb = TextEmbeddingCache.load(key, str(te1), "hint0")
    mask = TextEmbeddingCache.load(key, str(te2), "hint0")
    assert emb is not None and emb.shape == (8, 12)
    assert mask is not None and mask.shape == (8,)
    # Raw caption is NOT the disk key.
    assert TextEmbeddingCache.load("an omnigen2 caption", str(te1), "hint0") is None


def test_template_fingerprint_derived_from_system_prompt_text():
    from app.engine.models.families.omnigen2.driver import te_template_fingerprint
    from app.engine.models.families.omnigen2.trainer import _TE_TEMPLATE_ID

    assert te_template_fingerprint() in _TE_TEMPLATE_ID


def test_template_fingerprint_changes_if_prompt_text_changes():
    import hashlib

    from app.engine.models.families.omnigen2.driver import (
        OMNIGEN2_SYSTEM_PROMPT,
        te_template_fingerprint,
    )

    real_fp = te_template_fingerprint()
    mutated_fp = hashlib.sha256(
        (OMNIGEN2_SYSTEM_PROMPT + " mutated").encode("utf-8"),
    ).hexdigest()[:16]
    assert real_fp != mutated_fp


# ── Edit trainer (deliberately minimal — no composite TE keys) ───────────────


def test_edit_trainer_does_not_override_encode_text():
    """OmniGen2's text embeddings are CONTROL-INDEPENDENT (mllm never sees
    pixels — driver.py recon §1), so the edit trainer must NOT override
    encode_text with boogu-style composite (caption, control) keys, and
    must keep the base disk pre-cache."""
    from app.engine.models.families.omnigen2.trainer import OmniGen2Trainer
    from app.engine.models.families.omnigen2.trainer_edit import OmniGen2EditTrainer

    assert OmniGen2EditTrainer.encode_text is OmniGen2Trainer.encode_text
    assert (
        OmniGen2EditTrainer._pre_cache_text_embeddings
        is OmniGen2Trainer._pre_cache_text_embeddings
    )
