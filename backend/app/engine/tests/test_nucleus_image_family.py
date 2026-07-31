"""Tests for the nucleus_image family (diffusers-0.39-native Nucleus-Image).

TDD order (mirrors test_lumina2_family.py / test_chroma_family.py):
  Task 1: family registration + definition loading
  Task 2: loader manifest (component specs + dtype policy)
  Task 3: driver (LoRA targets incl. router-gate exclusion, forward_pass
          timestep NON-reversal + output negation, compute_target default,
          encode_text replicating NucleusMoEImagePipeline's chat-template
          formatting + hidden_states[-8] tap)
  Task 4: trainer override trio (encode_text tuple, _update_primary_model
          driver sync, ragged TE-cache trim/re-pad) + disk-cache layout +
          template fingerprint
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

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


def _make_nucleus_definition(**kwargs) -> MagicMock:
    """Build a mock Nucleus-Image ModelDefinition for loader/driver tests."""
    definition = MagicMock(spec=ModelDefinition)
    definition.family = "nucleus_image"
    definition.id = kwargs.get("id", "nucleus-image-test")
    definition.components = {}
    definition.architecture_params = kwargs.get("architecture_params", {})
    definition.lora_targetable_modules = kwargs.get("lora_targetable_modules", [])
    return definition


# ── Task 1: Family Registration ──────────────────────────────────────────────


def test_family_registered():
    """nucleus_image family must appear in ModelRegistry with the correct
    archetype."""
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()

    fam = ModelRegistry._families.get("nucleus_image")
    assert fam is not None, "nucleus_image family not registered"
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


def test_definition_loaded_nucleus_image():
    """nucleus-image definition must load from its YAML with verified facts."""
    ModelRegistry = _reload_definitions()

    fam_defs = {
        d.id: d for d in ModelRegistry._definitions.values()
        if d.family == "nucleus_image"
    }
    assert "nucleus-image" in fam_defs, f"missing nucleus-image; found: {set(fam_defs)}"

    defn = fam_defs["nucleus-image"]
    assert defn.components["repo"].path == "huggingface:NucleusAI/Nucleus-Image"
    assert defn.control_inputs == 0

    arch = defn.architecture_params
    assert arch.get("transformer.num_layers") == 32
    assert arch.get("transformer.num_experts") == 64
    assert arch.get("transformer.in_channels") == 64
    assert arch.get("transformer.out_channels") == 16
    assert arch.get("transformer.patch_size") == 2
    assert arch.get("transformer.joint_attention_dim") == 4096
    assert arch.get("transformer.num_key_value_heads") == 4
    assert arch.get("transformer.dense_moe_strategy") == "leave_first_three_blocks_dense"
    assert len(arch.get("transformer.capacity_factors")) == 32
    assert arch.get("vae.latent_channels") == 16
    assert arch.get("te.hidden_size") == 4096
    assert arch.get("te.num_hidden_layers") == 36
    assert arch.get("te.hidden_state_tap_index") == -8
    assert arch.get("te.max_sequence_length") == 1024
    # STATIC shift (verified checkpoint fact) — 1.0, NOT lumina2's 6.0.
    assert arch.get("scheduler.use_dynamic_shifting") is False
    assert arch.get("scheduler.shift") == 1.0


def test_definition_ships_native_sample_defaults():
    """Pipeline __call__ native defaults: 50 steps, guidance 4.0, res 1024."""
    ModelRegistry = _reload_definitions()
    defn = ModelRegistry._definitions["nucleus-image"]
    assert defn.defaults["num_inference_steps"] == 50
    assert defn.defaults["guidance_scale"] == 4.0
    assert defn.defaults["resolution"] == 1024


# ── Task 2: Loader Manifest ──────────────────────────────────────────────────


def test_manifest_components():
    """NucleusImageLoader manifest declares all four diffusers-native
    components, including the FULL Qwen3VLProcessor (not a bare tokenizer)."""
    from app.engine.models.families.nucleus_image.loader import NucleusImageLoader

    loader = NucleusImageLoader(torch.device("cpu"))
    definition = _make_nucleus_definition()
    specs = loader.get_component_manifest(definition)

    keys = {s.key for s in specs}
    assert {"tokenizer", "text_encoder", "vae", "unet"} <= keys, (
        f"missing required manifest keys; got {keys}"
    )

    spec_map = {s.key: s for s in specs}

    assert spec_map["tokenizer"].hf_class == "transformers.Qwen3VLProcessor", (
        f"tokenizer hf_class wrong: {spec_map['tokenizer'].hf_class}"
    )
    assert spec_map["tokenizer"].subfolder == "processor"
    assert spec_map["tokenizer"].is_torch_model is False

    assert spec_map["text_encoder"].hf_class == (
        "transformers.Qwen3VLForConditionalGeneration"
    )
    assert spec_map["text_encoder"].subfolder == "text_encoder"

    assert spec_map["vae"].hf_class == "diffusers.AutoencoderKLQwenImage"

    assert spec_map["unet"].hf_class == (
        "diffusers.NucleusMoEImageTransformer2DModel"
    )
    assert spec_map["unet"].subfolder == "transformer"


def test_loader_dtype_policy_is_generic():
    """Dtype policy is generic (no per-component override)."""
    import torch as _torch

    from app.engine.core.pipeline.loader_base import GenericComponentLoader
    from app.engine.models.families.nucleus_image.loader import NucleusImageLoader

    assert NucleusImageLoader._resolve_dtype is GenericComponentLoader._resolve_dtype

    loader = NucleusImageLoader(_torch.device("cpu"))
    definition = _make_nucleus_definition()
    for spec in loader.get_component_manifest(definition):
        assert spec.dtype_override is None, (
            f"{spec.key} must not force a dtype override"
        )


# ── Task 3: Driver ───────────────────────────────────────────────────────────

_TINY_CFG = dict(
    patch_size=2,
    in_channels=16,  # packed: unpacked C=4, patch_size=2 -> 4*2*2=16
    out_channels=4,
    num_layers=4,
    attention_head_dim=8,
    num_attention_heads=2,
    num_key_value_heads=1,
    joint_attention_dim=12,
    # axes_dims_rope MUST sum to attention_head_dim (verified against the
    # real config: 16+56+56=128=attention_head_dim).
    axes_dims_rope=(2, 2, 4),
    mlp_ratio=4.0,
    moe_enabled=True,
    # First 3 blocks (idx 0-2) dense, last block (idx 3) MoE — exercises
    # BOTH block types with a small model.
    dense_moe_strategy="leave_first_three_blocks_dense",
    num_experts=4,
    moe_intermediate_dim=8,
    capacity_factors=[0.0, 0.0, 0.0, 2.0],
    use_sigmoid=False,
    route_scale=1.0,
    # CPU-safe: torch.nn.functional.grouped_mm requires CUDA SM>=80.
    use_grouped_mm=False,
)


def _build_tiny_model():
    """Build a tiny real ``NucleusMoEImageTransformer2DModel`` for driver tests.

    ``SwiGLUExperts.__init__`` (diffusers 0.39.0,
    ``transformer_nucleusmoe_image.py`` lines 391-392) allocates its routed-
    expert weights via bare ``nn.Parameter(torch.empty(...))`` with NO
    ``reset_parameters``/init call anywhere in the class — real checkpoints
    always overwrite this via ``from_pretrained``, so upstream never needed
    one. Constructing the model from scratch (as this fixture does, with no
    checkpoint) leaves those specific tensors as UNINITIALIZED memory, which
    can contain NaN/Inf bit patterns nondeterministically depending on prior
    allocator activity — this produced real, reproducibly-flaky
    ``pred.isfinite()`` failures in this test file when several tiny models
    were built back-to-back (confirmed via forward-hook tracing: NaNs
    originate at ``transformer_blocks.N.img_mlp.experts``, i.e. exactly
    these two raw Parameters). Explicitly initialize them so every model this
    fixture builds is deterministic and finite, matching how a real
    checkpoint load would always populate them.
    """
    from diffusers.models.transformers.transformer_nucleusmoe_image import (
        NucleusMoEImageTransformer2DModel,
        SwiGLUExperts,
    )

    torch.manual_seed(0)
    model = NucleusMoEImageTransformer2DModel(**_TINY_CFG).eval()
    for module in model.modules():
        if isinstance(module, SwiGLUExperts):
            nn.init.normal_(module.gate_up_proj, mean=0.0, std=0.02)
            nn.init.normal_(module.down_proj, mean=0.0, std=0.02)
    return model


def _make_driver(model=None, arch=None):
    from app.engine.models.families.nucleus_image.driver import NucleusImageDriver

    definition = _make_nucleus_definition(architecture_params=arch or {})
    drv = NucleusImageDriver(definition, torch.device("cpu"))
    drv.assign_components(
        {"unet": model, "vae": None, "text_encoder": None, "tokenizer": None},
    )
    return drv


def test_driver_lora_targets_are_attention_and_ffn_only():
    """Every default LoRA target pattern matches >=1 Linear module, and the
    curated list contains ONLY attention + FFN patterns (controller-pinned
    scope, driver.py module docstring §7) — NOT encoder_proj/img_mod/img_in/
    proj_out/norm_out (the ai-toolkit superset)."""
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

    # Deliberately excluded surface (in-scope for ai-toolkit, NOT for us).
    excluded_suffixes = ("encoder_proj", "img_mod.1", "img_in", "proj_out", "norm_out.linear")
    for t in targets:
        assert not any(t.endswith(suf) for suf in excluded_suffixes), (
            f"target {t!r} unexpectedly touches excluded surface"
        )


def test_driver_lora_targets_cover_dense_and_moe_blocks():
    """attn.* + img_mlp.net.* must match the 3 DENSE blocks; shared_expert.
    net.* must match the 1 MoE block — no cross-contamination."""
    model = _build_tiny_model()
    drv = _make_driver(model)
    targets = set(drv.get_lora_targets())

    linear_names = {
        n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    }

    dense_ffn_matches = [
        n for n in linear_names
        if n.endswith("img_mlp.net.0.proj") or n.endswith("img_mlp.net.2")
    ]
    moe_ffn_matches = [
        n for n in linear_names
        if n.endswith("shared_expert.net.0.proj") or n.endswith("shared_expert.net.2")
    ]
    assert len(dense_ffn_matches) == 3 * 2, (
        f"expected 3 dense blocks * 2 modules, got {dense_ffn_matches}"
    )
    assert len(moe_ffn_matches) == 1 * 2, (
        f"expected 1 MoE block * 2 modules, got {moe_ffn_matches}"
    )
    # No overlap between the two target families.
    assert not set(dense_ffn_matches) & set(moe_ffn_matches)
    assert "img_mlp.net.0.proj" in targets
    assert "shared_expert.net.0.proj" in targets


def test_lora_targets_never_match_router_gate():
    """PINS task-brief decision #1: the MoE router gate
    (``NucleusMoELayer.gate``) must NEVER be matched by any LoRA target
    pattern — neither by literal presence in the target list NOR by
    suffix-match against a live-introspected model's real module names."""
    from app.engine.models.families.nucleus_image.driver import (
        ROUTER_GATE_MODULE_NAME,
    )

    model = _build_tiny_model()
    drv = _make_driver(model)
    targets = drv.get_lora_targets()

    # 1. Literal check: the bare "gate" string is not itself a target.
    assert ROUTER_GATE_MODULE_NAME not in targets
    assert not any(t == "gate" or t.endswith(".gate") for t in targets)

    # 2. Live introspection: find the real router gate module name(s) and
    #    confirm no target pattern's suffix-match would ever hit them.
    gate_names = [
        n for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and n.endswith(f".{ROUTER_GATE_MODULE_NAME}")
    ]
    assert gate_names, "test setup sanity: tiny MoE model must have a router gate"
    for gate_name in gate_names:
        for t in targets:
            assert not (gate_name == t or gate_name.endswith("." + t)), (
                f"LoRA target {t!r} unexpectedly matches router gate {gate_name!r}"
            )


def test_routed_experts_are_raw_parameters_not_linear():
    """SwiGLUExperts.gate_up_proj/down_proj are raw nn.Parameter tensors —
    confirms they are structurally un-targetable by name-based LoRA
    injection regardless of any target pattern (recon §4)."""
    model = _build_tiny_model()
    linear_names = {
        n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    }
    assert not any("experts.gate_up_proj" in n for n in linear_names)
    assert not any("experts.down_proj" in n for n in linear_names)

    # Confirm they DO exist as raw Parameters on the MoE block.
    moe_block = model.transformer_blocks[3]
    assert moe_block.moe_enabled is True
    assert isinstance(moe_block.img_mlp.experts.gate_up_proj, torch.nn.Parameter)
    assert isinstance(moe_block.img_mlp.experts.down_proj, torch.nn.Parameter)


def test_driver_no_te_lora_and_no_exclude():
    """Qwen3-VL stays frozen; no PEFT exclude_modules needed (target list
    never touches the gate by construction — see the pinning test above)."""
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
    default; NucleusImageDriver deliberately does not override it — the
    negation lives entirely inside forward_pass)."""
    drv = _make_driver(None)
    latents = torch.randn(2, 4, 4, 4)
    noise = torch.randn(2, 4, 4, 4)
    target = drv.compute_target(latents, noise, torch.tensor([500.0, 250.0]))
    assert torch.equal(target, noise - latents)


def test_driver_forward_timestep_not_reversed_and_output_negated():
    """forward_pass must (a) feed the transformer ``t/1000`` UNCHANGED (NOT
    reversed — pipeline_nucleusmoe_image.py line 575, the INVERSE mistake
    from lumina2) and (b) return the NEGATED raw model output
    (pipeline_nucleusmoe_image.py line 599). This is the family's #1
    silent-LoRA-killer risk."""
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
        pH, pW = H // 2, W // 2
        x = noisy.reshape(B, C, pH, 2, pW, 2).permute(0, 2, 4, 1, 3, 5).reshape(
            B, pH * pW, C * 4,
        )
        raw_direct = model(
            hidden_states=x,
            img_shapes=[(1, pH, pW)] * B,
            encoder_hidden_states=emb,
            encoder_hidden_states_mask=mask,
            timestep=torch.tensor([0.5]),
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
        f"transformer must receive t/1000 = 0.5 for t=500 UNCHANGED, got {ts}"
    )
    assert pred.shape == (B, 4, H, W)
    assert pred.isfinite().all()

    raw_unpacked = raw_direct.reshape(B, pH, pW, 4, 2, 2).permute(
        0, 3, 1, 4, 2, 5,
    ).reshape(B, 4, H, W)
    assert torch.allclose(pred, -raw_unpacked, atol=1e-5), (
        "forward_pass must return the NEGATED raw model output"
    )


def test_driver_forward_extreme_timesteps_confirm_no_reversal_direction():
    """t=0 (framework convention: pure LATENTS) must feed the transformer
    timestep=0.0 UNCHANGED; t=1000 (pure NOISE) must feed timestep=1.0
    UNCHANGED — pins the (NON-reversed) DIRECTION, distinguishing this
    family from lumina2's reversed convention."""
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

    assert torch.allclose(seen[0], torch.tensor([0.0]))
    assert torch.allclose(seen[1], torch.tensor([1.0]))


def test_driver_forward_handles_missing_mask():
    """A mask-less (plain tensor) text_embeddings input must not crash."""
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


def _stub_processor_and_te(D: int = 12, L: int = 8):
    """Build a mock Qwen3VLProcessor (chat template + tokenizer) and Qwen3VL
    text encoder for encode_text tests."""
    processor = MagicMock()
    seen_templates: list[list[dict]] = []

    def _apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
        seen_templates.append(messages)
        # Return a deterministic formatted string derived from the raw
        # user text so tests can assert on it.
        user_text = messages[1]["content"][0]["text"]
        return f"<|system|>...<|user|>{user_text}<|assistant|>"

    processor.apply_chat_template.side_effect = _apply_chat_template

    def _call_processor(text, **kwargs):
        n = len(text)
        out = MagicMock()
        out.input_ids = torch.zeros(n, L, dtype=torch.long)
        out.attention_mask = torch.ones(n, L, dtype=torch.long)
        return out

    processor.side_effect = _call_processor

    te = MagicMock()

    def _fake_te(input_ids, attention_mask=None, use_cache=None, return_dict=None, output_hidden_states=None):
        b, seq = input_ids.shape
        out = MagicMock()
        out.hidden_states = [torch.randn(b, seq, D) for _ in range(10)]
        return out

    te.side_effect = _fake_te
    return processor, te, seen_templates


def test_driver_encode_text_uses_chat_template_and_return_index():
    """encode_text must wrap every caption in the system+user chat template
    via processor.apply_chat_template (NucleusMoEImagePipeline._format_
    prompt) and tap hidden_states[return_index] (default -8)."""
    from app.engine.models.families.nucleus_image.driver import (
        NUCLEUS_SYSTEM_PROMPT,
        NucleusImageDriver,
    )

    definition = _make_nucleus_definition(
        architecture_params={"te.max_sequence_length": 10},
    )
    drv = NucleusImageDriver(definition, torch.device("cpu"))
    processor, te, seen_templates = _stub_processor_and_te()
    drv.tokenizer = processor
    drv.text_encoder = te

    out = drv.encode_text(["a red bicycle"], torch.float32)

    assert out.embeddings is not None
    assert len(seen_templates) == 1
    messages = seen_templates[0]
    assert messages[0] == {"role": "system", "content": NUCLEUS_SYSTEM_PROMPT}
    assert messages[1] == {
        "role": "user", "content": [{"type": "text", "text": "a red bicycle"}],
    }

    # hidden_states[-8] — default return_index (driver.py module docstring §2).
    assert drv.return_index == -8


def test_driver_encode_text_no_positive_negative_asymmetry():
    """Encoding the SAME literal string always produces the SAME
    formatting — no apply_system_prompt-style flag exists on this driver's
    encode_text, unlike lumina2 (see module docstring §1)."""
    import inspect

    from app.engine.models.families.nucleus_image.driver import NucleusImageDriver

    sig = inspect.signature(NucleusImageDriver.encode_text)
    assert "apply_system_prompt" not in sig.parameters, (
        "nucleus_image has NO positive/negative encoding asymmetry — this "
        "driver must not carry a lumina2-style toggle"
    )


def test_driver_assign_components_maps_processor_to_tokenizer_slot():
    """assign_components must map the loader's 'tokenizer' component key
    (holding a full Qwen3VLProcessor) onto driver.tokenizer."""
    drv = _make_driver(None)
    processor = MagicMock()
    drv.assign_components(
        {"unet": None, "vae": None, "text_encoder": None, "tokenizer": processor},
    )
    assert drv.tokenizer is processor


# ── Task 4: Trainer override trio + TE cache ─────────────────────────────────


def test_trainer_update_primary_model_syncs_driver_transformer():
    """_update_primary_model must sync self.transformer, self.components,
    AND driver.transformer."""
    import torch.nn as nn

    from app.engine.models.families.nucleus_image.driver import NucleusImageDriver
    from app.engine.models.families.nucleus_image.trainer import NucleusImageTrainer

    definition = _make_nucleus_definition()
    trainer = MagicMock(spec=NucleusImageTrainer)
    trainer.driver = NucleusImageDriver(definition, torch.device("cpu"))
    trainer.components = {"unet": None}
    trainer.transformer = None

    class _FakePEFT(nn.Module):
        pass

    peft_wrapped = _FakePEFT()
    NucleusImageTrainer._update_primary_model(trainer, peft_wrapped)

    assert trainer.transformer is peft_wrapped
    assert trainer.components.get("unet") is peft_wrapped
    assert trainer.driver.transformer is peft_wrapped


def _build_real_trainer_shell():
    from app.engine.models.families.nucleus_image.driver import NucleusImageDriver
    from app.engine.models.families.nucleus_image.trainer import NucleusImageTrainer

    definition = _make_nucleus_definition(
        architecture_params={"te.max_sequence_length": 8},
    )

    trainer = MagicMock(spec=NucleusImageTrainer)
    trainer.device = torch.device("cpu")
    trainer.definition = definition
    trainer.config = {"cache_text_embeddings": False}
    trainer.text_cache = {}
    trainer.logger = MagicMock()

    drv = NucleusImageDriver(definition, torch.device("cpu"))
    processor, te, _ = _stub_processor_and_te(D=12, L=8)
    drv.assign_components({
        "unet": None, "vae": None, "text_encoder": te, "tokenizer": processor,
    })
    trainer.driver = drv

    trainer.encode_text = lambda captions, dtype, batch=None: (
        NucleusImageTrainer.encode_text(trainer, captions, dtype, batch)
    )
    trainer._encode_text_direct = lambda captions, dtype: (
        NucleusImageTrainer._encode_text_direct(trainer, captions, dtype)
    )
    trainer._get_cached_text_embeddings = lambda captions, dtype: (
        NucleusImageTrainer._get_cached_text_embeddings(trainer, captions, dtype)
    )
    # Wire the REAL static trim helper — a bare MagicMock(spec=...) attribute
    # access would otherwise return an auto-mock (not a tuple), breaking the
    # ragged-cache reassembly downstream.
    trainer._trim_entry = NucleusImageTrainer._trim_entry
    return trainer


def test_trainer_encode_to_forward_real_seam():
    """encode_text returns a (emb, mask) TUPLE consumable by
    driver.forward_pass — full encode->forward round trip is finite."""
    trainer = _build_real_trainer_shell()

    text_emb = trainer.encode_text(["a nucleus_image test caption"], torch.float32)
    assert isinstance(text_emb, tuple) and len(text_emb) == 2
    emb, mask = text_emb
    assert emb.ndim == 3 and emb.shape[1] == 8
    assert mask.ndim == 2 and mask.shape[1] == 8


def test_trainer_ragged_cache_trims_and_repads():
    """RAGGED cache: entries are TRIMMED to true length on store, re-padded
    to the batch max on retrieval — mixed-length captions must not crash a
    plain torch.stack (qwen_image 'W3-4' precedent)."""
    trainer = _build_real_trainer_shell()
    trainer.config = {"cache_text_embeddings": True}
    trainer.text_encoder = trainer.driver.text_encoder

    # Manually seed the cache with two DIFFERENT true lengths.
    trainer.text_cache["short"] = (torch.randn(3, 12), torch.ones(3, dtype=torch.long))
    trainer.text_cache["long"] = (torch.randn(7, 12), torch.ones(7, dtype=torch.long))

    emb, mask = trainer.encode_text(["short", "long"], torch.float32)
    assert emb.shape == (2, 7, 12)
    assert mask.shape == (2, 7)
    # Short entry's padding tail must be zeroed (mask=0).
    assert mask[0, :3].sum().item() == 3
    assert mask[0, 3:].sum().item() == 0
    assert torch.equal(emb[0, 3:], torch.zeros(4, 12))


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


def test_trainer_te_disk_cache_layout(tmp_path):
    """TE disk cache lands in embeddings/{te_quant}/te1|te2 (emb + mask),
    keyed by the TEMPLATE-BAKED disk cache key (not the raw caption)."""
    from app.engine.components.text_embeddings import TextEmbeddingCache
    from app.engine.models.families.nucleus_image.trainer import (
        NucleusImageTrainer,
        _disk_cache_key,
    )

    trainer = MagicMock(spec=NucleusImageTrainer)
    trainer.config = {
        "cache_text_embeddings": True,
        "te_quantization": "none",
        "sample_prompts": [],
        "sample_negative_prompt": "",
    }
    trainer.device = torch.device("cpu")
    trainer.text_cache = {}
    trainer.logger = MagicMock()
    trainer._log_writer = None
    trainer.text_encoder = MagicMock()
    trainer._build_caption_hints.return_value = {"a nucleus caption": "hint0"}
    trainer._resolve_te_cache_dirs.return_value = [str(tmp_path)]
    trainer._resolve_loading_dtype.return_value = torch.float32
    trainer._sample_prompt_texts.side_effect = lambda: (
        NucleusImageTrainer._sample_prompt_texts(trainer)
    )
    # Wire the REAL static trim helper (see _build_real_trainer_shell comment
    # above — a bare MagicMock(spec=...) attribute would otherwise return an
    # auto-mock, not a tuple).
    trainer._trim_entry = NucleusImageTrainer._trim_entry

    def _fake_encode(captions, dtype):
        b = len(captions)
        return torch.zeros(b, 8, 12), torch.ones(b, 8)

    trainer._encode_text_direct = _fake_encode

    NucleusImageTrainer._pre_cache_text_embeddings(trainer)

    te1 = tmp_path / "embeddings" / "none" / "te1"
    te2 = tmp_path / "embeddings" / "none" / "te2"
    assert te1.is_dir()
    assert te2.is_dir()

    key = _disk_cache_key("a nucleus caption")
    emb = TextEmbeddingCache.load(key, str(te1), "hint0")
    mask = TextEmbeddingCache.load(key, str(te2), "hint0")
    assert emb is not None and emb.shape == (8, 12)
    assert mask is not None and mask.shape == (8,)
    # Raw caption is NOT the disk key.
    assert TextEmbeddingCache.load("a nucleus caption", str(te1), "hint0") is None


def test_template_fingerprint_derived_from_system_prompt_text():
    """The template id embeds a fingerprint HASHED FROM the actual system-
    prompt string — editing the prompt text changes every disk-cache key
    automatically."""
    from app.engine.models.families.nucleus_image.driver import (
        te_template_fingerprint,
    )
    from app.engine.models.families.nucleus_image.trainer import _TE_TEMPLATE_ID

    assert te_template_fingerprint() in _TE_TEMPLATE_ID


def test_template_fingerprint_changes_if_prompt_text_changes():
    """A different system-prompt string must produce a different
    fingerprint (proves the hash is content-derived, not a static
    constant)."""
    import hashlib

    from app.engine.models.families.nucleus_image.driver import (
        NUCLEUS_SYSTEM_PROMPT,
        te_template_fingerprint,
    )

    real_fp = te_template_fingerprint()
    mutated_fp = hashlib.sha256(
        (NUCLEUS_SYSTEM_PROMPT + " mutated").encode("utf-8"),
    ).hexdigest()[:16]
    assert real_fp != mutated_fp


def test_precision_spec_disables_amp_for_grouped_mm():
    """AMP must be OFF for this family (GPU UAT crash, 2026-07-14).

    Under ``torch.autocast(bf16)`` LayerNorm runs in fp32, so the block's
    modulated hidden states reach the frozen (non-LoRA) MoE experts as fp32
    — and ``torch._grouped_mm`` is NOT on autocast's cast-policy list, so it
    receives the raw fp32 x against bf16 expert weights and raises
    "expected mat1 and mat2 to have the same dtype". The sampler already
    runs the native no-autocast bf16 regime (proven on real weights); the
    trainer must match it. Ideogram4 precedent: PrecisionSpec(use_amp=False).
    """
    drv = _make_driver(None)
    spec = drv.get_precision_spec("bf16")
    assert spec.use_amp is False
    assert spec.autocast_dtype == torch.bfloat16
    assert spec.grad_scaler_enabled is False
