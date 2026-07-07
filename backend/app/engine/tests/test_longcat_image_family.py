"""Tests for the longcat_image family (diffusers-0.39-native LongCat-Image).

TDD order (F2 plan):
  Task 1: family registration + definition
  Task 2: loader manifest (incl. the extra ``text_processor`` component)
  Task 3: driver — LoRA targets vs a tiny instantiated transformer,
          timestep ÷1000 exactly once, flow-match target, encode_text
          replicating ``LongCatImagePipeline._encode_prompt``
  Task 4: trainer — override trio + TE disk-cache layout
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.engine.core.definitions import ModelDefinition


@pytest.fixture(autouse=True)
def _restore_model_registry():
    """Snapshot + restore ``ModelRegistry`` class state around every test.

    Registration tests mutate the registry's class-level discovery caches
    inline (forcing a re-scan); left unrestored those mutations leak into
    later tests in the session (mirrors test_krea2_family.py's fixture).
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


def _make_definition(**kwargs) -> MagicMock:
    """Build a mock longcat_image ModelDefinition."""
    definition = MagicMock(spec=ModelDefinition)
    definition.family = "longcat_image"
    definition.id = kwargs.get("id", "longcat-image-test")
    definition.components = {}
    definition.architecture_params = kwargs.get("architecture_params", {})
    definition.lora_targetable_modules = kwargs.get("lora_targetable_modules", [])
    return definition


# ── Task 1: Family registration + definition ────────────────────────────────


def test_family_registered():
    """longcat_image family must appear in ModelRegistry as latent_diffusion."""
    from app.engine.models.registry import ModelRegistry

    # Reset discovery state so this test is hermetic
    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()

    fam = ModelRegistry._families.get("longcat_image")
    assert fam is not None, "longcat_image family not registered"
    assert fam.archetype == "latent_diffusion", (
        f"expected archetype='latent_diffusion', got {fam.archetype!r}"
    )


def test_definition_loaded():
    """longcat-image-base definition must load from its YAML file."""
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()

    defn = ModelRegistry.get_definition("longcat-image-base")
    assert defn is not None, "longcat-image-base definition not loaded"
    assert defn.family == "longcat_image"

    # Canonical checkpoint repo
    repo = defn.components["repo"]
    repo_path = repo["path"] if isinstance(repo, dict) else repo.path
    assert repo_path == "huggingface:meituan-longcat/LongCat-Image", (
        f"wrong repo path: {repo_path}"
    )

    # Verified transformer config facts (diffusers 0.39 defaults)
    arch = defn.architecture_params
    assert arch.get("transformer.num_layers") == 19
    assert arch.get("transformer.num_single_layers") == 38
    assert arch.get("transformer.joint_attention_dim") == 3584
    assert arch.get("transformer.in_channels") == 64
    assert arch.get("te.max_length") == 512


# ── Task 2: Loader manifest ──────────────────────────────────────────────────


def test_loader_manifest_components():
    """Manifest declares transformer/TE/tokenizer/text_processor/vae specs.

    LongCat has the EXTRA ``text_processor`` component (Qwen2VLProcessor)
    vs zimage — it is part of the pipeline's component contract.
    """
    import torch
    from app.engine.models.families.longcat_image.loader import LongCatImageLoader

    loader = LongCatImageLoader(torch.device("cpu"))
    specs = loader.get_component_manifest(_make_definition())
    spec_map = {s.key: s for s in specs}

    assert {"tokenizer", "text_encoder", "text_processor", "vae", "unet"} <= set(
        spec_map
    ), f"missing manifest keys; got {set(spec_map)}"

    # Transformer: native diffusers 0.39 class, mapped to "unet"
    assert "LongCatImageTransformer2DModel" in spec_map["unet"].hf_class
    assert spec_map["unet"].subfolder == "transformer"

    # Text encoder: SAME class as qwen_image (Qwen2.5-VL)
    assert "Qwen2_5_VLForConditionalGeneration" in spec_map["text_encoder"].hf_class
    assert spec_map["text_encoder"].subfolder == "text_encoder"

    # Tokenizer + processor are not torch modules
    assert spec_map["tokenizer"].is_torch_model is False
    assert spec_map["text_processor"].is_torch_model is False
    assert "Processor" in spec_map["text_processor"].hf_class
    assert spec_map["text_processor"].subfolder == "text_processor"

    # VAE: standard 16-channel AutoencoderKL
    assert "AutoencoderKL" in spec_map["vae"].hf_class
    assert spec_map["vae"].subfolder == "vae"


# ── Task 3: Driver ───────────────────────────────────────────────────────────

# Tiny LongCatImageTransformer2DModel config (CPU-friendly).
# axes_dims_rope entries must each be EVEN and sum to attention_head_dim.
_TINY_CFG = dict(
    patch_size=1,
    in_channels=4,  # packed dim → 1 latent channel × 2×2 packing
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=16,
    pooled_projection_dim=16,
    axes_dims_rope=[4, 2, 2],
)


def _build_tiny_transformer():
    from diffusers.models.transformers.transformer_longcat_image import (
        LongCatImageTransformer2DModel,
    )

    return LongCatImageTransformer2DModel(**_TINY_CFG).eval()


def _make_driver(model=None, tokenizer=None, text_encoder=None, arch=None):
    import torch
    from app.engine.models.families.longcat_image.driver import LongCatImageDriver

    definition = _make_definition(architecture_params=arch or {})
    drv = LongCatImageDriver(definition, torch.device("cpu"))
    drv.assign_components({
        "unet": model,
        "vae": None,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
    })
    return drv


def test_driver_wiring_and_dtype():
    """Driver wires components; bf16 loading; no external scheduler."""
    import torch

    drv = _make_driver(model=None)
    assert drv.get_primary_model() is None
    assert drv.init_scheduler() is None
    assert drv.resolve_loading_dtype() == torch.bfloat16
    assert drv.get_te_lora_targets() == []


def test_driver_lora_targets_cover_double_and_single_blocks():
    """Every LoRA target pattern matches >=1 named module of a tiny transformer,
    covering BOTH the double (transformer_blocks) and single
    (single_transformer_blocks) streams."""
    import torch

    model = _build_tiny_transformer()
    drv = _make_driver(model=model)
    targets = drv.get_lora_targets()

    linear_names = {
        n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    }

    for target in targets:
        matched = [
            n for n in linear_names if n == target or n.endswith("." + target)
        ]
        assert matched, f"LoRA target {target!r} matches no Linear module"

    # Both streams must be covered by the union of matched modules
    all_matched = {
        n
        for n in linear_names
        for t in targets
        if n == t or n.endswith("." + t)
    }
    assert any(n.startswith("transformer_blocks.") for n in all_matched), (
        "no double-block (transformer_blocks) modules matched"
    )
    assert any(n.startswith("single_transformer_blocks.") for n in all_matched), (
        "no single-block (single_transformer_blocks) modules matched"
    )


def test_driver_forward_divides_timesteps_by_1000_once():
    """forward_pass must pass timestep = raw/1000 to the transformer (the
    model itself multiplies ×1000 internally for the time embedding —
    an extra ×1000 or ÷1000 silently yields pure-noise LoRAs)."""
    import torch

    model = _build_tiny_transformer()
    seen = {}
    original_forward = model.forward

    def _spy(*args, **kwargs):
        seen["timestep"] = kwargs["timestep"].clone()
        return original_forward(*args, **kwargs)

    model.forward = _spy
    drv = _make_driver(model=model)

    B, C, H, W = 1, 1, 4, 4
    emb = torch.randn(B, 7, 16)
    mask = torch.ones(B, 7, dtype=torch.long)
    with torch.no_grad():
        drv.forward_pass(
            noisy_input=torch.randn(B, C, H, W),
            timesteps=torch.tensor([500.0]),
            text_embeddings=(emb, mask),
            batch={},
        )

    assert torch.allclose(seen["timestep"], torch.tensor([0.5])), (
        f"expected timestep 0.5 (raw 500 ÷ 1000), got {seen['timestep']}"
    )


def test_driver_forward_pass_shape():
    """forward_pass: [B,C,H,W] in → [B,C,H,W] finite non-degenerate out
    (patchify 2×2 → packed transformer → unpatchify)."""
    import torch

    model = _build_tiny_transformer()
    drv = _make_driver(model=model)

    B, C, H, W = 2, 1, 4, 6
    emb = torch.randn(B, 7, 16)
    mask = torch.ones(B, 7, dtype=torch.long)
    with torch.no_grad():
        pred = drv.forward_pass(
            noisy_input=torch.randn(B, C, H, W),
            timesteps=torch.tensor([500.0, 250.0]),
            text_embeddings=(emb, mask),
            batch={},
        )

    assert pred.shape == (B, C, H, W), f"unexpected shape: {pred.shape}"
    assert pred.isfinite().all(), "output contains NaN or inf"
    assert pred.float().std() > 0, "output is degenerate (zero std)"


def test_driver_compute_target_is_flow_match():
    """Flow-match target: noise - latents (standard convention, NOT inverted)."""
    import torch

    drv = _make_driver(model=None)
    latents = torch.randn(2, 4, 8, 8)
    noise = torch.randn(2, 4, 8, 8)
    target = drv.compute_target(latents, noise, torch.tensor([500.0, 100.0]))
    assert torch.equal(target, noise - latents)


def _stub_longcat_tokenizer(max_length):
    """Char-level stub tokenizer replicating the HF surface the driver uses.

    - ``tokenizer(text, add_special_tokens=False)["input_ids"]`` → one token
      per character.
    - ``tokenizer.pad({"input_ids": ...}, ...)`` → padded tensors + mask.
    """
    import torch

    class _Batch:
        def __init__(self, ids, mask):
            self.input_ids = ids
            self.attention_mask = mask

    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) % 100 + 1 for c in text]}

        def pad(self, encoded, max_length, padding, return_attention_mask,
                return_tensors):
            assert padding == "max_length"
            rows, masks = [], []
            for ids in encoded["input_ids"]:
                pad_n = max_length - len(ids)
                rows.append(list(ids) + [0] * pad_n)
                masks.append([1] * len(ids) + [0] * pad_n)
            return _Batch(
                torch.tensor(rows, dtype=torch.long),
                torch.tensor(masks, dtype=torch.long),
            )

    return _Tok()


def _stub_longcat_te(hidden_dim):
    """Stub Qwen2.5-VL TE: returns .hidden_states tuple keyed to input shape."""
    import torch

    class _TE:
        def __init__(self):
            self.captured = {}

        def __call__(self, input_ids=None, attention_mask=None,
                     output_hidden_states=False):
            assert output_hidden_states is True
            self.captured["input_ids"] = input_ids
            self.captured["attention_mask"] = attention_mask
            B, S = input_ids.shape

            class _Out:
                hidden_states = tuple(
                    torch.randn(B, S, hidden_dim) for _ in range(3)
                )

            return _Out()

        def parameters(self):
            return iter([torch.zeros(1)])

    return _TE()


def test_driver_encode_text_replicates_pipeline_encode_prompt():
    """encode_text mirrors LongCatImagePipeline._encode_prompt:

    - middle segment padded to EXACTLY te.max_length,
    - wrapped in the captioning-expert prefix/suffix template,
    - hidden_states[-1] with the prefix/suffix rows sliced off,
    - returns (B, max_length, D) embeddings + (B, max_length) mask.
    """
    import torch
    from app.engine.models.families.longcat_image import driver as drv_mod

    max_len = 16
    D = 16
    tok = _stub_longcat_tokenizer(max_len)
    te = _stub_longcat_te(D)
    drv = _make_driver(
        model=None, tokenizer=tok, text_encoder=te,
        arch={"te.max_length": max_len},
    )

    out = drv.encode_text(["a fox", "a 'quoted' cat"], torch.float32)

    prefix_len = len(drv_mod.PROMPT_TEMPLATE_PREFIX)
    suffix_len = len(drv_mod.PROMPT_TEMPLATE_SUFFIX)

    # TE consumed prefix + padded middle + suffix
    ids = te.captured["input_ids"]
    assert ids.shape == (2, prefix_len + max_len + suffix_len), (
        f"TE input shape {tuple(ids.shape)} != prefix+{max_len}+suffix"
    )
    # prefix/suffix positions are always attended
    attn = te.captured["attention_mask"]
    assert attn[:, :prefix_len].all() and attn[:, -suffix_len:].all()

    # Output: prefix/suffix sliced off → exactly max_length rows
    assert out.embeddings.shape == (2, max_len, D), (
        f"embeddings shape {tuple(out.embeddings.shape)} != (2, {max_len}, {D})"
    )
    assert out.embeddings.dtype == torch.float32
    assert out.attention_mask is not None
    assert out.attention_mask.shape == (2, max_len)
    # "a fox" → 5 char-tokens valid, rest padding
    assert out.attention_mask[0].sum().item() == len("a fox")


def test_driver_prompt_template_matches_pipeline_literal():
    """The prefix/suffix template strings must equal the pipeline literals."""
    from app.engine.models.families.longcat_image import driver as drv_mod

    assert drv_mod.PROMPT_TEMPLATE_PREFIX == (
        "<|im_start|>system\nAs an image captioning expert, generate a "
        "descriptive text prompt based on an image content, suitable for "
        "input to a text-to-image model.<|im_end|>\n<|im_start|>user\n"
    )
    assert drv_mod.PROMPT_TEMPLATE_SUFFIX == "<|im_end|>\n<|im_start|>assistant\n"


def test_driver_split_quotation_tokenizes_quoted_text_per_char():
    """split_quotation splits quoted segments (pipeline glyph-rendering path)."""
    from app.engine.models.families.longcat_image.driver import split_quotation

    parts = split_quotation("Please write 'Hello' on the blackboard.")
    assert ("'Hello'", True) in parts
    assert parts[0] == ("Please write ", False)
    # word-internal apostrophes are NOT quote pairs
    parts2 = split_quotation("it's fine")
    assert parts2 == [("it's fine", False)]


# ── Task 4: Trainer (override trio + TE cache) ───────────────────────────────


def test_trainer_setup_family_wires_driver_and_loader():
    """_setup_family must instantiate LongCatImageDriver + Loader + Saver."""
    import torch
    from app.engine.models.families.longcat_image.trainer import LongCatImageTrainer
    from app.engine.models.families.longcat_image.driver import LongCatImageDriver
    from app.engine.models.families.longcat_image.loader import LongCatImageLoader

    trainer = MagicMock(spec=LongCatImageTrainer)
    trainer.device = torch.device("cpu")
    trainer.definition = _make_definition()

    LongCatImageTrainer._setup_family(trainer)

    assert isinstance(trainer.driver, LongCatImageDriver)
    assert isinstance(trainer.loader, LongCatImageLoader)
    assert trainer.saver is not None


def test_trainer_encode_text_returns_tuple_real_seam():
    """C1/C2 contract: trainer.encode_text returns (embeddings, mask) tuple
    consumable by driver.forward_pass — real trainer methods, real driver."""
    import torch
    from app.engine.models.families.longcat_image.trainer import LongCatImageTrainer
    from app.engine.models.families.longcat_image.driver import LongCatImageDriver

    max_len = 16
    definition = _make_definition(architecture_params={"te.max_length": max_len})
    drv = LongCatImageDriver(definition, torch.device("cpu"))

    model = _build_tiny_transformer()
    drv.assign_components({
        "unet": model,
        "vae": None,
        "text_encoder": _stub_longcat_te(16),
        "tokenizer": _stub_longcat_tokenizer(max_len),
    })

    trainer = MagicMock(spec=LongCatImageTrainer)
    trainer.device = torch.device("cpu")
    trainer.definition = definition
    trainer.config = {"cache_text_embeddings": False}
    trainer.text_cache = {}
    trainer.driver = drv

    # Bind the real methods under test
    trainer.encode_text = lambda caps, dtype, batch=None: (
        LongCatImageTrainer.encode_text(trainer, caps, dtype, batch)
    )
    trainer._encode_text_direct = lambda caps, dtype: (
        LongCatImageTrainer._encode_text_direct(trainer, caps, dtype)
    )

    out = trainer.encode_text(["a longcat caption"], torch.float32)

    assert isinstance(out, tuple) and len(out) == 2, (
        f"encode_text must return a 2-tuple, got {type(out)}"
    )
    emb, mask = out
    assert emb.ndim == 3, f"embeddings must be [B, L, D], got {emb.ndim}-D"
    assert emb.shape == (1, max_len, 16)
    assert mask.shape == (1, max_len)

    # Round-trip through the real driver forward
    with torch.no_grad():
        pred = drv.forward_pass(
            noisy_input=torch.randn(1, 1, 4, 4),
            timesteps=torch.tensor([500.0]),
            text_embeddings=out,
            batch={},
        )
    assert pred.shape == (1, 1, 4, 4)
    assert pred.isfinite().all()


def test_trainer_peft_model_sync():
    """C3/C4 contract: _update_primary_model syncs driver.model, components,
    and the read-only ``transformer`` property (the historical krea2 bug)."""
    import torch
    import torch.nn as nn
    from app.engine.models.families.longcat_image.trainer import LongCatImageTrainer
    from app.engine.models.families.longcat_image.driver import LongCatImageDriver

    definition = _make_definition()
    trainer = MagicMock(spec=LongCatImageTrainer)
    trainer.device = torch.device("cpu")
    trainer.definition = definition

    drv = LongCatImageDriver(definition, torch.device("cpu"))
    tiny = _build_tiny_transformer()
    drv.assign_components({
        "unet": tiny, "vae": None, "text_encoder": None, "tokenizer": None,
    })
    trainer.driver = drv
    trainer.components = {"unet": tiny}
    trainer.model = tiny

    class _FakePEFT(nn.Module):
        pass

    wrapped = _FakePEFT()
    LongCatImageTrainer._update_primary_model(trainer, wrapped)

    assert trainer.driver.model is wrapped, "driver.model not synced"
    assert trainer.components["unet"] is wrapped, "components['unet'] not synced"
    transformer_val = LongCatImageTrainer.transformer.fget(trainer)
    assert transformer_val is wrapped, "transformer property stale"


def test_trainer_te_disk_cache_layout(tmp_path):
    """TE disk cache must be keyed under embeddings/{te_quant}/te1|te2
    (te1 = embeddings, te2 = attention masks) — qwen_image layout."""
    import os
    import torch
    from app.engine.models.families.longcat_image.trainer import LongCatImageTrainer

    trainer = MagicMock(spec=LongCatImageTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = {
        "cache_text_embeddings": True,
        "te_quantization": "fp8",
    }
    trainer.text_cache = {}
    trainer.logger = MagicMock()
    trainer._log_writer = None
    trainer.text_encoder = MagicMock()
    trainer._build_caption_hints.return_value = {"a caption": "hint0"}
    trainer._resolve_te_cache_dirs.return_value = [str(tmp_path)]
    trainer._resolve_loading_dtype.return_value = torch.float32

    def _fake_encode(captions, dtype):
        B = len(captions)
        return torch.zeros(B, 4, 8), torch.ones(B, 4, dtype=torch.long)

    trainer._encode_text_direct = _fake_encode

    LongCatImageTrainer._pre_cache_text_embeddings(trainer)

    te1 = os.path.join(str(tmp_path), "embeddings", "fp8", "te1")
    te2 = os.path.join(str(tmp_path), "embeddings", "fp8", "te2")
    assert os.path.isdir(te1) and os.listdir(te1), "te1 embedding cache missing"
    assert os.path.isdir(te2) and os.listdir(te2), "te2 mask cache missing"

    # In-memory cache stores the (emb, mask) tuple
    assert "a caption" in trainer.text_cache
    emb, mask = trainer.text_cache["a caption"]
    assert emb.shape == (4, 8)
    assert mask.shape == (4,)
