"""Tests for krea2 family: vendored transformer + conditioning helpers.

TDD order:
  1. test_krea2_vendor_imports_and_instantiates  — module exists + builds on CPU
  2. test_krea2_vendor_forward_shape             — forward pass produces correct shape
  3. test_krea2_conditioning_helpers             — pack/unpack/prepare_position_ids
  4. test_krea2_family_registered               — family in ModelRegistry
  5. test_krea2_manifest_components             — loader manifest has required keys
  6. test_krea2_definitions_loaded              — Raw + Turbo definitions loaded
"""

import torch
import pytest
from unittest.mock import MagicMock

from app.engine.core.definitions import ModelDefinition


def _make_krea2_definition(**kwargs) -> MagicMock:
    """Build a mock Krea2 ModelDefinition for loader tests."""
    definition = MagicMock(spec=ModelDefinition)
    definition.family = "krea2"
    definition.id = kwargs.get("id", "krea2-test")
    definition.components = {}
    definition.architecture_params = kwargs.get("architecture_params", {})
    return definition

# ── Tiny config shared by both model tests ──────────────────────────────────
_TINY_CFG = dict(
    in_channels=64,
    num_layers=2,
    attention_head_dim=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=256,
    timestep_embed_dim=256,
    text_hidden_dim=128,
    num_text_layers=12,
    text_num_attention_heads=4,
    text_num_key_value_heads=4,
    text_intermediate_size=128,
    num_layerwise_text_blocks=1,
    num_refiner_text_blocks=1,
    axes_dims_rope=(32, 48, 48),  # sum=128 == attention_head_dim
    rope_theta=1000.0,
    norm_eps=1e-5,
)


def test_krea2_vendor_imports_and_instantiates():
    from app.engine.models.families.krea2.vendor.transformer_krea2 import Krea2Transformer2DModel

    m = Krea2Transformer2DModel.from_config(_TINY_CFG)
    names = {n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)}
    assert any(n.endswith("attn.to_q") for n in names), f"attn.to_q not found. names={names}"
    assert any(n.endswith("ff.gate") for n in names), f"ff.gate not found. names={names}"


def test_krea2_vendor_forward_shape():
    from app.engine.models.families.krea2.vendor.transformer_krea2 import Krea2Transformer2DModel
    from app.engine.models.families.krea2.vendor.krea2_conditioning import prepare_position_ids

    m = Krea2Transformer2DModel.from_config(_TINY_CFG).eval()

    B, img_seq, txt_seq = 1, 16, 7  # 4×4 image grid
    hs = torch.randn(B, img_seq, 64)
    ehs = torch.randn(B, txt_seq, 12, 128)
    ts = torch.tensor([0.5])
    pos = prepare_position_ids(txt_seq, 4, 4, torch.device("cpu"))

    with torch.no_grad():
        out = m(
            hidden_states=hs,
            encoder_hidden_states=ehs,
            timestep=ts,
            position_ids=pos,
            return_dict=False,
        )[0]

    assert out.shape == (B, img_seq, 64), f"unexpected output shape: {out.shape}"
    assert out.isfinite().all(), "output contains NaN or inf"
    assert out.float().std() > 0, "output is degenerate (zero std)"


def test_krea2_conditioning_helpers():
    from app.engine.models.families.krea2.vendor.krea2_conditioning import (
        prepare_position_ids,
        pack_latents,
        unpack_latents,
    )

    # prepare_position_ids
    txt_seq, H, W = 7, 4, 6
    pos = prepare_position_ids(txt_seq, H, W, torch.device("cpu"))
    assert pos.shape == (txt_seq + H * W, 3), f"unexpected pos shape: {pos.shape}"
    # text rows should all be zero
    assert pos[:txt_seq].sum() == 0, "text rows should be all zeros"
    # image rows: t=0, h in [0,H), w in [0,W)
    img_pos = pos[txt_seq:]
    assert img_pos[:, 0].sum() == 0, "image t-axis should be 0"
    assert img_pos[:, 1].max() == H - 1
    assert img_pos[:, 2].max() == W - 1

    # pack_latents / unpack_latents roundtrip
    B, C, Hpx, Wpx = 2, 16, 8, 8
    patch_size = 2
    latents = torch.randn(B, C, Hpx, Wpx)
    packed = pack_latents(latents, patch_size=patch_size)
    assert packed.shape == (B, (Hpx // patch_size) * (Wpx // patch_size), C * patch_size * patch_size)

    # unpack at pixel dimensions (no VAE scale needed since we call the raw unpack)
    unpacked = unpack_latents(packed, Hpx, Wpx, patch_size=patch_size)
    # unpack_latents returns (B, C, 1, H, W) matching Krea2 pipeline convention
    assert unpacked.shape == (B, C, 1, Hpx, Wpx), f"unexpected unpack shape: {unpacked.shape}"


# ── Step A: Family Registration ──────────────────────────────────────────────

def test_krea2_family_registered():
    """krea2 family must appear in ModelRegistry with the correct archetype."""
    from app.engine.models.registry import ModelRegistry

    # Reset discovery state so this test is hermetic
    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()

    fam = ModelRegistry._families.get("krea2")
    assert fam is not None, "krea2 family not registered"
    assert fam.archetype == "latent_diffusion", (
        f"expected archetype='latent_diffusion', got {fam.archetype!r}"
    )


# ── Step B: Loader Manifest ───────────────────────────────────────────────────

def test_krea2_manifest_components():
    """Krea2Loader manifest must declare tokenizer, text_encoder, and vae."""
    from app.engine.models.families.krea2.loader import Krea2Loader

    loader = Krea2Loader(torch.device("cpu"))
    definition = _make_krea2_definition()
    specs = loader.get_component_manifest(definition)

    keys = {s.key for s in specs}
    assert {"tokenizer", "text_encoder", "vae"} <= keys, (
        f"missing required manifest keys; got {keys}"
    )

    # Verify specific hf_classes
    spec_map = {s.key: s for s in specs}
    assert "Qwen2Tokenizer" in spec_map["tokenizer"].hf_class, (
        f"tokenizer hf_class wrong: {spec_map['tokenizer'].hf_class}"
    )
    assert "Qwen3VLModel" in spec_map["text_encoder"].hf_class, (
        f"text_encoder hf_class wrong: {spec_map['text_encoder'].hf_class}"
    )
    assert "AutoencoderKLQwenImage" in spec_map["vae"].hf_class or (
        "AutoencoderKL" in spec_map["vae"].hf_class
    ), f"vae hf_class wrong: {spec_map['vae'].hf_class}"


# ── Step C: Definitions ───────────────────────────────────────────────────────

def test_krea2_definitions_loaded():
    """krea2-raw and krea2-turbo definitions must load from their YAML files."""
    from app.engine.models.registry import ModelRegistry

    # Full reset so definitions are re-scanned
    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()

    fam_defs = {d.id: d for d in ModelRegistry._definitions.values() if d.family == "krea2"}
    assert {"krea2-raw", "krea2-turbo"} <= set(fam_defs), (
        f"missing krea2 definitions; found: {set(fam_defs)}"
    )

    raw_def = fam_defs["krea2-raw"]
    turbo_def = fam_defs["krea2-turbo"]

    assert raw_def.defaults.get("is_distilled") is False, (
        f"krea2-raw.defaults.is_distilled should be False, got {raw_def.defaults.get('is_distilled')!r}"
    )
    assert turbo_def.defaults.get("is_distilled") is True, (
        f"krea2-turbo.defaults.is_distilled should be True, got {turbo_def.defaults.get('is_distilled')!r}"
    )

    # Both must have lora_targetable_modules from the arch JSON
    expected_suffixes = {
        "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_gate",
        "attn.to_out.0", "ff.gate", "ff.up", "ff.down",
    }
    for def_id, defn in [("krea2-raw", raw_def), ("krea2-turbo", turbo_def)]:
        actual_suffixes = {m.split(".", 2)[-1] for m in defn.lora_targetable_modules}
        missing = expected_suffixes - actual_suffixes
        assert not missing, (
            f"{def_id}: missing lora_targetable_modules suffixes: {missing}"
        )


# ── Phase 2 Driver Tests ─────────────────────────────────────────────────────

# Step 1 — 4-D TextEncoderOutput plumbing

def test_krea2_text_encoder_output_4d():
    """TextEncoderOutput must accept 4-D embeddings (B, seq, 12, 2560)."""
    import torch
    from app.engine.core.text_encoding import TextEncoderOutput

    emb = torch.zeros(2, 7, 12, 2560)
    out = TextEncoderOutput(
        embeddings=emb,
        attention_mask=torch.ones(2, 7, dtype=torch.long),
    )
    assert out.embeddings.shape == (2, 7, 12, 2560)


# Step 2 — Driver skeleton + component wiring

def _make_krea2_driver_with_model(model=None):
    """Build a Krea2Driver with optional tiny transformer assigned."""
    from unittest.mock import MagicMock
    import torch
    from app.engine.models.families.krea2.driver import Krea2Driver

    definition = MagicMock()
    definition.family = "krea2"
    definition.id = "krea2-test"
    definition.lora_targetable_modules = [
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.ff.gate",
    ]
    definition.architecture_params = {
        "te.text_encoder_select_layers": [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
    }

    drv = Krea2Driver(definition, torch.device("cpu"))

    components: dict = {"unet": model, "vae": None, "text_encoder": None, "tokenizer": None}
    drv.assign_components(components)
    return drv


def test_krea2_driver_wiring():
    """Driver must wire components and satisfy basic interface contracts."""
    import torch

    drv = _make_krea2_driver_with_model(model=None)

    # assign_components stores None model cleanly
    assert drv.get_primary_model() is None
    # init_scheduler returns None (flow matching, no external scheduler)
    assert drv.init_scheduler() is None
    # resolve_loading_dtype returns bf16
    assert drv.resolve_loading_dtype() == torch.bfloat16
    # lora_targets come from definition
    targets = drv.get_lora_targets()
    assert any("attn.to_q" in t for t in targets)


# Step 3 — encode_text: 12-layer stacked Qwen3-VL

def test_krea2_encode_text_stacks_12_layers():
    """encode_text must return 4-D embeddings with 12 layers on axis 2."""
    import torch
    from unittest.mock import MagicMock
    from app.engine.models.families.krea2.driver import Krea2Driver

    # Build a definition with the correct select_layers key.
    definition = MagicMock()
    definition.family = "krea2"
    definition.id = "krea2-test"
    definition.lora_targetable_modules = []
    definition.architecture_params = {
        "te.text_encoder_select_layers": [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
    }

    drv = Krea2Driver(definition, torch.device("cpu"))

    # Stub tokenizer: returns fixed-length token tensors.
    B, full_seq = 2, 46  # prefix=34 + 7 prompt + 5 suffix = 46
    stub_tok = MagicMock()

    def _fake_tokenize(texts, **kwargs):
        n = len(texts)
        tok_out = MagicMock()
        # Return a batch that looks like a real tokenizer output and can be `.to(device)`.
        tok_out.input_ids = torch.zeros(n, full_seq, dtype=torch.long)
        tok_out.attention_mask = torch.ones(n, full_seq, dtype=torch.long)
        tok_out.to = lambda device: tok_out
        return tok_out

    stub_tok.side_effect = _fake_tokenize

    # Stub text encoder: returns object with .hidden_states tuple, len >= 36.
    D = 2560  # real Krea-2 text_hidden_dim
    stub_te = MagicMock()
    # All hidden states same shape; layer indices go up to 35.
    fake_hs = tuple(torch.randn(B, full_seq, D) for _ in range(36))

    def _fake_forward(**kwargs):
        out = MagicMock()
        out.hidden_states = fake_hs
        return out

    stub_te.side_effect = _fake_forward
    stub_te.parameters = lambda: iter([torch.zeros(1)])

    drv.tokenizer = stub_tok
    drv.text_encoder = stub_te

    out = drv.encode_text(["a fox", "a cat"], torch.float32)

    # Must be 4-D with 12 layers on axis 2.
    assert out.embeddings.ndim == 4, f"expected 4-D, got {out.embeddings.ndim}-D"
    assert out.embeddings.shape[2] == 12, (
        f"expected 12 layers on axis 2, got {out.embeddings.shape[2]}"
    )
    assert out.embeddings.shape[0] == B, (
        f"batch dim mismatch: {out.embeddings.shape[0]} vs {B}"
    )
    assert out.attention_mask is not None
    assert out.attention_mask.shape[0] == B


# Step 4 — forward_pass: pack + 3-axis position ids + velocity

def test_krea2_forward_pass_shape():
    """forward_pass must return [B, C, H, W] finite non-degenerate velocity."""
    import torch
    from app.engine.models.families.krea2.vendor.transformer_krea2 import Krea2Transformer2DModel
    from app.engine.models.families.krea2.driver import Krea2Driver
    from unittest.mock import MagicMock

    # Build tiny transformer (same cfg as existing Phase-1 vendor test).
    tiny_cfg = dict(**_TINY_CFG)
    model = Krea2Transformer2DModel.from_config(tiny_cfg).eval()

    definition = MagicMock()
    definition.family = "krea2"
    definition.id = "krea2-test"
    definition.lora_targetable_modules = []
    definition.architecture_params = {
        "te.text_encoder_select_layers": [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
    }

    drv = Krea2Driver(definition, torch.device("cpu"))
    drv.assign_components({"unet": model, "vae": None, "text_encoder": None, "tokenizer": None})

    B, C, H, W = 1, 16, 8, 8  # 4×4 patch grid after p=2
    txt_seq = 7
    # 4-D text embeddings: (B, txt_seq, 12, text_hidden_dim_tiny=128)
    emb_4d = torch.randn(B, txt_seq, 12, 128)
    mask = torch.ones(B, txt_seq, dtype=torch.long)

    noisy_input = torch.randn(B, C, H, W)
    timesteps = torch.tensor([500.0])  # [0, 1000] scale

    with torch.no_grad():
        pred = drv.forward_pass(
            noisy_input=noisy_input,
            timesteps=timesteps,
            text_embeddings=(emb_4d, mask),
            batch={},
        )

    assert pred.shape == (B, C, H, W), f"unexpected shape: {pred.shape}"
    assert pred.isfinite().all(), "output contains NaN or inf"
    assert pred.float().std() > 0, "output is degenerate (zero std)"


# Step 5 — driver wired into trainer

def test_krea2_driver_wired_in_trainer():
    """Krea2Trainer._setup_family must instantiate a Krea2Driver without raising."""
    import torch
    from unittest.mock import MagicMock, patch

    # _setup_family lazy-imports Krea2Loader and Krea2Driver from their source
    # modules (not module-level names in trainer.py), so we patch the source.
    with patch(
        "app.engine.models.families.krea2.loader.Krea2Loader",
        autospec=True,
    ) as MockLoader, patch(
        "app.engine.models.families.krea2.driver.Krea2Driver",
        autospec=True,
    ) as MockDriver:
        mock_loader_instance = MagicMock()
        MockLoader.return_value = mock_loader_instance
        mock_driver_instance = MagicMock()
        MockDriver.return_value = mock_driver_instance

        from app.engine.models.families.krea2.trainer import Krea2Trainer

        trainer = MagicMock(spec=Krea2Trainer)
        trainer.device = torch.device("cpu")
        trainer.definition = MagicMock()

        # Call the real _setup_family: it must NOT raise.
        Krea2Trainer._setup_family(trainer)

        # loader and driver must both be assigned.
        assert trainer.loader is mock_loader_instance
        assert trainer.driver is mock_driver_instance
