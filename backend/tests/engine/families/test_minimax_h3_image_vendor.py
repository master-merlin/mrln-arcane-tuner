"""The vendored MiniMax-H3-Image transformer and VAE (``minimax_h3_image/vendor/``).

diffusers 0.41.0 ships neither ``MiniMaxH3ImageTransformer2DModel`` nor
``AutoencoderKLMiniMaxH3Image``; both come from huggingface/diffusers PR #15230's merge
commit ``875373ac7f5856071e345a3e394c74efccc6b9e6``. What this file holds them to:

* the config fixtures are the HF repo's ``transformer_diffusers/config.json`` and
  ``vae_diffusers/config.json`` at sha ``fbde899f`` byte for byte (pinned by sha256), so a
  shrunk or edited fixture fails here rather than silently testing another model;
* the tiny test models are built from those fixtures with the overrides written below, in
  test code only (2 layers, smaller widths);
* a CPU forward pass of the transformer returns one row per target (image) token, of width
  ``in_channels * prod(patch_size)``, through the head the step index names;
* the VAE encodes an image to ``latent_channels`` at 1/16 the size and decodes it back;
* the package re-exports both classes lazily, and REVISION names the merge commit.

The inventory rows are held by ``app/engine/tests/test_vendor_inventory.py``.
Everything runs on CPU against tiny configs, never the real weights.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
from pathlib import Path

import pytest
import torch

VENDOR_PKG = "app.engine.models.families.minimax_h3_image.vendor"
BACKEND = Path(__file__).resolve().parents[3]
VENDOR_DIR = BACKEND / "app" / "engine" / "models" / "families" / "minimax_h3_image" / "vendor"
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "minimax_h3_image_diffusers_configs"
TRANSFORMER_CONFIG = FIXTURES / "transformer_diffusers" / "config.json"
VAE_CONFIG = FIXTURES / "vae_diffusers" / "config.json"

MERGE_COMMIT = "875373ac7f5856071e345a3e394c74efccc6b9e6"

# sha256 of the two files as served by
# https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/raw/fbde899f/<folder>/config.json
FIXTURE_SHA256 = {
    TRANSFORMER_CONFIG: "f8a1f3f87e708d0d4268d7c1abd040a7af9f2b7c435827f14e0a2d6fc260bcf9",
    VAE_CONFIG: "6d809326cc9a46763e810777d71d2fdddbd5df820d5644e8fe778ea97d9913a4",
}

# The tiny-model overrides. Everything not named here stays the checkpoint's value
# (in_channels 24, patch_size [1, 2, 2], num_pdd_heads 6, the eps values, rope_theta).
# rope_freq_dim 4 rotates 2 * 3 * 4 = 24 of the 32 head channels.
TRANSFORMER_TINY = {
    "num_attention_heads": 2,
    "attention_head_dim": 32,
    "hidden_size": 48,
    "num_layers": 2,
    "num_refiner_layers": 2,
    "ffn_dim": 64,
    "text_dim": 16,
    "freq_dim": 32,
    "time_embed_hidden_dim": 32,
    "time_embed_dim": 16,
    "rope_freq_dim": 4,
}
# Spatial downsampling (2, 2, 2, 2, 1, 1) and the 24 latent channels with their mean/std stay
# the checkpoint's; only the widths and depths shrink. The decoder's rotary dim is
# 16 * 0.5 = 8, divisible by 2 * 2 axes.
VAE_TINY = {
    "block_out_channels": [8, 8, 8, 8, 16, 16],
    "layers_per_block": 1,
    "norm_num_groups": 4,
    "decoder_num_layers": 2,
    "decoder_num_attention_heads": 2,
    "decoder_attention_head_dim": 16,
}


def _fixture(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tiny_config(path: Path, overrides: dict) -> dict:
    config = _fixture(path)
    unknown = sorted(set(overrides) - set(config))
    assert not unknown, f"tiny overrides name keys the checkpoint config lacks: {unknown}"
    config.update(overrides)
    return config


def _vendor(name: str):
    return getattr(importlib.import_module(VENDOR_PKG), name)


@pytest.mark.parametrize("path", sorted(FIXTURE_SHA256), ids=lambda p: p.parent.name)
def test_fixture_is_the_checkpoint_config_byte_for_byte(path: Path):
    assert path.is_file(), f"missing fixture {path}"
    have = hashlib.sha256(path.read_bytes()).hexdigest()
    assert have == FIXTURE_SHA256[path], (
        f"{path.name} in {path.parent.name} is not the file at MiniMaxAI/MiniMax-H3-Image@fbde899f "
        f"(sha256 {have}); the fixture is kept verbatim, tiny overrides live in this test"
    )


def test_fixtures_name_the_vendored_classes():
    assert _fixture(TRANSFORMER_CONFIG)["_class_name"] == "MiniMaxH3ImageTransformer2DModel"
    assert _fixture(VAE_CONFIG)["_class_name"] == "AutoencoderKLMiniMaxH3Image"


def test_the_pinned_diffusers_lacks_both_classes():
    """The reason the files are vendored; when this fails, the vendor can retire."""
    import diffusers

    for name in ("MiniMaxH3ImageTransformer2DModel", "AutoencoderKLMiniMaxH3Image"):
        assert not hasattr(diffusers, name), (
            f"diffusers {diffusers.__version__} ships {name}: retire the vendored copy"
        )


def test_the_package_reexports_lazily():
    """The package's top level imports nothing, so it never raises at import; the classes
    resolve on first attribute access, and an unknown name raises AttributeError."""
    tree = ast.parse((VENDOR_DIR / "__init__.py").read_text(encoding="utf-8"))
    top_level_imports = [
        ast.unparse(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not top_level_imports, f"the vendor package imports at its top level: {top_level_imports}"

    package = importlib.import_module(VENDOR_PKG)
    assert sorted(package.__all__) == ["AutoencoderKLMiniMaxH3Image", "MiniMaxH3ImageTransformer2DModel"]
    for name in package.__all__:
        cls = getattr(package, name)
        assert cls.__name__ == name
        assert cls.__module__.startswith(VENDOR_PKG + ".")
    with pytest.raises(AttributeError):
        package.NoSuchClass  # noqa: B018 - the access is the test


def test_revision_names_the_merge_commit_for_both_files():
    revision = (VENDOR_DIR / "REVISION").read_text(encoding="utf-8")
    for upstream in (
        "src/diffusers/models/transformers/transformer_minimax_h3_image.py",
        "src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_image.py",
    ):
        assert f"huggingface/diffusers@{MERGE_COMMIT} {upstream}" in revision, (
            f"REVISION does not pin {upstream} at {MERGE_COMMIT}"
        )


def _tiny_transformer():
    cls = _vendor("MiniMaxH3ImageTransformer2DModel")
    torch.manual_seed(0)
    return cls.from_config(_tiny_config(TRANSFORMER_CONFIG, TRANSFORMER_TINY)).eval()


def _packed_inputs(model, batch: int = 2, n_text: int = 5, latent_hw: tuple[int, int] = (4, 6)):
    """One packed ``[text | image]`` sequence at one noise level, the layout the model expects."""
    cfg = model.config
    pt, ph, pw = cfg.patch_size
    rows_h, rows_w = latent_hw[0] // ph, latent_hw[1] // pw
    n_image = rows_h * rows_w
    patch_dim = cfg.in_channels * pt * ph * pw
    seq_len = n_text + n_image
    gen = torch.Generator().manual_seed(1)

    text_pos = torch.stack([torch.arange(n_text)] * 3, dim=-1)
    hh, ww = torch.meshgrid(torch.arange(rows_h), torch.arange(rows_w), indexing="ij")
    image_pos = torch.stack(
        [torch.full((n_image,), n_text), hh.flatten() + n_text, ww.flatten() + n_text], dim=-1
    )
    return {
        "hidden_states": torch.randn(batch, n_image, patch_dim, generator=gen),
        "encoder_hidden_states": torch.randn(batch, n_text, cfg.text_dim, generator=gen),
        "timestep": torch.tensor([0.25]),
        "timestep_indices": torch.zeros(seq_len, dtype=torch.long),
        "token_tags": torch.cat(
            [torch.ones(n_text, dtype=torch.long), torch.zeros(n_image, dtype=torch.long)]
        ),
        "position_ids": torch.cat([text_pos, image_pos], dim=0),
        "image_indices": torch.arange(n_text, seq_len),
        "text_indices": torch.arange(n_text),
    }


def test_tiny_transformer_keeps_the_checkpoint_structure():
    model = _tiny_transformer()
    assert len(model.transformer_blocks) == 2
    assert len(model.token_refiner.refiner_blocks) == 2
    # num_pdd_heads comes from the fixture (6), one output head per sampling step.
    assert len(model.pdd_proj_out) == _fixture(TRANSFORMER_CONFIG)["num_pdd_heads"] == 6
    patch_dim = 24 * 1 * 2 * 2
    assert all(h.out_features == patch_dim for h in model.pdd_proj_out)


def test_transformer_cpu_forward_returns_the_target_rows():
    model = _tiny_transformer()
    inputs = _packed_inputs(model)
    with torch.no_grad():
        out = model(**inputs, pdd_step_index=3).sample
    assert out.shape == inputs["hidden_states"].shape == (2, 6, 96)
    assert torch.isfinite(out).all()


def test_transformer_forward_runs_through_the_named_head():
    model = _tiny_transformer()
    inputs = _packed_inputs(model)
    with torch.no_grad():
        for i, head in enumerate(model.pdd_proj_out):
            head.weight.zero_()
            head.bias.fill_(float(i))
        for step in range(len(model.pdd_proj_out)):
            out = model(**inputs, pdd_step_index=step).sample
            assert torch.equal(out, torch.full_like(out, float(step))), (
                f"pdd_step_index={step} did not project through head {step}"
            )


def test_tiny_vae_encodes_and_decodes_an_image():
    cls = _vendor("AutoencoderKLMiniMaxH3Image")
    torch.manual_seed(0)
    vae = cls.from_config(_tiny_config(VAE_CONFIG, VAE_TINY)).eval()
    ratio = math.prod(vae.config.spatial_downsample_factors)
    assert ratio == vae.spatial_compression_ratio == 16
    image = torch.randn(1, 3, 32, 48, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        posterior = vae.encode(image).latent_dist
        latent = posterior.mode()
        decoded = vae.decode(latent).sample
    assert latent.shape == (1, vae.config.latent_channels, 32 // ratio, 48 // ratio) == (1, 24, 2, 3)
    assert decoded.shape == image.shape
    assert torch.isfinite(decoded).all()
