"""Vendor tests for the qwen_image2 family (LANE-132 Task 1 -- the vendor drop only).

``qwen_image2/vendor/`` carries diffusers ``6256aa7666`` ("Add Qwen-Image 2.1
(#14804)", Apache-2.0): the transformer, the VAE and the encode/pack helpers of
the pipeline. diffusers 0.41.0 (the pinned release since LANE-149) exports them;
LANE-150 retires the copy. These tests pin:

  1. The vendored transformer builds from the Qwen-Image-2.1
     ``transformer/config.json`` values (Evidence:
     ``.agent/workdir/lane-104/step1-weights.log`` line 4) at a tiny depth, and a
     narrow CPU copy runs one forward over the pipeline's own joint-sequence
     layout.
  2. The vendored VAE builds from the ``vae/config.json`` values at a narrow
     width and compresses 16x spatially into 64 channels.
  3. The pipeline helpers pack/unpack losslessly and normalise VAE latents with
     the config's mean/std.
  4. Retirement: the day diffusers exports any of the vendored classes, the
     vendor copy is due for retirement and a test says so by name.
  5. Provenance: each vendored file keeps the upstream Apache header and names
     the upstream commit; NOTICE carries the entry.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

VENDOR = "app.engine.models.families.qwen_image2.vendor"
VENDOR_DIR = Path(__file__).resolve().parents[1] / "models" / "families" / "qwen_image2" / "vendor"
REPO_ROOT = Path(__file__).resolve().parents[4]
UPSTREAM_COMMIT = "6256aa7666"

# Qwen/Qwen-Image-2.1 @ 790c92633540, transformer/config.json and vae/config.json
# (Evidence: .agent/workdir/lane-104/step1-weights.log lines 4-5).
TRANSFORMER_CONFIG = {
    "_class_name": "QwenImage21Transformer2DModel",
    "_diffusers_version": "0.37.0.dev0",
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "in_channels": 64,
    "num_attention_heads": 32,
    "num_layers": 32,
    "out_channels": 64,
    "patch_size": 1,
}
VAE_CONFIG = {
    "_class_name": "AutoencoderKLQwenImage21",
    "_diffusers_version": "0.37.0.dev0",
    "base_dim": 96,
    "dim_mult": [1, 2, 4, 8, 8],
    "in_channels": 4,
    "out_channels": 4,
    "patch_size": None,
    "temperal_downsample": [False, True, True, True],
    "z_dim": 64,
}

# The classes the vendor copy stands in for. When diffusers exports one, the
# family should use it and ``qwen_image2/vendor/`` retires.
VENDORED_UPSTREAM_CLASSES = (
    "QwenImage21Transformer2DModel",
    "AutoencoderKLQwenImage21",
    "QwenImage21Pipeline",
)


def _diffusers_exports(name: str) -> bool:
    """The retirement probe: does the installed diffusers export ``name``?"""
    import diffusers

    return hasattr(diffusers, name)


# --- 1. transformer ----------------------------------------------------------


def test_transformer_builds_from_the_21_config_at_one_layer():
    from app.engine.models.families.qwen_image2.vendor.transformer_qwenimage21 import (
        QwenImage21Transformer2DModel,
    )

    config = {**TRANSFORMER_CONFIG, "num_layers": 1}
    # Full width is ~0.3 B parameters per layer; the meta device builds the
    # graph and its shapes without allocating them.
    with torch.device("meta"):
        model = QwenImage21Transformer2DModel.from_config(config)

    assert model.config.num_layers == 1
    assert model.config.num_attention_heads == 32
    assert model.config.attention_head_dim == 128
    assert list(model.config.axes_dims_rope) == [16, 56, 56]
    assert model.inner_dim == 32 * 128
    assert len(model.transformer_blocks) == 1
    assert tuple(model.img_in.weight.shape) == (4096, 64)
    assert tuple(model.proj_out.weight.shape) == (64, 4096)
    # context_in_dim is not in the checkpoint config: the default must be the
    # Qwen3-VL hidden size the 2.1 text path produces.
    assert model.config.context_in_dim == 4096


def _tiny_transformer():
    from app.engine.models.families.qwen_image2.vendor.transformer_qwenimage21 import (
        QwenImage21Transformer2DModel,
    )

    torch.manual_seed(0)
    # Real head_dim and rope axes (they must sum to it), 2 heads, 2 layers, a
    # narrow text projection: everything else from the 2.1 config.
    config = {**TRANSFORMER_CONFIG, "num_layers": 2, "num_attention_heads": 2, "context_in_dim": 32}
    return QwenImage21Transformer2DModel.from_config(config).eval()


def test_tiny_transformer_forward_on_cpu_matches_the_pipeline_layout():
    model = _tiny_transformer()
    batch, text_len, h, w = 2, 5, 4, 6
    target_tokens = h * w
    hidden_states = torch.randn(batch, target_tokens, 64)
    encoder_hidden_states = torch.randn(batch, text_len, 32)
    # The pipeline appends one img_mask slot per 2x2 group of target latents
    # (upstream pipeline_qwenimage21.py lines 737-743).
    img_mask = torch.cat(
        [torch.zeros(batch, text_len, dtype=torch.bool), torch.ones(batch, target_tokens // 4, dtype=torch.bool)],
        dim=1,
    )
    timestep = torch.tensor([0.7, 0.2])

    with torch.no_grad():
        out = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_shapes=[[(1, h, w)]] * batch,
            img_mask=img_mask,
            return_dict=False,
        )[0]
        out_other_t = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=torch.tensor([0.1, 0.9]),
            img_shapes=[[(1, h, w)]] * batch,
            img_mask=img_mask,
            return_dict=False,
        )[0]

    # The transformer returns the WHOLE joint sequence (text + target); the
    # pipeline keeps the last target_tokens positions (upstream
    # pipeline_qwenimage21.py line 783: `noise_pred[:, -latents.size(1):]`).
    # A trainer that forgets the slice would compute its loss on text slots.
    assert out.shape == (batch, text_len + target_tokens, 64)
    noise_pred = out[:, -target_tokens:]
    assert torch.isfinite(noise_pred).all()
    # The timestep reaches the target tokens: the modulation path is wired.
    assert not torch.allclose(noise_pred, out_other_t[:, -target_tokens:])
    # causal_condition: text tokens modulate from t=0, so they do NOT depend on
    # the sampled timestep (upstream transformer docstring, "causal_condition").
    assert torch.allclose(out[:, :text_len], out_other_t[:, :text_len])


def test_transformer_declares_gradient_checkpointing_and_peft():
    from diffusers.loaders import PeftAdapterMixin

    model = _tiny_transformer()
    assert model._supports_gradient_checkpointing is True
    assert isinstance(model, PeftAdapterMixin)


# --- 2. VAE ------------------------------------------------------------------


def _tiny_vae():
    from app.engine.models.families.qwen_image2.vendor.autoencoder_kl_qwenimage21 import (
        AutoencoderKLQwenImage21,
    )

    torch.manual_seed(0)
    # The 2.1 layout (5 levels, 64 latent channels, 4 image channels) at a
    # narrow width.
    config = {**VAE_CONFIG, "base_dim": 8, "decoder_base_dim": 8, "num_res_blocks": 1}
    return AutoencoderKLQwenImage21.from_config(config).eval()


def test_vae_compresses_16x_into_64_channels_and_decodes_back():
    vae = _tiny_vae()
    assert vae.config.z_dim == 64
    assert len(vae.config.latents_mean) == 64
    assert len(vae.config.latents_std) == 64

    image = torch.rand(1, 4, 1, 32, 48) * 2 - 1
    with torch.no_grad():
        posterior = vae.encode(image).latent_dist
        latents = posterior.mode()
        decoded = vae.decode(latents).sample

    assert latents.shape == (1, 64, 1, 2, 3)
    assert decoded.shape == image.shape
    assert torch.isfinite(decoded).all()


# --- 3. pipeline helpers -----------------------------------------------------


def test_pack_and_unpack_latents_round_trip():
    from app.engine.models.families.qwen_image2.vendor.pipeline_qwenimage21 import QwenImage21PipelineHelpers

    helpers = QwenImage21PipelineHelpers()
    assert helpers.vae_scale_factor == 16
    latents = torch.randn(2, 64, 1, 4, 6)
    packed = helpers._pack_latents(latents, 2, 64, 4, 6)
    # 2.1 is unpatched: one token per latent pixel, 64 channels per token.
    assert packed.shape == (2, 24, 64)
    assert torch.equal(packed[0, 7], latents[0, :, 0, 1, 1])
    unpacked = helpers._unpack_latents(packed, 4 * 16, 6 * 16, helpers.vae_scale_factor)
    assert torch.equal(unpacked, latents)


def test_encode_vae_image_normalises_with_the_config_mean_and_std():
    from app.engine.models.families.qwen_image2.vendor.pipeline_qwenimage21 import (
        QwenImage21PipelineHelpers,
        retrieve_latents,
    )

    vae = _tiny_vae()
    helpers = QwenImage21PipelineHelpers(vae=vae)
    assert helpers.latent_channels == 64
    image = torch.rand(1, 4, 1, 32, 32) * 2 - 1
    with torch.no_grad():
        normalised = helpers._encode_vae_image(image, generator=None)
        raw = retrieve_latents(vae.encode(image), sample_mode="argmax")
    mean = torch.tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 64, 1, 1, 1)
    assert torch.allclose(normalised, (raw - mean) / std)
    assert not torch.allclose(normalised, raw)


def test_calculate_shift_matches_the_upstream_anchors():
    from app.engine.models.families.qwen_image2.vendor.pipeline_qwenimage21 import calculate_shift

    assert math.isclose(calculate_shift(256), 0.5)
    assert math.isclose(calculate_shift(4096), 1.15)


# --- 4. retirement -----------------------------------------------------------


@pytest.mark.parametrize("name", VENDORED_UPSTREAM_CLASSES)
def test_vendor_is_due_for_retirement_on_the_pinned_release(name):
    import diffusers

    assert _diffusers_exports(name), (
        f"LANE-149: diffusers {diffusers.__version__} does not export {name}; "
        "the pinned release must ship the classes qwen_image2/vendor/ stands in for (LANE-150 retires the copy)"
    )


# --- 5. provenance -----------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    ["transformer_qwenimage21.py", "autoencoder_kl_qwenimage21.py", "pipeline_qwenimage21.py"],
)
def test_vendored_file_keeps_the_upstream_header_and_names_the_commit(filename):
    text = (VENDOR_DIR / filename).read_text(encoding="utf-8")
    head = text[:3000]
    assert "The HuggingFace Team. All rights reserved." in head
    assert 'Licensed under the Apache License, Version 2.0 (the "License");' in head
    assert UPSTREAM_COMMIT in head
    # Absolute diffusers imports only: upstream's parent-relative imports would
    # resolve inside our package and fail at import.
    assert "from .." not in text


def test_notice_names_the_qwen_image2_vendor_copy():
    notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "qwen_image2" in notice
    assert UPSTREAM_COMMIT in notice


def test_vendor_imports_without_optional_flex_attention_breaking_startup():
    import importlib

    for name in ("transformer_qwenimage21", "autoencoder_kl_qwenimage21", "pipeline_qwenimage21"):
        importlib.import_module(f"{VENDOR}.{name}")


# --- 6. LANE-150 parity against the release (fixture-first retirement) --------
#
# The release classes must reproduce the vendored copy's output on a pinned seed
# before the copy is deleted. ``vendor_parity.npz`` holds the inputs and the
# vendored outputs (written on the first run); ``lora_keys.json`` the sorted key
# list the saver emits for the tiny vendored model.

import importlib.util  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

_TESTS = Path(__file__).resolve().parent
_FIXTURES = _TESTS / "fixtures" / "qwen_image2"
PARITY_NPZ = _FIXTURES / "vendor_parity.npz"
LORA_KEYS = _FIXTURES / "lora_keys.json"
PARITY_ATOL = 1e-6


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert_parity(tensor: str, release: torch.Tensor, reference: np.ndarray, atol: float = PARITY_ATOL):
    diff = float(np.abs(release.detach().float().numpy() - reference).max())
    assert diff <= atol, f"LANE-150 parity qwen_image2/{tensor}: max|release - fixture| = {diff:.3e} > {atol}"


def _fixture(produce) -> dict:
    """The stored arrays; ``produce`` (the vendored copy) adds its keys if absent."""
    stored = {}
    if PARITY_NPZ.exists():
        with np.load(PARITY_NPZ) as z:
            stored = {k: z[k] for k in z.files}
    if not PARITY_NPZ.exists() or not set(produce_keys(produce)) <= set(stored):
        stored.update(produce())
        PARITY_NPZ.parent.mkdir(parents=True, exist_ok=True)
        np.savez(PARITY_NPZ, **stored)
    return stored


def produce_keys(produce) -> tuple:
    return produce.keys


def _transformer_inputs():
    g = torch.Generator().manual_seed(150)
    b, text_len, h, w, c = 2, 5, 4, 6, 8
    tokens = h * w
    return {
        "t_hidden": torch.randn(b, tokens, c, generator=g),
        "t_text": torch.randn(b, text_len, 32, generator=g),
        "t_timestep": torch.tensor([0.7, 0.2]),
        "t_mask": torch.cat(
            [torch.zeros(b, text_len, dtype=torch.bool), torch.ones(b, tokens // 4, dtype=torch.bool)], dim=1
        ),
    }, (b, h, w)


def _run_transformer(model, inputs, hw):
    b, h, w = hw
    with torch.no_grad():
        return model(
            hidden_states=inputs["t_hidden"],
            encoder_hidden_states=inputs["t_text"],
            timestep=inputs["t_timestep"],
            img_shapes=[[(1, h, w)]] * b,
            img_mask=inputs["t_mask"],
            return_dict=False,
        )[0]


def test_release_transformer_matches_the_vendored_forward():
    import diffusers

    tiny = _load_module("qwen_image2_tiny_parity", _FIXTURES / "tiny.py")
    inputs, hw = _transformer_inputs()

    def produce():
        vendored = tiny.tiny_transformer().eval()
        return {**{k: v.numpy() for k, v in inputs.items() if k != "t_mask"},
                "t_mask": inputs["t_mask"].numpy(),
                "t_out": _run_transformer(vendored, inputs, hw).numpy()}

    produce.keys = ("t_out",)
    stored = _fixture(produce)
    vendored = tiny.tiny_transformer().eval()  # seed 0, the fixture's weights
    release = diffusers.QwenImage21Transformer2DModel.from_config(dict(vendored.config)).eval()
    release.load_state_dict(vendored.state_dict(), strict=True)
    out = _run_transformer(release, inputs, hw)
    _assert_parity("transformer", out, stored["t_out"])
    _assert_parity("transformer(vendored)", _run_transformer(vendored, inputs, hw), stored["t_out"])


def test_release_vae_matches_the_vendored_encode_decode():
    import diffusers

    tiny = _load_module("qwen_image2_tiny_parity_vae", _FIXTURES / "tiny.py")
    g = torch.Generator().manual_seed(150)
    image = torch.rand(1, 4, 1, 32, 48, generator=g) * 2 - 1

    def run(vae):
        with torch.no_grad():
            latents = vae.encode(image).latent_dist.mode()
            return latents, vae.decode(latents).sample

    def produce():
        lat, dec = run(tiny.tiny_vae())
        return {"v_image": image.numpy(), "v_latents": lat.numpy(), "v_decoded": dec.numpy()}

    produce.keys = ("v_decoded",)
    stored = _fixture(produce)
    vendored = tiny.tiny_vae()
    release = diffusers.AutoencoderKLQwenImage21.from_config(dict(vendored.config)).eval()
    release.load_state_dict(vendored.state_dict(), strict=True)
    lat, dec = run(release)
    _assert_parity("vae_latents", lat, stored["v_latents"])
    _assert_parity("vae_decoded", dec, stored["v_decoded"])
    v_lat, v_dec = run(vendored)
    _assert_parity("vae_latents(vendored)", v_lat, stored["v_latents"])
    _assert_parity("vae_decoded(vendored)", v_dec, stored["v_decoded"])


def test_saver_key_set_is_pinned(tmp_path):
    from safetensors import safe_open

    training = _load_module("qwen_image2_training_for_keys", _TESTS / "test_qwen_image2_training.py")
    path = training._trained_lora_file(tmp_path)
    with safe_open(str(path), "pt") as f:
        keys = sorted(f.keys())
    if not LORA_KEYS.exists():
        LORA_KEYS.write_text(json.dumps(keys, indent=1) + "\n", encoding="utf-8")
    pinned = json.loads(LORA_KEYS.read_text(encoding="utf-8"))
    extra, missing = sorted(set(keys) - set(pinned)), sorted(set(pinned) - set(keys))
    assert keys == pinned, (
        f"LANE-150 lora keys qwen_image2: saved keys differ from fixtures/qwen_image2/lora_keys.json: "
        f"+{extra} -{missing}"
    )
