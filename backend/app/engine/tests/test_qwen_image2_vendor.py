"""Tests for the qwen_image2 family on diffusers 0.41.0's own classes (LANE-150).

LANE-132 vendored diffusers ``6256aa7666`` ("Add Qwen-Image 2.1 (#14804)") as
``qwen_image2/vendor/``; diffusers 0.41.0 ships the transformer, the VAE and the
pipeline helpers, so LANE-150 deleted the copy. These tests pin:

  1. The release transformer builds from the Qwen-Image-2.1
     ``transformer/config.json`` values (Evidence:
     ``.agent/workdir/lane-104/step1-weights.log`` line 4) and a tiny CPU copy
     runs one forward over the pipeline's own joint-sequence layout.
  2. The release VAE compresses 16x spatially into 64 channels.
  3. The pipeline helpers pack/unpack losslessly and normalise VAE latents.
  4. Parity: the release output equals the pinned-seed fixture
     (``fixtures/qwen_image2/vendor_parity.npz``, captured from the release
     copy before its deletion, LANE-150 commit 1) at ``atol=1e-6``.
  5. Retirement: the vendor package is gone, the loader names the release
     classes, nothing imports them at registry import (ARCHITECTURE D1).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch


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

# --- 1. transformer ----------------------------------------------------------


def test_transformer_builds_from_the_21_config_at_one_layer():
    from diffusers import QwenImage21Transformer2DModel

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
    from diffusers import QwenImage21Transformer2DModel

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
    from diffusers import AutoencoderKLQwenImage21

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
    from diffusers import QwenImage21Pipeline

    latents = torch.randn(2, 64, 1, 4, 6)
    packed = QwenImage21Pipeline._pack_latents(latents, 2, 64, 4, 6)
    # 2.1 is unpatched: one token per latent pixel, 64 channels per token.
    assert packed.shape == (2, 24, 64)
    assert torch.equal(packed[0, 7], latents[0, :, 0, 1, 1])
    unpacked = QwenImage21Pipeline._unpack_latents(packed, 4 * 16, 6 * 16, 16)
    assert torch.equal(unpacked, latents)


def test_calculate_shift_matches_the_upstream_anchors():
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

    assert math.isclose(calculate_shift(256), 0.5)
    assert math.isclose(calculate_shift(4096), 1.15)


# --- 4. LANE-150 parity against the pinned-seed fixture ------------------------
#
# ``vendor_parity.npz`` holds the inputs and the outputs the VENDORED copy
# produced (captured in LANE-150 commit 1, before the copy was deleted);
# ``lora_keys.json`` the sorted key list the saver emitted for the tiny vendored
# model. Both are read-only here: a missing fixture is a failure, never a
# silent re-capture from the release.

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


def _fixture() -> dict:
    assert PARITY_NPZ.exists(), f"LANE-150: the parity fixture {PARITY_NPZ} is missing"
    with np.load(PARITY_NPZ) as z:
        return {k: z[k] for k in z.files}


def _transformer_inputs(stored: dict):
    t = {k: torch.from_numpy(stored[k]) for k in ("t_hidden", "t_text", "t_timestep", "t_mask")}
    b, tokens = t["t_hidden"].shape[:2]
    return t, (b, 4, 6)


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
    stored = _fixture()
    inputs, hw = _transformer_inputs(stored)
    release = tiny.tiny_transformer().eval()  # the release class, seed 0 = the fixture's weights
    assert type(release) is diffusers.QwenImage21Transformer2DModel
    _assert_parity("transformer", _run_transformer(release, inputs, hw), stored["t_out"])


def test_release_vae_matches_the_vendored_encode_decode():
    import diffusers

    tiny = _load_module("qwen_image2_tiny_parity_vae", _FIXTURES / "tiny.py")
    stored = _fixture()
    image = torch.from_numpy(stored["v_image"])
    release = tiny.tiny_vae()
    assert type(release) is diffusers.AutoencoderKLQwenImage21
    with torch.no_grad():
        lat = release.encode(image).latent_dist.mode()
        dec = release.decode(lat).sample
    _assert_parity("vae_latents", lat, stored["v_latents"])
    _assert_parity("vae_decoded", dec, stored["v_decoded"])


def test_saver_key_set_is_pinned(tmp_path):
    from safetensors import safe_open

    assert LORA_KEYS.exists(), f"LANE-150: {LORA_KEYS} is missing"
    training = _load_module("qwen_image2_training_for_keys", _TESTS / "test_qwen_image2_training.py")
    path = training._trained_lora_file(tmp_path)
    with safe_open(str(path), "pt") as f:
        keys = sorted(f.keys())
    pinned = json.loads(LORA_KEYS.read_text(encoding="utf-8"))
    extra, missing = sorted(set(keys) - set(pinned)), sorted(set(pinned) - set(keys))
    assert keys == pinned, (
        f"LANE-150 lora keys qwen_image2: saved keys differ from fixtures/qwen_image2/lora_keys.json: "
        f"+{extra} -{missing}"
    )


# --- 5. LANE-150 retirement record ---------------------------------------------

import subprocess  # noqa: E402
import sys  # noqa: E402

_BACKEND = Path(__file__).resolve().parents[3]
_FAMILY_DIR = Path(__file__).resolve().parents[1] / "models" / "families" / "qwen_image2"
_RETIRED_FILE = "backend/app/engine/models/families/qwen_image2/vendor/transformer_qwenimage21.py"


def test_vendor_copy_is_retired_and_the_loader_names_the_release():
    from app.engine.models.families.qwen_image2.loader import QwenImage2Loader

    manifest = QwenImage2Loader(torch.device("cpu")).get_component_manifest(None)
    declared = {spec.key: spec.hf_class for spec in manifest}
    problems = []
    if (_FAMILY_DIR / "vendor" / "transformer_qwenimage21.py").exists():
        problems.append(f"{_RETIRED_FILE} still exists")
    if declared["unet"] != "diffusers.QwenImage21Transformer2DModel" or declared["vae"] != (
        "diffusers.AutoencoderKLQwenImage21"
    ):
        problems.append(
            f"the loader must name diffusers.QwenImage21Transformer2DModel, got {declared['unet']}"
        )
    elif problems:
        problems[-1] += "; the loader must name diffusers.QwenImage21Transformer2DModel"
    assert not problems, "LANE-150: " + "; ".join(problems)


def test_no_vendor_import_remains():
    """No import of the retired vendor package of this family under backend/, no ``vendor/`` directory."""
    assert not (_FAMILY_DIR / "vendor").exists(), "LANE-150: qwen_image2/vendor/ still exists"
    needle = "qwen_image2" + ".vendor"
    hits = [
        str(p.relative_to(_BACKEND))
        for p in _BACKEND.rglob("*.py")
        if "venv" not in p.relative_to(_BACKEND).parts
        and ".ruff_cache" not in p.parts
        and needle in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert not hits, f"LANE-150: modules still import the retired qwen_image2 vendor package: {hits}"


def test_the_release_transformer_is_not_imported_at_registry_import():
    """ARCHITECTURE D1: the release module guards flex_attention with ``except ImportError``
    only (transformer_qwenimage21.py:46-51), so the project resolves it lazily (hf_class
    string) and never at startup. A fresh interpreter: this process may have imported diffusers."""
    module = "diffusers.models.transformers.transformer_qwenimage21"
    # safety-net print: the fresh-interpreter probe's stdout IS the assertion
    code = (
        "import sys, app.engine.models.registry, app.engine.models.families.qwen_image2; "
        f"print({module!r} in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=_BACKEND, capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    assert proc.stdout.strip().splitlines()[-1] == "False", (
        f"LANE-150 (ARCHITECTURE D1): {module} is in sys.modules after importing the registry; "
        "its flex_attention guard catches ImportError only (transformer_qwenimage21.py:46-51), "
        "so it must load lazily"
    )


def test_notice_no_longer_lists_the_retired_qwen_image2_vendor_copy():
    notice = (Path(__file__).resolve().parents[4] / "NOTICE").read_text(encoding="utf-8")
    assert "qwen_image2" not in notice, "LANE-150: NOTICE still names the retired qwen_image2 vendor copy"
    assert "6256aa7666" not in notice
