"""LANE-150 parity: krea2's conditioning helpers on diffusers 0.41.0's ``Krea2Pipeline``.

``fixtures/krea2/vendor_parity.npz`` holds the OUTPUTS the vendored
``vendor/krea2_conditioning.py`` produced on pinned inputs (seed 150, CPU, the tiny
Qwen3-VL text encoder of ``fixtures/qwen_image2/tiny.py``); ``record_vendored()`` is the one
writer. Each test builds the release's pipeline (all heavy components ``None`` except the tiny
text encoder and tokenizer) and requires the stored arrays back within 1e-6.
``lora_keys.json`` pins the key list ``Krea2Saver`` writes for the PEFT-wrapped tiny model.

Recorded difference (VENDOR_INVENTORY.md): the vendored ``unpack_latents(latents, height, width)``
takes LATENT dims; the release's ``_unpack_latents`` takes PIXEL dims and divides by
``vae_scale_factor * patch_size``. The release call below therefore feeds latent dims * 8.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

_TESTS = Path(__file__).resolve().parent
_FIXTURES = _TESTS / "fixtures" / "krea2"
PARITY_NPZ = _FIXTURES / "vendor_parity.npz"
LORA_KEYS = _FIXTURES / "lora_keys.json"
ATOL = 1e-6
PATCH = 2
VAE_SCALE = 8
SELECT_LAYERS = (0, 1, 2)  # indices into the tiny encoder's 2-layer hidden_states tuple
MAX_SEQ = 256  # the byte-level prefix is ~150 tokens; a shorter budget would truncate the prompt
CAPTIONS = ["a red fox in snow", "two cups on a wooden table, soft light"]
LAT_B, LAT_C, LAT_H, LAT_W = 2, 16, 8, 12

_TINY_CFG = dict(
    in_channels=64, num_layers=2, attention_head_dim=128, num_attention_heads=4,
    num_key_value_heads=2, intermediate_size=256, timestep_embed_dim=256, text_hidden_dim=128,
    num_text_layers=12, text_num_attention_heads=4, text_num_key_value_heads=4,
    text_intermediate_size=128, num_layerwise_text_blocks=1, num_refiner_text_blocks=1,
    axes_dims_rope=(32, 48, 48), rope_theta=1000.0, norm_eps=1e-5,
)
_LORA_TARGETS = [
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_gate", "attn.to_out.0",
    "ff.gate", "ff.up", "ff.down",
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny():
    return _load_module("lane150_krea2_tiny", _TESTS / "fixtures" / "qwen_image2" / "tiny.py")


def _latents() -> torch.Tensor:
    g = torch.Generator().manual_seed(150)
    return torch.randn(LAT_B, LAT_C, LAT_H, LAT_W, generator=g)


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().numpy()


def _release_pipeline(te, tokenizer):
    from diffusers import Krea2Pipeline

    return Krea2Pipeline(
        scheduler=None, vae=None, text_encoder=te, tokenizer=tokenizer, transformer=None,
        text_encoder_select_layers=SELECT_LAYERS, patch_size=PATCH,
    )


def compute_release() -> dict[str, np.ndarray]:
    tiny = _tiny()
    te = tiny.tiny_qwen3vl_te().eval()
    pipe = _release_pipeline(te, tiny.tiny_processor().tokenizer)
    lat = _latents()
    with torch.no_grad():
        hs, mask = pipe.get_text_hidden_states(CAPTIONS, max_sequence_length=MAX_SEQ, device=torch.device("cpu"))
    packed = pipe._pack_latents(lat, LAT_B, LAT_C, LAT_H, LAT_W)
    # the release takes PIXEL dims (pipeline_krea2.py _unpack_latents), the vendored copy latent dims
    try:
        unpacked = pipe._unpack_latents(packed, LAT_H * VAE_SCALE, LAT_W * VAE_SCALE)
    except RuntimeError as exc:
        # wrong-unit dims make the reshape fail; name both shapes instead of torch's bare error
        raise AssertionError(
            f"LANE-150 parity krea2/unpacked: release _unpack_latents could not rebuild "
            f"{tuple(packed.shape)} into the latent grid {(LAT_B, LAT_C, 1, LAT_H, LAT_W)}: {exc}"
        ) from exc
    pos = pipe.prepare_position_ids(5, LAT_H // PATCH, LAT_W // PATCH, torch.device("cpu"))
    return {
        "text_hidden_states": _np(hs), "text_mask": mask.numpy().astype(np.float32),
        "packed": _np(packed), "unpacked": _np(unpacked), "position_ids": _np(pos),
    }


def compute_vendored() -> dict[str, np.ndarray]:
    from app.engine.models.families.krea2.vendor import krea2_conditioning as v

    tiny = _tiny()
    te = tiny.tiny_qwen3vl_te().eval()
    lat = _latents()
    with torch.no_grad():
        hs, mask = v.get_text_hidden_states(
            tiny.tiny_processor().tokenizer, te, CAPTIONS, SELECT_LAYERS,
            max_sequence_length=MAX_SEQ, device=torch.device("cpu"),
        )
    packed = v.pack_latents(lat, PATCH)
    unpacked = v.unpack_latents(packed, LAT_H, LAT_W, PATCH)
    pos = v.prepare_position_ids(5, LAT_H // PATCH, LAT_W // PATCH, torch.device("cpu"))
    return {
        "text_hidden_states": _np(hs), "text_mask": mask.numpy().astype(np.float32),
        "packed": _np(packed), "unpacked": _np(unpacked), "position_ids": _np(pos),
    }


def record_vendored() -> None:
    """Write the fixture from the vendored copy (run once, before the copy is deleted)."""
    PARITY_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PARITY_NPZ, **compute_vendored())


def _stored() -> dict:
    with np.load(PARITY_NPZ) as z:
        return {k: z[k] for k in z.files}


def _check(name: str, out: np.ndarray, ref: np.ndarray, who: str = "release"):
    assert out.shape == ref.shape, (
        f"LANE-150 parity krea2/{name}: shape {tuple(out.shape)} != fixture shape {tuple(ref.shape)}"
    )
    diff = float(np.abs(out - ref).max())
    assert diff <= ATOL, f"LANE-150 parity krea2/{name}: max|{who} - fixture| = {diff:.3e} > {ATOL:g}"


@pytest.fixture(scope="module")
def release_outputs():
    return compute_release()


@pytest.mark.parametrize("name", ["text_hidden_states", "text_mask", "packed", "unpacked", "position_ids"])
def test_krea2_release_matches_vendored_fixture(name, release_outputs):
    _check(name, release_outputs[name], _stored()[name])


def test_krea2_fixture_is_the_vendored_copy():
    """While the vendored copy exists it must still reproduce the fixture (the fixture is genuine)."""
    pytest.importorskip("app.engine.models.families.krea2.vendor.krea2_conditioning")
    out = compute_vendored()
    for name, ref in _stored().items():
        _check(name, out[name], ref, who="vendored")


def test_krea2_lora_keys_pinned(tmp_path):
    from unittest.mock import MagicMock

    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file

    from diffusers import Krea2Transformer2DModel
    from app.engine.models.families.krea2.driver import Krea2Driver

    model = get_peft_model(
        Krea2Transformer2DModel.from_config(_TINY_CFG),
        LoraConfig(r=4, lora_alpha=4, target_modules=_LORA_TARGETS),
    )
    definition = MagicMock()
    definition.family = "krea2"
    definition.id = "krea2-test"
    definition.lora_targetable_modules = _LORA_TARGETS
    definition.architecture_params = {}
    path = tmp_path / "t.safetensors"
    Krea2Driver(definition, torch.device("cpu")).get_saver().save(
        components={"unet": model, "config": {}}, path=path
    )
    keys = sorted(load_file(str(path)).keys())
    pinned = json.loads(LORA_KEYS.read_text(encoding="utf-8"))
    extra, missing = sorted(set(keys) - set(pinned)), sorted(set(pinned) - set(keys))
    assert keys == pinned, (
        f"LANE-150 lora keys krea2: saved keys differ from fixtures/krea2/lora_keys.json: "
        f"+{extra[:5]} -{missing[:5]} ({len(extra)} extra, {len(missing)} missing)"
    )
