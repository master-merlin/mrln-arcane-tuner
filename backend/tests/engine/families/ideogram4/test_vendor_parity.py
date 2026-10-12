"""LANE-150 parity: diffusers 0.41.0's ideogram4 classes against the (soon retired) vendored copy.

The fixtures under ``app/engine/tests/fixtures/ideogram4/`` hold what the VENDORED modules produced on
pinned inputs (probe: ``.agent/workdir/lane-150/probes/ideogram4_parity_probe.py``); they are frozen, so
these tests keep passing after ``vendor/`` is deleted. Nothing here imports ``vendor/``.

* transformer: the release model is built from the vendored state dict by the measured mapping
  (``attention.qkv.weight`` chunked in thirds into ``to_q/to_k/to_v``, ``attention.o`` -> ``to_out.0``).
  Test-local ``_split_fused_state_dict`` until the loader's ``utils.convert_transformer_state_dict`` exists.
* VAE: ``AutoencoderKLFlux2`` against the vendored ``Ideogram4AutoEncoder`` from one diffusers-layout state dict.
* LoRA keys: the key list the saver writes for the PEFT-wrapped vendored tiny model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

_FIXTURES = Path(__file__).resolve().parents[4] / "app" / "engine" / "tests" / "fixtures" / "ideogram4"
PARITY_NPZ = _FIXTURES / "vendor_parity.npz"
VAE_NPZ = _FIXTURES / "vae_parity.npz"
LORA_KEYS = _FIXTURES / "lora_keys.json"
LEGACY_LORA = _FIXTURES / "legacy_lora_rank4.safetensors"

ATOL_FP32 = 1e-6
ATOL_BF16 = 1e-2
# bf16 has 8 significand bits: one ulp is up to 2**-7 of the value (outputs reach ~2.4, ulp 1.6e-2), so a
# different rounding order between the two forwards moves a value by one ulp. A relative term, not a bigger atol.
RTOL_BF16 = 2.0**-7
# The release spells the same tiny config attention_head_dim x num_attention_heads / adaln_dim.
RELEASE_CONFIG = dict(
    in_channels=8, num_layers=2, attention_head_dim=8, num_attention_heads=2, intermediate_size=32,
    adaln_dim=8, llm_features_dim=12, rope_theta=10000, mrope_section=(1, 1, 1), norm_eps=1e-5,
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _split_fused_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The measured vendored -> release mapping (spec: 0 missing / 0 unexpected keys)."""
    out: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if key.endswith("attention.qkv.weight"):
            base = key[: -len("qkv.weight")]
            q, k, v = value.chunk(3, dim=0)
            out[base + "to_q.weight"], out[base + "to_k.weight"], out[base + "to_v.weight"] = q, k, v
        elif key.endswith("attention.o.weight"):
            out[key[: -len("o.weight")] + "to_out.0.weight"] = value
        else:
            out[key] = value
    return out


def _release_model(dtype: torch.dtype):
    from diffusers import Ideogram4Transformer2DModel

    stored = _load(PARITY_NPZ)
    sd = {k[3:]: torch.from_numpy(v) for k, v in stored.items() if k.startswith("sd.")}
    model = Ideogram4Transformer2DModel(**RELEASE_CONFIG)
    model.load_state_dict(_split_fused_state_dict(sd), strict=True)
    return model.eval().to(dtype)


def _inputs(dtype: torch.dtype) -> dict[str, torch.Tensor]:
    s = _load(PARITY_NPZ)
    f = lambda n: torch.from_numpy(s["in." + n]).to(dtype)  # noqa: E731
    return dict(
        hidden_states=f("x"), timestep=f("t"), encoder_hidden_states=f("llm_features"),
        position_ids=torch.from_numpy(s["in.position_ids"]), segment_ids=torch.from_numpy(s["in.segment_ids"]),
        indicator=torch.from_numpy(s["in.indicator"]),
    )


def _release_forward(dtype: torch.dtype) -> np.ndarray:
    model = _release_model(dtype)
    with torch.no_grad():
        out = model(**_inputs(dtype), return_dict=False)[0]
    return out.to(torch.float32).numpy()


def _check(name: str, out: np.ndarray, ref: np.ndarray, atol: float, rtol: float = 0.0) -> None:
    assert out.shape == ref.shape, (
        f"LANE-150 parity ideogram4/{name}: shape {tuple(out.shape)} != fixture shape {tuple(ref.shape)}"
    )
    diff = float(np.abs(out - ref).max())
    assert np.all(np.abs(out - ref) <= atol + rtol * np.abs(ref)), (
        f"LANE-150 parity ideogram4/{name}: max|release - fixture| = {diff:.3e} > {atol:g}"
        + (f" + {rtol:g}*|fixture|" if rtol else "")
    )


def test_transformer_fp32_matches_vendored_fixture() -> None:
    _check("transformer", _release_forward(torch.float32), _load(PARITY_NPZ)["out_fp32"], ATOL_FP32)


def test_transformer_bf16_matches_vendored_fixture() -> None:
    _check("transformer_bf16", _release_forward(torch.bfloat16), _load(PARITY_NPZ)["out_bf16"], ATOL_BF16, RTOL_BF16)


def test_split_mapping_loads_strictly_and_needs_the_o_rename() -> None:
    """Without the ``o`` -> ``to_out.0`` rename the strict load names both sides."""
    from diffusers import Ideogram4Transformer2DModel

    stored = _load(PARITY_NPZ)
    sd = {k[3:]: torch.from_numpy(v) for k, v in stored.items() if k.startswith("sd.")}
    mapped = _split_fused_state_dict(sd)
    broken = {k.replace("to_out.0.weight", "o.weight"): v for k, v in mapped.items()}
    with pytest.raises(RuntimeError, match=r"Missing key\(s\).*to_out\.0\.weight"):
        Ideogram4Transformer2DModel(**RELEASE_CONFIG).load_state_dict(broken, strict=True)


def test_the_fixture_adapter_changes_the_vendored_forward() -> None:
    """The adapter half of the fixture is not a no-op (a zero adapter would pin nothing)."""
    s = _load(PARITY_NPZ)
    assert float(np.abs(s["adapter_forward"] - s["out_fp32"]).max()) > 1e-2
    assert any(k.endswith("lora_B") and np.abs(v).max() > 0 for k, v in s.items() if k.startswith("adapter."))


# ---- VAE ---------------------------------------------------------------------------------------------


def _release_vae():
    from diffusers import AutoencoderKLFlux2

    stored = _load(VAE_NPZ)
    vae = AutoencoderKLFlux2(
        in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(32, 32, 64, 64),
        layers_per_block=1, latent_channels=4, norm_num_groups=32, sample_size=32,
    )
    vae.load_state_dict({k[3:]: torch.from_numpy(v) for k, v in stored.items() if k.startswith("sd.")}, strict=True)
    return vae.eval(), stored


def test_vae_encode_mode_matches_vendored_encode() -> None:
    vae, s = _release_vae()
    with torch.no_grad():
        out = vae.encode(torch.from_numpy(s["image"])).latent_dist.mode().numpy()
    _check("vae_encode", out, s["encode"], ATOL_FP32)


def test_vae_decode_matches_vendored_decode() -> None:
    vae, s = _release_vae()
    with torch.no_grad():
        out = vae.decode(torch.from_numpy(s["z"])).sample.numpy()
    _check("vae_decode", out, s["decode"], ATOL_FP32)


def test_bn_is_not_in_encode_decode_path() -> None:
    """Re-pinned on the release: encode/decode never touch the ``bn`` head (the driver owns the latent norm)."""
    vae, s = _release_vae()
    before = (vae.bn.running_mean.clone(), vae.bn.running_var.clone())
    with torch.no_grad():
        vae.encode(torch.from_numpy(s["image"]))
        vae.decode(torch.from_numpy(s["z"]))
    assert int(vae.bn.num_batches_tracked) == 0
    assert torch.equal(vae.bn.running_mean, before[0]) and torch.equal(vae.bn.running_var, before[1])


# ---- LoRA keys ----------------------------------------------------------------------------------------


def test_ideogram4_lora_keys_pinned(tmp_path) -> None:
    """The saver writes the released fused key set for the PEFT-wrapped tiny model."""
    from unittest.mock import MagicMock

    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file

    from app.engine.models.families.ideogram4.driver import IdeogramV4Driver
    from app.engine.models.families.ideogram4.vendor.modeling_ideogram4 import Ideogram4Transformer2DModel

    # Fused-layout module names are what the saver pins today; this builds them from the vendored class and
    # disappears with it (commit 2 re-points the test at the release model + the re-fusing saver).
    targets = ["qkv", "o", "w1", "w2", "w3"]
    model = get_peft_model(
        Ideogram4Transformer2DModel(
            emb_dim=16, num_layers=2, num_heads=2, intermediate_size=32, adanln_dim=8, in_channels=8,
            llm_features_dim=12, rope_theta=10000, mrope_section=(1, 1, 1), norm_eps=1e-5,
        ),
        LoraConfig(r=4, lora_alpha=4, target_modules=targets),
    )
    definition = MagicMock()
    definition.family, definition.id = "ideogram4", "ideogram4-test"
    definition.lora_targetable_modules = targets
    definition.architecture_params = {}
    path = tmp_path / "t.safetensors"
    IdeogramV4Driver(definition, torch.device("cpu")).get_saver().save(
        components={"unet": model, "config": {}}, path=path
    )
    keys = sorted(load_file(str(path)).keys())
    pinned = json.loads(LORA_KEYS.read_text(encoding="utf-8"))
    extra, missing = sorted(set(keys) - set(pinned)), sorted(set(pinned) - set(keys))
    assert keys == pinned, (
        f"LANE-150 lora keys ideogram4: saved keys differ from fixtures/ideogram4/lora_keys.json: "
        f"+{extra[:5]} -{missing[:5]} ({len(extra)} extra, {len(missing)} missing)"
    )


def test_legacy_rank4_file_is_the_shared_a_fused_layout() -> None:
    """The pre-lane file: one shared ``lora_A`` (r x H) and one ``lora_B`` (3H x r) per fused qkv."""
    from safetensors.torch import load_file

    sd = load_file(str(LEGACY_LORA))
    a = sd["diffusion_model.layers.0.attention.qkv.lora_A.weight"]
    b = sd["diffusion_model.layers.0.attention.qkv.lora_B.weight"]
    assert tuple(a.shape) == (4, 16) and tuple(b.shape) == (48, 4)
