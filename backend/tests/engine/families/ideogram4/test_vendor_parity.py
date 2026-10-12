"""LANE-150 parity: diffusers 0.41.0's ideogram4 classes against the retired vendored copy.

The fixtures under ``app/engine/tests/fixtures/ideogram4/`` hold what the VENDORED modules produced on
pinned inputs (probe: ``.agent/workdir/lane-150/probes/ideogram4_parity_probe.py``); they are frozen, so
these tests keep passing after ``vendor/`` is deleted. Nothing here imports ``vendor/``.

* transformer: the release model is built from the vendored state dict by the loader's own
  ``utils.convert_transformer_state_dict`` (``attention.qkv.weight`` chunked in thirds into
  ``to_q/to_k/to_v``, ``attention.o`` -> ``to_out.0``).
* VAE: ``AutoencoderKLFlux2`` against the vendored ``Ideogram4AutoEncoder`` from one diffusers-layout state dict.
* adapter: the PEFT-wrapped release model carrying the fixture's split adapter == the vendored fused
  model carrying it as a merged delta (``adapter_forward``); the saver's half is in test_ideogram4_family.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

_FIXTURES = Path(__file__).resolve().parents[4] / "app" / "engine" / "tests" / "fixtures" / "ideogram4"
PARITY_NPZ = _FIXTURES / "vendor_parity.npz"
VAE_NPZ = _FIXTURES / "vae_parity.npz"
LEGACY_LORA = _FIXTURES / "legacy_lora_rank4.safetensors"

ATOL_FP32 = 1e-6
ATOL_BF16 = 1e-2
# bf16 has 8 significand bits: one ulp is up to 2**-7 of the value. A different rounding order between the
# two forwards moves a value by about one ulp of the LARGEST intermediate, so the relative term scales with
# the output tensor's magnitude, max|fixture| (the same contract as the boogu_image and omnigen2 whole-model
# tests): atol 1e-2 + 2**-7 * max|fixture|, never a larger atol.
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
    """The loader's vendored -> release mapping (spec: 0 missing / 0 unexpected keys)."""
    from app.engine.models.families.ideogram4.utils import convert_transformer_state_dict

    return convert_transformer_state_dict(sd)


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
    bound = atol + rtol * float(np.abs(ref).max())
    assert diff <= bound, (
        f"LANE-150 parity ideogram4/{name}: max|release - fixture| = {diff:.3e} > {bound:.3e}"
        + (f" ({atol:g} + {rtol:g}*max|fixture|)" if rtol else "")
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


def test_peft_split_adapter_matches_the_vendored_merged_delta() -> None:
    """The release PEFT model with the fixture's split adapter (rank 4, alpha 4: scale 1) is the
    vendored fused model carrying the same adapter merged (``W_qkv + B_fused @ A_fused``)."""
    from peft import LoraConfig, get_peft_model

    s = _load(PARITY_NPZ)
    model = get_peft_model(
        _release_model(torch.float32),
        LoraConfig(r=4, lora_alpha=4, target_modules=["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"]),
    )
    modules = dict(model.base_model.model.named_modules())
    with torch.no_grad():
        for key, value in s.items():
            if key.startswith("adapter."):
                path, which = key[len("adapter."):].rsplit(".", 1)
                getattr(modules[path], which)["default"].weight.copy_(torch.from_numpy(value))
        out = model(**_inputs(torch.float32), return_dict=False)[0].numpy()
    _check("adapter_forward", out, s["adapter_forward"], 1e-5)


def test_legacy_rank4_file_is_the_shared_a_fused_layout() -> None:
    """The pre-lane file: one shared ``lora_A`` (r x H) and one ``lora_B`` (3H x r) per fused qkv."""
    from safetensors.torch import load_file

    sd = load_file(str(LEGACY_LORA))
    a = sd["diffusion_model.layers.0.attention.qkv.lora_A.weight"]
    b = sd["diffusion_model.layers.0.attention.qkv.lora_B.weight"]
    assert tuple(a.shape) == (4, 16) and tuple(b.shape) == (48, 4)
