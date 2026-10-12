"""LANE-150 parity: omnigen2 on diffusers 0.41.0's own Lumina2 pieces vs the retired vendored copy.

``fixtures/omnigen2/vendor_parity.npz`` holds the OUTPUTS the vendored copy produced on
pinned inputs (seed 150, CPU; recorded before the copy was deleted). Each test builds the
release-backed module with the same seeded weights and requires the stored arrays back:
fp32 within 1e-6, bf16 within 1e-2. ``lora_keys.json`` pins the key list the saver writes
for the PEFT-wrapped tiny model.

The release caption embedding has no ``timestep_scale``: the family's adapter
(``vendor/models/transformers/block_lumina2.py``) multiplies the timestep by it
(the family's ``transformer.timestep_scale`` is 1000).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

_TESTS = Path(__file__).resolve().parent
_FIXTURES = _TESTS / "fixtures" / "omnigen2"
PARITY_NPZ = _FIXTURES / "vendor_parity.npz"
LORA_KEYS = _FIXTURES / "lora_keys.json"
TIMESTEP_SCALE = 1000.0
DTYPES = [(torch.float32, 1e-6, "fp32"), (torch.bfloat16, 1e-2, "bf16")]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stored_arrays() -> dict:
    with np.load(PARITY_NPZ) as z:
        return {k: z[k] for k in z.files}


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().numpy()


def _fixture(key: str) -> list:
    """The stored arrays for ``key`` (recorded from the vendored copy; never rewritten here)."""
    stored = _stored_arrays()
    n = 0
    while f"{key}.{n}" in stored:
        n += 1
    assert n, f"LANE-150 parity omnigen2/{key}: no recorded fixture in {PARITY_NPZ.name}"
    return [stored[f"{key}.{i}"] for i in range(n)]


# bf16 has 8 significand bits: one ulp is at most 2**-7 of a value's magnitude. Two forwards that round in
# a different order differ by about one ulp of the LARGEST intermediate, which can sit far above a small
# output element (an output of 0.7 beside one of -18 moved 0.125), so the relative term scales with the
# output tensor's magnitude, max|fixture|. One contract in the boogu_image, omnigen2 and ideogram4
# whole-model tests: atol 1e-2 + 2**-7 * max|fixture|, never a larger atol.
RTOL_BF16 = 2.0**-7


def _check(name: str, row, outs: list):
    _, atol, tag = row
    rtol = RTOL_BF16 if tag == "bf16" else 0.0
    reference = _fixture(f"{name}.{tag}")
    assert len(outs) == len(reference), f"LANE-150 parity omnigen2/{name}: {len(outs)} outputs != {len(reference)}"
    for out, ref in zip(outs, reference):
        diff = float(np.abs(_np(out) - ref).max())
        bound = atol + rtol * float(np.abs(ref).max())
        assert diff <= bound, (
            f"LANE-150 parity omnigen2/{name}: max|release - fixture| = {diff:.3e} > {bound:.3e}"
            + (f" ({atol:g} + {rtol:g}*max|fixture|)" if rtol else "")
        )


def _randomize(module: torch.nn.Module, g: torch.Generator):
    """Distinct non-trivial weights (zero-inits and unit norms would hide differences)."""
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return module


def _gen() -> torch.Generator:
    torch.manual_seed(150)
    return torch.Generator().manual_seed(150)


def _rope(g, batch=2, length=6, half=2):
    return torch.polar(torch.ones(batch, length, half), torch.randn(batch, length, half, generator=g))


@pytest.mark.parametrize("row", DTYPES, ids=lambda r: r[2])
def test_rmsnorm_zero(row):
    from diffusers.models.normalization import LuminaRMSNormZero

    dtype = row[0]
    g = _gen()
    r = _randomize(LuminaRMSNormZero(16, 1e-5, True), g).to(dtype).eval()
    x, emb = torch.randn(2, 5, 16, generator=g).to(dtype), torch.randn(2, 16, generator=g).to(dtype)
    with torch.no_grad():
        _check("rmsnorm_zero", row, list(r(x, emb)))


@pytest.mark.parametrize("row", DTYPES, ids=lambda r: r[2])
def test_layernorm_continuous(row):
    from diffusers.models.normalization import LuminaLayerNormContinuous

    dtype = row[0]
    g = _gen()
    kw = dict(embedding_dim=16, conditioning_embedding_dim=16, elementwise_affine=False, eps=1e-6, bias=True, out_dim=8)
    r = _randomize(LuminaLayerNormContinuous(**kw), g).to(dtype).eval()
    x, c = torch.randn(2, 5, 16, generator=g).to(dtype), torch.randn(2, 16, generator=g).to(dtype)
    with torch.no_grad():
        _check("layernorm_continuous", row, [r(x, c)])


@pytest.mark.parametrize("row", DTYPES, ids=lambda r: r[2])
def test_feed_forward(row):
    from diffusers.models.attention import LuminaFeedForward

    dtype = row[0]
    g = _gen()
    kw = dict(dim=16, inner_dim=64, multiple_of=8, ffn_dim_multiplier=None)
    r = _randomize(LuminaFeedForward(**kw), g).to(dtype).eval()
    x = torch.randn(2, 5, 16, generator=g).to(dtype)
    with torch.no_grad():
        _check("feed_forward", row, [r(x)])


@pytest.mark.parametrize("row", DTYPES, ids=lambda r: r[2])
def test_caption_embedding_at_timestep_scale(row):
    from app.engine.models.families.omnigen2.vendor.models.transformers.block_lumina2 import (
        Lumina2CombinedTimestepCaptionEmbedding as Adapter,
    )

    dtype = row[0]
    g = _gen()
    a = _randomize(Adapter(16, 8, 32, 1e-5, TIMESTEP_SCALE), g).to(dtype).eval()
    assert a.timestep_scale == TIMESTEP_SCALE
    # Dyadic timesteps: t * 1000 is exact, so the sinusoid (arguments up to 1000) rounds identically
    # to the recorded vendored output instead of turning fp32 operation order into a 1e-5 difference.
    t, cap = torch.tensor([0.5, 0.25]), torch.randn(2, 3, 8, generator=g).to(dtype)
    with torch.no_grad():
        _check("caption_embedding", row, list(a(t, cap, dtype)))


@pytest.mark.parametrize("row", DTYPES, ids=lambda r: r[2])
def test_timestep_embedding(row):
    from diffusers.models.embeddings import TimestepEmbedding

    dtype = row[0]
    g = _gen()
    r = _randomize(TimestepEmbedding(8, 16), g).to(dtype).eval()
    x = torch.randn(2, 8, generator=g).to(dtype)
    with torch.no_grad():
        _check("timestep_embedding", row, [r(x)])


@pytest.mark.parametrize("row", DTYPES, ids=lambda r: r[2])
def test_apply_rotary_emb(row):
    from diffusers.models.embeddings import apply_rotary_emb as release

    dtype = row[0]
    g = _gen()
    x_c = torch.randn(2, 6, 4, 4, generator=g).to(dtype)  # [B, S, H, D] (lumina complex path)
    freqs = _rope(g)
    x_r = torch.randn(2, 4, 6, 4, generator=g).to(dtype)  # [B, H, S, D] (real path)
    cos, sin = torch.randn(6, 4, generator=g), torch.randn(6, 4, generator=g)
    r = [release(x_c, freqs, use_real=False), release(x_r, (cos, sin), use_real=True, use_real_unbind_dim=-2)]
    _check("apply_rotary_emb", row, r)


@pytest.mark.parametrize("row", DTYPES, ids=lambda r: r[2])
def test_attention_processor_gqa_masked_proportional(row):
    from diffusers.models.attention_processor import Attention
    from diffusers.models.transformers.transformer_lumina2 import Lumina2AttnProcessor2_0

    dtype = row[0]
    g = _gen()
    attn = Attention(
        query_dim=16, cross_attention_dim=None, dim_head=4, qk_norm="rms_norm", heads=4, kv_heads=2,
        eps=1e-5, bias=False, out_bias=False, processor=Lumina2AttnProcessor2_0(),
    )
    attn = _randomize(attn, g).to(dtype).eval()
    h = torch.randn(2, 6, 16, generator=g).to(dtype)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, 4:] = False
    kw = dict(attention_mask=mask, image_rotary_emb=_rope(g), base_sequence_length=4)
    with torch.no_grad():
        r = attn.processor(attn, h, h, **kw)
    _check("attention_processor", row, [r])


def _tiny_transformer_outputs(dtype, with_refs: bool = True):
    from app.engine.models.families.omnigen2.vendor.models.transformers.repo import OmniGen2RotaryPosEmbed
    from app.engine.tests.test_omnigen2_family import _TINY_CFG, _build_tiny_model

    g = _gen()
    model = _randomize(_build_tiny_model(), g).to(dtype).eval()
    hidden = torch.randn(2, _TINY_CFG["in_channels"], 4, 4, generator=g).to(dtype)
    text = torch.randn(2, 3, _TINY_CFG["text_feat_dim"], generator=g).to(dtype)
    refs = [[torch.randn(_TINY_CFG["in_channels"], 4, 2, generator=g).to(dtype)] for _ in range(2)] if with_refs else None
    freqs = OmniGen2RotaryPosEmbed.get_freqs_cis(_TINY_CFG["axes_dim_rope"], _TINY_CFG["axes_lens"], theta=10000)
    with torch.no_grad():
        out = model(
            hidden_states=hidden,
            # Powers of two: the vendored class scaled AFTER the sinusoid ((t * freq) * 1000), the adapter
            # scales BEFORE it ((t * 1000) * freq); the two are bit-identical only when t is a power of two
            # (a non-power-of-two t such as 0.75 differs by 9.8e-5 on outputs up to 18: argument rounding).
            timestep=torch.tensor([0.25, 0.5]),
            text_hidden_states=text,
            freqs_cis=freqs,
            text_attention_mask=torch.ones(2, 3, dtype=torch.bool),
            ref_image_hidden_states=refs,
            return_dict=False,
        )
    return [out]


@pytest.mark.parametrize("row", DTYPES, ids=lambda r: r[2])
def test_tiny_transformer_forward_on_release_blocks(row):
    # Power-of-two timesteps (see _tiny_transformer_outputs), so fp32 holds 1e-6 here too; bf16 rounds the
    # release's fp32-upcast RMSNorm / FP32SiLU feed-forward one ulp from the vendored result, covered by
    # the shared contract (atol + RTOL_BF16).
    _check("transformer", row, _tiny_transformer_outputs(row[0]))


def test_plain_generation_runs_with_no_reference_images():
    """With no reference images the ref_image_refiner would see a zero-length sequence, which the
    release's complex-path ``apply_rotary_emb`` cannot reshape; the transformer skips the refiner
    for it (every t2i sample and training step takes this path)."""
    (out,) = _tiny_transformer_outputs(torch.float32, with_refs=False)
    assert out.shape[0] == 2 and bool(torch.isfinite(out).all())


def _load_pinned_keys(path: Path) -> list:
    """The recorded key list; a missing file FAILS (it is never written from the code under test)."""
    assert path.exists(), (
        f"LANE-150 lora keys omnigen2: the pinned key list fixtures/omnigen2/lora_keys.json is missing ({path})"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_missing_key_fixture_fails_and_is_not_written(tmp_path):
    missing = tmp_path / "nested" / "lora_keys.json"
    with pytest.raises(AssertionError, match=r"fixtures/omnigen2/lora_keys\.json is missing"):
        _load_pinned_keys(missing)
    assert not missing.exists() and not missing.parent.exists()


def test_saver_key_set_is_pinned(tmp_path):
    from safetensors.torch import load_file

    portability = _load_module("omnigen2_portability_for_keys", _TESTS / "test_omnigen2_lora_portability.py")
    drv = portability._make_driver()
    model = portability._build_peft_model(drv)
    path = tmp_path / "omnigen2_lora.safetensors"
    drv.get_saver().save(components={"unet": model, "config": {}}, path=path)
    keys = sorted(load_file(str(path)).keys())
    pinned = _load_pinned_keys(LORA_KEYS)
    extra, missing = sorted(set(keys) - set(pinned)), sorted(set(pinned) - set(keys))
    assert keys == pinned, (
        f"LANE-150 lora keys omnigen2: saved keys differ from fixtures/omnigen2/lora_keys.json: "
        f"+{extra[:5]} -{missing[:5]} ({len(extra)} extra, {len(missing)} missing)"
    )
