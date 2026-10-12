"""LANE-150 parity: boogu_image's vendored Lumina2 pieces vs diffusers 0.41.0's own.

Fixture-first retirement. ``fixtures/boogu_image/vendor_parity.npz`` holds the
vendored copy's outputs on pinned inputs (seed 150, CPU; written on the first
run, compared on every later one). Each test builds the vendored module, loads
its state dict into the release class, and requires BOTH to reproduce the stored
arrays: fp32 within 1e-6, bf16 within 1e-2. ``lora_keys.json`` pins the key list
the saver writes for the PEFT-wrapped tiny vendored model.

The release caption embedding has no ``timestep_scale``: it is fed
``timestep * timestep_scale`` (the family's ``transformer.timestep_scale`` is 1000).
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

_TESTS = Path(__file__).resolve().parent
_FIXTURES = _TESTS / "fixtures" / "boogu_image"
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
    if not PARITY_NPZ.exists():
        return {}
    with np.load(PARITY_NPZ) as z:
        return {k: z[k] for k in z.files}


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().numpy()


def _fixture(key: str, vendored_outs: list) -> list:
    """The stored arrays for ``key``; the vendored outputs are written if absent."""
    stored = _stored_arrays()
    if f"{key}.0" not in stored:
        stored.update({f"{key}.{i}": _np(o) for i, o in enumerate(vendored_outs)})
        PARITY_NPZ.parent.mkdir(parents=True, exist_ok=True)
        np.savez(PARITY_NPZ, **stored)
    n = 0
    while f"{key}.{n}" in stored:
        n += 1
    return [stored[f"{key}.{i}"] for i in range(n)]


def _assert_parity(name: str, outs: list, reference: list, atol: float):
    assert len(outs) == len(reference), f"LANE-150 parity boogu_image/{name}: {len(outs)} outputs != {len(reference)}"
    for out, ref in zip(outs, reference):
        diff = float(np.abs(_np(out) - ref).max())
        assert diff <= atol, f"LANE-150 parity boogu_image/{name}: max|release - fixture| = {diff:.3e} > {atol:g}"


def _check(name: str, row, vendored_outs: list, release_outs: list):
    _, atol, tag = row
    reference = _fixture(f"{name}.{tag}", vendored_outs)
    _assert_parity(name, release_outs, reference, atol)
    _assert_parity(f"{name}(vendored)", vendored_outs, reference, atol)


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


def _pair(vendored, make_release, dtype):
    release = make_release()
    release.load_state_dict(vendored.state_dict(), strict=True)
    return vendored.to(dtype).eval(), release.to(dtype).eval()


@pytest.mark.parametrize("row", DTYPES, ids=lambda r: r[2])
def test_rmsnorm_zero(row):
    from diffusers.models.normalization import LuminaRMSNormZero as Release

    from app.engine.models.families.boogu_image.vendor.models.transformers.block_lumina2 import (
        LuminaRMSNormZero as Vendored,
    )

    dtype = row[0]
    g = _gen()
    v = _randomize(Vendored(16, 1e-5, True), g)
    v, r = _pair(v, lambda: Release(16, 1e-5, True), dtype)
    x, emb = torch.randn(2, 5, 16, generator=g).to(dtype), torch.randn(2, 16, generator=g).to(dtype)
    with torch.no_grad():
        _check("rmsnorm_zero", row, list(v(x, emb)), list(r(x, emb)))


@pytest.mark.parametrize("row", DTYPES, ids=lambda r: r[2])
def test_layernorm_continuous(row):
    from diffusers.models.normalization import LuminaLayerNormContinuous as Release

    from app.engine.models.families.boogu_image.vendor.models.transformers.block_lumina2 import (
        LuminaLayerNormContinuous as Vendored,
    )

    dtype = row[0]
    g = _gen()
    kw = dict(embedding_dim=16, conditioning_embedding_dim=16, elementwise_affine=False, eps=1e-6, bias=True, out_dim=8)
    v = _randomize(Vendored(**kw), g)
    v, r = _pair(v, lambda: Release(**kw), dtype)
    x, c = torch.randn(2, 5, 16, generator=g).to(dtype), torch.randn(2, 16, generator=g).to(dtype)
    with torch.no_grad():
        _check("layernorm_continuous", row, [v(x, c)], [r(x, c)])


@pytest.mark.parametrize("row", DTYPES, ids=lambda r: r[2])
def test_feed_forward(row):
    from diffusers.models.attention import LuminaFeedForward as Release

    from app.engine.models.families.boogu_image.vendor.models.transformers.block_lumina2 import (
        LuminaFeedForward as Vendored,
    )

    dtype = row[0]
    g = _gen()
    kw = dict(dim=16, inner_dim=64, multiple_of=8, ffn_dim_multiplier=None)
    v = _randomize(Vendored(**kw), g)
    v, r = _pair(v, lambda: Release(**kw), dtype)
    x = torch.randn(2, 5, 16, generator=g).to(dtype)
    with torch.no_grad():
        _check("feed_forward", row, [v(x)], [r(x)])


@pytest.mark.parametrize("row", DTYPES, ids=lambda r: r[2])
def test_caption_embedding_at_timestep_scale(row):
    from diffusers.models.transformers.transformer_lumina2 import (
        Lumina2CombinedTimestepCaptionEmbedding as Release,
    )

    from app.engine.models.families.boogu_image.vendor.models.transformers.block_lumina2 import (
        Lumina2CombinedTimestepCaptionEmbedding as Vendored,
    )

    dtype = row[0]
    g = _gen()
    v = _randomize(Vendored(16, 8, 32, 1e-5, TIMESTEP_SCALE), g)
    v, r = _pair(v, lambda: Release(hidden_size=16, cap_feat_dim=8, frequency_embedding_size=32, norm_eps=1e-5), dtype)
    # Dyadic timesteps: t * 1000 and the vendored Timesteps(scale=1000) then round identically, so the
    # sinusoid (arguments up to 1000) does not turn fp32 operation order into a 1e-5 difference.
    t, cap = torch.tensor([0.5, 0.25]), torch.randn(2, 3, 8, generator=g).to(dtype)
    with torch.no_grad():
        v_out = list(v(t, cap, dtype))
        r_out = list(r(torch.zeros(2, 1, 16, dtype=dtype), t * TIMESTEP_SCALE, cap))
        _check("caption_embedding", row, v_out, r_out)


@pytest.mark.parametrize("row", DTYPES, ids=lambda r: r[2])
def test_timestep_embedding(row):
    from diffusers.models.embeddings import TimestepEmbedding as Release

    from app.engine.models.families.boogu_image.vendor.models.embeddings import TimestepEmbedding as Vendored

    dtype = row[0]
    g = _gen()
    v = _randomize(Vendored(8, 16), g)
    v, r = _pair(v, lambda: Release(8, 16), dtype)
    x = torch.randn(2, 8, generator=g).to(dtype)
    with torch.no_grad():
        _check("timestep_embedding", row, [v(x)], [r(x)])


@pytest.mark.parametrize("row", DTYPES, ids=lambda r: r[2])
def test_apply_rotary_emb(row):
    from diffusers.models.embeddings import apply_rotary_emb as release

    from app.engine.models.families.boogu_image.vendor.models.embeddings import apply_rotary_emb as vendored

    dtype = row[0]
    g = _gen()
    x_c = torch.randn(2, 6, 4, 4, generator=g).to(dtype)  # [B, S, H, D] (lumina complex path)
    freqs = _rope(g)
    x_r = torch.randn(2, 4, 6, 4, generator=g).to(dtype)  # [B, H, S, D] (real path)
    cos, sin = torch.randn(6, 4, generator=g), torch.randn(6, 4, generator=g)
    v = [vendored(x_c, freqs, use_real=False), vendored(x_r, (cos, sin), use_real=True, use_real_unbind_dim=-2)]
    r = [release(x_c, freqs, use_real=False), release(x_r, (cos, sin), use_real=True, use_real_unbind_dim=-2)]
    _check("apply_rotary_emb", row, v, r)


@pytest.mark.parametrize("row", DTYPES, ids=lambda r: r[2])
def test_attention_processor_gqa_masked_proportional(row):
    from diffusers.models.attention_processor import Attention
    from diffusers.models.transformers.transformer_lumina2 import Lumina2AttnProcessor2_0

    from app.engine.models.families.boogu_image.vendor.models.attention_processor import BooguImageAttnProcessor

    dtype = row[0]
    g = _gen()
    attn = Attention(
        query_dim=16, cross_attention_dim=None, dim_head=4, qk_norm="rms_norm", heads=4, kv_heads=2,
        eps=1e-5, bias=False, out_bias=False, processor=BooguImageAttnProcessor(),
    )
    _randomize(attn, g)
    release_attn = copy.deepcopy(attn)
    release_attn.set_processor(Lumina2AttnProcessor2_0())
    attn, release_attn = attn.to(dtype).eval(), release_attn.to(dtype).eval()
    h = torch.randn(2, 6, 16, generator=g).to(dtype)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, 4:] = False
    kw = dict(attention_mask=mask, image_rotary_emb=_rope(g), base_sequence_length=4)
    with torch.no_grad():
        v = attn.processor(attn, h, h, **kw)
        r = release_attn.processor(release_attn, h, h, **kw)
    _check("attention_processor", row, [v], [r])


def _tiny_transformer_outputs(dtype, release: bool):
    from diffusers.models.transformers.transformer_lumina2 import Lumina2AttnProcessor2_0

    from app.engine.models.families.boogu_image.vendor.models.transformers.rope import (
        BooguImageDoubleStreamRotaryPosEmbed,
    )
    from app.engine.tests.test_boogu_image_vendor import (
        TINY_AXES_DIM_ROPE,
        TINY_AXES_LENS,
        TINY_IN_CHANNELS,
        TINY_INSTRUCTION_FEAT_DIM,
        _build_tiny_transformer,
    )

    g = _gen()
    model = _randomize(_build_tiny_transformer(), g)
    if release:
        for block in list(model.single_stream_layers) + list(model.double_stream_layers):
            for name in ("attn", "img_self_attn"):
                if hasattr(block, name):
                    getattr(block, name).set_processor(Lumina2AttnProcessor2_0())
    model = model.to(dtype).eval()
    hidden = [
        torch.randn(TINY_IN_CHANNELS, 4, 4, generator=g).to(dtype),
        torch.randn(TINY_IN_CHANNELS, 4, 6, generator=g).to(dtype),
    ]
    instr = torch.randn(2, 3, TINY_INSTRUCTION_FEAT_DIM, generator=g).to(dtype)
    freqs = BooguImageDoubleStreamRotaryPosEmbed.get_freqs_cis(TINY_AXES_DIM_ROPE, TINY_AXES_LENS, theta=10000)
    with torch.no_grad():
        return model(
            hidden_states=hidden,
            timestep=torch.rand(2, generator=g),
            instruction_hidden_states=instr,
            freqs_cis=freqs,
            instruction_attention_mask=torch.ones(2, 3, dtype=torch.bool),
            ref_image_hidden_states=None,
            return_dict=False,
        )


@pytest.mark.parametrize("row", DTYPES, ids=lambda r: r[2])
def test_tiny_transformer_forward_with_release_attention_processor(row):
    dtype = row[0]
    _check(
        "transformer",
        row,
        _tiny_transformer_outputs(dtype, release=False),
        _tiny_transformer_outputs(dtype, release=True),
    )


def test_saver_key_set_is_pinned(tmp_path):
    from safetensors import safe_open

    saver_tests = _load_module("boogu_image_saver_for_keys", _TESTS / "test_boogu_image_saver.py")
    targets = saver_tests._load_curated_targets()
    model = saver_tests._build_peft_model(targets)
    path = tmp_path / "boogu_lora.safetensors"
    saver_tests._get_saver().save(components={"unet": model, "config": {}}, path=path)
    with safe_open(str(path), "pt") as f:
        keys = sorted(f.keys())
    if not LORA_KEYS.exists():
        LORA_KEYS.parent.mkdir(parents=True, exist_ok=True)
        LORA_KEYS.write_text(json.dumps(keys, indent=1) + "\n", encoding="utf-8")
    pinned = json.loads(LORA_KEYS.read_text(encoding="utf-8"))
    extra, missing = sorted(set(keys) - set(pinned)), sorted(set(pinned) - set(keys))
    assert keys == pinned, (
        f"LANE-150 lora keys boogu_image: saved keys differ from fixtures/boogu_image/lora_keys.json: "
        f"+{extra[:5]} -{missing[:5]} ({len(extra)} extra, {len(missing)} missing)"
    )
