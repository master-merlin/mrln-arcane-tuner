"""Guards for the grouped-query-attention math-backend workaround.

The defect this protects against is silent by construction: nothing raises,
nothing warns, the run is simply 2.4x slower and needs 2.7x the VRAM. So the
tests have to hold two separate things down.

1. **Our expansion means the same thing as ``enable_gqa=True``.** If it did not,
   we would be shipping a fast wrong answer, which is worse than a slow right
   one.
2. **The upstream shape the patch attaches to still exists.** The patch replaces
   a module-level ``dispatch_attention_fn`` symbol and only fires on the
   ``enable_gqa`` keyword. Either could be renamed by a diffusers upgrade, and
   the failure mode would be a no-op — back to the slow path, with no error.
   That is the same silence the fix exists to break, so it gets a test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from app.engine.core.optimization.gqa_attention import (
    _HEAD_DIM,
    expand_gqa_dispatch,
    fused_backends_refuse_gqa,
    install_gqa_expansion,
    model_gqa_head_config,
)

REPO = Path(__file__).resolve().parents[4]
FAMILIES = REPO / "backend" / "app" / "engine" / "models" / "families"


def _native_like(query, key, value, *, enable_gqa=False, **kwargs):
    """Stand-in for diffusers' native backend: (B, S, H, D) in, SDPA underneath.

    Mirrors ``_native_attention_forward_op`` — the permute to (B, H, S, D) is
    unconditional there, which is what makes heads dim -2 for every caller.
    """
    q, k, v = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=enable_gqa)
    return out.permute(0, 2, 1, 3)


# ── 1. The expansion is numerically what enable_gqa asks for ────────────────


@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads"),
    [(48, 12), (32, 8), (28, 7), (8, 1)],  # krea2, ace_step15, boogu, extreme MQA
)
def test_expansion_matches_enable_gqa_exactly(num_heads, num_kv_heads):
    """Same numbers, different backend — the whole premise of the workaround.

    Uses float64 so the comparison is about the head-mapping being right, not
    about bf16 rounding hiding a wrong one.
    """
    torch.manual_seed(0)
    batch, seq, head_dim = 2, 16, 8
    query = torch.randn(batch, seq, num_heads, head_dim, dtype=torch.float64)
    key = torch.randn(batch, seq, num_kv_heads, head_dim, dtype=torch.float64)
    value = torch.randn(batch, seq, num_kv_heads, head_dim, dtype=torch.float64)

    reference = _native_like(query, key, value, enable_gqa=True)
    wrapped = expand_gqa_dispatch(_native_like)(query, key, value, enable_gqa=True)

    torch.testing.assert_close(wrapped, reference, rtol=0, atol=0)


def test_expansion_would_notice_a_wrong_head_mapping():
    """Positive control for the test above.

    ``repeat_interleave`` groups KV heads consecutively (query head ``i`` reads
    KV head ``i // groups``); ``repeat`` tiles them instead. If the assertion
    above could not tell those apart it would be proving nothing.
    """
    torch.manual_seed(0)
    query = torch.randn(1, 16, 8, 8, dtype=torch.float64)
    key = torch.randn(1, 16, 2, 8, dtype=torch.float64)
    value = torch.randn(1, 16, 2, 8, dtype=torch.float64)

    reference = _native_like(query, key, value, enable_gqa=True)
    tiled = _native_like(
        query, key.repeat(1, 1, 4, 1), value.repeat(1, 1, 4, 1), enable_gqa=False
    )
    assert not torch.allclose(tiled, reference)


def test_the_kernel_is_handed_plain_mha():
    """The point is not the numbers, it is which backend torch may then pick."""
    seen = {}

    def spy(query, key, value, *, enable_gqa=False, **kwargs):
        seen.update(
            enable_gqa=enable_gqa,
            q_heads=query.shape[-2],
            kv_heads=key.shape[-2],
            v_heads=value.shape[-2],
        )
        return query

    query = torch.zeros(1, 4, 48, 8)
    key = value = torch.zeros(1, 4, 12, 8)
    expand_gqa_dispatch(spy)(query, key, value, enable_gqa=True)

    assert seen == {"enable_gqa": False, "q_heads": 48, "kv_heads": 48, "v_heads": 48}


def test_non_gqa_calls_pass_through_untouched():
    """Prove the negative: an MHA family must not pay for this."""
    seen = {}

    def spy(query, key, value, *, enable_gqa=False, **kwargs):
        seen.update(enable_gqa=enable_gqa, kv_heads=key.shape[-2], key_id=id(key))
        return query

    query = key = value = torch.zeros(1, 4, 32, 8)
    expand_gqa_dispatch(spy)(query, key, value, enable_gqa=False)

    assert seen["enable_gqa"] is False
    assert seen["kv_heads"] == 32
    assert seen["key_id"] == id(key), "key was copied for a model that has no GQA"


def test_wrapping_twice_is_the_same_object():
    """prepare_for_training can run more than once per process."""
    once = expand_gqa_dispatch(_native_like)
    assert expand_gqa_dispatch(once) is once


# ── 2. Head-config discovery ────────────────────────────────────────────────


class _Attn(torch.nn.Module):
    def __init__(self, num_heads, num_kv_heads, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


class _Model(torch.nn.Module):
    def __init__(self, *attns):
        super().__init__()
        self.blocks = torch.nn.ModuleList(attns)


def test_head_config_finds_the_gqa_module():
    model = _Model(_Attn(20, 20), _Attn(48, 12))
    assert model_gqa_head_config(model) == (48, 12, 128)


def test_head_config_is_none_without_gqa():
    assert model_gqa_head_config(_Model(_Attn(32, 32), _Attn(20, 20))) is None


def test_head_config_skips_non_divisible_groupings():
    """Not a grouping repeat_interleave can express — leave it to the kernel."""
    assert model_gqa_head_config(_Model(_Attn(10, 4))) is None


# ── 3. The install decision ─────────────────────────────────────────────────


def test_probe_reports_supported_without_cuda():
    """No fused kernel to lose on CPU, so nothing to work around."""
    if torch.cuda.is_available():
        pytest.skip("CUDA present — this pins the CPU branch")
    assert fused_backends_refuse_gqa(48, 12, 128, torch.bfloat16) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="asks this GPU/build directly")
def test_on_this_build_gqa_is_what_the_fused_kernels_refuse():
    """The empirical claim the whole workaround rests on, kept honest.

    Same dtype, same head_dim, same everything except the KV-head count. If
    this ever reports that GQA is accepted, a fused kernel grew GQA support and
    the workaround should be deleted rather than carried — which is why it is
    written as a skip, not an inverted assertion.
    """
    if not fused_backends_refuse_gqa(48, 12, 128, torch.bfloat16):
        pytest.skip("this build's fused kernels accept GQA — workaround is dead weight")
    assert fused_backends_refuse_gqa(48, 48, 128, torch.bfloat16) is False, (
        "fused kernels refuse this shape even without GQA, so the head-count "
        "mismatch is not the discriminator and the diagnosis needs re-checking"
    )


def test_install_is_a_no_op_for_a_model_without_gqa():
    assert install_gqa_expansion(_Model(_Attn(32, 32))) is False


def test_install_patches_the_defining_module(monkeypatch):
    """The symbol replaced is the one the processor actually calls.

    Processors do ``from ..attention_dispatch import dispatch_attention_fn`` at
    import, so the live name lives in the transformer's own module — patching
    ``attention_dispatch`` itself would miss it.
    """
    module = type(sys)("fake_transformer_module")
    module.dispatch_attention_fn = _native_like
    sys.modules[module.__name__] = module
    monkeypatch.setattr(
        "app.engine.core.optimization.gqa_attention.fused_backends_refuse_gqa",
        lambda *a, **k: True,
    )
    try:
        model = _Model(_Attn(48, 12))
        model.__class__ = type("FakeTransformer", (_Model,), {"__module__": module.__name__})

        assert install_gqa_expansion(model) is True
        assert module.dispatch_attention_fn._mrln_gqa_expanded is True
        # Second call must not double-wrap.
        assert install_gqa_expansion(model) is False
    finally:
        del sys.modules[module.__name__]


def test_install_never_raises_on_a_hostile_model():
    """Losing speed is acceptable; killing a multi-hour run over it is not."""

    class Exploding:
        __module__ = "does.not.exist"

        def modules(self):
            raise RuntimeError("boom")

    assert install_gqa_expansion(Exploding()) is False


# ── 4. The upstream shape the patch attaches to ─────────────────────────────


@pytest.mark.parametrize("module_name", ["transformer_krea2", "ace_step_transformer"])
def test_diffusers_still_dispatches_gqa_the_way_the_patch_expects(module_name):
    """Each module must be in one of TWO healthy states, never in neither.

    The original version of this test asserted one rule for both modules: the
    module must still pass ``enable_gqa=``. diffusers 0.40 broke that for
    ``transformer_krea2`` -- and the break was *upstream adopting our fix*,
    expanding the KV heads itself with the same reasoning our patch carries. The
    test was right to fire and wrong about what it meant, so it now names both
    healthy states:

    * **passes enable_gqa** -> the kernel still gets a GQA problem, so our patch
      is load-bearing and must have something to replace;
    * **expands KV itself** -> upstream does the job, our wrapper never sees
      ``enable_gqa=True`` and degrades to a pass-through, which is correct.

    What must never happen is NEITHER: a module that has quietly dropped GQA
    handling altogether, where the math backend returns and nothing says so.

    ``dispatch_attention_fn`` must be a module-level name either way -- that is
    the symbol ``install_gqa_expansion`` replaces.
    """
    pytest.importorskip("diffusers")
    import importlib

    module = importlib.import_module(f"diffusers.models.transformers.{module_name}")
    assert hasattr(module, "dispatch_attention_fn"), (
        f"{module_name} no longer imports dispatch_attention_fn at module level; "
        "install_gqa_expansion has nothing to replace and silently does nothing"
    )
    source = Path(module.__file__).read_text(encoding="utf-8")
    passes_flag = "enable_gqa=" in source
    expands_itself = "repeat_interleave" in source
    assert passes_flag or expands_itself, (
        f"{module_name} neither passes enable_gqa nor expands the KV heads itself. "
        "Grouped-query attention has silently lost its handling: SDPA will fall back "
        "to the math backend, which is quadratic in image area and raises no warning. "
        "That regression cost krea2 2.4x wall clock and 2.7x peak VRAM once already."
    )


def test_the_wrapper_cannot_double_expand_when_upstream_already_did():
    """The exact conflict the diffusers 0.40 bump had to be cleared of.

    On 0.40 ``transformer_krea2`` expands the KV heads itself and then calls
    ``dispatch_attention_fn`` WITHOUT ``enable_gqa``. If our wrapper expanded on
    anything other than that flag, K and V would be repeated a second time and
    every attention head would read the wrong KV group -- silently, with correct
    shapes throughout, which is the worst way for this to fail.

    Called here exactly the way 0.40's processor calls it: no ``enable_gqa``
    kwarg at all, K/V already at full head width.
    """
    seen = {}

    def fake_dispatch(query, key, value, *args, **kwargs):
        seen["key_heads"] = key.shape[_HEAD_DIM]
        seen["value_heads"] = value.shape[_HEAD_DIM]
        seen["enable_gqa"] = kwargs.get("enable_gqa")
        return query

    wrapped = expand_gqa_dispatch(fake_dispatch)
    query = torch.zeros(1, 8, 48, 4)
    key = torch.zeros(1, 8, 48, 4)  # already expanded by upstream: 48 == 48
    wrapped(query, key, key.clone(), attn_mask=None)

    assert seen["key_heads"] == 48, (
        f"the wrapper expanded already-expanded KV heads to {seen['key_heads']}; "
        "on diffusers 0.40 krea2 this would silently misalign every head group"
    )
    assert seen["value_heads"] == 48
    assert seen["enable_gqa"] is False


def test_the_wrapper_still_forwards_a_kwarg_diffusers_accepts():
    """A pass-through must still be a VALID call.

    The wrapper always forwards ``enable_gqa=`` to the real dispatch function,
    including on the 0.40 krea2 path where the caller omitted it. If diffusers
    ever drops that parameter, the wrapper turns a working forward into a
    ``TypeError`` mid-training -- so the parameter's existence is the pin, not
    an assumption.
    """
    pytest.importorskip("diffusers")
    import inspect

    from diffusers.models.attention_dispatch import dispatch_attention_fn

    target = getattr(dispatch_attention_fn, "_mrln_gqa_wrapped", dispatch_attention_fn)
    assert "enable_gqa" in inspect.signature(target).parameters, (
        "diffusers' dispatch_attention_fn no longer accepts enable_gqa, but "
        "expand_gqa_dispatch always passes it. Every patched family would raise "
        "TypeError on its first attention call."
    )


@pytest.mark.parametrize(
    ("definition", "heads_key", "kv_key"),
    [
        ("krea2/definitions/krea2_raw.yaml", "transformer.num_attention_heads", "transformer.num_key_value_heads"),
        ("krea2/definitions/krea2_turbo.yaml", "transformer.num_attention_heads", "transformer.num_key_value_heads"),
        ("ace_step15/definitions/base.yaml", "transformer.num_attention_heads", "transformer.num_key_value_heads"),
        ("ace_step15/definitions/xl_base.yaml", "transformer.num_attention_heads", "transformer.num_key_value_heads"),
    ],
)
def test_the_families_this_was_written_for_still_use_gqa(definition, heads_key, kv_key):
    """Anti-vacuity: the tests above are only worth their runtime while some
    shipped family actually has more query heads than KV heads."""
    import yaml

    config = yaml.safe_load((FAMILIES / definition).read_text(encoding="utf-8"))
    params = config["architecture_params"]
    heads, kv_heads = params[heads_key], params[kv_key]
    assert heads > kv_heads, f"{definition} no longer uses GQA ({heads}q/{kv_heads}kv)"
    assert heads % kv_heads == 0, f"{definition} grouping is not expandable"


def test_prepare_for_training_configures_gqa_before_peft():
    """Order matters only in one direction, but the call has to be there at all.

    A helper nothing calls is the most expensive kind of dead code: it looks
    like the bug is fixed.
    """
    source = (
        REPO / "backend" / "app" / "engine" / "core" / "pipeline" / "pipeline_optimization.py"
    ).read_text(encoding="utf-8")
    gqa = source.index("self._configure_gqa_attention()")
    peft = source.index("self._apply_peft()")
    assert gqa < peft
