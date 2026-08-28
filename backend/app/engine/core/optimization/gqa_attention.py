"""Keep grouped-query attention off SDPA's math backend.

Some families use grouped-query attention (GQA): more query heads than
key/value heads. Diffusers expresses that by passing ``enable_gqa=True`` to
``dispatch_attention_fn`` and letting the kernel broadcast the KV heads::

    dispatch_attention_fn(query, key, value, enable_gqa=attn.num_heads != attn.num_kv_heads, ...)

Every *fused* backend diffusers offers raises on that flag (cuDNN, native
flash, flash-attn 2/3, sage — see ``attention_dispatch.py``), so a GQA model is
already restricted to the ``native`` backend, i.e. plain
``torch.nn.functional.scaled_dot_product_attention``. Torch then applies its own
rule, from ``check_batch_size_and_num_heads_dense``:

    For dense input, both fused kernels require query, key and value to have
    the same num_heads. [...] To broadcast dense inputs, try using unsqueeze
    and expand_to before passing them into the kernel.

So on a build whose flash kernel is missing or refuses GQA, ``enable_gqa=True``
silently selects the **math** backend, which materialises the full
``(B, H, S, S)`` score matrix — and diffusers' native op recomputes the whole
forward in its backward, so that matrix is built twice per step. The cost is
quadratic in sequence length, i.e. quadratic in image area, and invisible: no
warning is raised, the run is simply slower and needs far more VRAM.

Measured on krea2-raw (48 query heads / 12 KV heads, 28 layers) against
flux2-klein-9b (no GQA) over the same dataset, same bucket ladder, same 6000
steps: 9.49 h vs 3.98 h, and 67 GB vs 25 GB peak allocation. Fitting step time
against bucket area separates the terms — the linear (weights) term is 1.75x
klein's, which architecture explains, while the quadratic (attention) term is
3.32x, which it does not.

The fix is the one torch's own message asks for, and the one upstream reached
independently: expand the KV heads to match the query heads and hand the kernel
a plain MHA problem. ``omnigen2`` and ``boogu_image`` already carry it in their
vendored processors, with the same finding in the comment::

    # explicitly repeat key and value to match query length, otherwise using
    # enable_gqa=True results in MATH backend of sdpa in our test of pytorch2.6

The families affected here get their transformer from diffusers, not from
``vendor/``, so there is no source to edit. Instead we wrap the
``dispatch_attention_fn`` symbol *in the module that defines the model's class*
— the name each processor actually calls, since they do
``from ..attention_dispatch import dispatch_attention_fn`` at import. Scoped to
that module: an unaffected family loaded in the same process is untouched.

Expanding costs one copy of K and V at full head width, which is small next to
the score matrix it avoids (at 7632 tokens, 48 heads: ~93 MB per tensor versus
~5.6 GB per layer per pass).

This is a workaround for a kernel-availability gap, not a permanent behaviour
change, so it is **decided by measurement, not by hardcoding**: we ask torch
whether a fused kernel would take the model's own head configuration, and patch
only when the answer is no. A torch build that grows GQA support in flash turns
this off by itself — no pin to bump, no rule to remember.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Any

import torch

__all__ = [
    "expand_gqa_dispatch",
    "fused_backends_refuse_gqa",
    "install_gqa_expansion",
    "model_gqa_head_config",
]

# Attribute names diffusers uses for the query- and KV-head counts. Ordered
# most-specific first; the first pair that resolves on a module wins.
_HEAD_ATTRS = ("num_heads", "heads", "num_attention_heads")
_KV_HEAD_ATTRS = ("num_kv_heads", "kv_heads", "num_key_value_heads")

# The layout every caller of dispatch_attention_fn uses: (batch, seq, heads,
# dim). Not a guess — ``_native_attention_forward_op`` permutes (0, 2, 1, 3)
# unconditionally before calling SDPA, so it is the contract for all callers.
_HEAD_DIM = -2


def _first_attr(module: Any, names: tuple[str, ...]) -> int | None:
    for name in names:
        value = getattr(module, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def model_gqa_head_config(model: Any) -> tuple[int, int, int] | None:
    """Return ``(num_heads, num_kv_heads, head_dim)`` for the first GQA attention
    module found, or ``None`` if the model does not use grouped-query attention.

    One representative module is enough: the head configuration is a property of
    the architecture, and the kernels' answer does not vary per layer.
    """
    modules = model.modules() if hasattr(model, "modules") else ()
    for module in modules:
        heads = _first_attr(module, _HEAD_ATTRS)
        kv_heads = _first_attr(module, _KV_HEAD_ATTRS)
        if heads is None or kv_heads is None or heads == kv_heads:
            continue
        if heads % kv_heads:
            # Not a GQA grouping we can expand; leave it to the kernel.
            continue
        head_dim = _first_attr(module, ("head_dim",)) or 128
        return heads, kv_heads, head_dim
    return None


@lru_cache(maxsize=None)
def fused_backends_refuse_gqa(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> bool:
    """True when no fused CUDA kernel will accept this GQA shape.

    Asks torch's own host-side predicates — no attention kernel is launched and
    the probe tensors are a few kilobytes. Cached, because the answer depends
    only on the build and the head configuration.

    Returns False when CUDA is unavailable: there is no fused kernel to lose on
    CPU, so expanding would be cost without benefit.
    """
    if not torch.cuda.is_available():
        return False
    try:
        from torch.backends.cuda import (
            SDPAParams,
            can_use_efficient_attention,
            can_use_flash_attention,
        )

        # Sequence length does not enter the head-count check; keep it tiny.
        def probe(heads: int) -> torch.Tensor:
            return torch.zeros(1, heads, 8, head_dim, dtype=dtype, device="cuda")

        query, key, value = probe(num_heads), probe(num_kv_heads), probe(num_kv_heads)
        params = SDPAParams(query, key, value, None, 0.0, False, True)
        return not (
            can_use_flash_attention(params, False)
            or can_use_efficient_attention(params, False)
        )
    except Exception:  # noqa: BLE001 — a probe must never take a run down
        return False


def expand_gqa_dispatch(dispatch_fn):
    """Wrap ``dispatch_attention_fn`` so GQA reaches the kernel as plain MHA.

    ``repeat_interleave`` reproduces exactly what ``enable_gqa=True`` asks the
    kernel to do: query head ``i`` reads KV head ``i // (Hq // Hkv)``, so the KV
    heads repeat in consecutive groups. The result is numerically identical; only
    the backend torch picks changes.
    """
    if getattr(dispatch_fn, "_mrln_gqa_expanded", False):
        return dispatch_fn

    def wrapper(query, key, value, *args, enable_gqa: bool = False, **kwargs):
        if enable_gqa:
            groups = query.shape[_HEAD_DIM] // key.shape[_HEAD_DIM]
            if groups > 1 and query.shape[_HEAD_DIM] % key.shape[_HEAD_DIM] == 0:
                key = key.repeat_interleave(groups, dim=_HEAD_DIM)
                value = value.repeat_interleave(groups, dim=_HEAD_DIM)
                enable_gqa = False
        return dispatch_fn(query, key, value, *args, enable_gqa=enable_gqa, **kwargs)

    wrapper._mrln_gqa_expanded = True
    wrapper._mrln_gqa_wrapped = dispatch_fn
    return wrapper


def install_gqa_expansion(model: Any, logger: Any = None) -> bool:
    """Patch the model's own transformer module if GQA would fall to math SDPA.

    Returns True when the patch was installed. Never raises: a failure here
    costs speed, and taking a training run down over it would be worse.
    """
    try:
        heads = model_gqa_head_config(model)
        if heads is None:
            return False
        num_heads, num_kv_heads, head_dim = heads

        module = sys.modules.get(type(model).__module__)
        dispatch_fn = getattr(module, "dispatch_attention_fn", None)
        if dispatch_fn is None:
            return False
        if getattr(dispatch_fn, "_mrln_gqa_expanded", False):
            return False

        dtype = getattr(model, "dtype", None)
        if not isinstance(dtype, torch.dtype):
            dtype = torch.bfloat16
        if not fused_backends_refuse_gqa(num_heads, num_kv_heads, head_dim, dtype):
            return False

        module.dispatch_attention_fn = expand_gqa_dispatch(dispatch_fn)
        if logger is not None:
            logger.info(
                "gqa_expansion_installed",
                module=module.__name__,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                reason="no fused SDPA kernel accepts enable_gqa on this build",
            )
        return True
    except Exception as exc:  # noqa: BLE001 — optimisation, never fatal
        if logger is not None:
            logger.warning("gqa_expansion_failed", error=str(exc))
        return False
