"""Taylor-expansion feature cache for the ``boogu_image`` transformer.

PROVENANCE — CLEAN-ROOM IMPLEMENTATION
--------------------------------------
This module was written from, and only from:

* *From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers*
  (ICCV 2025) — the published method;
* ``_harness/research/taylorseer-cache-behavioural-spec.md`` — the behavioural
  specification derived from that paper and from the caller below;
* ``vendor/models/transformers/transformer_boogu.py`` — Boogu-Image's own
  **Apache-2.0** code. It is the caller, it is not TaylorSeer-derived, and it
  fixes the names, signatures and dictionary keys reproduced here.

It is **not** derived from the GPL-3.0 TaylorSeer implementation it replaces,
and was written without reading it. The interfaces and data formats are
reproduced deliberately, because the caller depends on them; the implementation
is new expression.

This module lives **outside** ``vendor/`` on purpose. ``vendor/`` means
"upstream code we do not own" and ``ruff.toml`` excludes it from linting on that
basis — so first-party code placed there would silently sit outside the gate.
The two package paths the caller imports, ``vendor/cache_functions`` and
``vendor/taylorseer_utils``, remain where they are and re-export from here;
those paths are frozen public surface.

WHAT IT DOES
------------
Diffusion sampling runs the transformer once per denoising step, and the
intermediate feature tensors change smoothly from step to step. The
acceleration: compute the transformer fully only on some steps, and on the steps
between, *predict* each cached feature from its recent history rather than
computing it.

The prediction is a truncated Taylor series in the step index. Each full-compute
step stores the feature as the zeroth-order term and extends a ladder of
finite-difference derivative estimates; each cached step evaluates the series at
the current offset from the last full compute.

Two properties govern correctness:

* The leading steps are always computed fully — a derivative estimate needs
  history, and the early steps carry the most structure.
* Derivative order grows only as history allows. One stored observation supports
  order 0 alone (reuse the last value); every subsequent full step permits one
  further order, up to a configured maximum.
"""

from __future__ import annotations

import math
from typing import Any, TypeAlias

import torch

__all__ = [
    "cache_init",
    "cal_type",
    "derivative_approximation",
    "derivative_approximation_4_double_stream",
    "force_scheduler",
    "taylor_cache_init",
    "taylor_formula",
    "taylor_formula_4_double_stream",
]

#: A cached feature is either one tensor or a fixed-arity tuple of tensors. Both
#: are held internally as tuples so the maths is written once; the two exported
#: variants differ only in whether they unwrap the single-element case.
Parts: TypeAlias = tuple[torch.Tensor, ...]

#: The cache is addressed as ``cache[_ROOT][stream][layer][module]``. ``_ROOT``
#: is a dictionary key, not a list index — the store is a dict at every level.
_ROOT = -1

#: Step classifications the caller acts on. It compares ``current["type"]``
#: against these two strings and nothing else.
_FULL = "full"
_TAYLOR = "Taylor"


def _reject(key: str, value: object, expected: str) -> None:
    """Raise for an invalid configuration value, naming the key that is wrong.

    Validation belongs at init: a bad value that survives to the cache internals
    surfaces as a shape or key error deep inside the transformer, where the
    configuration key responsible is no longer visible.
    """
    raise ValueError(
        f"cache_init: {key}={value!r} is invalid; expected {expected}"
    )


def _discover_num_layers(pipe: Any) -> int | None:
    """Best-effort transformer layer count, for pre-creating the layer slots.

    Pre-creation is an optimisation, not a requirement: the store creates any
    missing level on demand (see :func:`_slot`), so returning ``None`` here is a
    supported outcome and not a silent failure. The pipeline object is typed
    loosely on purpose — annotating it would require importing the pipeline,
    which imports this module.
    """
    transformer = getattr(pipe, "transformer", pipe)
    for probe in (
        lambda: int(transformer.config.num_layers),
        lambda: int(transformer.num_layers),
        lambda: len(transformer.layers),
    ):
        try:
            count = probe()
        except (AttributeError, TypeError, ValueError):
            continue
        if count > 0:
            return count
    return None


def cache_init(
    self: Any,
    num_steps: int,
    *,
    taylor_cache: bool = True,
    max_order: int = 4,
    first_enhance: int = 5,
    fresh_threshold: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the ``(cache_dic, current)`` pair the transformer reads each step.

    ``self`` is the pipeline. The keyword arguments carry the shipped
    configuration as defaults; they exist so a caller — or a test — can supply
    another configuration and have it validated here rather than at depth.
    """
    if not isinstance(num_steps, int) or num_steps < 1:
        _reject("num_steps", num_steps, "an int >= 1")
    if not isinstance(max_order, int) or max_order < 0:
        _reject("max_order", max_order, "an int >= 0")
    if not isinstance(first_enhance, int) or first_enhance < 0:
        _reject("first_enhance", first_enhance, "an int >= 0")
    if not isinstance(fresh_threshold, int) or fresh_threshold < 1:
        _reject("fresh_threshold", fresh_threshold, "an int >= 1")
    if not taylor_cache:
        # Rejected here rather than at the first cached step so the run fails
        # before the GPU spins up: this configuration cannot produce a result
        # either way, and the later failure costs the whole warm-up first.
        # cal_type keeps the same guard as a backstop for a flag flipped after
        # init.
        raise NotImplementedError(
            "cache_init: taylor_cache=False selects a feature-cache mode that "
            "is not implemented; pass taylor_cache=True (the shipped "
            "configuration) or leave the feature cache disabled entirely"
        )

    cache: dict[int, dict[Any, Any]] = {_ROOT: {}}
    cache_index: dict[int, dict[Any, Any]] = {_ROOT: {}}

    num_layers = _discover_num_layers(self)
    if num_layers is not None:
        cache[_ROOT]["layers_stream"] = {layer: {} for layer in range(num_layers)}
        cache_index[_ROOT] = {layer: {} for layer in range(num_layers)}

    cache_dic: dict[str, Any] = {
        "cache": cache,
        "cache_index": cache_index,
        "cache_counter": 0,
        "taylor_cache": taylor_cache,
        "max_order": max_order,
        "first_enhance": first_enhance,
        "fresh_threshold": fresh_threshold,
        # Set by force_scheduler below so there is exactly one place that
        # decides the applied interval.
        "cal_threshold": fresh_threshold,
        # Carried for the non-Taylor modes. They are unreachable while
        # taylor_cache is on, which is the shipped configuration; see cal_type.
        "fresh_ratio": 0.0,
        "fresh_ratio_schedule": "ToCa",
        "soft_fresh_weight": 0.0,
        "cache_type": "random",
        "Delta-DiT": False,
    }

    current: dict[str, Any] = {
        "step": 0,
        "num_steps": num_steps,
        # Step 0 is always a full compute, so it is seeded here; cal_type will
        # not duplicate it. See the invariant on activated_steps in cal_type.
        "activated_steps": [0],
        "type": _FULL,
        # The address of the feature being cached. The transformer sets all
        # three before every call into this module.
        "stream": None,
        "layer": None,
        "module": None,
    }

    force_scheduler(cache_dic, current)
    return cache_dic, current


def force_scheduler(cache_dic: dict[str, Any], current: dict[str, Any]) -> None:
    """Set the full-compute interval actually applied, ``cal_threshold``.

    Upstream carried a step-position weighting term here that is disabled in
    this configuration and contributes nothing to the result; a computation with
    no effect is not part of the behaviour, so it is not reproduced. The applied
    interval is therefore exactly the configured one.

    ``cal_threshold`` is a plain ``int``. Every comparison against it — here and
    in the caller — is with a Python int, so the 0-dimensional tensor it used to
    be bought nothing and cost a host synchronisation on each read.

    ``current`` is unused; it is part of the signature the caller depends on.
    """
    _ = current
    cache_dic["cal_threshold"] = int(cache_dic["fresh_threshold"])


def cal_type(cache_dic: dict[str, Any], current: dict[str, Any]) -> None:
    """Classify this step as a full compute or a cached prediction, in place.

    Called once per denoising step, before any layer runs. Mutates
    ``current["type"]`` and ``current["activated_steps"]``, and
    ``cache_dic["cache_counter"]`` and ``cache_dic["cal_threshold"]``.
    """
    step = current["step"]

    if step < cache_dic["first_enhance"]:
        # Warm-up: no derivative history exists yet and the early steps carry
        # the most structure, so they are never predicted.
        is_full = True
    else:
        is_full = cache_dic["cache_counter"] == cache_dic["cal_threshold"] - 1

    if is_full:
        current["type"] = _FULL
        cache_dic["cache_counter"] = 0
        activated = current["activated_steps"]
        # Invariant: activated_steps is strictly ascending and holds exactly one
        # entry per full step. Step 0 is seeded at init because it is always
        # full, so the guard below keeps the count exact instead of doubling it.
        if not activated or activated[-1] != step:
            activated.append(step)
        force_scheduler(cache_dic, current)
        return

    if not cache_dic["taylor_cache"]:
        # The non-Taylor modes ("ToCa", "Delta-Cache") are unreachable in every
        # configuration this project ships and were never exercised upstream.
        # Guessing at their behaviour would be worse than not having them, and
        # mislabelling the step would make the caller silently reuse a stale
        # feature — so this fails loudly instead.
        raise NotImplementedError(
            "boogu_image feature cache: only the Taylor path is implemented; "
            "set taylor_cache=True (the shipped configuration) or compute every "
            "step fully"
        )

    current["type"] = _TAYLOR
    cache_dic["cache_counter"] += 1


def _slot(cache_dic: dict[str, Any], current: dict[str, Any]) -> dict[int, Parts]:
    """Return the per-``(stream, layer, module)`` term store, creating it if absent.

    Creating on demand keeps the store self-healing: a stream or layer the
    pipeline addresses without a prior init call still works. An address that
    was never written is caught where it matters instead — :func:`_predict`
    refuses to predict from an empty store.
    """
    node: dict[Any, Any] = cache_dic["cache"][_ROOT]
    for key in (current["stream"], current["layer"], current["module"]):
        child = node.get(key)
        if child is None:
            child = {}
            node[key] = child
        node = child
    return node


def taylor_cache_init(cache_dic: dict[str, Any], current: dict[str, Any]) -> None:
    """Prepare the term store for the feature currently being addressed.

    Only the first step creates it. The store must survive across full steps:
    the divided-difference ladder in :func:`_store_derivatives` reads the
    previous full step's terms, so re-creating it later would silently reset the
    series to order 0 and quietly discard the acceleration.
    """
    if not cache_dic["taylor_cache"] or current["step"] != 0:
        return
    _slot(cache_dic, current)


def _ladder_allowed(cache_dic: dict[str, Any], current: dict[str, Any]) -> bool:
    """Whether derivative orders above 0 may be extended on this step.

    Differences taken across the warm-up region are estimated from the densely
    computed early steps, where the feature is changing fastest; extrapolating
    from them degrades the image without ever raising. The ladder therefore
    starts only once warm-up is over.

    ``>=`` and ``>`` against ``first_enhance`` select the same steps in the
    shipped configuration, because no full step falls on the warm-up boundary
    there; ``>=`` is used because "not a warm-up step" is the actual invariant.
    """
    return current["step"] >= cache_dic["first_enhance"]


def _store_derivatives(
    cache_dic: dict[str, Any], current: dict[str, Any], parts: Parts
) -> None:
    """Store ``parts`` as order 0 and extend the divided-difference ladder."""
    max_order = cache_dic["max_order"]
    store = _slot(cache_dic, current)
    # Snapshot: the new terms are built from the previous full step's terms, so
    # the store must not be mutated until every new order is computed.
    previous = dict(store)
    updated: dict[int, Parts] = {0: parts}

    activated = current["activated_steps"]
    if _ladder_allowed(cache_dic, current) and len(activated) >= 2:
        span = activated[-1] - activated[-2]
        # activated_steps is strictly ascending, so the gap cannot be zero and
        # the division below cannot be by zero.
        assert span >= 1, f"activated_steps not ascending: {activated[-2:]}"

        order = 0
        while order < max_order and order in previous:
            older = previous[order]
            if len(older) != len(updated[order]):
                raise ValueError(
                    "boogu_image feature cache: cached feature arity changed "
                    f"mid-run at order {order} "
                    f"({len(older)} -> {len(updated[order])})"
                )
            updated[order + 1] = tuple(
                (new - old) / span
                for new, old in zip(updated[order], older, strict=True)
            )
            order += 1

    # The store is bounded by max_order rather than assumed to be: the ladder
    # above is the only writer, and this is the assertion that says so.
    assert max(updated) <= max_order, (
        f"retained order {max(updated)} exceeds max_order={max_order}"
    )

    store.clear()
    store.update(updated)


def _predict(cache_dic: dict[str, Any], current: dict[str, Any]) -> Parts:
    """Evaluate the stored series at this step's offset from the last full compute."""
    store = _slot(cache_dic, current)
    if not store:
        raise RuntimeError(
            "boogu_image feature cache: nothing cached for "
            f"stream={current['stream']!r} layer={current['layer']!r} "
            f"module={current['module']!r}; a full step must run before a "
            "cached one"
        )

    offset = current["step"] - current["activated_steps"][-1]
    orders = sorted(store)
    arity = len(store[orders[0]])

    return tuple(
        sum(
            store[order][index] * (offset**order / math.factorial(order))
            for order in orders
        )
        for index in range(arity)
    )


def derivative_approximation(
    cache_dic: dict[str, Any], current: dict[str, Any], feature: torch.Tensor
) -> None:
    """Record ``feature`` for the current address on a full-compute step."""
    _store_derivatives(cache_dic, current, (feature,))


def derivative_approximation_4_double_stream(
    cache_dic: dict[str, Any],
    current: dict[str, Any],
    feature: tuple[torch.Tensor, ...],
) -> None:
    """Double-stream variant of :func:`derivative_approximation`.

    Identical semantics; the feature is a tuple of tensors and each element is
    tracked independently.
    """
    _store_derivatives(cache_dic, current, tuple(feature))


def taylor_formula(
    cache_dic: dict[str, Any], current: dict[str, Any]
) -> torch.Tensor:
    """Predict the feature for the current address on a cached step."""
    return _predict(cache_dic, current)[0]


def taylor_formula_4_double_stream(
    cache_dic: dict[str, Any], current: dict[str, Any]
) -> tuple[torch.Tensor, ...]:
    """Double-stream variant of :func:`taylor_formula`.

    Returns a tuple of the same arity as the tuple that was recorded, each
    element predicted exactly as the single-tensor variant predicts its one.
    """
    return _predict(cache_dic, current)
