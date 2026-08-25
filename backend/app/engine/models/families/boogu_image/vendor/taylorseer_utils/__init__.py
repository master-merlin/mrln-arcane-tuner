"""Taylor-series storage and evaluation for the ``boogu_image`` transformer.

The import path is the public surface — ``transformer_boogu.py`` imports all
five names below from this package — so it is kept exactly. The implementation
lives in :mod:`..taylor_cache`, shared with :mod:`..cache_functions`.

This is a CLEAN-ROOM replacement for a GPL-3.0-derived implementation, written
from the ICCV 2025 TaylorSeer paper, from
``_harness/research/taylorseer-cache-behavioural-spec.md``, and from
``models/transformers/transformer_boogu.py`` (Boogu's own Apache-2.0 caller,
which is not TaylorSeer-derived). It is not derived from the code it replaces.
See :mod:`..taylor_cache` for the full provenance note.
"""

from __future__ import annotations

from ..taylor_cache import (
    derivative_approximation,
    derivative_approximation_4_double_stream,
    taylor_cache_init,
    taylor_formula,
    taylor_formula_4_double_stream,
)

__all__ = [
    "derivative_approximation",
    "derivative_approximation_4_double_stream",
    "taylor_cache_init",
    "taylor_formula",
    "taylor_formula_4_double_stream",
]
