"""Feature-cache scheduling for the ``boogu_image`` transformer.

The import path is the public surface — ``transformer_boogu.py`` does
``from ...cache_functions import cal_type`` — so it is kept exactly. The
implementation lives in :mod:`...taylor_cache`, one module for both this package
and :mod:`..taylorseer_utils`, because the single- and double-stream paths
differ only in whether a cached feature is one tensor or a tuple of them.

It sits **outside** ``vendor/`` deliberately. ``vendor/`` means "upstream code
we do not own", and ``ruff.toml`` excludes it from linting on exactly that
basis; this implementation is ours, so it lives on a first-party path where the
gate lints it like any other module. Only these shims stay here, because the
caller's import paths are frozen.

This is a CLEAN-ROOM replacement for a GPL-3.0-derived implementation, written
from the ICCV 2025 TaylorSeer paper, from
``_harness/research/taylorseer-cache-behavioural-spec.md``, and from
``models/transformers/transformer_boogu.py`` (Boogu's own Apache-2.0 caller,
which is not TaylorSeer-derived). It is not derived from the code it replaces.
See :mod:`...taylor_cache` for the full provenance note.
"""

from __future__ import annotations

from ...taylor_cache import cache_init, cal_type, force_scheduler

__all__ = ["cache_init", "cal_type", "force_scheduler"]
