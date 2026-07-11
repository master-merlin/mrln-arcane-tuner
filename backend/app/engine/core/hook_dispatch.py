"""Shared detection of *meaningful* driver hook overrides.

Single source of truth for BOTH:

* the RUNTIME auto-delegation in
  :class:`app.engine.core.pipeline.pipeline_base.PipelineBaseMixin` (each
  clobber-capable base hook delegates to ``self.driver.<hook>`` when — and only
  when — the driver meaningfully overrides it), and
* the structural META-GUARD (``tests/engine/test_hook_wiring_meta.py``) that
  proves the mechanism engages across every registered family.

Deriving the "does the driver *meaningfully* override this hook?" predicate in
ONE place is deliberate: the guard and the live dispatch must agree byte-for-byte
about what counts as an override, or the guard would stop guarding the code that
actually runs.

WHAT "MEANINGFUL" MEANS
-----------------------
A driver class *meaningfully overrides* a hook when it defines the method below
:class:`IModelDriver` AND its normalized body is NOT one of:

* byte-identical to the ``IModelDriver`` interface default (a redundant
  re-statement of the base behavior),
* a known trivial-equivalent baseline (see :data:`TRIVIAL_BODIES` — e.g.
  ``init_scheduler`` is ``@abstractmethod`` on ``IModelDriver`` so its
  equivalent baseline is the ``PipelineBaseMixin`` default ``return None``),
* a bare ``return super().<hook>(...)`` delegation to the base default.

PURE INTROSPECTION — no GPU, no model loads, no driver instantiation. Results are
cached per ``(driver_cls, hook)`` so the per-training-step hooks (``add_noise``,
``sample_timesteps``) pay the AST-parse cost at most once per family class.

DEGRADED-SOURCE SAFETY
----------------------
When source is unavailable (e.g. a frozen/compiled deployment), the detectors
fail CLOSED — ``driver_meaningfully_overrides`` returns ``False`` — so the base
hook falls back to its self-contained default. That is exactly today's shipped
behavior (the base default), so a source-less deployment is never *worse* than
main; it merely forgoes the new auto-delegation.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from functools import lru_cache

from app.engine.core.interfaces import IModelDriver

# Hooks whose PipelineBaseMixin default is a self-contained no-op/None baseline
# that is NOT comparable to the IModelDriver body (``init_scheduler`` is
# ``@abstractmethod`` there). Their equivalent-baseline bodies live here so a
# driver that merely restates that baseline is not treated as a real override.
TRIVIAL_BODIES: dict[str, set[str]] = {
    "init_scheduler": {"return None"},
}


def normalize_body(fn) -> str | None:
    """Return a normalized method body (docstring/comment/whitespace-free).

    Parses the function via :mod:`ast` and re-emits its statements with
    ``ast.unparse`` — robust to multi-line signatures, comments, and formatting.
    The leading docstring statement is dropped. Returns ``None`` when the source
    cannot be read/parsed.
    """
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    func = tree.body[0] if tree.body else None
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    stmts = list(func.body)
    if (
        stmts
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        stmts = stmts[1:]  # drop docstring
    return " ".join(ast.unparse(s) for s in stmts).strip()


@lru_cache(maxsize=None)
def driver_meaningfully_overrides(driver_cls: type, hook: str) -> bool:
    """True iff ``driver_cls`` provides a non-trivial override of ``hook``.

    See the module docstring for the exact definition of "meaningful". Cached per
    ``(driver_cls, hook)``; safe to call on the hot training path.
    """
    fn = getattr(driver_cls, hook, None)
    base_fn = getattr(IModelDriver, hook, None)
    if fn is None or fn is base_fn:
        return False
    body = normalize_body(fn)
    if not body:
        # Source unavailable / unparseable → fail closed (base default runs).
        return False
    base_body = normalize_body(base_fn)
    if base_body and body == base_body:
        return False  # byte-identical re-statement of the base default
    if body in TRIVIAL_BODIES.get(hook, set()):
        return False
    if re.fullmatch(rf"return super\(\)\.{re.escape(hook)}\([^)]*\)", body):
        return False  # pure delegation to the (base) default
    return True


@lru_cache(maxsize=None)
def driver_hook_accepts_kwarg(driver_cls: type, hook: str, kwarg: str) -> bool:
    """True iff ``driver_cls.<hook>`` declares a parameter named ``kwarg``.

    Used to forward optional call-convention kwargs (e.g. the ``progress``
    training-fraction that boogu-style ``sample_timesteps`` drivers accept but
    the ``IModelDriver`` signature does not) only when the concrete driver's
    signature actually declares them.
    """
    fn = getattr(driver_cls, hook, None)
    if fn is None:
        return False
    try:
        return kwarg in inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False
