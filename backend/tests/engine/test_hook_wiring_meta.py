"""Repo-wide META-GUARD against the dead-dispatch (clobber) bug class.

WHAT THIS PINS
--------------
The training pipeline dispatches lifecycle hooks on the TRAINER via MRO
(:class:`app.engine.core.pipeline.pipeline_base.PipelineBaseMixin`). Some of
those hooks have a **self-contained base default** that does NOT delegate to the
family driver's same-named method (e.g. ``init_scheduler`` returns ``None``,
``add_noise`` delegates to a generic ``NoiseInterpolation`` component, NOT the
driver). For those hooks, a driver-level override is **DEAD CODE on the real
training path** unless the family's TRAINER also overrides the hook to delegate
to the driver.

This exact gap bit the project repeatedly (boogu ``init_scheduler`` clobber; k5
I2V frame-0 trained NOISED; ltx2 I2V same; sdxl/flux1 pooled-TE caches never
persisted; and — found by THIS guard — WAN 2.2 dual-expert timesteps were sampled
full-range instead of truncated to the active expert's boundary). Every one of
those was invisible to unit tests because the family's ``*_driver`` tests called
``driver.<hook>(...)`` **directly**, never through ``trainer.<hook>`` (the real
loop's call), so the MRO-resolution gap never surfaced.

THE INVARIANT
-------------
For every registered family, for every *clobber-capable* hook: **if the family's
DRIVER class meaningfully overrides the hook, the family's TRAINER class must also
override it** (i.e. ``trainer.<hook>`` must NOT resolve to the
``PipelineBaseMixin`` default). Otherwise the driver override is dead.

The clobber-capable hook list is DERIVED programmatically (:func:`_clobber_hooks`)
— it is every method present on BOTH ``PipelineBaseMixin`` and ``IModelDriver``
whose ``PipelineBaseMixin`` implementation does NOT delegate to
``self.driver.<hook>``. Deriving it (rather than hardcoding) means a newly added
self-contained hook is guarded automatically. As of writing it resolves to:
``add_noise, build_batch_extra, compute_target, get_te_cache, init_scheduler,
sample_timesteps, set_te_cache`` (7 hooks).

INVERSE SHAPE — ``_resolve_loading_dtype`` (documented, see the test below)
-------------------------------------------------------------------------
``_resolve_loading_dtype`` has the *inverse* dispatch: the model-LOADING pipeline
(``pipeline_loading.py``) calls ``self.driver.resolve_loading_dtype()`` — the
DRIVER — so a TRAINER-level override of ``_resolve_loading_dtype`` is dead **for
loading**. A blanket "no trainer may override ``_resolve_loading_dtype``" would be
WRONG (sdxl legitimately overrides it for its *text-embedding-cache* dtype, a
separate concern, AND its driver overrides ``resolve_loading_dtype`` to match).
So instead we pin the narrow, sound inverse guard: a trainer that overrides
``_resolve_loading_dtype`` MUST have a driver that overrides
``resolve_loading_dtype`` (evidence the author knew loading routes through the
driver, keeping the loading + TE-cache dtypes consistent).

PURE INTROSPECTION — no GPU, no model loads, no driver instantiation.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import textwrap
from types import SimpleNamespace

import pytest

from app.engine.core.interfaces import IModelDriver
from app.engine.core.pipeline.pipeline_base import PipelineBaseMixin
from app.engine.models.registry import registry

# ── Expected family floor ────────────────────────────────────────────────────
# Guards against silent registry breakage hollowing the test out to a no-op.
MIN_EXPECTED_FAMILIES = 21


# ── ALLOWLIST ────────────────────────────────────────────────────────────────
# (family, hook) pairs where the driver overrides a clobber-capable hook but the
# trainer intentionally does NOT delegate — because the driver's override is
# PROVABLY EQUIVALENT to the base ``PipelineBaseMixin`` path. Every entry carries
# a one-line justification. Adding an entry is a deliberate, reviewed act.
ALLOWLIST: dict[tuple[str, str], str] = {
    ("wan21", "add_noise"): (
        "WanDriverBase.add_noise is the family's contract-pinned flow-match lerp "
        "in [0,1000] space: t=timesteps/1000; t*noise+(1-t)*latents. This is "
        "ALGEBRAICALLY IDENTICAL to the base PipelineBaseMixin.add_noise path "
        "(NoiseInterpolation('linear'), whose _linear ALSO divides t by 1000), so "
        "the real-path result is bit-identical whether or not the trainer "
        "delegates. The driver method is exercised directly by the wan precision-"
        "contract tests (test_wan21_precision_contracts.py) — it is a tested, "
        "contract-defining method, not dead unused code; the trainer leaving "
        "add_noise at base default is safe and intentional."
    ),
    ("wan22", "add_noise"): (
        "Inherits WanDriverBase.add_noise (see wan21 justification): "
        "algebraically identical to the base NoiseInterpolation('linear') path, "
        "so no clobber. Pinned by test_wan22_precision_contracts.py."
    ),
}


# ── Hook-list derivation ─────────────────────────────────────────────────────
def _delegates_to_driver(fn, hook: str) -> bool:
    """True if PipelineBaseMixin.<hook>'s body calls ``self.driver.<hook>``."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return bool(re.search(rf"self\.driver\.{re.escape(hook)}\b", src)) or bool(
        re.search(rf"\bdriver\.{re.escape(hook)}\b", src)
    )


def _clobber_hooks() -> list[str]:
    """Every method on BOTH PipelineBaseMixin and IModelDriver whose
    PipelineBaseMixin default is self-contained (does NOT delegate to the
    driver's same-named method) — i.e. the clobber-capable hooks."""
    tr = {n for n, _ in inspect.getmembers(PipelineBaseMixin, inspect.isfunction)}
    dr = {n for n, _ in inspect.getmembers(IModelDriver, inspect.isfunction)}
    hooks = []
    for name in sorted(tr & dr):
        base_fn = getattr(PipelineBaseMixin, name)
        if not _delegates_to_driver(base_fn, name):
            hooks.append(name)
    return hooks


CLOBBER_HOOKS = _clobber_hooks()

# Normalized method bodies that are considered EQUIVALENT-to-base (no clobber)
# for a given hook, beyond the automatic "identical to IModelDriver body" check.
# init_scheduler is @abstractmethod on IModelDriver (no comparable body), so its
# equivalent baseline is the PipelineBaseMixin default "return None".
_TRIVIAL_BODIES: dict[str, set[str]] = {
    "init_scheduler": {"return None"},
}


# ── Source-body helpers ──────────────────────────────────────────────────────
def _normalize_body(fn) -> str | None:
    """Return a normalized method body (docstring/comments/whitespace-free).

    Parses the function via :mod:`ast` and re-emits its statements with
    ``ast.unparse`` — robust to multi-line signatures, comments, and formatting
    (a naive line-based skip breaks on the multi-line ``def foo(\\n  self,\\n
    ...\\n):`` signatures every one of these hooks uses). The leading docstring
    statement is dropped. Used to decide whether a driver override is *meaningful*
    vs a trivial redundant re-statement of the base default.
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


def _driver_meaningfully_overrides(driver_cls, hook: str) -> bool:
    """True iff ``driver_cls`` provides a non-trivial override of ``hook``.

    Non-trivial = defined below IModelDriver AND its normalized body differs from
    both the IModelDriver base body and any known trivial-equivalent body, and is
    not a bare ``return super().<hook>(...)`` delegation.
    """
    fn = getattr(driver_cls, hook, None)
    base_fn = getattr(IModelDriver, hook, None)
    if fn is None or fn is base_fn:
        return False
    body = _normalize_body(fn)
    if not body:
        return False
    base_body = _normalize_body(base_fn)
    if base_body and body == base_body:
        return False  # byte-identical re-statement of the base default
    if body in _TRIVIAL_BODIES.get(hook, set()):
        return False
    if re.fullmatch(rf"return super\(\)\.{re.escape(hook)}\([^)]*\)", body):
        return False  # pure delegation to the (base) default
    return True


def _trainer_overrides(trainer_cls, hook: str) -> bool:
    """True iff ``trainer_cls`` resolves ``hook`` to something other than the
    ``PipelineBaseMixin`` default (i.e. the trainer overrides the hook)."""
    return getattr(trainer_cls, hook, None) is not getattr(PipelineBaseMixin, hook, None)


# ── Family → trainer(s) → driver resolution ──────────────────────────────────
def _all_driver_subclasses() -> dict[str, type]:
    """Index every loaded IModelDriver subclass by class name.

    Family drivers are frequently imported LAZILY inside the trainer's
    ``_setup_family`` (``from .driver import XxxDriver``), so we force-import each
    family's ``driver`` / ``driver_base`` submodule first to populate the class
    tree before walking it.
    """
    registry.discover_families()
    for fam in registry._families:
        for sub in ("driver", "driver_base"):
            try:
                importlib.import_module(f"app.engine.models.families.{fam}.{sub}")
            except ImportError:
                pass
    # Shared driver bases live outside the registered-family set.
    for shared in ("wan_shared.driver_base", "prx_shared.driver"):
        try:
            importlib.import_module(f"app.engine.models.families.{shared}")
        except ImportError:
            pass

    idx: dict[str, type] = {}

    def _walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            idx.setdefault(sub.__name__, sub)
            _walk(sub)

    _walk(IModelDriver)
    return idx


_DRIVER_INDEX = _all_driver_subclasses()


def _trainer_variants(family_cls) -> list[type]:
    """All distinct trainer classes a family can dispatch.

    ``get_trainer_class`` reads ``self.definition.control_inputs`` for families
    that split standard vs image-edit trainers (flux1 Kontext, qwen edit); probe
    both branches with a stub definition to collect every variant.
    """
    seen: dict[str, type] = {}
    for control_inputs in (0, 1):
        inst = object.__new__(family_cls)
        inst.definition = SimpleNamespace(control_inputs=control_inputs)
        try:
            tc = family_cls.get_trainer_class(inst)
        except Exception:  # noqa: BLE001 — a family may hard-require more state
            continue
        if tc is not None:
            seen[tc.__name__] = tc
    return list(seen.values())


def _driver_for_trainer(trainer_cls) -> type | None:
    """Resolve the driver class a trainer wires via ``self.driver = XxxDriver(``.

    Searches the trainer's MRO source (edit/kontext trainers inherit
    ``_setup_family`` from their base trainer) for the first assignment, then
    resolves the class name through the global driver index.
    """
    for klass in trainer_cls.__mro__:
        try:
            src = inspect.getsource(klass)
        except (OSError, TypeError):
            continue
        m = re.search(r"self\.driver\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", src)
        if m:
            return _DRIVER_INDEX.get(m.group(1))
    return None


def _families() -> dict[str, type]:
    registry.discover_families()
    return dict(registry._families)


# Build the (family, hook) parameter grid once at collection time.
_FAMILIES = _families()
_PARAMS = [
    (fam, hook) for fam in sorted(_FAMILIES) for hook in CLOBBER_HOOKS
]


# ── Structural guards (fail loud if the introspection basis erodes) ──────────
def test_registry_has_expected_family_floor() -> None:
    assert len(_FAMILIES) >= MIN_EXPECTED_FAMILIES, (
        f"registry discovered {len(_FAMILIES)} families "
        f"(< {MIN_EXPECTED_FAMILIES}); registry breakage would silently hollow "
        f"out the meta-guard. Families: {sorted(_FAMILIES)}"
    )


def test_clobber_hook_list_is_nonempty_and_sane() -> None:
    # If this shrinks unexpectedly, the base-mixin dispatch shape changed and the
    # guard may no longer be watching the hooks it thinks it is.
    assert CLOBBER_HOOKS, "derived clobber-capable hook list is empty"
    for expected in ("init_scheduler", "add_noise", "compute_target",
                     "sample_timesteps", "get_te_cache", "set_te_cache",
                     "build_batch_extra"):
        assert expected in CLOBBER_HOOKS, (
            f"{expected!r} unexpectedly absent from the derived clobber-hook "
            f"list {CLOBBER_HOOKS} — dispatch shape changed?"
        )


def test_every_family_resolves_a_trainer_and_driver() -> None:
    """The guard is only meaningful if we actually resolve each family's
    trainer + driver classes; a silent None would make its hook checks vacuous."""
    missing = []
    for fam, fcls in sorted(_FAMILIES.items()):
        variants = _trainer_variants(fcls)
        if not variants:
            missing.append(f"{fam}: no trainer variants")
            continue
        for tc in variants:
            if _driver_for_trainer(tc) is None:
                missing.append(f"{fam}/{tc.__name__}: driver unresolved")
    assert not missing, "unresolved trainer/driver classes:\n  " + "\n  ".join(missing)


# ── The core meta-guard ──────────────────────────────────────────────────────
@pytest.mark.parametrize("family,hook", _PARAMS, ids=lambda v: v if isinstance(v, str) else None)
def test_driver_hook_override_is_wired_to_trainer(family: str, hook: str) -> None:
    """If the family DRIVER meaningfully overrides a clobber-capable hook, the
    family TRAINER must also override it (delegate) — else the driver override is
    dead code on the real training path."""
    fcls = _FAMILIES[family]
    for trainer_cls in _trainer_variants(fcls):
        driver_cls = _driver_for_trainer(trainer_cls)
        if driver_cls is None:
            continue  # covered (and failed) by the resolution guard above
        if not _driver_meaningfully_overrides(driver_cls, hook):
            continue
        if _trainer_overrides(trainer_cls, hook):
            continue
        # Driver overrides, trainer does not — either an allowlisted equivalence
        # or a LIVE dead-dispatch bug.
        if (family, hook) in ALLOWLIST:
            continue
        pytest.fail(
            f"DEAD DISPATCH: {driver_cls.__name__}.{hook} is a meaningful "
            f"override, but {trainer_cls.__name__} does NOT override "
            f"'{hook}', so the real training loop (self.{hook}) resolves to "
            f"PipelineBaseMixin.{hook} (the self-contained base default) and the "
            f"driver's override is DEAD CODE. Fix: add a trainer-level '{hook}' "
            f"that delegates to self.driver.{hook}(...) (see boogu_image / "
            f"wan22 sample_timesteps precedent), or — if the driver override is "
            f"provably equivalent to the base path — delete it (hv15 precedent) "
            f"or add an ALLOWLIST entry with justification."
        )


# ── Inverse-shape guard: _resolve_loading_dtype ──────────────────────────────
def test_resolve_loading_dtype_inverse_shape() -> None:
    """LOADING routes through ``driver.resolve_loading_dtype()``; a trainer-only
    override of ``_resolve_loading_dtype`` would be dead for loading. Pin the
    narrow sound guard: any trainer overriding ``_resolve_loading_dtype`` MUST
    have a driver overriding ``resolve_loading_dtype`` (see module docstring for
    why a blanket 'no trainer override' assertion would be wrong)."""
    offenders = []
    for fam, fcls in sorted(_FAMILIES.items()):
        for trainer_cls in _trainer_variants(fcls):
            t_over = getattr(trainer_cls, "_resolve_loading_dtype", None) is not getattr(
                PipelineBaseMixin, "_resolve_loading_dtype", None
            )
            if not t_over:
                continue
            driver_cls = _driver_for_trainer(trainer_cls)
            d_over = driver_cls is not None and (
                getattr(driver_cls, "resolve_loading_dtype", None)
                is not getattr(IModelDriver, "resolve_loading_dtype", None)
            )
            # resolve_loading_dtype is @abstractmethod on IModelDriver, so every
            # concrete driver defines it (d_over is True) — this guard therefore
            # mainly documents the inverse trap and catches a trainer override
            # paired with a driver that somehow left it abstract.
            if not d_over:
                offenders.append(f"{fam}/{trainer_cls.__name__}")
    assert not offenders, (
        "trainer overrides _resolve_loading_dtype but its driver does NOT "
        "override resolve_loading_dtype (loading path would ignore the trainer "
        f"override): {offenders}"
    )
