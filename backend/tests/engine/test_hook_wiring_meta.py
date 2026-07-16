"""Repo-wide META-GUARD for the (now-cured) dead-dispatch bug class.

HISTORY — THE BUG CLASS
-----------------------
The training pipeline dispatches lifecycle hooks on the TRAINER via MRO
(:class:`app.engine.core.pipeline.pipeline_base.PipelineBaseMixin`). Several of
those hooks historically had a **self-contained base default** that did NOT
delegate to the family driver's same-named method (``init_scheduler`` returned
``None``; ``add_noise`` went to a generic ``NoiseInterpolation`` component; etc.).
For those hooks a driver-level override was **DEAD CODE on the real training
path** unless the family's TRAINER also overrode the hook to delegate to the
driver. That exact gap bit the project repeatedly (boogu ``init_scheduler``
clobber; k5 / ltx2 I2V frame-0 trained NOISED; sdxl / flux1 pooled-TE caches
never persisted; WAN 2.2 dual-expert timesteps sampled full-range).

THE STRUCTURAL CURE (W5-1)
--------------------------
Each clobber-capable ``PipelineBaseMixin`` hook default now **auto-delegates**:
via :meth:`PipelineBaseMixin._driver_hook_override` it dispatches to
``self.driver.<hook>`` whenever the driver *meaningfully overrides* the hook
(:func:`app.engine.core.hook_dispatch.driver_meaningfully_overrides` — the SAME
predicate this guard uses), and otherwise runs the base default. Priority is:

    trainer explicit override  >  driver override (auto-delegated)  >  base default

A driver override without trainer wiring is therefore **no longer dead** — the
base hook dispatches it. This guard's job flips accordingly: instead of proving
"every meaningful driver override is trainer-wired", it now proves the
**mechanism itself works** and pins **which families depend on it**.

WHAT THIS PINS
--------------
1. Structural floors (family count, hook-list sanity, trainer/driver resolution)
   so registry breakage can't silently hollow the guard into a no-op.
2. The auto-delegation MECHANISM, functionally, on the real ``PipelineBaseMixin``
   MRO: a driver that overrides a hook is dispatched with no trainer wiring; a
   driver that does not falls back to the base default; a trainer override still
   wins over the driver's.
3. The exact set of real (family, hook) pairs that rely on auto-delegation
   (driver meaningfully overrides AND trainer does not) — a reviewed allowlist-
   equivalent. A new entry means a family started depending on the mechanism.
4. The inverse-shape guard for ``_resolve_loading_dtype`` (loading routes through
   the DRIVER; see :func:`test_resolve_loading_dtype_inverse_shape`).

PURE INTROSPECTION + tiny synthetic dispatches — no GPU, no model loads, no real
driver instantiation beyond attribute-free ``object.__new__`` shells.
"""

from __future__ import annotations

import importlib
import inspect
import re
from types import SimpleNamespace

import structlog
import torch

from app.engine.core.hook_dispatch import driver_meaningfully_overrides
from app.engine.core.interfaces import IModelDriver
from app.engine.core.pipeline.pipeline_base import PipelineBaseMixin
from app.engine.models.registry import registry
from app.engine.strategies.noise_interpolation import NoiseInterpolation

# ── Expected family floor ────────────────────────────────────────────────────
# Guards against silent registry breakage hollowing the test out to a no-op.
MIN_EXPECTED_FAMILIES = 21

# ── Reviewed set of families that rely on the auto-delegation MECHANISM ───────
# (family, hook) pairs where the DRIVER meaningfully overrides a clobber-capable
# hook and the TRAINER does NOT — so the ONLY thing wiring the driver override to
# the real training path is the base auto-delegation. Every entry is a reviewed,
# proven-safe reliance on the mechanism. Adding one is a deliberate act.
#
# wan21/wan22 ``add_noise``: WanDriverBase.add_noise is the family's contract-
# pinned flow-match lerp in [0,1000] space (t=timesteps/1000; t*noise+(1-t)*
# latents) — ALGEBRAICALLY IDENTICAL to the base NoiseInterpolation('linear')
# path. Auto-delegation therefore changes ZERO training math vs the old base
# default; it merely makes the (previously dead-but-equivalent) driver method the
# one that actually runs. Pinned by test_wan2{1,2}_precision_contracts.py.
#
# wan22_ti2v_5b ``add_noise``: Wan22Ti2v5bDriver.add_noise is the SAME base
# flow-match lerp for non-i2v-engaged steps (byte-identical — it delegates to
# ``super().add_noise`` in that branch), but on an i2v-engaged step it pins
# frame 0's noise scale to 0 (the TI2V-5B ``expand_timesteps`` conditioning
# scheme — see driver.py's module docstring). This is a REAL, deliberate
# behavior difference from the base NoiseInterpolation path (not merely a
# dead-but-equivalent restatement like wan21/wan22), and the auto-delegation
# mechanism is exactly what's relied on to route it onto the real training
# path with no trainer wiring. Pinned by
# test_wan22_ti2v_5b_i2v_conditioning.py.
# bernini_r ``add_noise``: BerniniRDriver extends WanDriverBase and inherits its
# flow-match lerp verbatim (raw [0,1000] space; ALGEBRAICALLY IDENTICAL to the
# base NoiseInterpolation('linear') path, exactly as wan21/wan22). The trainer
# overrides ``sample_timesteps`` (SD3-mode) but NOT ``add_noise``, so
# auto-delegation is the only thing routing the wan lerp onto bernini_r's real
# training path — and it changes zero training math vs the base default.
AUTODELEGATED_FAMILY_HOOKS: set[tuple[str, str]] = {
    ("wan21", "add_noise"),
    ("wan22", "add_noise"),
    ("wan22_ti2v_5b", "add_noise"),
    ("bernini_r", "add_noise"),
}


# ── Clobber-hook derivation (the auto-delegating hooks) ──────────────────────
def _clobber_hooks() -> list[str]:
    """Every method on BOTH PipelineBaseMixin and IModelDriver whose
    PipelineBaseMixin default is wired for auto-delegation (its body routes
    through :meth:`PipelineBaseMixin._driver_hook_override`). Deriving the list
    from the wiring itself — rather than hardcoding — means a newly auto-
    delegated hook is guarded automatically, and a hook that LOSES its wiring
    (regressing to a self-contained non-delegating default) drops out and trips
    :func:`test_clobber_hook_list_is_nonempty_and_sane`."""
    tr = {n for n, _ in inspect.getmembers(PipelineBaseMixin, inspect.isfunction)}
    dr = {n for n, _ in inspect.getmembers(IModelDriver, inspect.isfunction)}
    hooks = []
    for name in sorted(tr & dr):
        base_fn = getattr(PipelineBaseMixin, name)
        try:
            src = inspect.getsource(base_fn)
        except (OSError, TypeError):
            continue
        if "_driver_hook_override" in src:
            hooks.append(name)
    return hooks


CLOBBER_HOOKS = _clobber_hooks()


def _trainer_overrides(trainer_cls, hook: str) -> bool:
    """True iff ``trainer_cls`` resolves ``hook`` to something other than the
    ``PipelineBaseMixin`` default (i.e. the trainer overrides the hook)."""
    return getattr(trainer_cls, hook, None) is not getattr(
        PipelineBaseMixin, hook, None
    )


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


_FAMILIES = _families()


# ── Structural guards (fail loud if the introspection basis erodes) ──────────
def test_registry_has_expected_family_floor() -> None:
    assert len(_FAMILIES) >= MIN_EXPECTED_FAMILIES, (
        f"registry discovered {len(_FAMILIES)} families "
        f"(< {MIN_EXPECTED_FAMILIES}); registry breakage would silently hollow "
        f"out the meta-guard. Families: {sorted(_FAMILIES)}"
    )


def test_clobber_hook_list_is_nonempty_and_sane() -> None:
    # If this shrinks unexpectedly, a base hook LOST its auto-delegation wiring
    # (regressed to a self-contained non-delegating default) and the dead-
    # dispatch trap is reopened for that hook.
    assert CLOBBER_HOOKS, "derived auto-delegating hook list is empty"
    for expected in (
        "init_scheduler",
        "add_noise",
        "compute_target",
        "sample_timesteps",
        "get_te_cache",
        "set_te_cache",
        "build_batch_extra",
    ):
        assert expected in CLOBBER_HOOKS, (
            f"{expected!r} unexpectedly absent from the derived auto-delegating "
            f"hook list {CLOBBER_HOOKS} — did its base default lose the "
            f"_driver_hook_override wiring?"
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


# ── Synthetic mechanism fixtures ─────────────────────────────────────────────
class _MechDriver(IModelDriver):
    """Minimal concrete driver with NO meaningful clobber-hook override.

    Every clobber-capable hook is left at (or trivially equal to) the
    ``IModelDriver`` default, so :func:`driver_meaningfully_overrides` reports
    ``False`` for all of them — the trainer must fall back to the base default.
    """

    def __init__(self) -> None:
        self.text_cache: dict = {}

    def assign_components(self, components):  # noqa: D102
        pass

    def get_components(self):  # noqa: D102
        return {}

    def get_primary_model(self):  # noqa: D102
        return None

    def get_text_encoders(self):  # noqa: D102
        return {}

    def get_lora_targets(self):  # noqa: D102
        return []

    def init_scheduler(self):  # noqa: D102 — trivial (== base default None)
        return None

    def resolve_loading_dtype(self):  # noqa: D102
        return torch.float32

    def encode_text(self, captions, dtype):  # noqa: D102
        return None

    def get_te_lora_targets(self):  # noqa: D102
        return []

    def forward_pass(self, noisy_input, timesteps, text_embeddings, batch):  # noqa: D102
        return None

    def get_saver(self):  # noqa: D102
        return None


class _MechOverrideDriver(_MechDriver):
    """Driver that MEANINGFULLY overrides every clobber-capable hook with a
    recognizable sentinel — used to prove auto-delegation dispatches each one."""

    def add_noise(self, latents, noise, timesteps):
        return "D:add_noise"

    def build_batch_extra(self, items):
        return {"src": "driver"}

    def compute_target(self, latents, noise, timesteps):
        return "D:compute_target"

    def get_te_cache(self):
        return {"driver_cache": {"k": torch.zeros(1)}}

    def init_scheduler(self):
        return "D:scheduler"

    def sample_timesteps(self, batch_size, device, config, latents=None):
        return "D:sample_timesteps"

    def set_te_cache(self, caches):
        self.restored = caches


class _MechTrainer(PipelineBaseMixin):
    """Concrete PipelineBaseMixin subclass (all abstracts stubbed) so we can
    exercise the REAL base-hook MRO with ``object.__new__`` + manual state."""

    def _setup_family(self):  # noqa: D102
        pass

    async def setup(self):  # noqa: D102
        pass

    async def load_model(self):  # noqa: D102
        pass

    async def prepare_data(self):  # noqa: D102
        pass

    async def train(self):  # noqa: D102
        pass


class _MechTrainerOverridesAddNoise(_MechTrainer):
    """A trainer that explicitly overrides add_noise — must win over the
    driver's override (precedence: trainer > driver > base)."""

    def add_noise(self, latents, noise, timesteps):
        return "T:add_noise"


def _make_trainer(driver, *, cls=_MechTrainer):
    t = object.__new__(cls)
    t.driver = driver
    t.device = torch.device("cpu")
    t.config = {}
    t.text_cache = {}
    t.noise_interpolation = NoiseInterpolation("linear")
    t.logger = structlog.get_logger("mech")
    return t


# ── The core mechanism guards ────────────────────────────────────────────────
def test_autodelegation_dispatches_driver_override_for_every_hook() -> None:
    """A driver that overrides each clobber hook is dispatched on the REAL base
    MRO path with NO trainer wiring — the structural cure for dead dispatch."""
    t = _make_trainer(_MechOverrideDriver())
    latents = torch.randn(2, 4, 4, 4)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([500.0, 500.0])

    assert t.add_noise(latents, noise, timesteps) == "D:add_noise"
    assert t.compute_target(latents, noise, timesteps) == "D:compute_target"
    assert t.build_batch_extra([]) == {"src": "driver"}
    assert t.init_scheduler() == "D:scheduler"
    assert t.sample_timesteps(2) == "D:sample_timesteps"
    assert t.get_te_cache() == {"driver_cache": {"k": torch.zeros(1)}}

    sentinel = {"te": {"cap": torch.ones(1)}}
    t.set_te_cache(sentinel)
    assert t.driver.restored is sentinel


def test_base_default_runs_when_driver_does_not_override() -> None:
    """A driver with no meaningful override leaves each hook at the base
    default — auto-delegation must NOT fire and change behavior."""
    t = _make_trainer(_MechDriver())
    latents = torch.randn(2, 4, 4, 4)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([500.0, 500.0])

    assert torch.equal(
        t.add_noise(latents, noise, timesteps),
        t.noise_interpolation.add_noise(latents, noise, timesteps),
    )
    assert torch.equal(t.compute_target(latents, noise, timesteps), noise - latents)
    assert t.build_batch_extra([]) == {}
    assert t.init_scheduler() is None

    # get/set_te_cache fall back to the trainer-level text_cache dict.
    assert t.get_te_cache() is None  # empty text_cache
    t.text_cache = {"cap": torch.ones(1)}
    got = t.get_te_cache()
    assert set(got) == {"te"} and "cap" in got["te"]
    t.set_te_cache({"te": {"other": torch.zeros(1)}})
    assert set(t.text_cache) == {"other"}

    # sample_timesteps falls back to the base TimestepSampler (a real tensor).
    out = t.sample_timesteps(2, latents=latents)
    assert isinstance(out, torch.Tensor)


def test_trainer_override_wins_over_driver_override() -> None:
    """Precedence: an explicit TRAINER override shadows the base auto-delegating
    hook entirely via MRO, so the driver's override never runs."""
    t = _make_trainer(_MechOverrideDriver(), cls=_MechTrainerOverridesAddNoise)
    latents = torch.randn(2, 4, 4, 4)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([500.0, 500.0])
    assert t.add_noise(latents, noise, timesteps) == "T:add_noise"


def test_missing_driver_falls_back_to_base_default() -> None:
    """A trainer shell without a ``driver`` attribute (early setup / hv15-style
    dispatch tests) must still resolve the base default, not AttributeError."""
    t = object.__new__(_MechTrainer)
    t.noise_interpolation = NoiseInterpolation("linear")
    latents = torch.randn(2, 4, 4, 4)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([250.0, 900.0])
    assert torch.equal(
        t.add_noise(latents, noise, timesteps),
        t.noise_interpolation.add_noise(latents, noise, timesteps),
    )


def test_real_family_autodelegation_engages_on_wan() -> None:
    """Non-vacuity: the mechanism actually engages on a REAL family. WAN's
    driver add_noise is dispatched through the base hook with no trainer wiring,
    and is bit-identical to both the driver method and the base linear path."""
    from app.engine.models.families.wan21.driver import Wan21Driver
    from app.engine.models.families.wan21.trainer import Wan21Trainer

    assert driver_meaningfully_overrides(Wan21Driver, "add_noise")
    assert not _trainer_overrides(Wan21Trainer, "add_noise")

    t = _make_trainer(object.__new__(Wan21Driver))
    # The base hook must SELECT the driver method (mechanism engaged), not fall
    # back — wan's driver output is bit-identical to the base linear path, so
    # this identity check is what actually proves delegation (not the values).
    selected = t._driver_hook_override("add_noise")
    assert selected is not None
    assert selected.__func__ is Wan21Driver.add_noise

    latents = torch.randn(2, 16, 1, 4, 4)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([700.0, 700.0])

    via_trainer = t.add_noise(latents, noise, timesteps)
    via_driver = Wan21Driver.add_noise(t.driver, latents, noise, timesteps)
    assert torch.equal(via_trainer, via_driver), (
        "base auto-delegation must dispatch WAN's real driver add_noise"
    )
    # Equivalent to the base NoiseInterpolation('linear') path it replaced.
    assert torch.allclose(
        via_trainer, t.noise_interpolation.add_noise(latents, noise, timesteps)
    )


def test_autodelegated_family_hook_set_is_exactly_expected() -> None:
    """Sweep every real family: the set of (family, hook) relying on auto-
    delegation (driver meaningfully overrides AND trainer does NOT) must match
    the reviewed :data:`AUTODELEGATED_FAMILY_HOOKS`. A NEW entry means a family
    started depending on the base mechanism — a deliberate, reviewed event that
    must be added here with a safety justification. A MISSING entry means a
    family that used to rely on it grew a trainer override (harmless) or lost its
    driver override (investigate)."""
    got: set[tuple[str, str]] = set()
    for fam, fcls in sorted(_FAMILIES.items()):
        for tc in _trainer_variants(fcls):
            dc = _driver_for_trainer(tc)
            if dc is None:
                continue
            for hook in CLOBBER_HOOKS:
                if driver_meaningfully_overrides(dc, hook) and not _trainer_overrides(
                    tc, hook
                ):
                    got.add((fam, hook))
    assert got == AUTODELEGATED_FAMILY_HOOKS, (
        "auto-delegated (family, hook) set drifted from the reviewed "
        f"expectation.\n  unexpected (newly relying on the mechanism): "
        f"{sorted(got - AUTODELEGATED_FAMILY_HOOKS)}\n  missing (no longer "
        f"relying): {sorted(AUTODELEGATED_FAMILY_HOOKS - got)}"
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
            t_over = getattr(
                trainer_cls, "_resolve_loading_dtype", None
            ) is not getattr(PipelineBaseMixin, "_resolve_loading_dtype", None)
            if not t_over:
                continue
            driver_cls = _driver_for_trainer(trainer_cls)
            d_over = driver_cls is not None and (
                getattr(driver_cls, "resolve_loading_dtype", None)
                is not getattr(IModelDriver, "resolve_loading_dtype", None)
            )
            if not d_over:
                offenders.append(f"{fam}/{trainer_cls.__name__}")
    assert not offenders, (
        "trainer overrides _resolve_loading_dtype but its driver does NOT "
        "override resolve_loading_dtype (loading path would ignore the trainer "
        f"override): {offenders}"
    )
