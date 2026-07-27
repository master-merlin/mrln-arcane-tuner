"""``PipelineBaseMixin.is_video_family`` / ``is_audio_family`` must resolve
ONCE per run and must never swallow a resolver misconfiguration as False.

``pipeline_base.py`` previously re-ran ``resolve_video_profile`` /
``resolve_capabilities`` (a registry lookup + capability merge +
``build_field_visibility`` over ~30 field rules) on EVERY property access —
2-4x per accumulation step. Worse, ``except Exception: return False`` turned
a registry/definition misconfiguration into "image family", so a broken
video family would silently collate 4D and mis-train instead of failing
loudly.

``functools.cached_property`` fixes the perf issue (resolve once) and
dropping the blanket ``except`` lets resolver errors propagate. Bare test
harnesses that construct these mixins without a ``definition`` (e.g.
``_Harness`` in ``test_control_video_latents.py``) must keep returning
``False`` without raising — an explicit early-out on missing ``definition``
covers that, distinct from masking a real resolver failure.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import structlog

import app.engine.core.archetypes as archetypes_mod
import app.engine.core.video_contract as video_contract_mod
import app.engine.models.registry as registry_mod
from app.engine.core.pipeline.pipeline_base import PipelineBaseMixin
from app.engine.core.pipeline.pipeline_data import PipelineDataMixin


class _Harness(PipelineBaseMixin):
    """Concrete shell for exercising the capability-flag properties directly
    (no driver/loader/saver wiring needed for this unit)."""

    def _setup_family(self):  # pragma: no cover - abstract stub
        pass

    async def setup(self):  # pragma: no cover - abstract stub
        pass

    async def load_model(self):  # pragma: no cover - abstract stub
        pass

    async def prepare_data(self):  # pragma: no cover - abstract stub
        pass

    async def train(self):  # pragma: no cover - abstract stub
        pass


def _make(definition=None) -> _Harness:
    t = object.__new__(_Harness)
    t.logger = structlog.get_logger("test_pipeline_base_capability_flags")
    if definition is not None:
        t.definition = definition
    return t


# ── (a) resolution happens ONCE across repeated access ──────────────────────


def test_is_video_family_resolves_once_across_repeated_access(monkeypatch):
    calls = {"n": 0}

    def _fake_resolve(definition):
        calls["n"] += 1
        return SimpleNamespace(is_video=True)

    monkeypatch.setattr(video_contract_mod, "resolve_video_profile", _fake_resolve)

    t = _make(definition=SimpleNamespace(family="fake"))
    for _ in range(5):
        assert t.is_video_family is True

    assert calls["n"] == 1


def test_is_audio_family_resolves_once_across_repeated_access(monkeypatch):
    calls = {"n": 0}

    def _fake_resolve(definition):
        calls["n"] += 1
        return {"capabilities": {"is_audio_family": True}}

    monkeypatch.setattr(archetypes_mod, "resolve_capabilities", _fake_resolve)

    t = _make(definition=SimpleNamespace(family="fake"))
    for _ in range(5):
        assert t.is_audio_family is True

    assert calls["n"] == 1


# ── (b) a raising resolver propagates instead of returning False ───────────


def test_is_video_family_propagates_resolver_errors(monkeypatch):
    def _boom(definition):
        raise ValueError("video family misconfigured")

    monkeypatch.setattr(video_contract_mod, "resolve_video_profile", _boom)

    t = _make(definition=SimpleNamespace(family="broken"))
    with pytest.raises(ValueError, match="video family misconfigured"):
        _ = t.is_video_family


def test_is_audio_family_propagates_resolver_errors(monkeypatch):
    def _boom(definition):
        raise KeyError("unregistered family")

    monkeypatch.setattr(archetypes_mod, "resolve_capabilities", _boom)

    t = _make(definition=SimpleNamespace(family="broken"))
    with pytest.raises(KeyError, match="unregistered family"):
        _ = t.is_audio_family


# ── (c) the bare-harness path (no definition) still works ──────────────────


def test_bare_harness_without_definition_is_false_for_both_flags():
    t = _make(definition=None)

    assert t.is_video_family is False
    assert t.is_audio_family is False


def test_bare_harness_never_calls_the_resolver(monkeypatch):
    calls = {"n": 0}

    def _fake_resolve(definition):
        calls["n"] += 1
        return SimpleNamespace(is_video=True)

    monkeypatch.setattr(video_contract_mod, "resolve_video_profile", _fake_resolve)

    t = _make(definition=None)
    assert t.is_video_family is False
    assert calls["n"] == 0


# ── (d) resolver misconfiguration must fail loudly, REGARDLESS of which
#      call site touches the property first (Finding 2, W2.T8 review) ──────
#
# The three real CONSUMER sites (`pipeline_data.py:1007,1172`,
# `pipeline_train.py:409-410`) read the flags via
# `getattr(self, "is_video_family", False)`. Python's `getattr(obj, name,
# default)` cannot distinguish "attribute genuinely absent" from "the
# descriptor's getter raised AttributeError" -- it swallows BOTH into
# `default`. `resolve_capabilities` (`archetypes.py:250`) does
# `ARCHETYPES[family_cls.archetype]`, and `archetype` is a bare, undefaulted
# `ModelFamily` class attr (`definitions.py:63-69`) every family sets
# individually -- a family that forgets it raises exactly that
# AttributeError.
#
# Today this is inert only because `prepare_data()`'s UNGUARDED read
# (`pipeline_data.py:377`) happens to run before any guarded site, so the
# raw AttributeError propagates loudly before it can be swallowed. Nothing
# pinned that ordering -- a routine refactor (reordering `prepare_data()`,
# making the read lazy, moving the video-contract check) could silently
# reintroduce the exact silent-mis-training bug this task exists to prevent.
#
# Fix: convert a resolver-raised `AttributeError` into a `RuntimeError`
# INSIDE the property, before it can ever reach an outer `getattr(...,
# default)`. `getattr` only swallows `AttributeError`, so a `RuntimeError`
# propagates through the guarded pattern unconditionally -- the guarantee
# becomes structural (true for every call site, in every order), not an
# accident of which statement happens to run first.
#
# NOTE: replacing the 3 guarded call sites with direct reads (the brief's
# named "preferred approach") was investigated and rejected -- real,
# non-contrived bare harnesses construct `PipelineDataMixin`/
# `PipelineTrainMixin` WITHOUT `PipelineBaseMixin` in their MRO and without
# ever setting `is_video_family`/`is_audio_family` as an instance attribute
# (`test_control_video_latents.py::_Harness`,
# `test_edit_pipeline_data.py::_Harness`,
# `test_nan_window_skip.py::_ScriptedTrainer` -- the last one drives the
# REAL `train()` coroutine end-to-end). A direct read on those objects
# raises a bare "no such attribute" `AttributeError` immediately -- the
# property's own `definition is None` early-out never even gets a chance to
# run, because the descriptor doesn't exist in the class's MRO at all. That
# would BREAK all three currently-green test files. See the fix-wave report
# section for the concrete verification.


class _FamilyMissingArchetype:
    """A family class that forgot to declare `archetype`.

    `ModelFamily` (`definitions.py:63-69`) leaves `archetype` a bare,
    undefaulted annotation -- exactly the shape of misconfiguration the
    brief named. Accessing `.archetype` on this class raises a plain
    `AttributeError`, mirroring `ARCHETYPES[family_cls.archetype]`
    (`archetypes.py:250`) for a family that never set it.
    """


def _patch_broken_registry(monkeypatch) -> None:
    monkeypatch.setattr(
        registry_mod.registry,
        "get_family_class",
        lambda family_id: _FamilyMissingArchetype,
    )


def test_is_video_family_resolver_attributeerror_is_not_a_bare_attributeerror(
    monkeypatch,
):
    _patch_broken_registry(monkeypatch)

    t = _make(definition=SimpleNamespace(family="broken"))
    with pytest.raises(Exception) as exc_info:
        _ = t.is_video_family

    # Must never be a bare AttributeError -- that's exactly what
    # `getattr(obj, name, default)` (the pattern at the 3 guarded consumer
    # sites) silently swallows as "not a video family".
    assert not isinstance(exc_info.value, AttributeError)


def test_is_audio_family_resolver_attributeerror_is_not_a_bare_attributeerror(
    monkeypatch,
):
    _patch_broken_registry(monkeypatch)

    t = _make(definition=SimpleNamespace(family="broken"))
    with pytest.raises(Exception) as exc_info:
        _ = t.is_audio_family

    assert not isinstance(exc_info.value, AttributeError)


def test_is_video_family_resolver_error_survives_the_guarded_getattr_pattern(
    monkeypatch,
):
    """Mirrors the EXACT pattern at pipeline_data.py:1007/1172 and
    pipeline_train.py:409-410. Before the fix this silently returned False;
    after, the misconfiguration still raises through it -- proving the
    guarantee no longer depends on call order."""
    _patch_broken_registry(monkeypatch)

    t = _make(definition=SimpleNamespace(family="broken"))
    with pytest.raises(Exception):
        bool(getattr(t, "is_video_family", False))


def test_is_audio_family_resolver_error_survives_the_guarded_getattr_pattern(
    monkeypatch,
):
    _patch_broken_registry(monkeypatch)

    t = _make(definition=SimpleNamespace(family="broken"))
    with pytest.raises(Exception):
        bool(getattr(t, "is_audio_family", False))


def test_missing_archetype_family_raises_loudly_through_real_prepare_data(
    monkeypatch,
):
    """Integration-level pin of Finding 2's deliverable: a misconfigured
    family must fail LOUDLY through the REAL `prepare_data()` path (not a
    re-implementation, not just the isolated property). Drives the actual
    `PipelineDataMixin.prepare_data()` (`pipeline_data.py:366`), whose first
    real statement after allocating the bucket manager
    (`resolve_still_resolutions(self.config, self.is_video_family)`, line
    377) resolves capabilities for a family that forgot `archetype`.

    Deliberately does NOT assert on which specific line inside
    `prepare_data()` raises -- only that the coroutine as a whole raises
    instead of completing/silently defaulting -- so this test does not
    depend on statement order inside `prepare_data()` and keeps catching a
    future reordering of its body.
    """
    _patch_broken_registry(monkeypatch)

    class _DataHarness(PipelineDataMixin, PipelineBaseMixin):
        """Mirrors the real `GenericTrainingPipeline`'s MRO closely enough
        to exercise the REAL `prepare_data()` with a broken family."""

        def _setup_family(self):  # pragma: no cover - abstract stub
            pass

        async def setup(self):  # pragma: no cover - abstract stub
            pass

        async def load_model(self):  # pragma: no cover - abstract stub
            pass

        async def train(self):  # pragma: no cover - abstract stub
            pass

    t = object.__new__(_DataHarness)
    t.logger = structlog.get_logger("test_missing_archetype_integration")
    t.config = {"resolutions": [512]}
    t.definition = SimpleNamespace(family="broken")

    with pytest.raises(Exception):
        asyncio.run(t.prepare_data())
