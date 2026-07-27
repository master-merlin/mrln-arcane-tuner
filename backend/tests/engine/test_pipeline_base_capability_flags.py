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

from types import SimpleNamespace

import pytest
import structlog

import app.engine.core.archetypes as archetypes_mod
import app.engine.core.video_contract as video_contract_mod
from app.engine.core.pipeline.pipeline_base import PipelineBaseMixin


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
