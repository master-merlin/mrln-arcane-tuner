"""Tests for the shared GPU-plugin unload helper (P2c / B-CLEAN-9).

CaptionService, MaskingService, and ScoringService each triplicated the same
unload shape (unload every plugin, reset active-model bookkeeping,
gc.collect() + torch.cuda.synchronize() + torch.cuda.empty_cache()). This
pins the shared helper's contract directly; per-service call sites are
covered by their own service tests.

Follows the torch.cuda stubbing pattern used elsewhere in the suite
(``monkeypatch.setattr(torch.cuda, "is_available", ...)``) rather than
mocking the whole torch module.
"""
from unittest.mock import MagicMock

import torch

from app.core.gpu_unload import unload_gpu_plugins


class _Owner:
    """Stand-in for a service class/instance carrying active-key bookkeeping."""
    _active_model_id = None


def test_unloads_every_plugin_in_the_dict():
    owner = _Owner()
    plugin_a, plugin_b = MagicMock(), MagicMock()
    unload_gpu_plugins(
        owner,
        plugins={"a": plugin_a, "b": plugin_b},
        active_attr="_active_model_id",
        service_label="test",
    )
    plugin_a.unload.assert_called_once()
    plugin_b.unload.assert_called_once()


def test_resets_the_active_key_on_owner():
    owner = _Owner()
    owner._active_model_id = "loaded-model"
    unload_gpu_plugins(
        owner, plugins={}, active_attr="_active_model_id", service_label="test",
    )
    assert owner._active_model_id is None


def test_synchronize_gc_and_empty_cache_invoked_when_cuda_available(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))

    import app.core.gpu_unload as gpu_unload_mod
    monkeypatch.setattr(gpu_unload_mod.gc, "collect", lambda: calls.append("gc"))

    owner = _Owner()
    unload_gpu_plugins(owner, plugins={}, active_attr="_active_model_id", service_label="test")

    assert calls == ["gc", "synchronize", "empty_cache"]


def test_skips_cuda_calls_when_cuda_unavailable(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))

    owner = _Owner()
    unload_gpu_plugins(owner, plugins={}, active_attr="_active_model_id", service_label="test")

    assert calls == []


def test_empty_plugins_dict_does_not_raise(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    owner = _Owner()
    unload_gpu_plugins(owner, plugins={}, active_attr="_active_model_id", service_label="test")
    assert owner._active_model_id is None
