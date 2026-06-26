"""Tests for the training-loop VRAM safety valve.

``GenericTrainingPipeline._maybe_reclaim_vram`` releases free cached blocks
(``empty_cache``) when the caching-allocator reserved pool crosses a configured
fraction of total VRAM — preventing bucket-shape fragmentation from ratcheting
reserved past the card into Windows shared memory (a freeze). It must be a
no-op when disabled or when memory is comfortable.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from app.engine.core.pipeline.pipeline_train import PipelineTrainMixin


class _Probe(PipelineTrainMixin):
    """Bind just enough state to exercise _maybe_reclaim_vram in isolation."""

    def __init__(self, fraction):
        self.config = {"training_vram_reclaim_fraction": fraction}
        self.device = torch.device("cuda", 0)
        self.logger = SimpleNamespace(info=lambda *a, **k: None)


_TOTAL = 96_000 * 1024**2  # ~94 GB card


def _patch_total(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda dev: SimpleNamespace(total_memory=_TOTAL),
    )


def _count_empty_cache(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: calls.__setitem__("n", calls["n"] + 1)
    )
    return calls


def test_reclaims_when_reserved_above_fraction(monkeypatch):
    _patch_total(monkeypatch)
    calls = _count_empty_cache(monkeypatch)
    _Probe(fraction=0.9)._maybe_reclaim_vram(int(0.95 * _TOTAL))
    assert calls["n"] == 1


def test_no_reclaim_when_reserved_below_fraction(monkeypatch):
    _patch_total(monkeypatch)
    calls = _count_empty_cache(monkeypatch)
    _Probe(fraction=0.9)._maybe_reclaim_vram(int(0.80 * _TOTAL))
    assert calls["n"] == 0


def test_disabled_with_zero_fraction(monkeypatch):
    _patch_total(monkeypatch)
    calls = _count_empty_cache(monkeypatch)
    _Probe(fraction=0.0)._maybe_reclaim_vram(_TOTAL)  # fully reserved, but disabled
    assert calls["n"] == 0


def test_safe_when_device_props_raise(monkeypatch):
    def _boom(_dev):
        raise RuntimeError("no device")

    monkeypatch.setattr(torch.cuda, "get_device_properties", _boom)
    calls = _count_empty_cache(monkeypatch)
    _Probe(fraction=0.9)._maybe_reclaim_vram(_TOTAL)  # must not raise
    assert calls["n"] == 0
