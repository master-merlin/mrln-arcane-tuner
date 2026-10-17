"""LANE-157: a trainer worker must survive the text-encoder offload.

The reproduction (ltx2-5-base, 2026-10-15; flux3 jobs 8dc8d0d8 / 80a5bb96 on
LANE-144) died silently right after ``text_encoders_offloaded_to_cpu``. The
root cause is the offload doubling the process's host commit charge; guard 5
below pins the sliced offload (``offload_module_to_cpu``).

This branch carries only guard 5 of LANE-157's file (cherry-picked from
3d1779cf): guards 1-4 (faulthandler on the worker's command line, the
watchdog's exit code, the offload sub-step logging, the ltx2 connector
instrumentation) pin LANE-157 commits this branch does not carry, and arrive
with LANE-157's merge to main.

Every assertion carries the marker ``LANE-157 loud-death guard``.
"""

from __future__ import annotations

import app.engine.core.pipeline.pipeline_loading as pl
from app.engine.models.families.qwen_image.driver import QwenImageDriver
from app.engine.models.families.qwen_image.trainer import QwenImageTrainer

MARKER = "LANE-157 loud-death guard"


class _RecordingLogger:
    def __init__(self, events: list) -> None:
        self._events = events

    def info(self, event, **kw):  # noqa: ANN001
        self._events.append(("log", event, kw))

    def debug(self, event, **kw):  # noqa: ANN001
        self._events.append(("log", event, kw))

    def warning(self, event, **kw):  # noqa: ANN001
        self._events.append(("log", event, kw))

    error = warning


# ── 5. the root cause: the offload must not double the commit charge ────
#
# Measured on the reproduction host (Windows, WDDM; .agent/workdir/lane-157/
# cuda_commit_probe.py + e1-mem.log, 2026-10-15): every CUDA allocation charges
# the SAME number of bytes to the process's host commit as a CPU allocation does,
# and the caching allocator keeps a freed block charged until empty_cache(). So
# `te.to("cpu")` of the 24 GB Gemma4 encoder charged 24 GB of new CPU tensors
# while the 24 GB it copied from (plus the ~9 GB the text-encoding forward left
# in the cache) stayed charged: commit available fell from 13.6 GB to 0.0 GB and
# the worker died 0xC0000005 inside `convert` with 62 GB of RAM still free.
# The offload must release the cache first and then move the encoder in bounded
# chunks, releasing each chunk's device blocks before the next one is copied.


class _CommitLedger:
    """Host commit as WDDM charges it: live device + cached device + CPU copies."""

    def __init__(self, device_live: int, cached: int) -> None:
        self.device_live = device_live
        self.cached = cached
        self.cpu = 0
        self.baseline = self.charged
        self.peak = self.charged

    @property
    def charged(self) -> int:
        return self.device_live + self.cached + self.cpu

    def move_to_cpu(self, nbytes: int) -> None:
        self.device_live -= nbytes
        self.cached += nbytes  # freed into the caching allocator, still charged
        self.cpu += nbytes
        self.peak = max(self.peak, self.charged)

    def empty_cache(self) -> None:
        self.cached = 0


def _ledger_leaf(ledger: _CommitLedger, numel: int):
    import torch

    class _Leaf(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(numel), requires_grad=False)
            self._on_device = True

        def _apply(self, fn, recurse=True):  # noqa: ANN001
            out = super()._apply(fn, recurse)
            if self._on_device:  # the first move off the (modelled) device
                self._on_device = False
                ledger.move_to_cpu(self.weight.numel() * self.weight.element_size())
            return out

    return _Leaf()


def test_text_encoder_offload_does_not_double_the_commit_charge(monkeypatch):
    import torch

    leaf_numel, n_layers = 1024, 8
    leaf_bytes = leaf_numel * 4
    total = leaf_bytes * n_layers * 2
    forward_cache = leaf_bytes * 3  # what the text-encoding forward left cached
    ledger = _CommitLedger(device_live=total, cached=forward_cache)

    te = torch.nn.Module()
    te.layers = torch.nn.ModuleList()
    for _ in range(n_layers):
        block = torch.nn.Module()
        block.attn = _ledger_leaf(ledger, leaf_numel)
        block.mlp = _ledger_leaf(ledger, leaf_numel)
        te.layers.append(block)

    events: list = []
    driver = object.__new__(QwenImageDriver)
    driver.text_encoder = te
    trainer = object.__new__(QwenImageTrainer)
    trainer.driver = driver
    trainer.components = {"text_encoder": te}
    trainer.config = {"cache_text_embeddings": True, "unload_text_encoder": False}
    trainer.text_cache = {}
    trainer._te_unloaded = False
    trainer.device = "cuda"
    trainer.logger = _RecordingLogger(events)

    # One chunk = one block (attn + mlp): the bound the offload must keep.
    monkeypatch.setattr(pl, "_OFFLOAD_CHUNK_BYTES", 2 * leaf_bytes, raising=False)
    monkeypatch.setattr(pl.torch.cuda, "empty_cache", ledger.empty_cache)
    monkeypatch.setattr(pl.gc, "collect", lambda *a: 0)

    trainer._offload_text_encoders()

    assert ledger.cpu == total, (
        f"{MARKER}: the offload did not move the whole encoder ({ledger.cpu} of "
        f"{total} bytes)"
    )
    extra = ledger.peak - ledger.baseline
    assert extra <= 2 * leaf_bytes, (
        f"{MARKER}: the text-encoder offload charged {extra} bytes of host commit "
        f"above its starting point (bound: one {2 * leaf_bytes}-byte chunk; "
        f"the whole encoder is {total}) -- it copies to CPU while the device "
        f"copy and the forward's cached blocks stay charged, which exhausted "
        f"the commit limit and killed the LTX-2.5 worker with 0xC0000005"
    )
