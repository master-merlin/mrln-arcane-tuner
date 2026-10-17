"""LANE-157: a trainer worker that dies must die LOUDLY.

The reproduction (ltx2-5-base, 2026-10-15) ended ``stopped`` with
``Process exited unexpectedly (detected by watchdog)`` and nothing else: no
traceback, no exit code, and a last log line (``text_encoders_offloaded_to_cpu``)
emitted BEFORE the work it names. Three guards make the next death name itself:

1. the worker is launched with ``-X faulthandler`` (right after ``-u``), so a
   native crash dumps every thread's Python stack into ``trainer_stdout.log``;
2. the job manager keeps the ``Popen`` handle per job (``_processes``) and the
   pid watchdog writes the handle's exit code into ``job.error``, then
   releases the handle;
3. ``_offload_text_encoders`` logs one ``text_encoder_offload_step`` before
   each sub-step and ``text_encoders_offloaded_to_cpu`` only AFTER the moves;
   ``_move_component_to_gpu`` logs ``moving_component_to_gpu`` before its
   ``.to()``.

Every assertion carries the marker ``LANE-157 loud-death guard``.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import app.core.job_manager as jm
import app.engine.core.pipeline.pipeline_loading as pl
import app.engine.models.training_plugin as tp
from app.core.job import JobStatus
from app.core.job_manager import JobManager
from app.engine.models.families.qwen_image.driver import QwenImageDriver
from app.engine.models.families.qwen_image.trainer import QwenImageTrainer
from app.engine.models.training_plugin import StandardPlugin

MARKER = "LANE-157 loud-death guard"

# STATUS_ACCESS_VIOLATION as Windows reports it through Popen.returncode.
ACCESS_VIOLATION = 0xC0000005


# ── 1. faulthandler on the worker's command line ─────────────────────────


def test_trainer_is_launched_with_faulthandler(monkeypatch, tmp_path):
    captured: dict = {}

    def _recording_popen(cmd, **kwargs):  # noqa: ANN001
        captured["argv"] = list(cmd)
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr(tp.subprocess, "Popen", _recording_popen)

    StandardPlugin().start_training(
        {
            "definition_id": "ltx2-5-base",
            "output_dir": str(tmp_path),
            "job_id": "lane-157-job",
        }
    )
    stdout = captured.get("kwargs", {}).get("stdout")
    if hasattr(stdout, "close"):
        stdout.close()  # the boot log handle start_training opened

    argv = captured["argv"]
    script_idx = next(
        i for i, a in enumerate(argv) if str(a).endswith("run_trainer.py")
    )
    assert argv[1:script_idx] == ["-u", "-X", "faulthandler"], (
        f"{MARKER}: the worker is launched without -X faulthandler right after "
        f"-u (interpreter flags before the script: {argv[1:script_idx]!r}), so a "
        "native crash leaves no stack in trainer_stdout.log"
    )


# ── 2. the watchdog reports the kept handle's exit code ─────────────────


class _SyncThread:
    """Runs the watchdog body inline, once, instead of on a daemon thread."""

    def __init__(self, target=None, daemon=None, name=None, **_kw):  # noqa: ANN001
        self._target = target

    def start(self) -> None:
        self._target()


def test_watchdog_reports_the_kept_handles_exit_code(monkeypatch):
    mgr = JobManager()
    job = mgr.create_job("flux/dev", {"output_dir": "outputs", "lora_name": "loud"})
    job.status = JobStatus.RUNNING
    job.pid = 100

    handle = MagicMock()
    handle.pid = 100
    handle.poll.return_value = ACCESS_VIOLATION
    handle.returncode = ACCESS_VIOLATION
    processes = getattr(mgr, "_processes", None)
    if processes is None:  # pre-fix tree: the manager never kept the handle
        processes = {}
        mgr._processes = processes
    processes[job.id] = handle

    # Narrow, module-local doubles: only job_manager's view of threading/time
    # changes (the real modules stay untouched for every other thread).
    monkeypatch.setattr(jm, "threading", SimpleNamespace(Thread=_SyncThread))
    monkeypatch.setattr(
        jm, "time", SimpleNamespace(sleep=lambda _s: None, time=time.time),
    )
    with patch.object(JobManager, "_is_pid_alive", side_effect=lambda p: False), \
         patch.object(JobManager, "_resolve_worker_pid", return_value=200), \
         patch.object(JobManager, "_terminate_process_tree", return_value=[]), \
         patch.object(mgr, "_persist_status"), \
         patch.object(mgr, "schedule_advance_queue"):
        mgr._start_pid_watchdog(job.id, 100)

    assert job.status == JobStatus.STOPPED, (
        f"{MARKER}: the watchdog did not retire the dead job (status {job.status!r})"
    )
    assert job.error == (
        "Process exited unexpectedly (exit code 0xC0000005, detected by watchdog)"
    ), (
        f"{MARKER}: the watchdog error does not carry the kept handle's exit "
        f"code: {job.error!r}"
    )
    assert job.id not in getattr(mgr, "_processes", {}), (
        f"{MARKER}: the Popen handle was not released on the terminal transition"
    )


def test_watchdog_reaps_a_really_exited_worker_even_when_its_pid_looks_alive(
    monkeypatch,
):
    """A REAL abnormal exit, with the pid check lying 'alive' (the POSIX zombie)."""
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])
    proc.wait()  # exited; deliberately not polled before the watchdog runs

    mgr = JobManager()
    job = mgr.create_job("flux/dev", {"output_dir": "outputs", "lora_name": "loud"})
    job.status = JobStatus.RUNNING
    job.pid = proc.pid
    mgr._processes[job.id] = proc

    cycles = {"n": 0}

    def _bounded_sleep(_s):  # noqa: ANN001
        cycles["n"] += 1
        assert cycles["n"] < 20, (
            f"{MARKER}: the watchdog never noticed the exited handle "
            "(pid looks alive)"
        )

    monkeypatch.setattr(jm, "threading", SimpleNamespace(Thread=_SyncThread))
    monkeypatch.setattr(
        jm, "time", SimpleNamespace(sleep=_bounded_sleep, time=time.time),
    )
    with patch.object(JobManager, "_is_pid_alive", return_value=True), \
         patch.object(JobManager, "_resolve_worker_pid", return_value=proc.pid), \
         patch.object(JobManager, "_terminate_process_tree", return_value=[]), \
         patch.object(mgr, "_persist_status"), \
         patch.object(mgr, "schedule_advance_queue"):
        mgr._start_pid_watchdog(job.id, proc.pid)

    assert job.status == JobStatus.STOPPED, (
        f"{MARKER}: an exited worker whose pid still looks alive was never "
        f"retired (status {job.status!r})"
    )
    assert job.error == (
        "Process exited unexpectedly (exit code 0x00000003, detected by watchdog)"
    ), f"{MARKER}: the real exit code is missing from the error: {job.error!r}"
    assert job.id not in mgr._processes, (
        f"{MARKER}: the exited handle was not released"
    )


def test_delete_job_releases_the_kept_handle():
    mgr = JobManager()
    job = mgr.create_job("flux/dev", {"output_dir": "outputs", "lora_name": "loud"})
    job.status = JobStatus.COMPLETED
    handle = MagicMock()
    handle.pid = 100
    mgr._processes[job.id] = handle

    with patch.object(mgr, "_persist_delete"):
        mgr.delete_job(job.id)

    assert job.id not in mgr._processes, (
        f"{MARKER}: delete_job left the job's Popen handle in _processes"
    )


def test_failed_start_after_spawn_releases_the_kept_handle(monkeypatch):
    mgr = JobManager()
    job = mgr.create_job("flux/dev", {"output_dir": "outputs", "lora_name": "loud"})
    handle = MagicMock()
    handle.pid = 4242
    plugin = MagicMock()
    plugin.start_training.return_value = handle
    monkeypatch.setattr(
        jm, "plugin_manager", SimpleNamespace(get_plugin=lambda _pid: plugin)
    )
    monkeypatch.setattr(jm, "LogTailer", MagicMock())

    with patch.object(mgr, "_require_available_definition"), \
         patch.object(mgr, "_persist_status"), \
         patch.object(
             mgr, "_start_pid_watchdog", side_effect=RuntimeError("watchdog boom")
         ):
        try:
            mgr.start_job(job.id, clear_stale_signal=False, preflight=False)
        except RuntimeError as exc:
            assert "watchdog boom" in str(exc), f"{MARKER}: unexpected error {exc!r}"
        else:
            raise AssertionError(f"{MARKER}: the injected launch failure did not raise")

    assert job.status == JobStatus.FAILED, (
        f"{MARKER}: the failed launch did not reset the job (status {job.status!r})"
    )
    assert job.id not in mgr._processes, (
        f"{MARKER}: a launch that failed after the spawn left its Popen handle "
        f"in _processes"
    )


def test_start_job_keeps_the_spawned_handle_for_the_watchdog(monkeypatch, tmp_path):
    """The REAL launch (start_job -> StandardPlugin.start_training -> Popen)
    must keep the Popen handle the watchdog polls; without it the watchdog
    finds no handle and a dead worker's exit code is lost again."""
    handle = MagicMock()
    handle.pid = 5151
    handle.poll.return_value = None
    handle.returncode = None
    captured: dict = {}

    def _popen(cmd, **kwargs):  # noqa: ANN001
        captured["kwargs"] = kwargs
        return handle

    monkeypatch.setattr(tp.subprocess, "Popen", _popen)
    monkeypatch.setattr(
        jm, "plugin_manager", SimpleNamespace(get_plugin=lambda _pid: StandardPlugin())
    )
    monkeypatch.setattr(jm, "LogTailer", MagicMock())

    mgr = JobManager()
    job = mgr.create_job(
        "ltx2-5-base",
        {
            "definition_id": "ltx2-5-base",
            "output_dir": str(tmp_path),
            "lora_name": "loud",
        },
    )
    watchdog = MagicMock()
    try:
        with patch.object(mgr, "_require_available_definition"), \
             patch.object(mgr, "_persist_status"), \
             patch.object(mgr, "_start_pid_watchdog", watchdog):
            mgr.start_job(job.id, clear_stale_signal=False, preflight=False)
    finally:
        stdout = captured.get("kwargs", {}).get("stdout")
        if hasattr(stdout, "close"):
            stdout.close()  # the boot log handle start_training opened

    assert "kwargs" in captured, f"{MARKER}: start_job never reached Popen"
    assert mgr._processes.get(job.id) is handle, (
        f"{MARKER}: start_job did not keep the spawned Popen handle in "
        f"_processes (got {mgr._processes.get(job.id)!r}); the watchdog would "
        "find no handle and report a dead worker without its exit code"
    )
    watchdog.assert_called_once_with(job.id, handle.pid)


# ── 3. offload sub-steps are logged before the work, the summary after ──


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


class _RecordingModule:
    def __init__(self, events: list, tag: str) -> None:
        self._events = events
        self._tag = tag

    def to(self, device):  # noqa: ANN001
        self._events.append(("to", self._tag, str(device)))
        return self


def _project(events: list, names: set[str]) -> list[tuple]:
    out: list[tuple] = []
    for ev in events:
        if ev[0] == "log":
            _, name, kw = ev
            if name in names:
                out.append((name, kw.get("step"), kw.get("name")))
        else:
            out.append(ev)
    return out


def test_text_encoder_offload_logs_after_the_move_with_substeps(monkeypatch):
    events: list = []
    te = _RecordingModule(events, "text_encoder")
    vae = _RecordingModule(events, "vae")

    driver = object.__new__(QwenImageDriver)
    driver.text_encoder = te
    trainer = object.__new__(QwenImageTrainer)
    trainer.driver = driver
    trainer.components = {"text_encoder": te, "vae": vae}
    trainer.config = {"cache_text_embeddings": True, "unload_text_encoder": False}
    trainer.text_cache = {}
    trainer._te_unloaded = False
    trainer.device = "cuda"  # fake: the stubs only record the target
    trainer.logger = _RecordingLogger(events)

    monkeypatch.setattr(pl.torch.cuda, "empty_cache", lambda: events.append(("call", "empty_cache")))
    monkeypatch.setattr(pl.gc, "collect", lambda *a: events.append(("call", "gc_collect")) or 0)

    trainer._offload_text_encoders()

    seq = _project(events, {"text_encoder_offload_step", "text_encoders_offloaded_to_cpu"})
    expected = [
        ("text_encoder_offload_step", "to_cpu", "text_encoder"),
        ("to", "text_encoder", "cpu"),
        ("text_encoder_offload_step", "empty_cache", None),
        ("call", "empty_cache"),
        ("text_encoder_offload_step", "gc_collect", None),
        ("call", "gc_collect"),
        ("text_encoders_offloaded_to_cpu", None, None),
    ]
    assert seq == expected, (
        f"{MARKER}: the offload must log each sub-step before it runs and "
        f"text_encoders_offloaded_to_cpu exactly once, last; got {seq!r}"
    )

    events.clear()
    trainer._move_component_to_gpu("vae")
    seq = _project(events, {"moving_component_to_gpu", "component_moved_to_gpu"})
    assert seq[:2] == [
        ("moving_component_to_gpu", None, "vae"),
        ("to", "vae", "cuda"),
    ], (
        f"{MARKER}: moving_component_to_gpu name=vae must precede the VAE "
        f".to(); got {seq!r}"
    )


# ── 4. the LTX2 connector offload (the override after super()) ──────────


def test_ltx2_connector_offload_is_instrumented(monkeypatch):
    from app.engine.models.families.ltx2.driver import Ltx2Driver
    from app.engine.models.families.ltx2.trainer import Ltx2Trainer

    events: list = []
    connectors = _RecordingModule(events, "connectors")

    driver = object.__new__(Ltx2Driver)
    driver.text_encoder = None  # base offload runs against zero encoders
    driver.connectors = connectors
    trainer = object.__new__(Ltx2Trainer)
    trainer.driver = driver
    trainer.components = {}
    trainer.config = {"cache_text_embeddings": True, "unload_text_encoder": False}
    trainer.text_cache = {}
    trainer._te_unloaded = False
    trainer.device = "cuda"  # fake: the stub only records the target
    trainer.logger = _RecordingLogger(events)

    # The override only empties the cache when CUDA is present; pretend it is.
    monkeypatch.setattr(pl.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pl.torch.cuda, "empty_cache", lambda: events.append(("call", "empty_cache")))
    monkeypatch.setattr(pl.gc, "collect", lambda *a: events.append(("call", "gc_collect")) or 0)

    trainer._offload_text_encoders()

    seq = _project(events, {"text_encoder_offload_step"})
    move = ("to", "connectors", "cpu")
    assert move in seq, (
        f"{MARKER}: the LTX2 override never moved the connectors to CPU; got {seq!r}"
    )
    move_at = seq.index(move)
    step_to_cpu = ("text_encoder_offload_step", "to_cpu", "connectors")
    assert step_to_cpu in seq[:move_at], (
        f"{MARKER}: no text_encoder_offload_step step=to_cpu name=connectors "
        f"before the connectors .to('cpu'); got {seq!r}"
    )
    after = seq[move_at + 1:]
    cache_call = ("call", "empty_cache")
    assert cache_call in after, (
        f"{MARKER}: the override did not empty the cache after the connectors "
        f"move; got {seq!r}"
    )
    cache_at = after.index(cache_call)
    assert any(
        ev[0] == "text_encoder_offload_step" and ev[1] == "empty_cache"
        for ev in after[:cache_at]
    ), (
        f"{MARKER}: no text_encoder_offload_step step=empty_cache before the "
        f"override's own empty_cache (after the connectors move); got {seq!r}"
    )


def test_ltx2_real_module_connectors_offload_names_the_component(monkeypatch):
    """A real nn.Module takes the sliced path: its events must name connectors."""
    import torch
    from app.engine.models.families.ltx2.driver import Ltx2Driver
    from app.engine.models.families.ltx2.trainer import Ltx2Trainer

    events: list = []
    driver = object.__new__(Ltx2Driver)
    driver.text_encoder = None
    driver.connectors = torch.nn.Linear(2, 2)
    trainer = object.__new__(Ltx2Trainer)
    trainer.driver = driver
    trainer.components = {}
    trainer.config = {"cache_text_embeddings": True, "unload_text_encoder": False}
    trainer.text_cache = {}
    trainer._te_unloaded = False
    trainer.device = "cuda"
    trainer.logger = _RecordingLogger(events)

    monkeypatch.setattr(pl.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pl.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(pl.gc, "collect", lambda *a: 0)

    trainer._offload_text_encoders()

    sliced = [
        kw for ev in events if ev[0] == "log" and ev[1] == "text_encoder_offload_step"
        for kw in [ev[2]] if kw.get("step") == "slice_to_cpu"
    ]
    assert sliced, f"{MARKER}: the real-module connectors move emitted no slice event"
    assert all(kw.get("name") == "connectors" for kw in sliced), (
        f"{MARKER}: sliced connector events do not carry name=connectors: {sliced!r}"
    )


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


# ── 6. the sliced offload names every slice and every empty_cache ───────


def _sliced_offload_trainer(events: list):
    import torch

    class _Leaf(torch.nn.Module):
        def __init__(self, tag: str) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1024), requires_grad=False)
            self._tag = tag
            self._moved = False

        def _apply(self, fn, recurse=True):  # noqa: ANN001
            if not self._moved:
                self._moved = True
                events.append(("copy", self._tag))
            return super()._apply(fn, recurse)

    te = torch.nn.Module()
    te.layers = torch.nn.ModuleList()
    for i in range(3):
        block = torch.nn.Module()
        block.attn = _Leaf(f"layers.{i}.attn")
        block.mlp = _Leaf(f"layers.{i}.mlp")
        te.layers.append(block)

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
    return trainer, 2 * 1024 * 4  # one chunk = one block (attn + mlp)


def test_sliced_offload_logs_before_every_slice_and_empty_cache(monkeypatch):
    events: list = []
    trainer, chunk = _sliced_offload_trainer(events)
    monkeypatch.setattr(pl, "_OFFLOAD_CHUNK_BYTES", chunk)
    monkeypatch.setattr(
        pl.torch.cuda, "empty_cache", lambda: events.append(("call", "empty_cache"))
    )
    monkeypatch.setattr(pl.gc, "collect", lambda *a: 0)

    trainer._offload_text_encoders()

    seq = [
        ev if ev[0] != "log" else ("log", ev[1], ev[2].get("step"), ev[2].get("name"))
        for ev in events
    ]
    calls = [i for i, ev in enumerate(seq) if ev == ("call", "empty_cache")]
    copies = [i for i, ev in enumerate(seq) if ev[0] == "copy"]
    assert len(copies) == 6, f"{MARKER}: expected 6 leaf copies; got {seq!r}"
    for i in calls:
        prev = seq[i - 1]
        assert prev[:3] == ("log", "text_encoder_offload_step", "empty_cache"), (
            f"{MARKER}: empty_cache call #{i} is not immediately preceded by a "
            f"text_encoder_offload_step step=empty_cache; got {seq!r}"
        )
    # Every empty_cache inside the sliced offload names the component.
    inner = [seq[i - 1] for i in calls[:-1]]  # the last one is the base method's
    assert inner and all(ev[3] == "text_encoder" for ev in inner), (
        f"{MARKER}: the sliced offload's empty_cache events do not name the "
        f"component; got {inner!r}"
    )
    for i in copies:
        before = [ev for ev in seq[:i] if ev[0] != "copy"]
        assert before[-1][:4] == (
            "log", "text_encoder_offload_step", "slice_to_cpu", "text_encoder"
        ), (
            f"{MARKER}: slice copy {seq[i]!r} is not preceded by a "
            f"text_encoder_offload_step step=slice_to_cpu name=text_encoder; "
            f"got {seq!r}"
        )


def test_injected_empty_cache_failure_leaves_a_line_naming_it(monkeypatch):
    events: list = []
    trainer, chunk = _sliced_offload_trainer(events)
    monkeypatch.setattr(pl, "_OFFLOAD_CHUNK_BYTES", chunk)
    n = {"calls": 0}

    def _failing_empty_cache() -> None:
        n["calls"] += 1
        if n["calls"] == 3:  # mid-offload, after a slice copy
            raise RuntimeError("injected empty_cache failure")
        events.append(("call", "empty_cache"))

    monkeypatch.setattr(pl.torch.cuda, "empty_cache", _failing_empty_cache)
    monkeypatch.setattr(pl.gc, "collect", lambda *a: 0)

    try:
        trainer._offload_text_encoders()
    except RuntimeError as exc:
        assert "injected" in str(exc), f"{MARKER}: unexpected error {exc!r}"
    else:
        raise AssertionError(f"{MARKER}: the injected empty_cache failure did not raise")

    logs = [ev for ev in events if ev[0] == "log"]
    last = logs[-1]
    assert (last[1], last[2].get("step"), last[2].get("name")) == (
        "text_encoder_offload_step", "empty_cache", "text_encoder"
    ), (
        f"{MARKER}: the last line before the failing empty_cache does not name "
        f"it (step=empty_cache name=text_encoder); got {last!r}"
    )
