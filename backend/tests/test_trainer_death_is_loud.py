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
