"""The GPU health record on a real job's way out, and everyone who reads its log.

The exit paths run the real ``run_trainer.main`` in a process of its own, with a fake
trainer and a fake registry, so that ``sys.exit``, ``atexit`` and a daemon thread left
behind all do what they do in a job. Nothing here touches a card or starts a training.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.core.job import JobStatus
from app.core.job_manager import (
    JobManager,
    _parse_persisted_log_lines,
    _trainer_msg_for_server_log,
)
from app.core.log_tailer import LogTailer
from app.engine.components.job_log_writer import JobLogWriter

BACKEND = Path(__file__).resolve().parents[1]
MUST_END_S = 150.0  # only spent when a test is about to fail

# Runs ``run_trainer.main`` as a job would. CUDA_VISIBLE_DEVICES stays at -1 while the
# imports run and torch's own check is replaced, so no card is touched by any of it.
JOB_SCRIPT = r"""
import atexit, json, os, sys, threading, time, types

spec = json.loads(sys.argv[1])
root = sys.argv[2]
probe = open(os.path.join(root, "probe.txt"), "a", encoding="utf-8", buffering=1)
ready = threading.Event()
training = threading.Event()


def note(*parts):
    probe.write(" ".join(str(p) for p in parts) + "\n")


if spec.get("nvml") == "hangs":
    fake = types.ModuleType("pynvml")

    def init():
        note("init_on", threading.current_thread().name)
        ready.set()
        threading.Event().wait()

    fake.nvmlInit = init
    sys.modules["pynvml"] = fake

import run_trainer
import torch

torch.cuda.is_available = lambda: False
if spec.get("card", True):
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_DEVICE_ORDER", None)
else:
    ready.set()
if spec.get("ready_at_start"):
    ready.set()

# What torch says of the card it trains on, once CUDA is up: never, unless the spec says.
torch_bus = spec.get("torch_bus")
torch.cuda.is_initialized = lambda: torch_bus is not None


def properties(index):
    note("properties_of", index)
    return types.SimpleNamespace(pci_domain_id=0, pci_bus_id=torch_bus, pci_device_id=0)


torch.cuda.get_device_properties = properties
torch.cuda.current_device = lambda: 0


def job_log_text():
    for folder, _dirs, files in os.walk(root):
        if "job_log.jsonl" in files:
            with open(os.path.join(folder, "job_log.jsonl"), encoding="utf-8") as f:
                return f.read()
    return ""


real_finalize = run_trainer._finalize_before_exit


def finalize(log_writer=None):
    text = job_log_text()
    note(
        "finalize", "summaries", text.count("gpu_health_summary"),
        "exits", text.count('"type":"exit"'), "writer", log_writer is not None,
    )
    if spec.pop("finalize", None) == "raises":
        raise RuntimeError("the finaliser broke")
    return real_finalize(log_writer)


run_trainer._finalize_before_exit = finalize

if "readings" in spec:
    from app.core import gpu_health
    from app.engine.components import gpu_health_recorder as ghr

    gpu_health._NVML_AVAILABLE = True
    readings = list(spec["readings"])
    real_recorder = ghr.GpuHealthRecorder

    # The cards NVML lists, by their bus numbers in NVML's order.
    buses = spec.get("cards", [0x01])
    nvml = types.ModuleType("pynvml")
    nvml.nvmlDeviceGetCount = lambda: len(buses)
    nvml.nvmlDeviceGetHandleByIndex = lambda index: index
    nvml.nvmlDeviceGetUUID = lambda handle: f"GPU-{handle}"
    nvml.nvmlDeviceGetPciInfo = lambda handle: types.SimpleNamespace(
        domain=0, bus=buses[handle], device=0,
    )
    sys.modules["pynvml"] = nvml

    def reader(_index):
        note("read", _index)
        # A stalling disk is to stall on the way out, not while the job still starts up.
        if "stall_s" in spec:
            assert training.wait(60)
        if readings:
            return readings.pop(0)
        ready.set()
        if spec.get("then") == "hang":
            threading.Event().wait()
        return None

    ghr.GpuHealthRecorder = lambda writer: real_recorder(writer, reader=reader, interval_s=0.0)
    if "write_wait_s" in spec:
        ghr.WRITE_WAIT_S = spec["write_wait_s"]

if "lock_wait_s" in spec:
    from app.engine.components import job_log_writer as jlw

    jlw.LOCK_WAIT_S = spec["lock_wait_s"]

if "note_wait_s" in spec:
    from app.engine.components import job_log_writer as jlw

    jlw.NOTE_WAIT_S = spec["note_wait_s"]

# stdout and stderr, which a job sends to trainer_stdout.log, stall once this is set.
streams_stall = threading.Event()
if spec.get("streams_stall"):
    import logging

    class StallingStream:
        def __init__(self, inner):
            self.inner = inner

        def write(self, text):
            if streams_stall.is_set():
                note("stream_stalled")
                threading.Event().wait(3600)
            return self.inner.write(text)

        def flush(self):
            self.inner.flush()

    real_streams = (sys.stdout, sys.stderr)
    sys.stdout, sys.stderr = StallingStream(sys.stdout), StallingStream(sys.stderr)
    for handler in logging.getLogger().handlers:
        if getattr(handler, "stream", None) is real_streams[0]:
            handler.stream = sys.stdout
        elif getattr(handler, "stream", None) is real_streams[1]:
            handler.stream = sys.stderr

if "stall_s" in spec:
    from app.engine.components import job_log_writer as jlw

    class StallingFile:
        def __init__(self, inner):
            self.inner = inner

        @property
        def closed(self):
            return self.inner.closed

        def write(self, text):
            if spec.get("stall_on", '"gpu_health"') in text:
                ready.set()
                threading.Event().wait(spec["stall_s"])
            self.inner.write(text)

        def flush(self):
            self.inner.flush()

        def close(self):
            self.inner.close()

    class StallingWriter(jlw.JobLogWriter):
        def __init__(self, output_dir):
            super().__init__(output_dir)
            self._file = StallingFile(self._file)

    jlw.JobLogWriter = StallingWriter

if spec.get("writer") == "fails":
    from app.engine.components import job_log_writer as jlw

    def no_writer(_output_dir):
        raise OSError("the output folder cannot be written")

    jlw.JobLogWriter = no_writer

if spec.get("recorder") == "fails":
    from app.engine.components import gpu_health_recorder as ghr

    def no_recorder(_writer):
        raise RuntimeError("no recorder today")

    ghr.GpuHealthRecorder = no_recorder


class Trainer:
    def __init__(self, definition, run_config):
        self.config = run_config

    async def setup(self):
        pass

    async def load_model(self):
        if spec.get("train") == "load_raises":
            raise RuntimeError("the load failed")

    async def prepare_data(self):
        pass

    def _get_text_encoders(self):
        return {}

    def get_te_cache(self):
        return None

    def _pre_cache_text_embeddings(self):
        pass

    def _offload_text_encoders(self):
        pass

    def _move_component_to_gpu(self, _name):
        pass

    def _validate_latent_cache(self):
        pass

    async def _pre_cache_latents(self):
        pass

    def _pre_cache_aux(self):
        pass

    def _offload_vae(self):
        pass

    async def prepare_for_training(self):
        pass

    async def train(self):
        training.set()
        assert ready.wait(60), "the recorder never got that far"
        streams_stall.set()
        how = spec.get("train", "returns")
        if how == "raises":
            raise RuntimeError("the trainer broke")
        if how == "interrupt":
            raise KeyboardInterrupt
        if how == "sys_exit":
            sys.exit(3)
        if how == "rebuild":
            self._rebuild_exit = True
        if how == "logs":
            self._log_writer.step({"step": 1})
            self._log_writer.log("a line after the stall")
            note("trained_on")
        if how == "soft_stop":
            from app.engine.components.signal_manager import TrainingSignalManager

            manager = TrainingSignalManager(root)
            TrainingSignalManager.send_signal(root, "soft_stop")
            assert manager.handle_signals() == "soft_stop"
            self._soft_stop_exit = True


class Family:
    def __init__(self, definition, config):
        pass

    def get_trainer_class(self):
        return Trainer


def family_class(_name):
    if spec.get("family") == "unknown":
        raise ValueError("no such family")
    return Family


run_trainer.registry.initialize = lambda: None
run_trainer.registry.get_definition = (
    lambda _id: None if spec.get("definition") == "missing" else types.SimpleNamespace(family="fake")
)
run_trainer.registry.get_family_class = family_class

config = "not json" if spec.get("config") == "bad" else json.dumps(
    {"output_dir": root, "lora_name": "job"}
)
sys.argv = ["run_trainer.py", "--definition_id", "fake/job", "--config", config]
atexit.register(lambda: note("system_monitor_imported", "app.core.system_monitor" in sys.modules))
run_trainer.main()
"""


def good(**fields: Any) -> dict[str, Any]:
    return {
        "temperature_c": 60,
        "temperature_margin_c": 20,
        "fan_percent": 40,
        "power_draw_w": 300.0,
        "power_limit_w": 600.0,
        "thermal_throttle_ms": 0.0,
        "power_throttle_ms": 0.0,
        "throttle_reasons": [],
        "readable": True,
        **fields,
    }


def bad(error: str = "GPU is lost") -> dict[str, Any]:
    return {"readable": False, "error": error}


class Job:
    def __init__(self, done: subprocess.CompletedProcess, root: Path) -> None:
        self.code = done.returncode
        self.stderr = done.stderr
        logs = list(root.rglob("job_log.jsonl"))
        assert len(logs) <= 1
        self.has_log = bool(logs)
        text = logs[0].read_text(encoding="utf-8") if logs else ""
        self.entries = [json.loads(line) for line in text.splitlines()]
        probe = root / "probe.txt"
        self.notes = probe.read_text(encoding="utf-8").splitlines() if probe.exists() else []

    def types(self) -> list[str]:
        return [entry["type"] for entry in self.entries]

    def summaries(self) -> list[str]:
        return [
            entry["data"]
            for entry in self.entries
            if entry["type"] == "log" and str(entry["data"]).startswith("gpu_health_summary")
        ]

    def summary(self) -> str:
        found = self.summaries()
        assert len(found) == 1, (found, self.stderr)
        return found[0]

    def index_of_summary(self) -> int:
        self.summary()
        return next(
            i for i, entry in enumerate(self.entries)
            if str(entry["data"]).startswith("gpu_health_summary")
        )

    def finalizing(self) -> list[str]:
        return [note for note in self.notes if note.startswith("finalize")]


@pytest.fixture
def run_job(tmp_path: Path):
    def _run(**spec: Any) -> Job:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": str(BACKEND)}
        done = subprocess.run(
            [sys.executable, "-c", JOB_SCRIPT, json.dumps(spec), str(tmp_path)],
            cwd=BACKEND,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=spec.get("must_end_s", MUST_END_S),
            check=False,
        )
        return Job(done, tmp_path)

    return _run


def assert_written_once_before_the_finaliser_and_the_exit_line(job: Job, code: int) -> None:
    """The order of Acceptance 6, on the paths that finalise."""
    assert job.code == code, job.stderr
    types = job.types()
    assert types[-1] == "exit", job.stderr
    assert types.count("exit") == 1
    assert job.entries[-1]["data"]["code"] == code
    assert job.index_of_summary() < len(types) - 1
    assert job.finalizing() == ["finalize summaries 1 exits 0 writer True"]


# ── Every way out ───────────────────────────────────────────────────────


@pytest.mark.parametrize("how", ["returns", "soft_stop", "rebuild"])
def test_a_job_that_ends_normally(run_job, how: str) -> None:
    job = run_job(train=how, readings=[good()])
    assert_written_once_before_the_finaliser_and_the_exit_line(job, 0)
    assert "error" not in job.entries[-1]["data"]
    # Only a soft stop's exit tells the backend the job stopped rather than finished.
    assert job.entries[-1]["data"].get("stopped", False) is (how == "soft_stop")


def test_the_rebuild_exit_still_says_so_before_the_summary(run_job) -> None:
    job = run_job(train="rebuild", readings=[good()])
    rebuild = next(
        i for i, entry in enumerate(job.entries)
        if "Adaptive rebuild restart" in str(entry["data"])
    )
    assert rebuild < job.index_of_summary()


@pytest.mark.parametrize(
    ("how", "error"), [("raises", "the trainer broke"), ("load_raises", "the load failed")],
)
def test_a_job_whose_trainer_raises(run_job, how: str, error: str) -> None:
    job = run_job(train=how, readings=[good()])
    assert_written_once_before_the_finaliser_and_the_exit_line(job, 1)
    assert error in job.entries[-1]["data"]["error"]
    # The traceback still reaches the log, and stays what a reader sees last.
    messages = [str(entry["data"]) for entry in job.entries]
    header = next(i for i, message in enumerate(messages) if "Trainer crashed" in message)
    assert job.index_of_summary() < header


def test_an_interrupt_writes_the_summary_and_otherwise_ends_as_before(run_job) -> None:
    job = run_job(train="interrupt", readings=[good()])
    assert job.summary().startswith("gpu_health_summary")
    assert job.code != 0
    # As before this change: no finaliser and no exit line, the watchdog takes over.
    assert "exit" not in job.types()
    assert job.finalizing() == []


def test_a_sys_exit_inside_the_trainer_writes_the_summary_and_keeps_its_code(run_job) -> None:
    job = run_job(train="sys_exit", readings=[good()])
    assert job.summary().startswith("gpu_health_summary")
    assert job.code == 3
    assert "exit" not in job.types()


@pytest.mark.parametrize(
    ("spec", "error"),
    [
        ({"definition": "missing"}, "not found"),
        ({"family": "unknown"}, "no such family"),
    ],
)
def test_the_early_refusals_write_the_summary_before_their_exit_line(
    run_job, spec: dict[str, str], error: str,
) -> None:
    job = run_job(readings=[good()], **spec)
    assert job.code == 1
    assert job.types()[-1] == "exit"
    assert job.types().count("exit") == 1
    assert error in job.entries[-1]["data"]["error"]
    assert job.index_of_summary() < len(job.entries) - 1


def test_a_failure_after_the_summary_does_not_write_it_again(run_job) -> None:
    job = run_job(finalize="raises", readings=[good()])
    assert job.code == 1, job.stderr
    assert job.summary().startswith("gpu_health_summary")
    assert job.types()[-1] == "exit"
    assert "the finaliser broke" in job.entries[-1]["data"]["error"]


def test_a_job_that_fails_before_the_writer_exists_exits_as_it_did(run_job) -> None:
    job = run_job(config="bad")
    assert job.code == 1
    assert job.has_log is False
    assert "Unhandled exception" in job.stderr


def test_a_writer_that_cannot_open_ends_the_job_as_it_did(run_job) -> None:
    job = run_job(writer="fails")
    assert job.code == 1
    assert job.has_log is False
    assert "cannot be written" in job.stderr
    assert job.finalizing() == ["finalize summaries 0 exits 0 writer False"]


def test_a_recorder_that_cannot_be_made_does_not_reach_the_job(run_job) -> None:
    job = run_job(recorder="fails", card=False)
    assert job.code == 0, job.stderr
    assert job.types()[-1] == "exit"
    assert job.summaries() == []
    assert "no recorder today" in job.stderr


def test_a_job_shown_no_card_says_there_is_no_reading(run_job) -> None:
    job = run_job(card=False)
    assert job.code == 0, job.stderr
    assert "gpu_health" not in job.types()
    assert "no reading" in job.summary()


# ── What the summary says, from a real exit path ────────────────────────


def test_the_summary_of_known_values(run_job) -> None:
    job = run_job(
        readings=[
            good(
                temperature_c=60, temperature_margin_c=20, fan_percent=40,
                thermal_throttle_ms=2.0, power_throttle_ms=100.0,
            ),
            good(
                temperature_c=71, temperature_margin_c=9, fan_percent=55,
                thermal_throttle_ms=5.0, power_throttle_ms=2600.0,
            ),
        ],
    )
    line = job.summary()
    assert "peak temperature 71 C" in line
    assert "lowest margin 9 C" in line
    assert "peak fan 55%" in line
    assert "thermal throttling grew by 3 ms" in line
    assert "power throttling grew by 2.5 s" in line
    assert job.types().count("gpu_health") == 2


def test_the_summary_of_a_card_lost_during_the_job(run_job) -> None:
    job = run_job(train="raises", readings=[good(temperature_c=70), bad(), bad()])
    assert_written_once_before_the_finaliser_and_the_exit_line(job, 1)
    line = job.summary()
    assert "70 C" in line
    assert "unreadable" in line
    assert "GPU is lost" in line
    assert job.types().count("gpu_health") == 2  # one good, one for the change


def test_a_hung_read_cannot_hold_the_summary_or_the_exit(run_job) -> None:
    job = run_job(train="raises", readings=[good(temperature_c=70)], then="hang")
    assert_written_once_before_the_finaliser_and_the_exit_line(job, 1)
    line = job.summary()
    assert "70 C" in line
    assert "had not returned" in line


def test_a_hanging_nvml_init_holds_up_neither_the_job_nor_its_exit(run_job) -> None:
    job = run_job(nvml="hangs")
    assert job.code == 0, job.stderr
    assert job.types()[-1] == "exit"
    assert "init_on gpu-health-recorder" in job.notes
    line = job.summary()
    assert "no reading" in line
    assert "had not returned" in line
    assert "system_monitor_imported False" in job.notes


def test_a_write_stuck_for_good_cannot_hold_the_process(run_job) -> None:
    job = run_job(readings=[good()], stall_s=3600, write_wait_s=0.1, lock_wait_s=0.1)
    assert job.code == 0, job.stderr
    assert "job log is stuck" in job.stderr
    # The process ended with no line after the stuck one, and the finaliser wrote nothing.
    assert "exit" not in job.types()
    assert job.finalizing() == ["finalize summaries 0 exits 0 writer False"]


def test_a_write_stuck_for_good_also_bounds_the_failure_path(run_job) -> None:
    job = run_job(
        train="raises", readings=[good()], stall_s=3600, write_wait_s=0.1, lock_wait_s=0.1,
    )
    assert job.code == 1, job.stderr
    assert "the trainer broke" in job.stderr


@pytest.mark.parametrize("how", ["logs", "rebuild"])
def test_a_write_stuck_for_good_cannot_hold_the_trainers_next_line(run_job, how: str) -> None:
    # The trainer writes again while the recorder's write is stuck: it waits once, for a
    # bound, and the job then ends as it would on a stuck log, rather than not at all.
    job = run_job(
        train=how, readings=[good()], stall_s=3600, write_wait_s=0.1, lock_wait_s=0.1,
        must_end_s=60,
    )
    assert job.code == 0, job.stderr
    assert "job log is stuck" in job.stderr
    assert "exit" not in job.types()
    assert job.finalizing() == ["finalize summaries 0 exits 0 writer False"]
    if how == "logs":
        assert "trained_on" in job.notes


# ── Which card it is ────────────────────────────────────────────────────


def test_the_trainer_names_its_card_once_torch_has_chosen_it(run_job) -> None:
    # Two cards, in CUDA's own order: only torch knows which one trains.
    job = run_job(readings=[good()], cards=[0x01, 0x41], torch_bus=0x41)
    assert job.code == 0, job.stderr
    assert [note for note in job.notes if note.startswith("read")] == ["read 1", "read 1"]
    readings = [entry["data"] for entry in job.entries if entry["type"] == "gpu_health"]
    assert [data["pci_bus_id"] for data in readings] == ["00000000:41:00.0"]
    assert "peak temperature 60 C" in job.summary()


def test_naming_the_card_starts_nothing_in_cuda(run_job) -> None:
    # CUDA not up: nothing is asked of torch, and of two cards none is guessed.
    job = run_job(readings=[good()], cards=[0x01, 0x41], ready_at_start=True)
    assert job.code == 0, job.stderr
    assert not [note for note in job.notes if note.startswith(("properties_of", "read"))]
    assert "gpu_health" not in job.types()
    assert "no reading" in job.summary()


@pytest.mark.parametrize(("how", "code"), [("returns", 0), ("raises", 1)])
def test_a_summary_write_stuck_on_the_disk_cannot_hold_the_way_out(
    run_job, how: str, code: int,
) -> None:
    # The read hangs, so stop returns False; then the summary's own write never returns,
    # with the writer's lock free. The job still ends, with what it could not write lost.
    job = run_job(
        train=how, readings=[good()], then="hang", stall_s=3600,
        stall_on="gpu_health_summary", lock_wait_s=0.5, must_end_s=60,
    )
    assert job.code == code, job.stderr
    assert "job log is stuck" in job.stderr
    assert job.summaries() == []
    assert "exit" not in job.types()
    assert job.finalizing() == ["finalize summaries 0 exits 0 writer False"]


@pytest.mark.parametrize(("how", "code"), [("returns", 0), ("raises", 1)])
def test_a_stalled_stdout_and_stderr_cannot_hold_the_way_out(run_job, how: str, code: int) -> None:
    # As above, and the stdout and stderr every warning and log line of the way out goes
    # to stall as well, from the moment training ends.
    job = run_job(
        train=how, readings=[good()], then="hang", stall_s=3600,
        stall_on="gpu_health_summary", lock_wait_s=0.5, note_wait_s=0.5,
        streams_stall=True, must_end_s=60,
    )
    assert job.code == code, job.stderr
    assert "stream_stalled" in job.notes
    assert job.summaries() == []
    assert "exit" not in job.types()
    assert job.finalizing() == ["finalize summaries 0 exits 0 writer False"]


def test_a_rebuild_message_stuck_on_the_disk_cannot_hold_the_relaunch(run_job) -> None:
    # train() asks for a rebuild restart, and the message saying so stalls on the disk
    # itself, with the lock free and no recorder write in the way. The job must still end,
    # so that the backend can relaunch it from its checkpoint.
    job = run_job(
        train="rebuild", readings=[good()], stall_s=3600,
        stall_on="Adaptive rebuild restart", lock_wait_s=0.5, note_wait_s=0.5, must_end_s=60,
    )
    assert job.code == 0, job.stderr
    assert "job log is stuck" in job.stderr
    assert "exit" not in job.types()


def test_a_disk_that_comes_back_gets_the_summary_and_the_exit_line(run_job) -> None:
    job = run_job(readings=[good()], stall_s=3.0, write_wait_s=0.1, lock_wait_s=30.0)
    assert job.code == 0, job.stderr
    assert job.types()[-1] == "exit"
    assert job.types().count("gpu_health") == 1
    assert job.index_of_summary() < len(job.entries) - 1


def test_a_force_stop_keeps_whole_lines_and_can_write_no_summary(tmp_path: Path) -> None:
    code = """
import sys, threading, types
nvml = types.ModuleType("pynvml")  # one card, so that only the writer is on trial
nvml.nvmlDeviceGetCount = lambda: 1
nvml.nvmlDeviceGetHandleByIndex = lambda index: index
nvml.nvmlDeviceGetPciInfo = lambda _h: types.SimpleNamespace(domain=0, bus=1, device=0)
from app.core import gpu_health
sys.modules["pynvml"] = nvml
from app.engine.components.gpu_health_recorder import GpuHealthRecorder
from app.engine.components.job_log_writer import JobLogWriter
gpu_health._NVML_AVAILABLE = True
writer = JobLogWriter(sys.argv[1])
reading = {"readable": True, "pad": "x" * 500}
GpuHealthRecorder(writer, reader=lambda _i: reading, interval_s=0.001).start()
threading.Event().wait()
"""
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    env["PYTHONPATH"] = str(BACKEND)
    process = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], cwd=BACKEND, env=env)
    log = tmp_path / "job_log.jsonl"
    try:
        deadline = time.monotonic() + MUST_END_S
        while time.monotonic() < deadline:
            if log.exists() and log.stat().st_size > 20_000:
                break
            time.sleep(0.05)
    finally:
        process.kill()  # what the job manager's force stop comes to
        process.wait(MUST_END_S)

    lines = log.read_text(encoding="utf-8").splitlines()
    # Line-buffered and flushed: a kill can only cut the last line.
    entries = [json.loads(line) for line in lines[:-1]]
    assert len(entries) > 10
    assert {entry["type"] for entry in entries} == {"gpu_health"}


# ── Everyone who reads the log ──────────────────────────────────────────


@pytest.fixture
def manager_job():
    manager = JobManager()
    broadcasts: list[tuple[str, dict]] = []
    with patch("app.core.job_manager.event_manager") as events, \
            patch("app.core.job_manager.asyncio.run_coroutine_threadsafe"), \
            patch.object(manager, "_persist_status"), \
            patch.object(manager, "_persist_config"), \
            patch.object(manager, "schedule_advance_queue"), \
            patch.object(manager, "_handle_exit_message") as exits:
        events.broadcast.side_effect = (
            lambda topic, payload: broadcasts.append((topic, payload))
        )
        manager.set_loop(MagicMock(spec=asyncio.AbstractEventLoop))
        job = manager.create_job(
            "flux/dev",
            {"output_dir": "outputs", "lora_name": "x", "definition_id": "flux/dev"},
        )
        job.status = JobStatus.RUNNING
        yield manager, job, broadcasts, exits


def _mixed_log(tmp_path: Path) -> Path:
    writer = JobLogWriter(str(tmp_path))
    writer.status("Training")
    writer.log("first log")
    writer.emit("gpu_health", good())
    writer.step({"step": 1, "loss": 0.5, "status": "training"})
    writer.emit("gpu_health", bad())
    writer.log("second log")
    writer.warning("careful")
    writer.emit("gpu_health", good(temperature_c=99))
    writer.step({"step": 2, "loss": 0.4, "status": "training"})
    writer.log("gpu_health_summary: peak temperature 99 C")
    writer.exit(0)
    return tmp_path / "job_log.jsonl"


def _replay(manager: JobManager, job: Any, log: Path) -> tuple[list[str], list[Exception]]:
    """Through the real tailer into the real handler, until the exit line."""
    seen: list[str] = []
    errors: list[Exception] = []
    done = threading.Event()

    def dispatch(job_id: str, entry: dict[str, Any]) -> None:
        seen.append(entry["type"])
        try:
            manager._dispatch_log_entry(job_id, entry)
        except Exception as exc:  # the tailer would swallow it and die
            errors.append(exc)
            raise
        finally:
            if entry["type"] == "exit":
                done.set()

    tailer = LogTailer(job.id, str(log), dispatch, poll_interval=0.01)
    tailer.start()
    assert done.wait(MUST_END_S), (seen, errors)
    tailer.stop()
    return seen, errors


def test_a_replay_passes_the_readings_by_and_keeps_the_order(tmp_path: Path, manager_job) -> None:
    manager, job, broadcasts, exits = manager_job
    seen, errors = _replay(manager, job, _mixed_log(tmp_path))

    assert errors == []
    assert seen.count("gpu_health") == 3  # they do pass through the tailer
    exits.assert_called_once()

    # The job's visible log has the log lines, in order, and no reading.
    assert job.logs == [
        "first log", "second log", "gpu_health_summary: peak temperature 99 C",
    ]
    assert job.warnings == ["careful"]
    assert job.status_label == "Training"

    # What the UI is sent: everything but the readings, in the order written.
    sent = [
        payload.get("message") if topic != "job_update" else "job_update"
        for topic, payload in broadcasts
    ]
    assert sent == [
        "job_update",
        "first log",
        json.dumps({"step": 1, "loss": 0.5, "status": "training"}),
        "second log",
        "careful",
        json.dumps({"step": 2, "loss": 0.4, "status": "training"}),
        "gpu_health_summary: peak temperature 99 C",
    ]


def test_a_reading_is_not_mirrored_into_the_server_log() -> None:
    assert _trainer_msg_for_server_log("gpu_health", good()) is None
    assert _trainer_msg_for_server_log("gpu_health", bad()) is None


def test_the_persisted_tail_shows_no_reading_and_the_summary(tmp_path: Path) -> None:
    lines = _parse_persisted_log_lines(str(_mixed_log(tmp_path)))
    assert lines == [
        "first log",
        "second log",
        "[WARNING] careful",
        "gpu_health_summary: peak temperature 99 C",
    ]


def test_twelve_hours_of_readings_push_no_log_line_out_of_the_tail(tmp_path: Path) -> None:
    writer = JobLogWriter(str(tmp_path))
    for number in range(1000):
        writer.log(f"line {number}")
        for _ in range(5):  # 5000 readings in all, more than a 12-hour job's 4320
            writer.emit("gpu_health", good())
    writer.close()

    lines = _parse_persisted_log_lines(str(tmp_path / "job_log.jsonl"), limit=1000)
    assert lines == [f"line {number}" for number in range(1000)]


def test_readings_take_no_place_in_the_live_buffer(manager_job) -> None:
    manager, job, _broadcasts, _exits = manager_job
    job.logs.extend(f"line {number}" for number in range(1000))
    for _ in range(5000):
        manager._dispatch_log_entry(job.id, {"type": "gpu_health", "t": 1.0, "data": good()})
    assert job.logs == [f"line {number}" for number in range(1000)]


def test_a_time_written_out_of_order_changes_nothing(tmp_path: Path, manager_job) -> None:
    """The writer takes ``t`` before its lock, so two threads' lines can swap."""
    manager, job, broadcasts, _exits = manager_job
    log = tmp_path / "job_log.jsonl"
    entries = [
        {"t": 10.0, "type": "log", "data": "before"},
        {"t": 9.999, "type": "gpu_health", "data": good()},
        {"t": 10.001, "type": "step", "data": {"step": 1}},
        {"t": 9.5, "type": "log", "data": "after"},
        {"t": 10.2, "type": "exit", "data": {"code": 0}},
    ]
    log.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")

    _seen, errors = _replay(manager, job, log)

    assert errors == []
    assert job.logs == ["before", "after"]  # the file's order, not the clock's
    assert [payload["timestamp"] for _topic, payload in broadcasts] == [10.0, 10.001, 9.5]
