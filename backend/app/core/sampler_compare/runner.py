"""One run of the comparison, and the queue of them.

Each run gets a backend and a trainer of its own as child processes, on the comparison's port and
its own copies of the user's stores (``isolation``), watched every ``Seams.poll_s`` and stopped on
any limit (``telemetry.breach``), a wrong sampler or a wrong split. Both children and everything
they started are killed in ``finally``, whichever way the run ends.

A finished run is reused only when its recorded identity equals the one asked for now, its verdict
accepted it, and its own files pass the sampler, seed and split checks again; anything else stops
the queue by name. Nothing is deleted: a partial run is moved aside.
"""
from __future__ import annotations

import datetime
import hashlib
import contextlib
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import port_resolver
import psutil
import restart_launcher

from app.core.container_config import auth_token
from app.core.db.engine import DatabaseEngine
from app.core.naming import model_part_from_definition_id
from app.core.sampler_compare import facts as facts_module
from app.core.sampler_compare import isolation, plan, telemetry
from app.core.sampler_compare.plan import BACKEND, Comparison, ConfigError, Limits
from app.core.sampler_compare.telemetry import Card, Peaks, Sample, TelemetryFile
from app.engine.core import validation_compare as vc

GIB = 2**30
MIB = 2**20
LOOPBACK = "127.0.0.1"
READY_WAIT_S = 240
STATUS_EVERY_S = 30
IDLE_POLL_S = 30
KILL_WAIT_S = 30
PORT_FREE_WAIT_S = 15
PORT_FREE_POLL_S = 0.2
POWER_TOLERANCE_W = 1.0
STATE_VERSION = 1
UNREADABLE_SPLIT = "its split file can't be read"
_LOG_GROWTH = "job_log.jsonl"


def port_taken(port: int) -> bool:
    """Whether something listens on ``port``. A bind probe (``restart_launcher.port_is_free``)
    answers at once; a port it can't bind is then asked by connecting, since a socket a killed
    server left waiting blocks a bind on some systems but takes no connection."""
    if restart_launcher.port_is_free(LOOPBACK, port):
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        return probe.connect_ex((LOOPBACK, port)) == 0


@dataclass
class Seams:
    """Everything the outside world supplies to a comparison."""

    read_card: Callable[[], Card | None]
    available_ram_bytes: Callable[[], int]
    clock: Callable[[], float]
    sleep: Callable[[float], None]
    backend_argv: Callable[[Comparison], list[str]]
    trainer_argv: Callable[[Comparison, str, dict], list[str]]
    port_taken: Callable[[int], bool]
    facts: Callable[[str, dict, dict], dict]
    user_db: Path
    user_settings: Path
    poll_s: float = 3.0
    ready_wait_s: float = READY_WAIT_S
    card_problem: str | None = None  # why the card read can't be told to be the trainer's

    @classmethod
    def real(cls, c: Comparison) -> Seams:
        python = sys.executable
        card, card_problem = telemetry.card_choice(telemetry.nvml_device_count(), os.environ)

        def backend_argv(c: Comparison) -> list[str]:
            return [python, "-m", "uvicorn", "app.main:app", "--app-dir", str(BACKEND),
                    "--host", LOOPBACK, "--port", str(c.port)]

        def trainer_argv(c: Comparison, definition_id: str, config: dict) -> list[str]:
            return [python, str(BACKEND / "run_trainer.py"), "--definition_id", definition_id,
                    "--config", json.dumps(config)]

        return cls(
            read_card=lambda: telemetry.read_card(card),
            available_ram_bytes=lambda: psutil.virtual_memory().available,
            clock=time.monotonic,
            sleep=time.sleep,
            backend_argv=backend_argv,
            trainer_argv=trainer_argv,
            port_taken=port_taken,
            facts=lambda definition_id, arms, env: facts_module.facts_in_child(definition_id, arms, env, python),
            user_db=Path(DatabaseEngine().db_path),  # where the backend's own engine looks; opens nothing
            user_settings=Path(port_resolver.settings_path()),
            card_problem=card_problem,
        )


# ── The reference job and what it asks for ──────────────────────────────────


def read_reference(db: Path, job_id: str) -> tuple[dict, str]:
    """(config, definition_id) of the reference job, from the user's database opened read-only."""
    conn = isolation.read_only(db)
    try:
        row = conn.execute("select config, definition_id from job_history where id=?", (job_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ConfigError(f"the database {db} has no job {job_id!r}")
    return json.loads(row[0]), row[1]


def source_dataset(db: Path, name: str) -> Path:
    conn = isolation.read_only(db)
    try:
        row = conn.execute("select path from datasets where name=?", (name,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ConfigError(f"the database {db} has no dataset {name!r}")
    return Path(row[0])


@dataclass
class Prepared:
    definition_id: str
    dataset: str
    source: Path
    facts: dict
    manifest: dict[str, str]
    requested: dict[str, dict]
    settings: bytes | None = None  # the user's settings as read once; the runs' copy is written from these


def requested_identities(c: Comparison, base: dict, definition_id: str, facts: dict,
                         dataset_digest: str, model_source: dict, metadata_digest: str) -> dict[str, dict]:
    """label -> the identity each run of the queue is asked for now."""
    return {
        plan.label(arm, seed): plan.identity(c, definition_id, dataset_digest, plan.run_config(base, c, arm, seed),
                                             facts["arms"][arm]["effective"], model_source, metadata_digest)
        for arm, seed in plan.queue(c)
    }


def prepare(c: Comparison, seams: Seams, copy: bool) -> Prepared:
    """Read the reference job and the user's settings, copy (or, without ``copy``, only hash) the
    dataset, ask the product for its facts, and resolve every run's requested identity. The
    settings copy is written later (``run_queue``), once no finished run is refused."""
    base, definition_id = read_reference(seams.user_db, c.reference_job)
    settings = isolation.read_settings(seams.user_settings)
    model_source = isolation.model_source(settings, definition_id)
    first_arm, first_seed = plan.queue(c)[0]
    dataset = plan.dataset_name(plan.run_config(base, c, first_arm, first_seed))
    source = source_dataset(seams.user_db, dataset)
    manifest = isolation.copy_dataset(source, c.work_dir, copy=copy)
    snapshot = c.work_dir / isolation.METADATA_SNAPSHOT
    if copy:
        snapshot = isolation.snapshot_database(seams.user_db, c.work_dir, dataset)
    elif not snapshot.exists():  # what the first launch would take
        snapshot = seams.user_db
    metadata_digest = isolation.dataset_metadata_digest(snapshot, dataset)
    if copy:
        facts =seams.facts(definition_id, c.arms, isolation.child_env(c, _db_folder(c) / "facts.db"))
    else:  # nothing is written into the work folder: the fact child's stores go to a scratch folder
        with tempfile.TemporaryDirectory(prefix="sampler-compare-", ignore_cleanup_errors=True) as scratch:
            root = Path(scratch)
            facts = seams.facts(definition_id, c.arms, isolation.child_env(
                c, root / isolation.ISOLATED_PATHS["MRLN_DB_PATH"] / "facts.db", root=root))
    if int(facts["api_port"]) != c.port:
        raise ConfigError(f"the trainer's loopback calls would go to port {facts['api_port']}, not {c.port}")
    digest = isolation.dataset_digest(manifest)
    return Prepared(definition_id, dataset, source, facts, manifest,
                    requested_identities(c, base, definition_id, facts, digest, model_source, metadata_digest),
                    settings)


# ── A run's files ───────────────────────────────────────────────────────────


def _db_folder(c: Comparison) -> Path:
    return c.work_dir / isolation.ISOLATED_PATHS["MRLN_DB_PATH"]


def run_dir(c: Comparison, label: str, definition_id: str) -> Path:
    """Where the trainer writes a run: ``<output_dir>/<lora_name>_<model part>`` of ``plan.run_names``."""
    return c.work_dir / "runs" / f"{c.name}_{label}_{model_part_from_definition_id(definition_id)}"


def _log(c: Comparison, label: str, suffix: str) -> Path:
    return c.work_dir / "logs" / f"{label}.{suffix}"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def metrics(folder: Path) -> list[dict]:
    """The validation records on disk; a line still being written (or broken) is left out."""
    path = folder / vc.METRICS_FILE
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                out.append(record)
    return out


def _steps(records: list[dict]) -> list:
    return [r.get("step") for r in records]


def is_complete(folder: Path, c: Comparison, records: list[dict] | None = None) -> bool:
    return _steps(metrics(folder) if records is None else records) == plan.passes(c.steps, c.validation.every)


def applied_seed(log: Path) -> Any:
    seed = None
    if log.exists():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "run_seed_applied" not in line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("event") == "run_seed_applied":
                seed = event.get("seed")
    return seed


def logged_config(folder: Path) -> dict | None:
    log = _read_json(folder / "training_log.json")
    return log.get("config") if isinstance(log, dict) and isinstance(log.get("config"), dict) else None


def write_verdict(c: Comparison, label: str, accepted: bool, stopped_for: str | None) -> None:
    isolation.write_json(_log(c, label, "verdict.json"),
                         {"version": STATE_VERSION, "accepted": accepted, "stopped_for": stopped_for})


def _measured_digest(split_text: str) -> str:
    """The sha256 of what ``validation_compare.same_measurement`` compares (a test pins the two
    agreeing; that module's own reading is local to its function)."""
    split = json.loads(split_text)
    representatives = [{k: v for k, v in item.items() if k != "path"} for item in split.get("representatives") or []]
    measured = [representatives, *(split.get(k) for k in ("holdout", "split_seed", "global_triggerword"))]
    return hashlib.sha256(json.dumps(measured, sort_keys=True).encode("utf-8")).hexdigest()


def split_problem(run_dir: Path, c: Comparison, requested: dict, manifest: dict[str, str], record: bool = True,
                  records: list[dict] | None = None) -> str | None:
    """Why this run's split is not the measurement ``requested`` asks for, or None. The first split
    that passes everything is recorded in ``comparison.json`` (``record``), once."""
    config, definition_id = requested["config"], requested["definition_id"]
    path = run_dir / vc.SPLIT_FILE
    if not path.exists():
        return "the run wrote no split file"
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        split = json.loads(text)
    except ValueError:
        return UNREADABLE_SPLIT
    wanted = {
        "split_seed": config["validation_split_seed"],
        "holdout_fraction": config["validation_holdout_fraction"],
        "noise_draws": config["validation_noise_draws"],
        "timestep_bands": config["validation_timestep_bands"],
        "every_n_steps": config["validation_every_n_steps"],
    }
    for key, want in wanted.items():
        if split.get(key) != want:
            return f"its split records {key} {split.get(key)!r}, not {want!r}"
    if split.get("validation_dataset") is not None:
        return f"its split records validation_dataset {split['validation_dataset']!r}, not none"
    if split.get("model") is None:
        return "its split names no model"
    if split["model"] != definition_id:
        return f"its split names model {split['model']!r}, not {definition_id!r}"
    prefix = f"{plan.dataset_name(config)}/"
    for item in split.get("representatives") or []:
        key = item.get("key", "")
        if not item.get("sha256"):
            return f"its split records no content digest for {key}"
        name = key[len(prefix):] if key.startswith(prefix) else None
        if name is None or manifest.get(name) != item["sha256"]:
            return f"its held-out {key} is not the copied dataset's picture"
    for entry in metrics(run_dir) if records is None else records:
        draws, centres = entry.get("noise_draws"), entry.get("band_centres") or []
        if draws != wanted["noise_draws"]:
            return f"its metrics at step {entry.get('step')} records noise_draws {draws!r}, not {wanted['noise_draws']}"
        if len(centres) != wanted["timestep_bands"]:
            return (f"its metrics at step {entry.get('step')} records {len(centres)} bands, "
                    f"not {wanted['timestep_bands']}")
    measurement = c.work_dir / "comparison.json"
    recorded = _read_json(measurement)
    if recorded is None:
        if measurement.exists():
            return f"{measurement} can't be read"
        if record:
            isolation.write_json(measurement, {"version": STATE_VERSION, "label": run_dir.name, "split": str(path),
                                               "measured_sha256": _measured_digest(text)})
        return None
    reference = Path(recorded["split"])
    if not reference.exists():
        return f"the recorded measurement's split file {reference} is gone"
    reference_text = reference.read_text(encoding="utf-8", errors="replace")
    if _measured_digest(reference_text) != recorded["measured_sha256"]:
        return f"the recorded measurement's split file {reference} changed since it was recorded"
    if not vc.same_measurement(reference_text, text):
        return f"its split is not the same measurement as {recorded['label']} ({reference})"
    return None


def _sampler_and_seed(folder: Path, c: Comparison, label: str, requested: dict) -> str | None:
    logged = logged_config(folder)
    if logged is None:
        return "the run wrote no readable training_log.json"
    mismatch = facts_module.sampler_mismatch(logged, requested["effective_sampler"])
    if mismatch:
        return mismatch
    seed = applied_seed(_log(c, label, "log"))
    if seed != requested["config"]["seed"]:
        return f"the run applied seed {seed!r}, not {requested['config']['seed']}"
    return None


def finished_run_problem(folder: Path, c: Comparison, label: str, requested: dict, manifest: dict[str, str],
                         records: list[dict] | None = None, record: bool = True) -> str | None:
    """Why the run on disk is not a finished run of ``requested``, or None: every pass, the sampler
    and seed it ran, and its split (recorded as the measurement only with ``record``). Checked after
    a run, before a reuse and before the report."""
    if not folder.exists():
        return "the run wrote no run folder"
    records = metrics(folder) if records is None else records
    wanted = plan.passes(c.steps, c.validation.every)
    if _steps(records) != wanted:
        return f"incomplete metrics: passes {_steps(records)} of {wanted}"
    return _sampler_and_seed(folder, c, label, requested) or split_problem(
        folder, c, requested, manifest, record=record, records=records)


def _identity_problem(recorded: dict | None, requested: dict) -> str | None:
    if recorded is None:
        return "no recorded identity: nothing says which experiment it ran"
    difference = plan.identity_difference(recorded, requested)
    return f"it ran another experiment: its identity differs in {', '.join(difference)}" if difference else None


def _verdict_problem(c: Comparison, label: str) -> str | None:
    verdict = _read_json(_log(c, label, "verdict.json"))
    if verdict is None:
        return "no recorded verdict"
    if verdict.get("accepted") is not True:
        return verdict.get("stopped_for") or "its run was refused"
    return None


def reusable(folder: Path, c: Comparison, label: str, requested: dict, manifest: dict[str, str],
             records: list[dict] | None = None, record: bool = True) -> str | None:
    """Why a finished run on disk can't stand for the run asked for now, or None."""
    return (_identity_problem(_read_json(_log(c, label, "identity.json")), requested)
            or _verdict_problem(c, label)
            or finished_run_problem(folder, c, label, requested, manifest, records, record=record))


def move_aside(c: Comparison, label: str, folder: Path) -> list[Path]:
    """A partial run's folder and logs go under ``partial/<stamp>/``: a fresh run never appends."""
    target = c.work_dir / "partial" / f"{datetime.datetime.now():%Y%m%dT%H%M%S%f}"
    moving = ([folder] if folder.exists() else []) + sorted((c.work_dir / "logs").glob(f"{label}.*"))
    moved = []
    for path in moving:
        target.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target / path.name))
        moved.append(target / path.name)
    return moved


# ── Before a run ────────────────────────────────────────────────────────────


def _card_problem(limits: Limits, card: Card | None) -> str | None:
    if card is None:
        return "the card can't be read"
    if limits.power_limit_w is not None:
        if card.power_limit_w is None:
            return f"the card reads no power limit; the config asks for {limits.power_limit_w:g} W"
        if abs(card.power_limit_w - limits.power_limit_w) > POWER_TOLERANCE_W:
            return f"the power limit reads {card.power_limit_w:g} W, not {limits.power_limit_w:g} W"
    return None


def checks(c: Comparison, seams: Seams, card: Card | None) -> list[tuple[str | None, str]]:
    """Every check a run makes before it starts, on one card reading (the idle wait aside):
    (why it fails, or None; what was read)."""
    limits = c.limits
    available = seams.available_ram_bytes() / GIB
    shown = "none" if card is None or card.power_limit_w is None else f"{card.power_limit_w:g} W"
    card_text = "card: unreadable" if card is None else (
        f"card: {card.vram_mib} MiB in use (idle at {limits.idle_vram_mib}), {card.temperature_c:g} C, "
        f"power limit {shown}")
    return [
        (f"port {c.port} is already taken: refusing to use or touch that process" if seams.port_taken(c.port)
         else None, f"port {c.port}: free"),
        (seams.card_problem or _card_problem(limits, card), card_text),
        (f"host available RAM {available:.1f} GiB, under {limits.free_ram_gib:g} GiB"
         if available < limits.free_ram_gib else None,
         f"host RAM: {available:.1f} GiB available (floor {limits.free_ram_gib:g} GiB)"),
    ]


def preflight(c: Comparison, seams: Seams) -> str | None:
    """Why no run may start now (waiting up to ``idle_wait_minutes`` for an idle card), or None."""
    limits = c.limits
    started = seams.clock()
    while True:
        card = seams.read_card()
        problems = [problem for problem, _ in checks(c, seams, card) if problem]
        if problems:
            return problems[0]
        if card.vram_mib <= limits.idle_vram_mib:
            return None
        if seams.clock() - started >= limits.idle_wait_minutes * 60:
            return f"the card is not idle after {limits.idle_wait_minutes:g} min: {card.vram_mib} MiB in use"
        seams.sleep(IDLE_POLL_S)


# ── The children ────────────────────────────────────────────────────────────


@contextlib.contextmanager
def _pinned(affinity: tuple[int, ...] | None):
    """This process on ``affinity`` while the block runs, so a child started in it inherits the
    mask from its first instruction (a venv's launcher starts the real interpreter at once)."""
    if not affinity:
        yield
        return
    me = psutil.Process()
    before = me.cpu_affinity()
    me.cpu_affinity(list(affinity))
    try:
        yield
    finally:
        me.cpu_affinity(before)


class _Children:
    """The backend and trainer, and every process seen under them: each kept on ``affinity`` and
    all killed by ``stop``."""

    def __init__(self, affinity: tuple[int, ...] | list[int] | None = None) -> None:
        self.affinity = sorted(affinity) if affinity else None
        self.procs: list[subprocess.Popen] = []
        self.seen: dict[int, psutil.Process] = {}

    def start(self, argv: list[str], env: dict, log) -> subprocess.Popen:
        with _pinned(self.affinity):
            proc = subprocess.Popen(argv, cwd=BACKEND, env=env, stdout=log, stderr=subprocess.STDOUT)
        self.procs.append(proc)
        self.note(proc)
        return proc

    def note(self, proc: subprocess.Popen) -> None:
        """Every process under ``proc`` is remembered and, with an affinity, put on it."""
        for p in facts_module.tree(proc.pid):  # empty when gone: what it started was noted earlier
            p = self.seen.setdefault(p.pid, p)
            if self.affinity:
                try:
                    if sorted(p.cpu_affinity()) != self.affinity:
                        p.cpu_affinity(self.affinity)
                except psutil.Error:
                    pass  # ended meanwhile

    def rss_bytes(self) -> int:
        total = 0
        for p in list(self.seen.values()):
            try:
                if p.is_running():
                    total += p.memory_info().rss
            except psutil.Error:
                pass
        return total

    def stop(self) -> None:
        for proc in self.procs:
            self.note(proc)
        facts_module.kill_all(list(self.seen.values()), KILL_WAIT_S)
        for proc in self.procs:
            try:
                proc.wait(timeout=KILL_WAIT_S)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"process {proc.pid} survived a kill for {KILL_WAIT_S} s") from None


def _token_headers() -> dict[str, str]:
    """The header the trainer's own loopback calls send (``pipeline_data._internal_api_headers``)."""
    token = auth_token()
    return {"X-Auth-Token": token} if token else {}


def _wait_ready(c: Comparison, seams: Seams, backend: subprocess.Popen, dataset: str, log: Path) -> str | None:
    url = f"http://{LOOPBACK}:{c.port}/api/datasets/{urllib.parse.quote(dataset, safe='')}"
    request = urllib.request.Request(url, headers=_token_headers())
    deadline = seams.clock() + seams.ready_wait_s
    while True:
        if backend.poll() is not None:
            return f"the comparison's backend exited with {backend.returncode}; see {log}"
        if seams.port_taken(c.port):  # else not bound yet: a request would only wait for its refusal
            try:
                with urllib.request.urlopen(request, timeout=3):
                    return None
            except urllib.error.HTTPError as exc:
                return f"the comparison's backend answers {exc.code} for the dataset {dataset!r}; see {log}"
            except (urllib.error.URLError, OSError):
                pass  # bound, not serving yet
        if seams.clock() >= deadline:
            return f"the comparison's backend did not answer within {seams.ready_wait_s / 60:g} min; see {log}"
        seams.sleep(seams.poll_s)


def _wait_port_free(c: Comparison, seams: Seams) -> bool:
    deadline = seams.clock() + PORT_FREE_WAIT_S
    while seams.port_taken(c.port):
        if seams.clock() >= deadline:
            return False
        seams.sleep(PORT_FREE_POLL_S)
    return True


# ── One run ─────────────────────────────────────────────────────────────────


class _Status:
    """``status.json``: written at most every ``STATUS_EVERY_S`` and on every stop, with the lowest
    available RAM of every run so far."""

    def __init__(self, c: Comparison, seams: Seams):
        self.path = c.work_dir / "status.json"
        self.seams = seams
        self.written_at: float | None = None
        self.runs: list[Peaks] = []

    def new_run(self) -> Peaks:
        peaks = Peaks()
        self.runs.append(peaks)
        return peaks

    def write(self, **fields) -> None:
        lows = [p.min_ram_gib for p in self.runs if p.min_ram_gib is not None]
        isolation.write_json(self.path, {
            "version": STATE_VERSION,
            "at": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            "min_available_ram_gib": round(min(lows), 2) if lows else None,
            **fields,
        })
        self.written_at = self.seams.clock()

    def maybe(self, **fields) -> None:
        if self.written_at is None or self.seams.clock() - self.written_at >= STATUS_EVERY_S:
            self.write(**fields)


def _record(arm: str, seed: int, **fields) -> dict:
    return {"label": plan.label(arm, seed), "arm": arm, "seed": seed, "ran": False, "exit": None, "minutes": None,
            "stopped_for": None, "complete": False, "run_dir": None, "sampler": None, **fields}


@dataclass
class _Run:
    """What the watch over one run reads and writes."""

    c: Comparison
    seams: Seams
    label: str
    folder: Path
    requested: dict
    manifest: dict[str, str]
    file: TelemetryFile
    status: _Status
    log_path: Path
    children: _Children = field(default_factory=_Children)
    peaks: Peaks = field(default_factory=Peaks)


class _StepReader:
    """The trainer's last ``step_progress`` from its log, read as it grows; a line not yet ended
    waits for the rest of it."""

    def __init__(self, reader):
        self.reader = reader
        self.tail = ""
        self.step, self.step_time, self.events = 0, None, 0

    def poll(self) -> None:
        lines = (self.tail + self.reader.read()).split("\n")
        self.tail = lines.pop()
        for line in lines:
            if '"step_progress"' not in line:
                continue
            try:
                event = json.loads(line)
                self.step, self.step_time = int(event.get("step", self.step)), event.get("step_time")
            except (ValueError, TypeError, AttributeError):
                continue
            self.events += 1


def _watch(run: _Run, trainer: subprocess.Popen) -> tuple[str | None, Sample | None]:
    c, seams, folder = run.c, run.seams, run.folder
    started = seams.clock()
    grown_at, size_seen = started, -1
    sampler_checked = split_checked = False
    sample = None
    with open(run.log_path, encoding="utf-8", errors="replace") as reader:
        progress = _StepReader(reader)
        while trainer.poll() is None:
            seams.sleep(seams.poll_s)
            now = seams.clock()
            progress.poll()
            step = progress.step
            for proc in run.children.procs:
                run.children.note(proc)
            sample = Sample(now - started, step, progress.step_time, seams.read_card(),
                            seams.available_ram_bytes() / MIB, run.children.rss_bytes() / MIB)
            run.peaks.add(sample)
            size = run.log_path.stat().st_size + ((folder / _LOG_GROWTH).stat().st_size
                                                  if (folder / _LOG_GROWTH).exists() else 0)
            if size != size_seen:
                size_seen, grown_at = size, now
            stop = None
            if not sampler_checked and (folder / "training_log.json").exists():
                logged = logged_config(folder)
                if logged is not None:  # else still being written: look again
                    sampler_checked = True
                    stop = facts_module.sampler_mismatch(logged, run.requested["effective_sampler"])
            if not stop and not split_checked and (folder / vc.SPLIT_FILE).exists():
                problem = split_problem(folder, c, run.requested, run.manifest, record=False)
                if problem != UNREADABLE_SPLIT:  # else still being written: look again
                    split_checked, stop = True, problem
            stop = stop or telemetry.breach(sample, c.limits, now - grown_at)
            sample.note = stop or ""
            run.file.row(sample)
            fields = {"running": run.label, "minutes": round((now - started) / 60, 1), "step": step,
                      "stopping": stop}
            if stop:
                run.status.write(**fields)
                return stop, sample
            run.status.maybe(**fields)
    return None, sample


def run_one(c: Comparison, arm: str, seed: int, requested: dict, manifest: dict[str, str], seams: Seams,
            status: _Status | None = None) -> dict:
    """Train one arm at one seed beside the user's backend; -> the run's record."""
    label = plan.label(arm, seed)
    status = status or _Status(c, seams)
    refusal = preflight(c, seams)
    if refusal:
        return _record(arm, seed, stopped_for=f"before the run: {refusal}")
    definition_id = requested["definition_id"]
    config = {**requested["config"], **plan.run_names(c, arm, seed)}
    db_path = _db_folder(c) / f"{label}-{datetime.datetime.now():%Y%m%dT%H%M%S%f}.db"
    # From the comparison's snapshot, never the live database: every arm sees the same metadata.
    isolation.copy_database(c.work_dir / isolation.METADATA_SNAPSHOT, db_path, requested["dataset"],
                            c.work_dir / isolation.DATASET_FOLDER)
    env = isolation.child_env(c, db_path)
    (c.work_dir / "logs").mkdir(parents=True, exist_ok=True)
    isolation.write_json(_log(c, label, "identity.json"), requested)
    run = _Run(c, seams, label, run_dir(c, label, definition_id), requested, manifest,
               TelemetryFile(_log(c, label, "telemetry.csv")), status, _log(c, label, "log"),
               children=_Children(c.cpu_affinity), peaks=status.new_run())
    run.file.start()
    run.file.row(Sample(0.0, 0, None, seams.read_card(), seams.available_ram_bytes() / MIB, 0.0,
                        note="before the trainer starts"))
    started = seams.clock()
    stopped_for, exit_code, last, ended = None, None, None, "the runner failed"
    backend_log_path = _log(c, label, "backend.log")
    backend_log = open(backend_log_path, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
    trainer_log = open(run.log_path, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
    try:
        backend = run.children.start(seams.backend_argv(c), env, backend_log)
        stopped_for = _wait_ready(c, seams, backend, requested["dataset"], backend_log_path)
        if stopped_for is None:
            trainer = run.children.start(seams.trainer_argv(c, definition_id, config), env, trainer_log)
            stopped_for, last = _watch(run, trainer)
            exit_code = trainer.poll()
        ended = None
    finally:
        run.children.stop()
        backend_log.close()
        trainer_log.close()
        if ended:  # an exception is on its way out: the file still says how the run ended
            run.file.row(Sample(seams.clock() - started, 0, None, None, 0.0, 0.0, note=f"ended: {ended}"))
    port_free = _wait_port_free(c, seams)
    if exit_code is None and stopped_for is None:
        exit_code = -1
    folder = run.folder
    records = metrics(folder)
    complete = folder.exists() and is_complete(folder, c, records)
    if stopped_for is None:
        stopped_for = (f"the trainer exited with {exit_code}" if exit_code != 0
                       else finished_run_problem(folder, c, label, requested, manifest, records))
    final = last or Sample(seams.clock() - started, 0, None, None, seams.available_ram_bytes() / MIB, 0.0)
    final.note = f"ended: {stopped_for or 'finished'}"
    run.file.row(final)
    write_verdict(c, label, stopped_for is None, stopped_for)
    logged = logged_config(folder) if folder.exists() else None
    return _record(
        arm, seed, ran=True, exit=exit_code, minutes=round((seams.clock() - started) / 60, 1),
        stopped_for=stopped_for, complete=complete, run_dir=str(folder) if folder.exists() else None,
        sampler={"effective": requested["effective_sampler"],
                 "logged": facts_module.canonical_sampler(logged) if logged else None,
                 "applied_seed": applied_seed(run.log_path)},
        port_free=port_free, **run.peaks.summary(),
    )


# ── The queue ───────────────────────────────────────────────────────────────


def _append(c: Comparison, record: dict) -> None:
    with open(c.work_dir / "runs.jsonl", "a", encoding="utf-8") as out:
        out.write(json.dumps(record, default=str) + "\n")


def existing_run(c: Comparison, label: str, folder: Path, requested: dict,
                 manifest: dict[str, str], record: bool = True) -> tuple[str, str | None]:
    """What the queue does with a run already on disk: ("new" | "reuse" | "partial" | "refuse", why).
    A run of another identity is refused whether or not it finished: it is never trained over.
    Without ``record`` nothing is written (``split_problem``)."""
    if not (folder.exists() or any((c.work_dir / "logs").glob(f"{label}.*"))):
        return "new", None
    recorded = _read_json(_log(c, label, "identity.json"))
    identity = _identity_problem(recorded, requested)
    if recorded is not None and identity:
        return "refuse", identity
    records = metrics(folder)
    if not (folder.exists() and is_complete(folder, c, records)):
        return "partial", None
    why = identity or _verdict_problem(c, label) or finished_run_problem(folder, c, label, requested, manifest,
                                                                         records, record=record)
    return ("refuse", why) if why else ("reuse", None)


def run_queue(c: Comparison, seams: Seams, prepared: Prepared | None = None) -> str | None:
    """Run the queue to its end or its first stop; -> the stop's reason, or None."""
    for sub in ("runs", "logs", "partial"):
        (c.work_dir / sub).mkdir(parents=True, exist_ok=True)
    status = _Status(c, seams)
    try:
        prepared = prepared or prepare(c, seams, copy=True)
    except (ConfigError, RuntimeError, OSError, sqlite3.Error) as exc:
        status.write(running=None, queue_stop=str(exc))
        return str(exc)
    stop = refused_runs(c, prepared)
    if stop is None:  # the settings the identities were resolved from, and no others
        isolation.write_settings(prepared.settings, c.work_dir / isolation.ISOLATED_PATHS["MRLN_SETTINGS_PATH"])
    for arm, seed in plan.queue(c) if stop is None else ():
        label = plan.label(arm, seed)
        folder = run_dir(c, label, prepared.definition_id)
        requested = prepared.requested[label]
        state, why = existing_run(c, label, folder, requested, prepared.manifest)
        if state == "refuse":
            stop = f"{label} (run earlier): {why}"
            break
        if state == "reuse":
            _append(c, _record(arm, seed, reused=True, complete=True, run_dir=str(folder)))
            continue
        if state == "partial":
            move_aside(c, label, folder)
        try:
            record = run_one(c, arm, seed, requested, prepared.manifest, seams, status)
        except Exception as exc:  # noqa: BLE001 - the queue still ends with a verdict, a record and a status
            record = _failed_run(c, arm, seed, exc)
        _append(c, record)
        if record["stopped_for"]:
            stop = f"{label}: {record['stopped_for']}"
            break
    status.write(running=None, queue_stop=stop)
    return stop


def refused_runs(c: Comparison, prepared: Prepared) -> str | None:
    """Every run on disk the queue would refuse, each named, or None; nothing is written."""
    refusals = []
    for arm, seed in plan.queue(c):
        label = plan.label(arm, seed)
        state, why = existing_run(c, label, run_dir(c, label, prepared.definition_id), prepared.requested[label],
                                  prepared.manifest, record=False)
        if state == "refuse":
            refusals.append(f"{label} (run earlier): {why}")
    return "; ".join(refusals) or None


def _failed_run(c: Comparison, arm: str, seed: int, exc: Exception) -> dict:
    """The record of a run the runner failed in; its verdict refuses it, where it can be written."""
    error = f"{type(exc).__name__}: {exc}"
    why = f"the runner failed: {error}"
    with contextlib.suppress(OSError):
        write_verdict(c, plan.label(arm, seed), False, why)
    return _record(arm, seed, ran=True, stopped_for=why, error=error)
