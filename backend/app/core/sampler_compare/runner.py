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
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from app.core.naming import model_part_from_definition_id
from app.core.sampler_compare import facts as facts_module
from app.core.sampler_compare import isolation, plan, telemetry
from app.core.sampler_compare.plan import BACKEND, Comparison, ConfigError
from app.core.sampler_compare.telemetry import Card, Sample
from app.engine.core import validation_compare as vc

GIB = 2**30
MIB = 2**20
READY_WAIT_S = 240
STATUS_EVERY_S = 30
IDLE_POLL_S = 30
KILL_WAIT_S = 30
PORT_FREE_WAIT_S = 15
POWER_TOLERANCE_W = 1.0
STATE_VERSION = 1
_LOG_GROWTH = "job_log.jsonl"


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

    @classmethod
    def real(cls, c: Comparison) -> Seams:
        python = sys.executable

        def backend_argv(c: Comparison) -> list[str]:
            return [python, "-m", "uvicorn", "app.main:app", "--app-dir", str(BACKEND),
                    "--host", "127.0.0.1", "--port", str(c.port)]

        def trainer_argv(c: Comparison, definition_id: str, config: dict) -> list[str]:
            return [python, str(BACKEND / "run_trainer.py"), "--definition_id", definition_id,
                    "--config", json.dumps(config)]

        return cls(
            read_card=telemetry.read_card,
            available_ram_bytes=lambda: psutil.virtual_memory().available,
            clock=time.monotonic,
            sleep=time.sleep,
            backend_argv=backend_argv,
            trainer_argv=trainer_argv,
            port_taken=port_taken,
            facts=lambda definition_id, arms, env: facts_module.facts_in_child(definition_id, arms, env, python),
            user_db=Path(os.environ.get("MRLN_DB_PATH") or BACKEND / "app" / "arcane_tuner.db"),
            user_settings=Path(os.environ.get("MRLN_SETTINGS_PATH") or BACKEND / "settings.json"),
        )


def port_taken(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


# ── The reference job and what it asks for ──────────────────────────────────


def _read_only(db: Path) -> sqlite3.Connection:
    if not Path(db).exists():
        raise ConfigError(f"the database {db} does not exist")
    return sqlite3.connect(f"{Path(db).resolve().as_uri()}?mode=ro", uri=True)


def read_reference(db: Path, job_id: str) -> tuple[dict, str]:
    """(config, definition_id) of the reference job, from the user's database opened read-only."""
    conn = _read_only(db)
    try:
        row = conn.execute("select config, definition_id from job_history where id=?", (job_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ConfigError(f"the database {db} has no job {job_id!r}")
    return json.loads(row[0]), row[1]


def source_dataset(db: Path, name: str) -> Path:
    conn = _read_only(db)
    try:
        row = conn.execute("select path from datasets where name=?", (name,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ConfigError(f"the database {db} has no dataset {name!r}")
    return Path(row[0])


@dataclass
class Prepared:
    base: dict
    definition_id: str
    dataset: str
    source: Path
    facts: dict
    manifest: dict[str, str]
    digest: str
    requested: dict[str, dict]


def requested_identities(c: Comparison, base: dict, definition_id: str, facts: dict,
                         dataset_digest: str) -> dict[str, dict]:
    """label -> the identity each run of the queue is asked for now."""
    keys = frozenset(facts["timestep_keys"])
    return {
        plan.label(arm, seed): plan.identity(c, definition_id, dataset_digest,
                                             plan.run_config(base, c, arm, seed, keys),
                                             facts["arms"][arm]["effective"])
        for arm, seed in plan.queue(c)
    }


def prepare(c: Comparison, seams: Seams, copy: bool) -> Prepared:
    """Read the reference job, copy (or, without ``copy``, only hash) the dataset, ask the product
    for its facts, and resolve every run's requested identity."""
    base, definition_id = read_reference(seams.user_db, c.reference_job)
    first_arm, first_seed = plan.queue(c)[0]
    dataset = plan.dataset_name(plan.run_config(base, c, first_arm, first_seed, plan.timestep_keys_from_source()))
    source = source_dataset(seams.user_db, dataset)
    target = c.work_dir / isolation.DATASET_FOLDER
    if copy:
        manifest = isolation.copy_dataset(source, target)
        isolation.copy_settings(seams.user_settings, c.work_dir / isolation.ISOLATED_PATHS["MRLN_SETTINGS_PATH"])
    elif isolation.manifest_path(target).exists():
        manifest = isolation.read_manifest(target)
        if isolation.source_manifest(source) != manifest:
            raise ConfigError(f"the dataset changed since this comparison began: {source}")
    else:
        if not source.is_dir():
            raise ConfigError(f"the dataset folder {source} does not exist")
        manifest = isolation.source_manifest(source)
    facts = seams.facts(definition_id, c.arms, isolation.child_env(c, _db_folder(c) / "facts.db"))
    if set(facts["timestep_keys"]) != plan.timestep_keys_from_source():
        raise ConfigError("the sampler's source and the product disagree on the timestep keys")
    if int(facts["api_port"]) != c.port:
        raise ConfigError(f"the trainer's loopback calls would go to port {facts['api_port']}, not {c.port}")
    digest = isolation.dataset_digest(manifest)
    return Prepared(base, definition_id, dataset, source, facts, manifest, digest,
                    requested_identities(c, base, definition_id, facts, digest))


# ── A run's files ───────────────────────────────────────────────────────────


def _db_folder(c: Comparison) -> Path:
    return c.work_dir / isolation.ISOLATED_PATHS["MRLN_DB_PATH"]


def run_dir(c: Comparison, label: str, definition_id: str) -> Path:
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


def is_complete(folder: Path, c: Comparison) -> bool:
    return [r.get("step") for r in metrics(folder)] == plan.passes(c.steps, c.validation.every)


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
    """The sha256 of what ``validation_compare.same_measurement`` compares."""
    split = json.loads(split_text)
    representatives = [{k: v for k, v in item.items() if k != "path"} for item in split.get("representatives") or []]
    measured = [representatives, *(split.get(k) for k in ("holdout", "split_seed", "global_triggerword"))]
    return hashlib.sha256(json.dumps(measured, sort_keys=True).encode("utf-8")).hexdigest()


def split_problem(run_dir: Path, c: Comparison, config: dict, definition_id: str, manifest: dict[str, str],
                  record: bool = True) -> str | None:
    """Why this run's split is not the measurement asked for, or None. The first split that passes
    everything is recorded in ``comparison.json`` (``record``), once."""
    path = run_dir / vc.SPLIT_FILE
    if not path.exists():
        return "the run wrote no split file"
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        split = json.loads(text)
    except ValueError:
        return "its split file can't be read"
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
    for entry in metrics(run_dir):
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


def reusable(run_dir: Path, c: Comparison, label: str, requested_identity: dict, definition_id: str,
             manifest: dict[str, str]) -> str | None:
    """Why a finished run on disk can't stand for the run asked for now, or None."""
    if not is_complete(run_dir, c):
        return f"it lacks validation passes: {[r.get('step') for r in metrics(run_dir)]}"
    recorded = _read_json(_log(c, label, "identity.json"))
    if recorded is None:
        return "no recorded identity: nothing says which experiment it ran"
    difference = plan.identity_difference(recorded, requested_identity)
    if difference:
        return f"it ran another experiment: its identity differs in {', '.join(difference)}"
    verdict = _read_json(_log(c, label, "verdict.json"))
    if verdict is None:
        return "no recorded verdict"
    if verdict.get("accepted") is not True:
        return verdict.get("stopped_for") or "its run was refused"
    return _sampler_and_seed(run_dir, c, label, requested_identity) or split_problem(
        run_dir, c, requested_identity["config"], definition_id, manifest)


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


def preflight(c: Comparison, seams: Seams) -> str | None:
    """Why no run may start now (waiting up to ``idle_wait_minutes`` for an idle card), or None."""
    if seams.port_taken(c.port):
        return f"port {c.port} is already taken: refusing to use or touch that process"
    limits = c.limits
    started = seams.clock()
    while True:
        card = seams.read_card()
        if card is None:
            return "the card can't be read"
        if limits.power_limit_w is not None:
            if card.power_limit_w is None:
                return f"the card reads no power limit; the config asks for {limits.power_limit_w:g} W"
            if abs(card.power_limit_w - limits.power_limit_w) > POWER_TOLERANCE_W:
                return f"the power limit reads {card.power_limit_w:g} W, not {limits.power_limit_w:g} W"
        if card.vram_mib <= limits.idle_vram_mib:
            break
        if seams.clock() - started >= limits.idle_wait_minutes * 60:
            return f"the card is not idle after {limits.idle_wait_minutes:g} min: {card.vram_mib} MiB in use"
        seams.sleep(IDLE_POLL_S)
    available = seams.available_ram_bytes() / GIB
    if available < limits.free_ram_gib:
        return f"host available RAM {available:.1f} GiB, under {limits.free_ram_gib:g} GiB"
    return None


# ── The children ────────────────────────────────────────────────────────────


class _Children:
    """The backend and trainer, and every process seen under them: all killed by ``stop``."""

    def __init__(self) -> None:
        self.procs: list[subprocess.Popen] = []
        self.seen: dict[int, psutil.Process] = {}

    def start(self, argv: list[str], env: dict, log, affinity) -> subprocess.Popen:
        proc = subprocess.Popen(argv, cwd=BACKEND, env=env, stdout=log, stderr=subprocess.STDOUT)
        self.procs.append(proc)
        self.note(proc)
        if affinity:
            psutil.Process(proc.pid).cpu_affinity(list(affinity))
        return proc

    def note(self, proc: subprocess.Popen) -> None:
        try:
            parent = psutil.Process(proc.pid)
            for p in [parent, *parent.children(recursive=True)]:
                self.seen.setdefault(p.pid, p)
        except psutil.Error:
            pass  # gone already: what it started was noted on an earlier poll

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
        for p in self.seen.values():
            try:
                p.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(list(self.seen.values()), timeout=KILL_WAIT_S)
        for proc in self.procs:
            try:
                proc.wait(timeout=KILL_WAIT_S)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"process {proc.pid} survived a kill for {KILL_WAIT_S} s") from None


def _token_headers() -> dict[str, str]:
    token = os.environ.get("MRLN_AUTH_TOKEN", "").strip()
    return {"X-Auth-Token": token} if token else {}


def _wait_ready(c: Comparison, seams: Seams, backend: subprocess.Popen, dataset: str, log: Path) -> str | None:
    url = f"http://127.0.0.1:{c.port}/api/datasets/{urllib.parse.quote(dataset, safe='')}"
    request = urllib.request.Request(url, headers=_token_headers())
    deadline = seams.clock() + READY_WAIT_S
    while True:
        if backend.poll() is not None:
            return f"the comparison's backend exited with {backend.returncode}; see {log}"
        try:
            with urllib.request.urlopen(request, timeout=3):
                return None
        except urllib.error.HTTPError as exc:
            return f"the comparison's backend answers {exc.code} for the dataset {dataset!r}; see {log}"
        except (urllib.error.URLError, OSError):
            pass  # not up yet
        if seams.clock() >= deadline:
            return f"the comparison's backend did not answer within {READY_WAIT_S // 60} min; see {log}"
        seams.sleep(seams.poll_s)


def _wait_port_free(c: Comparison, seams: Seams) -> bool:
    deadline = time.monotonic() + PORT_FREE_WAIT_S
    while seams.port_taken(c.port):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.2)
    return True


# ── One run ─────────────────────────────────────────────────────────────────


class _Status:
    """``status.json``: written at most every ``STATUS_EVERY_S`` and on every stop."""

    def __init__(self, c: Comparison, seams: Seams):
        self.path = c.work_dir / "status.json"
        self.seams = seams
        self.written_at: float | None = None
        self.min_ram_gib: float | None = None

    def ram(self, available_bytes: float) -> None:
        gib = available_bytes / GIB
        self.min_ram_gib = gib if self.min_ram_gib is None else min(self.min_ram_gib, gib)

    def write(self, **fields) -> None:
        isolation.write_json(self.path, {
            "version": STATE_VERSION,
            "at": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            "min_available_ram_gib": None if self.min_ram_gib is None else round(self.min_ram_gib, 2),
            **fields,
        })
        self.written_at = self.seams.clock()

    def maybe(self, **fields) -> None:
        if self.written_at is None or self.seams.clock() - self.written_at >= STATUS_EVERY_S:
            self.write(**fields)


def _record(c: Comparison, arm: str, seed: int, **fields) -> dict:
    return {"label": plan.label(arm, seed), "arm": arm, "seed": seed, "ran": False, "exit": None, "minutes": None,
            "stopped_for": None, "complete": False, "run_dir": None, "sampler": None, **fields}


def _watch(c, seams, label, trainer, children, folder, config, definition_id, requested, manifest,
           file, peaks, status, log_path) -> tuple[str | None, Sample | None]:
    started = seams.clock()
    grown_at, size_seen = started, -1
    step, step_time = 0, None
    sampler_checked = split_checked = False
    sample = None
    with open(log_path, encoding="utf-8", errors="replace") as reader:
        while trainer.poll() is None:
            seams.sleep(seams.poll_s)
            now = seams.clock()
            for line in reader.readlines():
                if '"step_progress"' not in line:
                    continue
                try:
                    event = json.loads(line)
                    step, step_time = int(event.get("step", step)), event.get("step_time")
                except (ValueError, TypeError, AttributeError):
                    continue
            for proc in children.procs:
                children.note(proc)
            ram = seams.available_ram_bytes()
            status.ram(ram)
            sample = Sample(now - started, step, step_time, seams.read_card(), ram / MIB, children.rss_bytes() / MIB)
            peaks.add(sample)
            size = log_path.stat().st_size + ((folder / _LOG_GROWTH).stat().st_size
                                              if (folder / _LOG_GROWTH).exists() else 0)
            if size != size_seen:
                size_seen, grown_at = size, now
            stop = None
            if not sampler_checked and (folder / "training_log.json").exists():
                logged = logged_config(folder)
                if logged is not None:  # else still being written: look again
                    sampler_checked = True
                    stop = facts_module.sampler_mismatch(logged, requested["effective_sampler"])
            if not stop and not split_checked and (folder / vc.SPLIT_FILE).exists():
                if _read_json(folder / vc.SPLIT_FILE) is not None:
                    split_checked = True
                    stop = split_problem(folder, c, config, definition_id, manifest, record=False)
            stop = stop or telemetry.breach(sample, c.limits, now - grown_at)
            sample.note = stop or ""
            file.row(sample)
            fields = {"running": label, "minutes": round((now - started) / 60, 1), "step": step, "stopping": stop}
            if stop:
                status.write(**fields)
                return stop, sample
            status.maybe(**fields)
    return None, sample


def run_one(c: Comparison, arm: str, seed: int, definition_id: str, requested_identity: dict, seams: Seams,
            status: _Status | None = None) -> dict:
    """Train one arm at one seed beside the user's backend; -> the run's record."""
    label = plan.label(arm, seed)
    status = status or _Status(c, seams)
    refusal = preflight(c, seams)
    if refusal:
        return _record(c, arm, seed, stopped_for=f"before the run: {refusal}")
    config = {**requested_identity["config"], **plan.run_names(c, arm, seed)}
    folder = run_dir(c, label, definition_id)
    manifest = isolation.read_manifest(c.work_dir / isolation.DATASET_FOLDER)
    db_path = _db_folder(c) / f"{label}-{datetime.datetime.now():%Y%m%dT%H%M%S%f}.db"
    isolation.copy_database(seams.user_db, db_path, requested_identity["dataset"],
                            c.work_dir / isolation.DATASET_FOLDER)
    env = isolation.child_env(c, db_path)
    (c.work_dir / "logs").mkdir(parents=True, exist_ok=True)
    isolation.write_json(_log(c, label, "identity.json"), requested_identity)
    file = telemetry.TelemetryFile(_log(c, label, "telemetry.csv"))
    file.start()
    file.row(Sample(0.0, 0, None, seams.read_card(), seams.available_ram_bytes() / MIB, 0.0,
                    note="before the trainer starts"))
    peaks = telemetry.Peaks()
    children = _Children()
    log_path = _log(c, label, "log")
    started = seams.clock()
    stopped_for, exit_code, last, ended = None, None, None, "the runner failed"
    backend_log = open(_log(c, label, "backend.log"), "w", encoding="utf-8", errors="replace")  # noqa: SIM115
    trainer_log = open(log_path, "w", encoding="utf-8", errors="replace")  # noqa: SIM115
    try:
        backend = children.start(seams.backend_argv(c), env, backend_log, c.cpu_affinity)
        stopped_for = _wait_ready(c, seams, backend, requested_identity["dataset"], _log(c, label, "backend.log"))
        if stopped_for is None:
            trainer = children.start(seams.trainer_argv(c, definition_id, config), env, trainer_log, c.cpu_affinity)
            stopped_for, last = _watch(c, seams, label, trainer, children, folder, config, definition_id,
                                       requested_identity, manifest, file, peaks, status, log_path)
            exit_code = trainer.poll()
        ended = None
    finally:
        children.stop()
        backend_log.close()
        trainer_log.close()
        if ended:  # an exception is on its way out: the file still says how the run ended
            file.row(Sample(seams.clock() - started, 0, None, None, 0.0, 0.0, note=f"ended: {ended}"))
    port_free = _wait_port_free(c, seams)
    if exit_code is None and stopped_for is None:
        exit_code = -1
    complete = folder.exists() and is_complete(folder, c)
    if stopped_for is None:
        if exit_code != 0:
            stopped_for = f"the trainer exited with {exit_code}"
        elif not folder.exists():
            stopped_for = "the run wrote no run folder"
        elif not complete:
            steps = [r.get("step") for r in metrics(folder)]
            stopped_for = f"incomplete metrics: passes {steps} of {plan.passes(c.steps, c.validation.every)}"
        else:
            stopped_for = _sampler_and_seed(folder, c, label, requested_identity) or split_problem(
                folder, c, config, definition_id, manifest)
    final = last or Sample(seams.clock() - started, 0, None, None, seams.available_ram_bytes() / MIB, 0.0)
    final.note = f"ended: {stopped_for or 'finished'}"
    file.row(final)
    write_verdict(c, label, stopped_for is None, stopped_for)
    logged = logged_config(folder) if folder.exists() else None
    return _record(
        c, arm, seed, ran=True, exit=exit_code, minutes=round((seams.clock() - started) / 60, 1),
        stopped_for=stopped_for, complete=complete, run_dir=str(folder) if folder.exists() else None,
        sampler={"effective": requested_identity["effective_sampler"],
                 "logged": facts_module.canonical_sampler(logged) if logged else None,
                 "applied_seed": applied_seed(log_path)},
        port_free=port_free, **peaks.summary(),
    )


# ── The queue ───────────────────────────────────────────────────────────────


def _append(c: Comparison, record: dict) -> None:
    with open(c.work_dir / "runs.jsonl", "a", encoding="utf-8") as out:
        out.write(json.dumps(record, default=str) + "\n")


def _existing(c: Comparison, label: str, folder: Path) -> bool:
    return folder.exists() or any((c.work_dir / "logs").glob(f"{label}.*"))


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
    stop = None
    for arm, seed in plan.queue(c):
        label = plan.label(arm, seed)
        folder = run_dir(c, label, prepared.definition_id)
        requested = prepared.requested[label]
        if _existing(c, label, folder):
            recorded = _read_json(_log(c, label, "identity.json"))
            difference = plan.identity_difference(recorded, requested) if recorded is not None else []
            if difference:
                stop = f"{label} (run earlier): it ran another experiment: its identity differs in {', '.join(difference)}"
                break
            if folder.exists() and is_complete(folder, c):
                why = reusable(folder, c, label, requested, prepared.definition_id, prepared.manifest)
                if why:
                    stop = f"{label} (finished earlier): {why}"
                    break
                _append(c, _record(c, arm, seed, reused=True, complete=True, run_dir=str(folder)))
                continue
            move_aside(c, label, folder)
        record = run_one(c, arm, seed, prepared.definition_id, requested, seams, status)
        _append(c, record)
        if record["stopped_for"]:
            stop = f"{label}: {record['stopped_for']}"
            break
    status.write(running=None, queue_stop=stop)
    return stop
