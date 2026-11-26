"""Shared builders for the sampler comparison's tests: a config, and a world of fakes (a user
database, a dataset, settings, a fake card and host, and the fake trainer and backend as real
child processes)."""
import json
import socket
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from app.core.sampler_compare import facts as facts_module
from app.core.sampler_compare import isolation, plan, runner
from app.core.sampler_compare.runner import port_taken
from app.core.sampler_compare.telemetry import Card
from app.engine.strategies.timestep_keys import TIMESTEP_SCHEMA_DEFAULTS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fake_trainer  # noqa: E402

GIB = 2**30
DEFINITION = "moss-image-base"
CONFIG = {
    "version": 1,
    "name": "cmp",
    "reference_job": "job-1",
    "dataset": "Cars",
    "arms": {
        "uniform": {"timestep_sampling": "uniform"},
        "radc": {"timestep_sampling": "radc", "radc_start": 0.9, "radc_end": 0.3},
    },
    "seeds": [42, 43],
    "steps": 2000,
    "validation": {"every": 500, "noise_draws": 4, "holdout_fraction": 0.3, "split_seed": 0},
    "port": 8001,
    "limits": {"temperature_c": 94, "vram_mib": 88000, "free_ram_gib": 1, "stall_minutes": 15,
               "run_minutes": 120, "idle_vram_mib": 20000, "idle_wait_minutes": 30, "power_limit_w": 500},
}
BASE_JOB = {
    "learning_rate": 0.0001,
    "datasets": [{"dataset_name": "Original", "repeats": 1}],
    "timestep_sampling": "logit_normal",
    "logit_normal_mu": 0.3,
}


def write_config(folder: Path, **changes) -> Path:
    config = json.loads(json.dumps(CONFIG))
    config.setdefault("work_dir", str(folder / "work"))
    config.update(changes)
    path = folder / "comparison.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.fixture
def make_comparison(tmp_path):
    def make(**changes) -> plan.Comparison:
        return plan.load_comparison(write_config(tmp_path, **changes))

    return make


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class FakeClock:
    """A clock that moves ``scale`` times faster than the sleeps it is asked for."""

    def __init__(self, scale: float):
        self.now = 0.0
        self.scale = scale

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        time.sleep(min(seconds, 0.02))
        self.now += seconds * self.scale


def fake_facts(definition_id: str, arms: dict, env: dict) -> dict:
    """The facts for a definition that declares nothing: the arm's keys over the schema defaults."""
    resolved = {}
    for arm, settings in arms.items():
        given = facts_module.canonical_sampler(settings)
        effective = {**TIMESTEP_SCHEMA_DEFAULTS, **given}
        resolved[arm] = {"effective": effective,
                         "source": {key: "arm" if key in given else "schema" for key in effective}}
    return {"api_port": int(env["PORT"]), "arms": resolved}


class World:
    """The user's side (database, dataset, settings) and the seams the runner is given."""

    def __init__(self, root: Path):
        self.root = root
        self.dataset = root / "user-datasets" / "Cars"
        self.dataset.mkdir(parents=True)
        for i in range(20):
            (self.dataset / f"img{i:02d}.png").write_bytes(f"picture {i}".encode() * 50)
            (self.dataset / f"img{i:02d}.txt").write_text(f"caption {i}", encoding="utf-8")
        self.db = root / "user.db"
        self.settings = root / "settings.json"
        self.settings.write_text(json.dumps({"application": {"backend_port": 8000}}), encoding="utf-8")
        self.base = json.loads(json.dumps(BASE_JOB))
        self.write_db()
        self.port = free_port()
        self.card = Card(1000, 40.0, 100.0, 500.0)
        self.ram = 50 * GIB
        self.trainer_starts = 0
        self.backend_starts = 0
        self.modes: dict[str, str] = {}
        self.default_mode = "ok"

    def write_db(self, extra_jobs: dict | None = None) -> None:
        self._prepared: list[tuple[plan.Comparison, runner.Prepared]] = []  # the reference job may change
        if self.db.exists():
            self.db.unlink()
        db = sqlite3.connect(self.db)
        db.execute("create table datasets (id text primary key, name text not null unique, path text not null)")
        db.execute("create table job_history (id text primary key, status text, config text, definition_id text)")
        db.execute("insert into datasets values ('d1', 'Cars', ?)", (str(self.dataset),))
        db.execute("insert into datasets values ('d2', 'Other', ?)", (str(self.root / "other"),))
        db.execute("insert into job_history values ('job-1', 'completed', ?, ?)", (json.dumps(self.base), DEFINITION))
        db.execute("insert into job_history values ('job-0', 'running', '{}', ?)", (DEFINITION,))
        for job_id, config in (extra_jobs or {}).items():
            db.execute("insert into job_history values (?, 'completed', ?, ?)", (job_id, json.dumps(config), DEFINITION))
        db.commit()
        db.close()

    def set_base(self, **changes) -> None:
        self.base.update(changes)
        self.write_db()

    def set_settings(self, content: dict) -> None:
        """The user's settings file now holds ``content`` (the prepared runs are read afresh)."""
        self.settings.write_text(json.dumps(content), encoding="utf-8")
        self._prepared = []

    def comparison(self, **changes) -> plan.Comparison:
        changes.setdefault("port", self.port)
        return plan.load_comparison(write_config(self.root, **changes))

    def trainer_argv(self, c, definition_id, config):
        self.trainer_starts += 1
        label = config["lora_name"][len(c.name) + 1:]
        mode = self.modes.get(label, self.default_mode)
        return [sys.executable, str(HERE / "fake_trainer.py"), "--definition_id", definition_id,
                "--config", json.dumps(config), "--mode", mode]

    def backend_argv(self, c):
        self.backend_starts += 1
        return [sys.executable, str(HERE / "fake_backend.py")]

    def seams(self, **overrides):
        values = dict(
            read_card=lambda: self.card,
            available_ram_bytes=lambda: self.ram,
            clock=time.monotonic,
            sleep=time.sleep,
            backend_argv=self.backend_argv,
            trainer_argv=self.trainer_argv,
            port_taken=port_taken,
            facts=fake_facts,
            user_db=self.db,
            user_settings=self.settings,
            poll_s=0.05,
        )
        values.update(overrides)
        return runner.Seams(**values)

    def prepared(self, c) -> runner.Prepared:
        """``runner.prepare(c, copy=True)``, once per comparison while the reference job stands."""
        for known, prepared in self._prepared:
            if known == c:
                return prepared
        prepared = runner.prepare(c, self.seams(), copy=True)
        self._prepared.append((c, prepared))
        return prepared

    def place_all(self, c, mode="ok", modes=None, skip=()):
        """Every run of ``c``'s queue placed finished (``modes``: label -> mode), but ``skip``."""
        for arm, seed in plan.queue(c):
            name = plan.label(arm, seed)
            if name not in skip:
                self.place_finished(c, arm, seed, mode=(modes or {}).get(name, mode))

    def place_finished(self, c, arm, seed, mode="ok", accepted=True, stopped_for=None, identity=True):
        """A finished run on disk, as the queue leaves one: run files, its log, identity and verdict."""
        prepared = self.prepared(c)
        name = plan.label(arm, seed)
        requested = prepared.requested[name]
        config = {**requested["config"], **plan.run_names(c, arm, seed)}
        folder = c.work_dir / isolation.DATASET_FOLDER
        fake_trainer.write_run(config, DEFINITION, mode, folder, prepared.dataset)
        logs = c.work_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / f"{name}.log").write_text(fake_trainer.applied_seed_line(config, mode) + "\n", encoding="utf-8")
        if identity:
            isolation.write_json(logs / f"{name}.identity.json", requested)
        runner.write_verdict(c, name, accepted, stopped_for)
        return runner.run_dir(c, name, DEFINITION)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


@pytest.fixture(scope="module")
def ran_queue(tmp_path_factory):
    """One whole queue, trained by the fake trainer, for the tests that only read what it left."""
    world = World(tmp_path_factory.mktemp("ran"))
    c = world.comparison()
    return world, c, runner.run_queue(c, world.seams())


def started_pids(log: Path, event: str) -> list[int]:
    out = []
    if log.exists():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("event") == event:
                out.append(int(record["pid"]))
    return out
