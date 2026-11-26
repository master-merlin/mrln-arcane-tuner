"""Compare timestep samplers on one model and dataset, beside your running backend.

    python backend/scripts/compare_samplers.py my-comparison.json --dry-run
    python backend/scripts/compare_samplers.py my-comparison.json
    python backend/scripts/compare_samplers.py my-comparison.json --report-only

The config (see ``compare_samplers.example.json``) names a finished reference job, the arms (each
a set of timestep keys), two or more seeds, the validation cadence, a port of its own and the
limits that stop a run. Every run trains the reference job's settings with only the arm's sampler
and the seed changed, on its own copy of the dataset, database and settings under the work folder,
with a backend of its own on the configured port. ``report.txt`` and ``report.json`` in the work
folder give the noise floor and every pair's verdict.

``--dry-run`` checks everything and starts no backend or trainer; ``--report-only`` writes the
report over the runs that finished. Exit codes: 0 done, 1 the queue stopped or a check failed
(the reason is printed), 2 the config can't be used.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # the checkout is shared and only read while a comparison runs

import argparse  # noqa: E402
import os  # noqa: E402
import sqlite3  # noqa: E402
from pathlib import Path  # noqa: E402

_BACKEND = Path(__file__).absolute().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.core.sampler_compare import facts, isolation, plan, report, runner  # noqa: E402

_FAILURES = (plan.ConfigError, RuntimeError, OSError, sqlite3.Error, ValueError, KeyError)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", type=Path, help="the comparison's config file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="check everything; start no backend or trainer")
    mode.add_argument("--report-only", action="store_true", help="write the report over the finished runs")
    mode.add_argument("--facts", nargs=2, metavar=("DEFINITION", "ARMS"), help=argparse.SUPPRESS)
    return parser


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _weight_cache() -> str:
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return explicit
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return str(Path(home) / "hub")


def dry_run(c: plan.Comparison, seams: runner.Seams) -> int:
    """Every check a run makes before it starts, and what it would copy; starts no backend or trainer."""
    problems: list[str] = []

    def problem(text: str) -> None:
        problems.append(text)
        _say(f"PROBLEM: {text}")

    _say(f"comparison {c.name}: work folder {c.work_dir}")
    labels = [plan.label(arm, seed) for arm, seed in plan.queue(c)]
    _say(f"queue ({len(labels)} runs): {', '.join(labels)}")
    try:
        prepared = runner.prepare(c, seams, copy=False)
    except _FAILURES as exc:
        problem(str(exc))
        prepared = None
    if prepared is not None:
        _say(f"reference job {c.reference_job}: {prepared.definition_id}, read from {seams.user_db} (read-only)")
        for arm in c.arms:
            sampler = report.sampler_text(prepared.facts, arm)
            _say(f"arm {arm}: " + ", ".join(f"{key} {value}" for key, value in sampler.items()))
        work = c.work_dir
        _say(f"copy plan: dataset {prepared.dataset!r} from {prepared.source} ({len(prepared.manifest)} files, "
             f"without .cache and .thumbnails) -> {work / isolation.DATASET_FOLDER}; database {seams.user_db} "
             f"(read-only) -> {work / isolation.ISOLATED_PATHS['MRLN_DB_PATH']}/<run>.db for each run; settings "
             f"{seams.user_settings} -> {work / isolation.ISOLATED_PATHS['MRLN_SETTINGS_PATH']}; server log -> "
             f"{work / isolation.ISOLATED_PATHS['MRLN_SERVER_LOG_PATH']}; runs -> {work / 'runs'}")
        for arm, seed in plan.queue(c):
            name = plan.label(arm, seed)
            folder = runner.run_dir(c, name, prepared.definition_id)
            state, why = runner.existing_run(c, name, folder, prepared.requested[name], prepared.definition_id,
                                             prepared.manifest)
            if state == "refuse":
                problem(f"{name} (run earlier) would be refused: {why}")
            elif state == "reuse":
                _say(f"{name}: finished earlier; reused")
            elif state == "partial":
                _say(f"{name}: partial; would be moved aside to {c.work_dir / 'partial'}")
    if seams.port_taken(c.port):
        problem(f"port {c.port} is already taken: the queue would refuse to use or touch that process")
    else:
        _say(f"port {c.port}: free")
    card = seams.read_card()
    if card is None:
        problem("the card can't be read")
    else:
        _say(f"card: {card.vram_mib} MiB in use (idle at {c.limits.idle_vram_mib}), {card.temperature_c:g} C, "
             f"power limit {card.power_limit_w if card.power_limit_w is None else f'{card.power_limit_w:g} W'}")
        wanted = c.limits.power_limit_w
        if wanted is not None and (card.power_limit_w is None
                                   or abs(card.power_limit_w - wanted) > runner.POWER_TOLERANCE_W):
            shown = "none" if card.power_limit_w is None else f"{card.power_limit_w:g} W"
            problem(f"the power limit reads {shown}, not {wanted:g} W")
    available = seams.available_ram_bytes() / runner.GIB
    if available < c.limits.free_ram_gib:
        problem(f"host available RAM {available:.1f} GiB, under {c.limits.free_ram_gib:g} GiB")
    else:
        _say(f"host RAM: {available:.1f} GiB available (floor {c.limits.free_ram_gib:g} GiB)")
    if c.allow_downloads:
        _say(f"downloads on: a missing weight downloads, and the shared weight cache {_weight_cache()} is written to")
    else:
        _say("downloads off: a missing weight fails the run; the shared weight cache is only read")
    _say("dry run: " + ("OK" if not problems else "PROBLEMS above"))
    return 0 if not problems else 1


def main(argv: list[str] | None = None, seams: runner.Seams | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.facts:
        return facts.main(args.facts)
    if args.config is None:
        parser.error("the config file is required")
    try:
        c = plan.load_comparison(args.config)
    except plan.ConfigError as exc:
        sys.stderr.write(f"compare_samplers: {exc}\n")
        return 2
    seams = seams or runner.Seams.real(c)
    if args.dry_run:
        return dry_run(c, seams)
    try:
        prepared = runner.prepare(c, seams, copy=not args.report_only)
    except _FAILURES as exc:
        _say(f"stopped: {exc}")
        return 1
    stop = None if args.report_only else runner.run_queue(c, seams, prepared)
    code = report.write_report(c, stop, prepared.facts, prepared.requested, prepared.definition_id)
    _say(f"queue {'stopped: ' + stop if stop else 'done'}; report: {c.work_dir / 'report.txt'}")
    return 1 if stop else code


if __name__ == "__main__":
    sys.exit(main())
