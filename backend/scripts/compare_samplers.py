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
import sqlite3  # noqa: E402
from pathlib import Path  # noqa: E402

_BACKEND = Path(__file__).absolute().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from huggingface_hub.constants import HF_HUB_CACHE  # noqa: E402

from app.core.sampler_compare import isolation, plan, report, runner  # noqa: E402

_FAILURES = (plan.ConfigError, RuntimeError, OSError, sqlite3.Error, ValueError, KeyError)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", type=Path, help="the comparison's config file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="check everything; start no backend or trainer")
    mode.add_argument("--report-only", action="store_true", help="write the report over the finished runs")
    return parser


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def dry_run(c: plan.Comparison, seams: runner.Seams) -> int:
    """Every check a run makes before it starts, and what it would copy; starts no backend or trainer."""
    problems: list[str] = []

    def problem(text: str) -> None:
        problems.append(text)
        _say(f"PROBLEM: {text}")

    _say(f"comparison {c.name}: work folder {c.work_dir}")
    busy = runner.lock_problem(c)
    if busy:
        problem(busy)
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
             f"(read-only) -> {work / isolation.METADATA_SNAPSHOT} once, and from it "
             f"{work / isolation.ISOLATED_PATHS['MRLN_DB_PATH']}/<run>.db for each run; settings "
             f"{seams.user_settings} -> {work / isolation.ISOLATED_PATHS['MRLN_SETTINGS_PATH']}; server log -> "
             f"{work / isolation.ISOLATED_PATHS['MRLN_SERVER_LOG_PATH']}; runs -> {work / 'runs'}")
        for arm, seed in plan.queue(c):
            name = plan.label(arm, seed)
            folder = runner.run_dir(c, name, prepared.definition_id)
            state, why = runner.existing_run(c, name, folder, prepared.requested[name], prepared.manifest,
                                             record=False)
            if state == "refuse":
                problem(f"{name} (run earlier) would be refused: {why}")
            elif state == "reuse":
                _say(f"{name}: finished earlier; reused")
            elif state == "partial":
                _say(f"{name}: partial; would be moved aside to {c.work_dir / 'partial'}")
    for failed, read in runner.checks(c, seams, seams.read_card()):
        if failed:
            problem(failed)
        else:
            _say(read)
    if c.allow_downloads:
        _say(f"downloads on: a missing weight downloads, and the shared weight cache {HF_HUB_CACHE} is written to")
    else:
        _say("downloads off: a missing weight fails the run; the shared weight cache is only read")
    _say("dry run: " + ("OK" if not problems else "PROBLEMS above"))
    return 0 if not problems else 1


def main(argv: list[str] | None = None, seams: runner.Seams | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
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
        with runner.work_lock(c):  # held through the report; a second launch touches nothing
            return _launch(c, seams, args.report_only)
    except runner.WorkFolderBusy as exc:
        _say(f"stopped: {exc}")
        return 1


def _launch(c: plan.Comparison, seams: runner.Seams, report_only: bool) -> int:
    try:
        prepared = runner.prepare(c, seams, copy=not report_only)
    except _FAILURES as exc:
        _say(f"stopped: {exc}")
        return 1
    try:
        stop = None if report_only else runner.run_queue(c, seams, prepared)
        code = report.write_report(c, stop, prepared)
    except Exception as exc:  # noqa: BLE001 - the reason, never a traceback
        _say(f"stopped: {type(exc).__name__}: {exc}")
        return 1
    _say(f"queue {'stopped: ' + stop if stop else 'done'}; report: {c.work_dir / 'report.txt'}")
    return 1 if stop else code


if __name__ == "__main__":
    sys.exit(main())
