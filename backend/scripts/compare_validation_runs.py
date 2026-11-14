"""Compare the held-out validation of training runs.

    python backend/scripts/compare_validation_runs.py --run uniform=outputs/a --run radc=outputs/b

Each ``--run`` is ``<arm>=<output folder>``; give an arm several folders (one per training
seed) for a noise floor. Reads only ``validation_metrics.jsonl`` and ``validation_split.json``
in each folder, and writes only the ``--json`` file.

Exit codes: 0 with a report, 2 when the runs don't measure the same thing, 1 when a file
can't be read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_BACKEND = Path(__file__).absolute().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.engine.core import validation_compare as vc  # noqa: E402


def _run(text: str) -> tuple[str, str]:
    arm, sep, folder = text.partition("=")
    if not sep or not arm or not folder:
        raise argparse.ArgumentTypeError(f"{text!r} is not <arm>=<output folder>")
    return arm, folder


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=_run, action="append", required=True, metavar="ARM=FOLDER",
                        help="a run's output folder, under its arm's name (repeat; two or more arms)")
    parser.add_argument("--step", type=int, default=None, help="the step to compare (default: the last every run has)")
    parser.add_argument("--resamples", type=int, default=vc.DEFAULT_RESAMPLES, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0, help="seed of the bootstrap")
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON to this file")
    return parser


def write_json(path: Path, report: dict) -> None:
    """Write the report atomically: a temp file beside it, then ``os.replace``."""
    path = path.absolute()
    handle, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
        os.replace(temp, path)
    except BaseException:
        if os.path.exists(temp):
            os.remove(temp)
        raise


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = vc.compare(args.run, step=args.step, resamples=args.resamples, seed=args.seed)
    except vc.Refusal as refused:
        print(f"Not compared: {refused}", file=sys.stderr)
        return 2
    except vc.Unreadable as unreadable:
        print(f"Can't read: {unreadable}", file=sys.stderr)
        return 1
    sys.stdout.write(vc.format_report(report))
    if args.json is not None:
        try:
            write_json(args.json, report)
        except OSError as exc:
            print(f"Can't write: {args.json}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
