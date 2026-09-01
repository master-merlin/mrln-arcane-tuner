"""Record the FULL outcome vector of a pytest run — one line per test id.

``pass / fail / skip / xfail / xpass / error`` per nodeid, written as JSON to
the path in ``MRLN_OUTCOME_VECTOR``. junitxml cannot do this: it files a
non-strict xpass as a plain pass, and LANE-63's acceptance is that a serial
run and two parallel runs agree on EVERY outcome kind, not on counts.

Load it as a plugin on the run being measured::

    python -m pytest backend -p tests.support.outcome_vector   # + MRLN_OUTCOME_VECTOR=…

Compare vectors (exit 1 on any difference)::

    python -m tests.support.outcome_vector serial.json n8.json n16.json

Under xdist ``pytest_runtest_logreport`` fires in the controller for every
worker's report, so one file covers the whole run.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ENV = "MRLN_OUTCOME_VECTOR"

_OUTCOMES: dict[str, str] = {}


def _classify(report) -> str | None:
    if report.when == "call":
        if report.passed:
            return "xpass" if getattr(report, "wasxfail", None) is not None else "pass"
        if report.failed:
            return "fail"
        if report.skipped:
            return "xfail" if getattr(report, "wasxfail", None) is not None else "skip"
    elif report.when == "setup":
        if report.failed:
            return "error"
        if report.skipped:
            return "xfail" if getattr(report, "wasxfail", None) is not None else "skip"
    elif report.when == "teardown" and report.failed:
        return "error"
    return None


def pytest_runtest_logreport(report):
    kind = _classify(report)
    if kind is None:
        return
    # A teardown error overrides the call outcome; nothing overrides an error.
    if _OUTCOMES.get(report.nodeid) == "error":
        return
    _OUTCOMES[report.nodeid] = kind


def pytest_sessionfinish(session, exitstatus):
    target = os.environ.get(ENV)
    if not target or getattr(session.config, "workerinput", None) is not None:
        return  # unset, or an xdist worker (the controller writes)
    Path(target).write_text(json.dumps(_OUTCOMES, indent=0, sort_keys=True), encoding="utf-8")


def compare(paths: list[str]) -> int:
    vectors = {p: json.loads(Path(p).read_text(encoding="utf-8")) for p in paths}
    ids = [set(v) for v in vectors.values()]
    universe = set.union(*ids)
    rc = 0
    for p, v in vectors.items():
        counts = Counter(v.values())
        print(f"{p}: {len(v)} ids  " + "  ".join(f"{k}={counts[k]}" for k in
              ("pass", "fail", "skip", "xfail", "xpass", "error")))
        missing = universe - set(v)
        if missing:
            rc = 1
            print(f"  MISSING in {p}: {sorted(missing)[:20]} (+{max(0, len(missing) - 20)})")
    first = paths[0]
    for other in paths[1:]:
        diffs = [(nid, vectors[first].get(nid), vectors[other].get(nid))
                 for nid in sorted(universe) if vectors[first].get(nid) != vectors[other].get(nid)]
        if diffs:
            rc = 1
            print(f"DIFF {first} vs {other}: {len(diffs)}")
            for nid, a, b in diffs[:40]:
                print(f"  {nid}: {a} -> {b}")
    print("IDENTICAL" if rc == 0 else "DIFFERENT")
    return rc


if __name__ == "__main__":
    sys.exit(compare(sys.argv[1:]))
