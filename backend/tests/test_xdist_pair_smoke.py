"""The pinned pytest + pytest-xdist pair really runs parallel workers (LANE-63).

``pytest-xdist`` is a test-only dependency (DECISION-27) and the gate invokes
``-n auto --dist loadgroup``. Two failure modes would let the gate quietly go
serial or quietly break the serialisation the GPU/port groups rely on:

* a future pytest bump that xdist's hooks no longer support — historically
  this surfaces as a plugin that registers but schedules nothing, or as
  ``-n`` being rejected outright;
* ``--dist loadgroup`` no longer honouring ``xdist_group`` — every "serialise"
  decision in step 4 of the plan depends on it.

So this runs a tiny self-contained suite as a real subprocess under the
project venv, with the real installed plugin, and asserts on what the workers
themselves wrote to disk: (a) both workers actually ran, (b) every test of one
``xdist_group`` landed on ONE worker, (c) without ``-n`` the same suite reports
the serial marker — proving the stamp observes the environment rather than
always saying "parallel".

The suite lives under ``tmp_path`` with its own ``pytest.ini`` so rootdir and
confcutdir stay there and none of the backend conftests load.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SUITE = '''
import os
import pytest

OUT = r"{out}"


def _stamp(name):
    with open(os.path.join(OUT, name + ".txt"), "w", encoding="utf-8") as fh:
        fh.write(os.environ.get("PYTEST_XDIST_WORKER", "serial"))


@pytest.mark.xdist_group("alpha")
@pytest.mark.parametrize("i", range(6))
def test_alpha(i):
    _stamp(f"alpha-{{i}}")


@pytest.mark.xdist_group("beta")
@pytest.mark.parametrize("i", range(6))
def test_beta(i):
    _stamp(f"beta-{{i}}")


@pytest.mark.parametrize("i", range(12))
def test_free(i):
    _stamp(f"free-{{i}}")
'''


def _run_suite(tmp_path: Path, *pytest_args: str) -> tuple[subprocess.CompletedProcess, dict[str, str]]:
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "pytest.ini").write_text("[pytest]\naddopts = -p no:cacheprovider\n", encoding="utf-8")
    (tmp_path / "test_smoke.py").write_text(textwrap.dedent(_SUITE.format(out=out)), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path), "-q", *pytest_args],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=300,
    )
    stamps = {p.stem: p.read_text(encoding="utf-8") for p in out.glob("*.txt")}
    return proc, stamps


def _by_prefix(stamps: dict[str, str], prefix: str) -> set[str]:
    return {worker for name, worker in stamps.items() if name.startswith(prefix)}


def test_two_workers_run_and_loadgroup_keeps_a_group_on_one_worker(tmp_path):
    proc, stamps = _run_suite(tmp_path, "-n", "2", "--dist", "loadgroup")
    assert proc.returncode == 0, (
        "the pinned pytest/xdist pair could not run `-n 2 --dist loadgroup`:\n"
        + proc.stdout[-2000:] + proc.stderr[-2000:]
    )
    assert len(stamps) == 24, f"expected 24 stamped tests, got {sorted(stamps)}"

    workers = set(stamps.values())
    assert "serial" not in workers, "a test ran without PYTEST_XDIST_WORKER — xdist did not schedule it"
    assert workers == {"gw0", "gw1"}, (
        f"both workers must actually execute tests; saw {sorted(workers)} — "
        "a single worker means the gate silently went serial"
    )
    assert len(_by_prefix(stamps, "alpha-")) == 1, f"xdist_group('alpha') split across {_by_prefix(stamps, 'alpha-')}"
    assert len(_by_prefix(stamps, "beta-")) == 1, f"xdist_group('beta') split across {_by_prefix(stamps, 'beta-')}"


def test_without_dash_n_the_same_suite_reports_serial(tmp_path):
    """Prove the negative for the instrument: the stamp is not always 'parallel'."""
    proc, stamps = _run_suite(tmp_path)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert len(stamps) == 24
    assert set(stamps.values()) == {"serial"}


def test_the_installed_xdist_is_the_pinned_one():
    """`requirements.txt` says 3.8.0; a drifted venv would test a different pair."""
    from importlib import metadata

    req = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8")
    pinned = next(
        (line.split("==", 1)[1].split("#", 1)[0].strip() for line in req.splitlines()
         if line.startswith("pytest-xdist==")),
        None,
    )
    assert pinned, "backend/requirements.txt no longer pins pytest-xdist=="
    try:
        installed = metadata.version("pytest-xdist")
    except metadata.PackageNotFoundError:
        pytest.fail("pytest-xdist is pinned in requirements.txt but not installed in this venv")
    assert installed == pinned, f"requirements.txt pins pytest-xdist=={pinned}, venv has {installed}"
