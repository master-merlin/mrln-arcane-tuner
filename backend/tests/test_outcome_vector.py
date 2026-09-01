"""The outcome-vector plugin behind LANE-63's serial-vs-parallel acceptance.

A child pytest runs one test of every outcome kind with the plugin loaded and
the JSON it writes must name each kind — in particular xpass, which junitxml
files as a plain pass, and a teardown error, which overrides the call result.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]

_SUITE = '''
import pytest

def test_pass(): pass
def test_fail(): assert 0
def test_skip(): pytest.skip("s")
@pytest.mark.xfail(reason="x")
def test_xfail(): assert 0
@pytest.mark.xfail(reason="x")
def test_xpass(): pass
@pytest.fixture
def broken(): raise RuntimeError("setup")
def test_error(broken): pass
@pytest.fixture
def leaky():
    yield
    raise RuntimeError("teardown")
def test_teardown_error(leaky): pass
'''


def test_every_outcome_kind_is_named(tmp_path):
    (tmp_path / "test_kinds.py").write_text(_SUITE)
    out = tmp_path / "vec.json"
    env = dict(os.environ, MRLN_OUTCOME_VECTOR=str(out), PYTEST_XDIST_WORKER="gwVec")
    env["PYTHONPATH"] = str(BACKEND)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-p", "tests.support.outcome_vector", "--rootdir", str(tmp_path), str(tmp_path)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    assert out.exists(), proc.stdout[-2000:] + proc.stderr[-1000:]
    vec = {k.split("::")[-1]: v for k, v in json.loads(out.read_text()).items()}
    assert vec == {
        "test_pass": "pass", "test_fail": "fail", "test_skip": "skip",
        "test_xfail": "xfail", "test_xpass": "xpass", "test_error": "error",
        "test_teardown_error": "error",
    }, vec


def test_compare_reports_a_changed_outcome_and_a_missing_id(tmp_path, capsys):
    from tests.support.outcome_vector import compare

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"t::x": "pass", "t::y": "skip"}))
    b.write_text(json.dumps({"t::x": "fail"}))
    assert compare([str(a), str(b)]) == 1
    text = capsys.readouterr().out
    assert "t::x: pass -> fail" in text and "MISSING" in text
    b.write_text(json.dumps({"t::x": "pass", "t::y": "skip"}))
    assert compare([str(a), str(b)]) == 0
