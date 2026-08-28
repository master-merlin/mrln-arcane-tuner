"""Guards for the bash probe, and for the habit that made it necessary.

LANE-16: `main` @ `52a2f197` measured 9 failed / 5686 passed in one session and
0 failed / 5695 passed in another, on the same commit. Totals reconcile, so it
was the same nine tests flipping on which `bash` won the operator's PATH race.

Reproduced here with the fix as the only variable — same machine, same tree,
System32 prepended so WSL's bash wins `shutil.which`:

    old selector, WSL-first PATH   ->   9 failed / 44 passed
    new selector, WSL-first PATH   ->  53 passed / 0 failed

The interesting test in this file is the last one. The probe is only half the
fix; the other half is that `shutil.which("bash")` must not come back, and no
amount of correct code elsewhere prevents someone reaching for the obvious call
again next year.
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

from tests.support import bash_probe
from tests.support.bash_probe import bash_candidates, bash_skip_reason, find_bash

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[1]
ENTRYPOINT = REPO_ROOT / "entrypoint.sh"


# ── the probe answers the right question ────────────────────────────────────


def test_a_missing_target_raises_instead_of_condemning_every_shell():
    """The trap this module exists to avoid, one level up.

    `test -f` on a path that is not there fails for every shell alive, so a
    probe that accepted a bad target would report "no usable bash on this
    machine" — a confident, wrong statement about the platform. It cost one
    debugging round while this module was being written, against
    `docker/entrypoint.sh`, which is not where entrypoint.sh lives.
    """
    with pytest.raises(FileNotFoundError, match="does not exist"):
        find_bash(REPO_ROOT / "no" / "such" / "file.sh")


def test_the_skip_reason_distinguishes_its_three_causes():
    """A skip that does not say why is how nine failures went unexplained twice."""
    missing = bash_skip_reason(REPO_ROOT / "not_here.sh")
    assert missing and "not present in this checkout" in missing

    if ENTRYPOINT.exists() and bash_candidates():
        # On a machine with a working shell there is no reason at all, and that
        # is the case that must not silently become a skip.
        assert bash_skip_reason(ENTRYPOINT) is None or "none can open" in bash_skip_reason(ENTRYPOINT)


def test_no_usable_bash_reports_the_candidates_it_tried(monkeypatch):
    """Prove the negative: the useless-shell path names names.

    Two fakes, because the two failures are different and must read
    differently — nothing found at all, versus shells found that cannot see
    the path.
    """
    if not ENTRYPOINT.exists():
        pytest.skip("entrypoint.sh not present in this checkout")

    monkeypatch.setattr(bash_probe, "bash_candidates", lambda: ())
    assert bash_skip_reason(ENTRYPOINT) == (
        "no bash on PATH or in a known Git installation"
    )

    monkeypatch.setattr(bash_probe, "bash_candidates", lambda: (r"C:\fake\bash.exe",))
    monkeypatch.setattr(bash_probe, "_can_see", lambda *a: False)
    reason = bash_skip_reason(ENTRYPOINT)
    assert "none can open" in reason
    assert r"C:\fake\bash.exe" in reason, "the skip does not say which shell it tried"


@pytest.mark.skipif(not bash_candidates(), reason="no bash on this machine")
def test_the_probe_agrees_with_the_shell_it_picked():
    """End-to-end: whatever comes back can really open the file.

    Asserting on the shell's own answer rather than on its install path — the
    whole point of a probe over an allowlist.
    """
    if not ENTRYPOINT.exists():
        pytest.skip("entrypoint.sh not present in this checkout")
    chosen = find_bash(ENTRYPOINT)
    if chosen is None:
        pytest.skip(bash_skip_reason(ENTRYPOINT) or "no usable bash")

    proc = subprocess.run(
        [chosen, "-c", 'if test -f "$1"; then echo SEES_IT; fi', "_", str(ENTRYPOINT)],
        capture_output=True, text=True, timeout=60,
    )
    assert "SEES_IT" in proc.stdout, (
        f"find_bash returned {chosen}, which cannot open {ENTRYPOINT}"
    )


def test_candidates_are_not_collapsed_to_the_first_hit(tmp_path, monkeypatch):
    """`shutil.which` returns one. Returning one IS the defect.

    If enumeration stopped at the PATH winner there would be nothing to fall
    back to, and a machine whose winner is WSL would have no usable shell at
    all — which is the failure this whole module exists to prevent.

    Built on a synthetic PATH rather than on whatever this machine happens to
    have installed: the property is about the enumeration, and a test that
    skips on single-bash machines would quietly stop covering it in CI.
    """
    name = "bash.exe" if os.name == "nt" else "bash"
    first, second = tmp_path / "one", tmp_path / "two"
    for d in (first, second):
        d.mkdir()
        (d / name).write_bytes(b"")

    monkeypatch.setenv("PATH", os.pathsep.join([str(first), str(second)]))
    bash_candidates.cache_clear()
    try:
        found = bash_candidates()
        assert str(first / name) in found, "the PATH winner is missing"
        assert str(second / name) in found, (
            "enumeration stopped at the first hit — the losing shell is "
            "invisible, so the probe has nothing to fall back to"
        )
        assert found.index(str(first / name)) < found.index(str(second / name)), (
            "candidates are not in PATH order"
        )
    finally:
        bash_candidates.cache_clear()


# ── the habit that caused it ────────────────────────────────────────────────


def _which_bash_calls(source: str) -> list[int]:
    """Line numbers of real ``which("bash")`` CALLS in *source*.

    Deliberately an AST walk and not a regex over lines. The regex version was
    written first and its very first run flagged this file's own docstring and
    the two comments explaining why the call was removed — "a guard that forbids
    naming a thing punishes documenting its removal", which is already in
    LESSONS twice. Comments and strings are not in the AST, so the question it
    answers is the one that matters: is anyone *calling* it.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "which":
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value == "bash":
            hits.append(node.lineno)
    return hits


def test_no_test_reaches_for_shutil_which_bash_again():
    """The mechanical half of the fix.

    A correct probe sitting in `support/` prevents nothing on its own: the next
    person needing a shell writes the obvious call, and the suite goes back to
    passing or failing according to whose machine ran it. There is no honest use
    of ``which("bash")`` in a test — the question is always *which bash can open
    this path*.
    """
    offenders = []
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        source = path.read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()
        for n in _which_bash_calls(source):
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{n}: {lines[n - 1].strip()}")

    assert not offenders, (
        "these pick a bash by PATH order rather than by whether it can open the "
        "path they hand it, which is LANE-16 returning:\n  "
        + "\n  ".join(offenders)
        + "\n\nUse tests.support.bash_probe.find_bash(<the path you will pass>)."
    )


def test_that_guard_can_actually_fail():
    """Positive control (CONVENTIONS rule 11), both directions.

    A collect-offenders test that never collected one is indistinguishable from
    a test that cannot collect. And since this one was narrowed *away* from
    prose, the second half matters as much as the first: it has to still catch
    the call, and it has to stay quiet about text that merely mentions it.
    """
    assert _which_bash_calls('bash = shutil.which("bash")') == [1]
    assert _which_bash_calls("bash = shutil.which('bash')") == [1]
    assert _which_bash_calls("from shutil import which\nb = which('bash')") == [2]

    assert _which_bash_calls('cmd = shutil.which("cmd")') == []
    assert _which_bash_calls("bash = find_bash(path)") == []
    # The three shapes that broke the regex version.
    assert _which_bash_calls('"""do not use shutil.which("bash") here."""') == []
    assert _which_bash_calls('# shutil.which("bash") used to answer this') == []
    assert _which_bash_calls("MSG = 'call shutil.which(\"bash\") no more'") == []
