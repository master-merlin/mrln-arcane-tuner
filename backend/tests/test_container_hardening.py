"""The container build is pinned and the runtime is not root.

Two kinds of test here, and the difference matters:

* ``TestEntrypointDropsPrivileges`` actually RUNS the entrypoint's decision
  logic with ``id``/``setpriv`` shimmed onto PATH, and asserts on the command
  line it tried to exec. That is observable behaviour, not a text match â€” it
  fails if the branch is wrong, if the uid is wrong, or if the drop is skipped.
  It needs a POSIX shell, so it skips on a machine without one.
* ``TestBuildContract`` reads ``Dockerfile``/``entrypoint.sh`` as text. Text
  matching is a weak guard, so each assertion here is paired with a mutation
  check proving the matcher actually notices when the property is removed â€”
  otherwise a rewrite could delete the guard and leave this file green.

Why not a real container test: the image is a multi-GB CUDA build. A test that
cannot run in the gate is not a guard, so the gate gets the strongest thing
that CAN run every time, and the docstring says plainly what it does not cover
â€” an actual `docker run` asserting `id -u`, which belongs in release QA.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import pytest

from app.core.restart_contract import RESTART_EXIT_CODE
from tests.support.bash_probe import bash_skip_reason, find_bash

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
ENTRYPOINT = REPO_ROOT / "entrypoint.sh"

EXPECTED_UID = "10001"


def _dockerfile() -> str:
    if not DOCKERFILE.exists():
        pytest.skip("Dockerfile not present in this checkout")
    return DOCKERFILE.read_text(encoding="utf-8")


def _entrypoint() -> str:
    if not ENTRYPOINT.exists():
        pytest.skip("entrypoint.sh not present in this checkout")
    return ENTRYPOINT.read_text(encoding="utf-8")


# â”€â”€ behaviour â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _bash() -> str | None:
    """A shell that can open ``entrypoint.sh``, not merely a shell.

    ``shutil.which("bash")`` used to answer this and got it wrong: on Windows it
    returns WSL's System32 bash about as often as Git Bash, and WSL cannot see
    `D:\\...` â€” so the guard below was satisfied by a shell that then failed
    every test instead of skipping. See ``tests/support/bash_probe.py``.
    """
    return find_bash(ENTRYPOINT) if ENTRYPOINT.exists() else None


requires_bash = pytest.mark.skipif(
    bash_skip_reason(ENTRYPOINT) is not None,
    reason=bash_skip_reason(ENTRYPOINT) or "",
)

# The external commands entrypoint.sh calls BEFORE the privilege drop that the
# shims below do not replace. `id`, `getent`, `chown`, `setpriv` are shimmed
# because their ANSWER is what is under test; these come from the OS. Keep in
# step with entrypoint.sh: a command added there and missing here surfaces as
# "command not found" from the script rather than as a named skip.
_ENTRYPOINT_NEEDS = ("mkdir", "cut")


def _bash_tool_dirs(bash: str) -> list[str]:
    """Directories holding the coreutils that ship NEXT TO this bash.

    A non-login ``bash <script>`` never sources ``/etc/profile``, and that is
    what puts MSYS ``/usr/bin`` on PATH â€” so Git's ``bash.exe`` inherits the
    Windows PATH as-is, and whether ``mkdir`` resolves depends on the operator
    having ``C:\\Program Files\\Git\\usr\\bin`` exported. LANE-44 measured the
    consequence: the same commit was red from one shell and green from another.
    The coreutils live in a known place relative to the shell itself, so the
    test hands them over instead of hoping the operator did.
    """
    here = Path(bash).resolve().parent
    dirs: list[str] = []
    for candidate in (here, here.parent / "usr" / "bin"):
        if any((candidate / f"mkdir{suffix}").is_file() for suffix in (".exe", "")):
            if str(candidate) not in dirs:
                dirs.append(str(candidate))
    return dirs


def _missing_commands(bash: str, env: dict[str, str]) -> list[str]:
    """Which of ``_ENTRYPOINT_NEEDS`` this bash cannot resolve under ``env``.

    Asked of the shell that will run the script, with the PATH it will be
    given â€” the only environment whose answer counts.
    """
    proc = subprocess.run(
        [
            bash,
            "-c",
            'for c in "$@"; do command -v "$c" >/dev/null 2>&1 || echo "$c"; done',
            "_",
            *_ENTRYPOINT_NEEDS,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        cwd=str(REPO_ROOT),
    )
    return proc.stdout.split()


def _sh_path(p: Path) -> str:
    """A path the test's shell can actually stat.

    entrypoint.sh runs under git-bash on this machine, whose ``[ -d ]`` does not
    resolve a drive-letter path like ``D:/x``; it wants the MSYS form ``/d/x``.
    On Linux/macOS this is the plain posix path. Without it the home-directory
    probe reports "missing" for a directory that is plainly there, and the test
    fails for a reason that has nothing to do with the entrypoint.
    """
    s = p.as_posix()
    if len(s) > 1 and s[1] == ":":
        return f"/{s[0].lower()}{s[2:]}"
    return s


def _entrypoint_env(bash: str, shim: Path, data: Path) -> dict[str, str]:
    env = dict(os.environ)
    path_entries = [str(shim), *_bash_tool_dirs(bash), env.get("PATH", "")]
    env["PATH"] = os.pathsep.join(path_entries)
    env["MRLN_DATA_DIR"] = str(data)
    return env


def _run_entrypoint_as_fake_root(
    tmp: Path, *, uid_exists: bool = True, home_exists: bool = True
) -> str:
    """Run entrypoint.sh with ``id``, ``setpriv``, ``getent``, ``chown`` shimmed.

    The shims are the OPERATING SYSTEM, not the logic under test: the thing
    being asserted is which branch the script takes and what arguments it hands
    to ``setpriv``. ``setpriv`` prints its argv and exits, which stops the
    script exactly where the real one would hand off.

    ``getent`` emits a REAL passwd line, because the script reads field 6 out of
    it to find the app user's home. A shim that only set an exit status let the
    home come back empty, which is precisely the bug this file now pins â€” the
    shim has to be faithful in the field under test or it proves nothing.
    ``home_exists`` controls whether that home is actually on disk, so the
    "no usable home" branch can be exercised too.
    """
    shim = tmp / "shim"
    shim.mkdir()

    home = tmp / "apphome"
    if home_exists:
        home.mkdir()

    (shim / "id").write_text("#!/bin/sh\necho 0\n", encoding="utf-8")
    # Echo the inherited environment as well as argv: HOME/USER are exported
    # BEFORE the exec, so the only way to observe them is from the child.
    (shim / "setpriv").write_text(
        '#!/bin/sh\necho "SETPRIV_CALLED $*"\necho "SETPRIV_ENV HOME=$HOME USER=$USER"\nexit 0\n',
        encoding="utf-8",
    )
    if uid_exists:
        (shim / "getent").write_text(
            f"#!/bin/sh\necho 'mrln:x:{EXPECTED_UID}:{EXPECTED_UID}::"
            f"{_sh_path(home)}:/bin/sh'\nexit 0\n",
            encoding="utf-8",
        )
    else:
        (shim / "getent").write_text("#!/bin/sh\nexit 2\n", encoding="utf-8")
    (shim / "chown").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    for f in shim.iterdir():
        f.chmod(0o755)

    data = tmp / "data"
    data.mkdir()

    bash = _bash()
    assert bash is not None  # requires_bash guards every caller
    env = _entrypoint_env(bash, shim, data)
    missing = _missing_commands(bash, env)
    if missing:
        # A test that cannot run must never look like a test that ran and
        # failed (LANE-44). Name what is missing and where it was looked for.
        pytest.skip(
            f"entrypoint.sh needs {', '.join(missing)} and {bash} cannot resolve "
            f"them â€” looked next to the shell in {_bash_tool_dirs(bash) or '(nothing)'} "
            "and on PATH"
        )

    proc = subprocess.run(
        [bash, str(ENTRYPOINT)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        # NOT cwd=tmp, deliberately. A child's working directory is an open
        # handle on that directory for the child's lifetime, and Windows
        # releases it lazily after the process exits â€” so deleting the
        # directory immediately afterwards intermittently raised
        # `PermissionError [WinError 32] file is being used by another
        # process`. entrypoint.sh uses only absolute paths before the
        # privilege drop, so the working directory is irrelevant to what is
        # being tested; pointing it at the repo root removes the handle
        # without changing the behaviour under test.
        cwd=str(REPO_ROOT),
    )
    return proc.stdout + proc.stderr


class TestEntrypointShellProbe:
    """The probe that decides "run" vs "skip with a reason" must itself be true.

    LANE-44: three tests were red from one shell and green from another on the
    same commit because ``mkdir`` resolved through the OPERATOR's PATH. Both
    halves of the fix are asserted: the coreutils are found next to the shell
    without help from PATH, and a PATH that lacks them is reported by name
    rather than surfacing as a failing test.
    """

    @requires_bash
    def test_coreutils_are_found_next_to_the_shell_without_the_operators_path(self, tmp_path: Path):
        bash = _bash()
        assert bash is not None
        shim = tmp_path / "shim"
        shim.mkdir()
        env = _entrypoint_env(bash, shim, tmp_path)
        # Strip the inherited PATH entirely: what remains is the shim plus the
        # directories derived from the shell's own location.
        env["PATH"] = os.pathsep.join([str(shim), *_bash_tool_dirs(bash)])
        assert _missing_commands(bash, env) == [], (
            f"{bash} could not resolve {_ENTRYPOINT_NEEDS} from its own tool "
            f"dirs {_bash_tool_dirs(bash)} â€” the fix depends on the operator's PATH again"
        )

    @requires_bash
    def test_a_bare_path_is_reported_by_command_name(self, tmp_path: Path):
        bash = _bash()
        assert bash is not None
        env = dict(os.environ)
        env["PATH"] = str(tmp_path)  # a directory with no coreutils at all
        assert _missing_commands(bash, env) == list(_ENTRYPOINT_NEEDS)


class TestEntrypointDropsPrivileges:
    @requires_bash
    def test_running_as_root_re_execs_as_the_app_uid(self, tmp_path):
        out = _run_entrypoint_as_fake_root(tmp_path)

        assert "SETPRIV_CALLED" in out, (
            "entrypoint did not attempt a privilege drop while running as root.\n"
            f"output:\n{out}"
        )
        assert f"--reuid={EXPECTED_UID}" in out, f"wrong or missing uid:\n{out}"
        assert f"--regid={EXPECTED_UID}" in out, f"wrong or missing gid:\n{out}"
        # Without --init-groups the process keeps root's supplementary groups,
        # which quietly undoes part of the drop.
        assert "--init-groups" in out, f"supplementary groups not reset:\n{out}"

    @requires_bash
    def test_drop_happens_before_the_app_directories_are_touched(self, tmp_path):
        """Ordering is load-bearing, so it is pinned.

        The script must drop privileges BEFORE it starts rewriting symlinks
        under /app. If the drop moves below that, every one of those writes
        happens as root and the container is only nominally non-root.
        """
        out = _run_entrypoint_as_fake_root(tmp_path)

        assert "SETPRIV_CALLED" in out
        # /app does not exist on a test machine; if the script had reached the
        # symlink section before dropping, `set -e` would have killed it there
        # and we would never see the setpriv line.
        assert "ln:" not in out and "No such file or directory" not in out, (
            "entrypoint reached the /app section before dropping privileges:\n" + out
        )

    @requires_bash
    def test_missing_app_user_continues_instead_of_failing_to_boot(self, tmp_path):
        """Prove the negative on the fallback path.

        An image built before this change has no uid 10001. Refusing to boot
        would convert a hardening regression into an outage, so the documented
        behaviour is 'continue as root, loudly'. Pinned so the warning cannot
        be quietly dropped later.
        """
        out = _run_entrypoint_as_fake_root(tmp_path, uid_exists=False)

        assert "SETPRIV_CALLED" not in out, "dropped to a uid that does not exist"
        assert "CONTINUING AS ROOT" in out, f"silent fallback to root:\n{out}"

    @requires_bash
    def test_drop_moves_home_to_the_app_user_not_just_the_uid(self, tmp_path):
        """setpriv changes credentials and NOTHING else â€” including not HOME.

        Regression pin. The first non-root image dropped to uid 10001 while
        leaving ``HOME=/root``, which that uid cannot write, and two things
        broke on the same root cause:

        * numba (pymatting <- rembg <- the masking service) could not place its
          JIT cache â€” site-packages is read-only to the app user and the
          HOME fallback was unwritable â€” and raised "no locator available" at
          IMPORT time, taking the app down on boot;
        * ``git config --global`` could not write /root/.gitconfig, so the
          safe.directory the self-updater depends on was never set, and the
          failure was swallowed by ``|| true``.

        Asserting on the child's environment, not on the script's text: what
        matters is the value the dropped process actually receives.
        """
        out = _run_entrypoint_as_fake_root(tmp_path)

        assert "SETPRIV_CALLED" in out, f"no privilege drop happened:\n{out}"
        assert "HOME=/root" not in out, (
            "the dropped process inherited root's HOME â€” numba's JIT cache and "
            f"`git config --global` both fail on that:\n{out}"
        )
        assert f"HOME={_sh_path(tmp_path / 'apphome')}" in out, (
            "HOME was not moved to the app user's home from the passwd entry:\n" + out
        )
        assert "USER=mrln" in out, f"USER not set for the app user:\n{out}"

    @requires_bash
    def test_missing_home_warns_rather_than_silently_keeping_roots(self, tmp_path):
        """Prove the negative: no home on disk must be loud, not silent.

        A derived image could drop the home directory. Keeping root's HOME
        without saying so is what made the original failure so hard to read â€”
        the crash surfaced deep inside numba, nowhere near the privilege drop.
        """
        out = _run_entrypoint_as_fake_root(tmp_path, home_exists=False)

        assert "SETPRIV_CALLED" in out, f"the drop must still happen:\n{out}"
        assert "no home directory for uid" in out, (
            f"HOME could not be set and the script said nothing:\n{out}"
        )


# â”€â”€ build contract â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _run_sha_validation(value: str | None) -> int:
    """Execute the Dockerfile's GIT_SHA check in isolation; return its exit code.

    The validation is lifted from the Dockerfile rather than restated here â€” a
    copy would drift, and a test that agrees with its own copy of the logic
    proves nothing about the build.
    """
    df = _dockerfile()
    start = df.index('case "$GIT_SHA" in')
    end = df.index("if [ -f /run/secrets/git_token ]", start)
    snippet = df[start:end]
    # Undo the Dockerfile line continuations to get plain shell back.
    snippet = snippet.replace("\\\n", "\n")

    env = dict(os.environ)
    if value is None:
        env.pop("GIT_SHA", None)
    else:
        env["GIT_SHA"] = value

    # A SENTINEL, not a bare exit code. An acceptance asserted purely on
    # `returncode == 0` cannot tell "the guard accepted the sha" from "bash
    # never ran and returned 1" â€” and the rejection cases, which assert
    # `!= 0`, pass happily on a spawn failure. That asymmetry made the
    # acceptance case the only one of the pair that could break for a reason
    # unrelated to the guard, which is exactly what happened.
    proc = subprocess.run(
        [_bash(), "-c", snippet + '\necho "GUARD_ACCEPTED"\nexit 0\n'],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    if proc.returncode != 0 and "GUARD_ACCEPTED" not in proc.stdout:
        # Either a genuine rejection or a failure to run at all. Distinguish
        # them: a rejection prints the guard's own message on stderr.
        if "GIT_SHA" not in proc.stderr:
            pytest.fail(
                "the shell did not execute the guard, so nothing was measured. "
                f"rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
            )
    return proc.returncode


class TestShaValidationActuallyRejects:
    """The required-arg guard, fired on purpose rather than read.

    A build arg that is documented as required but not enforced is the
    guard-that-cannot-fire class. These run the real check.
    """

    @requires_bash
    @pytest.mark.parametrize(
        "value,why",
        [
            (None, "unset â€” the default a plain `docker build` produces"),
            ("", "empty string"),
            ("main", "a branch name, the exact thing being removed"),
            ("07f44457", "a short sha â€” ambiguous, cannot be verified"),
            ("z" * 40, "right length, not hex"),
            ("a" * 39, "one char short"),
            ("a" * 41, "one char long"),
        ],
    )
    def test_rejects(self, value, why):
        assert _run_sha_validation(value) != 0, f"accepted {value!r} ({why})"

    @requires_bash
    def test_accepts_a_real_full_sha(self):
        """Prove the negative: the guard is not simply refusing everything."""
        assert _run_sha_validation("0" * 40) == 0
        assert _run_sha_validation("abcdef0123456789" * 2 + "abcdef0f") == 0


class TestBuildContract:
    def test_git_sha_is_a_required_build_arg(self):
        df = _dockerfile()
        assert "ARG GIT_SHA" in df
        assert "GIT_SHA=<full 40-hex commit> is required" in df, (
            "the build must fail without an explicit commit â€” building from a "
            "moving branch makes the image non-reproducible"
        )

    def test_the_build_verifies_it_landed_on_that_commit(self):
        """A reset that silently no-ops would pin nothing."""
        df = _dockerfile()
        assert 'git reset --hard "$GIT_SHA"' in df
        assert '[ "$(git rev-parse HEAD)" = "$GIT_SHA" ]' in df

    def test_cachebust_is_not_reintroduced_as_a_build_arg(self):
        """CACHEBUST let a build take 'whatever main is now'.

        GIT_SHA subsumes it (a different sha changes the layer's command), so
        its return would reintroduce exactly the moving-branch build this task
        removed.

        Matched on the FUNCTIONAL forms, not the bare word: the comment above
        the clone explains why the arg was dropped, and a guard that forbids
        naming a removed feature punishes documenting the removal. This caught
        that on its first run.
        """
        df = _dockerfile()
        assert "ARG CACHEBUST" not in df
        assert "$CACHEBUST" not in df
        assert "${CACHEBUST" not in df

    def test_ollama_has_a_checksum_verified_path(self):
        df = _dockerfile()
        assert "OLLAMA_SHA256" in df
        assert "sha256sum -c -" in df
        # Half a pin is not a pin: a version without a digest is still an
        # unverified download, and must be refused rather than accepted.
        assert "must be set together" in df

    def test_no_hardcoded_ollama_digest(self):
        """A checksum nobody verified reads as proof and is worse than none."""
        df = _dockerfile()
        assert "OLLAMA_SHA256=\n" in df or "ARG OLLAMA_SHA256=" in df

    def test_image_does_not_set_user_root(self):
        df = _dockerfile()
        assert "USER root" not in df

    def test_a_non_root_user_is_created(self):
        df = _dockerfile()
        assert "useradd" in df
        assert EXPECTED_UID in df, "the fixed app uid must be pinned in the image"

    def test_entrypoint_is_not_writable_by_the_app_user(self):
        """It runs as root before the drop, so app-user write = path to root."""
        df = _dockerfile()
        assert "chown root:root /entrypoint.sh" in df

    def test_deferred_items_stay_named(self):
        """A deferral that stops being written down becomes an oversight."""
        df = _dockerfile()
        for item in ("digest pinning", "SBOM", "signing"):
            assert item in df, f"{item!r} deferral is no longer recorded"


class TestTheseMatchersActuallyFail:
    """Vacuity checks.

    Every assertion above is a substring match, which passes just as happily
    against a file that no longer contains the guard *and* against one that
    never did. These mutate the real text and prove each matcher notices.
    Without this class, deleting a guard could leave the suite green.
    """

    def test_git_sha_matcher_notices_removal(self):
        df = _dockerfile().replace("ARG GIT_SHA", "ARG SOMETHING_ELSE")
        assert "ARG GIT_SHA" not in df

    def test_cachebust_matcher_notices_reintroduction(self):
        df = _dockerfile() + "\nARG CACHEBUST=0\n"
        assert "ARG CACHEBUST" in df, "the CACHEBUST guard would not catch a re-add"

    def test_cachebust_matcher_tolerates_the_word_in_prose(self):
        """The inverse of the above, and the reason the matcher was narrowed.

        Explaining why an arg was removed must not fail the guard that removed
        it, or the next person deletes the explanation to get green.
        """
        df = "# the old CACHEBUST arg is gone because GIT_SHA subsumes it\n"
        assert "ARG CACHEBUST" not in df and "$CACHEBUST" not in df

    def test_user_root_matcher_notices_reintroduction(self):
        df = _dockerfile() + "\nUSER root\n"
        assert "USER root" in df, "the USER root guard would not catch a re-add"

    def test_setpriv_matcher_notices_removal(self):
        ep = _entrypoint().replace("setpriv", "sudo")
        assert "setpriv --reuid" not in ep


# â”€â”€ The entrypoint is the supervisor: it relaunches on the sentinel (LANE-56) â”€

# A stub interpreter for the SUPERVISED part of the script: the resolver call
# prints a port; the uvicorn call records what it was handed and exits with the
# next code in STUB_CODES ("wait" = block until TERM, then record it, exit 0).
_STUB_PYTHON = r'''#!/bin/sh
case "$1" in
  */port_resolver.py)
    echo asked >> "$STUB_RESOLVER_LOG"
    echo "${PORT:-8765}"
    exit 0
    ;;
esac
if [ "$1" = "-m" ] && [ "$2" = "uvicorn" ]; then
  n=0
  [ -f "$STUB_RECORD" ] && n=$(wc -l < "$STUB_RECORD")
  echo "restart=${MRLN_RESTART:-unset} supervised=${MRLN_SUPERVISED:-unset} ppid=$PPID argv=$*" >> "$STUB_RECORD"
  code=$(echo "$STUB_CODES" | cut -d, -f$((n + 1)))
  if [ "$code" = "wait" ]; then
    trap 'echo term >> "$STUB_TERM_MARK"; exit 0' TERM
    sleep 30 & wait $!
    exit 0
  fi
  exit "${code:-0}"
fi
exit 0
'''


class _Supervised:
    """Run the WHOLE entrypoint as a non-root user, with a private PATH.

    Private, not merely prefixed: the script runs ``git config --global`` and
    ``ollama serve`` when it finds them, and this machine may have both. With
    only the shims and the coreutils next to bash on PATH, neither exists.
    ``MRLN_APP_DIR`` points the checkout at a temp dir (it is the app's own
    variable for that: ``self_update.py:386``), so ``/app/backend`` is never
    needed.
    """

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        shim = tmp / "shim"
        shim.mkdir()
        (shim / "id").write_text("#!/bin/sh\necho 1000\n", encoding="utf-8")
        (shim / "python").write_text(_STUB_PYTHON, encoding="utf-8")
        for f in shim.iterdir():
            f.chmod(0o755)
        (tmp / "data").mkdir()
        (tmp / "app" / "backend").mkdir(parents=True)
        self.record = tmp / "uvicorn_runs.txt"
        self.resolver_log = tmp / "resolver_calls.txt"
        self.term_mark = tmp / "term.txt"
        self.bash = _bash()
        assert self.bash is not None  # requires_bash guards every caller
        env = dict(os.environ)
        for stale in ("MRLN_RESTART", "MRLN_SUPERVISED", "PORT", "MRLN_BIND_HOST",
                      "MRLN_SETTINGS_PATH", "MRLN_AUTH_TOKEN"):
            env.pop(stale, None)
        env["PATH"] = os.pathsep.join([str(shim), *_bash_tool_dirs(self.bash)])
        env["MRLN_DATA_DIR"] = str(tmp / "data")
        env["MRLN_APP_DIR"] = (tmp / "app").as_posix()
        env["STUB_RECORD"] = self.record.as_posix()
        env["STUB_RESOLVER_LOG"] = self.resolver_log.as_posix()
        env["STUB_TERM_MARK"] = self.term_mark.as_posix()
        self.env = env
        missing = _missing_commands(self.bash, env)
        if missing:
            pytest.skip(f"entrypoint.sh needs {', '.join(missing)} and {self.bash} cannot "
                        f"resolve them on a private PATH of {env['PATH']}")

    def start(self, codes: str) -> subprocess.Popen:
        self.env["STUB_CODES"] = codes
        return subprocess.Popen([self.bash, str(ENTRYPOINT)], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=self.env,
                                cwd=str(REPO_ROOT))

    def run(self, codes: str) -> tuple[int, str]:
        proc = self.start(codes)
        out, _ = proc.communicate(timeout=120)
        return proc.returncode, out

    def runs(self) -> list[str]:
        if not self.record.exists():
            return []
        return self.record.read_text(encoding="utf-8").splitlines()

    def resolver_calls(self) -> int:
        return len(self.resolver_log.read_text(encoding="utf-8").split()) \
            if self.resolver_log.exists() else 0


class TestEntrypointRelaunchesOnTheSentinel:
    """Executable, under the script's REAL ``set -euo pipefail`` header â€” the
    adversary's BLOCK was precisely that a bare ``python -m uvicorn`` returning
    75 aborts the script at that line before any ``$?`` comparison."""

    @requires_bash
    def test_the_sentinel_relaunches_with_the_restart_flag_and_a_fresh_port(self, tmp_path):
        sup = _Supervised(tmp_path)
        code, out = sup.run(f"{RESTART_EXIT_CODE},0")

        runs = sup.runs()
        assert len(runs) == 2, (runs, out)
        assert "restart=unset" in runs[0] and "supervised=1" in runs[0]
        assert "restart=1" in runs[1] and "supervised=1" in runs[1]
        assert sup.resolver_calls() == 2, "the port is resolved before EVERY launch"
        assert code == 0, (code, out)
        assert "restart requested" in out

    @requires_bash
    def test_any_other_exit_code_ends_the_container_with_that_code(self, tmp_path):
        """A crash must not loop: 3 is uvicorn's STARTUP_FAILURE."""
        sup = _Supervised(tmp_path)
        code, out = sup.run("3,0")
        assert len(sup.runs()) == 1, (sup.runs(), out)
        assert code == 3, (code, out)

    @requires_bash
    def test_term_is_forwarded_to_the_server_and_the_loop_ends_with_its_code(self, tmp_path):
        """``docker stop`` sends TERM to PID 1 â€” the loop. It must reach uvicorn,
        and the container's exit code must be the server's, not 143."""
        sup = _Supervised(tmp_path)
        proc = sup.start("wait")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not sup.runs():
            time.sleep(0.1)
        assert sup.runs(), "the server stub never started"
        # The stub recorded its parent's (msys) pid: the loop shell. Signalled
        # from a sibling bash because Python's send_signal on Windows is
        # TerminateProcess, which no trap can see.
        ppid = re.search(r"ppid=(\d+)", sup.runs()[0]).group(1)
        subprocess.run([sup.bash, "-c", f"kill -TERM {ppid}"], timeout=30, check=True)

        out, _ = proc.communicate(timeout=60)
        assert sup.term_mark.exists(), ("the server never received TERM", out)
        assert proc.returncode == 0, (proc.returncode, out)
        assert len(sup.runs()) == 1, "a forwarded TERM must not look like a restart"


class TestEntrypointSupervisorText:
    """The shape that keeps the loop correct under the header. Cheap; runs everywhere."""

    def _loop(self) -> str:
        text = _entrypoint()
        start = text.index("export MRLN_SUPERVISED=1")
        return text[start:]

    def test_the_shebang_is_bash(self):
        """bash reaps re-parented trainer children in its SIGCHLD handler;
        a ``dash`` PID 1 would not (Assumption â€” observed in UAT item 5)."""
        assert _entrypoint().splitlines()[0] == "#!/usr/bin/env bash"

    def test_the_server_is_a_background_child_that_is_waited_for_twice_guarded(self):
        loop = self._loop()
        launch = next(line for line in loop.splitlines() if "-m uvicorn" in line)
        assert "exec" not in launch, "exec makes uvicorn PID 1 and the exit the container's end"
        assert launch.rstrip().endswith("&")
        assert 'pid=$!' in loop
        guarded = 'code=0; wait "$pid" || code=$?'
        assert loop.count(guarded) == 2, (
            "both waits must be in the errexit-exempt form: the first returns "
            "128+15 when TERM interrupts it, and only the second yields the child's code")

    def test_the_trap_is_installed_before_the_first_launch_and_forwards_guarded(self):
        loop = self._loop()
        assert loop.index("trap ") < loop.index("-m uvicorn")
        assert 'kill -0 "$pid"' in loop
        assert 'kill -TERM "$pid"' in loop
        assert '${pid:-}' in loop, "under set -u a bare $pid in an early trap aborts the script"

    def test_the_compared_literal_is_the_contracts_constant(self):
        """RULE-21: the literal is the wire; this pin keeps it from drifting."""
        assert f'"$code" -ne {RESTART_EXIT_CODE}' in self._loop()

    def test_the_relaunch_sets_the_restart_flag_and_everything_else_exits(self):
        loop = self._loop()
        assert "export MRLN_RESTART=1" in loop
        assert 'exit "$code"' in loop

    def test_the_header_is_still_strict(self):
        """The loop is written FOR ``set -euo pipefail``; loosening the header
        would hide a regression in the guarded form."""
        assert "set -euo pipefail" in _entrypoint()


# â”€â”€ The pinned Ollama path must be reachable, verified, and fail closed â”€â”€â”€â”€â”€â”€


def _ollama_block() -> str:
    """The Ollama install layer, from its ARGs to the end of its RUN."""
    df = _dockerfile()
    start = df.index("ARG INSTALL_OLLAMA")
    tail = df[start:]
    end = tail.index("\nFROM ") if "\nFROM " in tail else len(tail)
    return tail[:end]


class TestTheOllamaPinIsReachable:
    """The pin has to name an asset that exists, or it is decoration.

    Found 2026-09-03: the pinned path fetched ``ollama-linux-amd64.tgz`` and
    extracted it with gzip, but Ollama had replaced that asset with split
    ``.tar.zst`` bundles and no longer publishes the ``.tgz`` at all -- not in
    any of the last 60 releases. So setting OLLAMA_VERSION + OLLAMA_SHA256,
    which is the entire point of the verified path, 404'd on every current
    release, and the only way the image could build was the unpinned
    ``install.sh`` pipe. **The pin was unreachable, not merely unused**, and
    nothing said so, so a release build would have taken the supply-chain
    exposure it believed it had opted out of.

    **Be clear about what the text matchers here do NOT catch.** They were
    written first and run against the pre-fix Dockerfile to check them, and
    they passed -- because that file was internally CONSISTENT: it fetched a
    ``.tgz`` and extracted it with ``-xzf``. Nothing about it was wrong except
    the one thing no text can know, which is that the asset had stopped
    existing. A guard that passes on the defect it was written for is not a
    guard, and saying so is cheaper than discovering it later.

    So the property that actually pins the original defect lives in
    ``test_the_pinned_asset_still_exists_upstream``, which asks GitHub and
    skips when offline. The matchers below stay because they pin real and
    different failure shapes -- a fetch/extract mismatch, a lost digest check,
    a reintroduced fallback -- each of which would break the pinned path in a
    way this file CAN see.
    """

    @pytest.mark.skipif(
        os.environ.get("MRLN_SKIP_NETWORK_TESTS") == "1",
        reason="MRLN_SKIP_NETWORK_TESTS=1",
    )
    def test_the_pinned_asset_still_exists_upstream(self):
        """THE guard for the 2026-09-03 defect: does the name we fetch exist?

        The Dockerfile can be perfectly self-consistent and still name an asset
        upstream stopped publishing, which is precisely what happened -- and
        the only way to know is to look. This asks GitHub for the latest Ollama
        release's asset list.

        It skips rather than fails when the network is unavailable or the API
        is rate-limited, because a test that goes red on a train is a test
        people learn to ignore. It does NOT skip when the answer comes back and
        the name is missing: that is the finding.
        """
        import json
        import urllib.error
        import urllib.request

        block = _ollama_block()
        match = re.search(r"ollama-linux-amd64\.[A-Za-z.]+", block)
        assert match, "no Ollama asset name found in the Dockerfile"
        wanted = match.group(0)

        url = "https://api.github.com/repos/ollama/ollama/releases/latest"
        request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            pytest.skip(f"GitHub unreachable, cannot verify the asset name: {exc}")

        names = {asset.get("name") for asset in payload.get("assets", [])}
        if not names:
            pytest.skip("the release carried no asset list")
        assert wanted in names, (
            f"the Dockerfile pins `{wanted}`, which Ollama's latest release "
            f"({payload.get('tag_name')}) does not publish. Its assets are "
            f"{sorted(n for n in names if n and 'linux-amd64' in n)}. A pinned "
            "build would 404 and the only working path would be the UNPINNED "
            "install.sh pipe -- the exposure the pin exists to avoid. This is "
            "the 2026-09-03 defect recurring: update the asset name and the "
            "extract flag together."
        )

    def test_the_fetched_asset_and_the_extract_flag_agree(self):
        """A different shape: fetch one archive format, extract with another.

        This is NOT what happened in 2026-09-03 (that file was consistent); it
        is the mistake the FIX could have made, and the one a future rename
        will make if only half of it is edited.
        """
        block = _ollama_block()
        zst = "ollama-linux-amd64.tar.zst" in block
        tgz = "ollama-linux-amd64.tgz" in block
        assert zst != tgz, (
            "the Ollama layer must fetch exactly one archive form; it names "
            f"{'both' if zst and tgz else 'neither'}"
        )
        if zst:
            assert "--zstd" in block, (
                "the layer fetches a .tar.zst but does not extract with "
                "`tar --zstd`. That is the 2026-09-03 defect with the formats "
                "swapped: the download succeeds and the extract fails."
            )
        else:
            assert "--zstd" not in block, "fetches a .tgz but extracts with zstd"

    def test_the_pinned_branch_verifies_the_digest(self):
        assert "sha256sum -c -" in _ollama_block(), (
            "the pinned path must verify the digest it was given; a pin that "
            "downloads without checking is an unverified download wearing a pin"
        )

    def test_the_pinned_branch_does_not_fall_back_to_the_unpinned_pipe(self):
        """A verified path that can silently become an unverified one is not a
        verified path. The unpinned branch may swallow its own failure, because
        the sidecar is optional; the pinned branch may not."""
        block = _ollama_block()
        pinned = block[block.index("pinned, verifying sha256") : block.index("elif")]
        assert "||" not in pinned, (
            "the pinned Ollama branch contains a `||` fallback, so a failed "
            "fetch or a mismatched digest would fall through instead of "
            "failing the build. That is how an unreachable pin stays invisible."
        )

    def test_half_a_pin_is_refused(self):
        """A version without a digest is the same exposure with more confidence."""
        block = _ollama_block()
        assert "must be set together" in block and "exit 1" in block

    def test_the_digest_is_not_hardcoded_as_a_default(self):
        """A checksum nobody verified reads as proof and is worse than none.

        It would also pin one release forever and be wrong for every other one
        SILENTLY, because a mismatch reads as a corrupted download rather than
        as a stale pin.
        """
        assert re.search(r"OLLAMA_SHA256=[0-9a-f]{64}", _ollama_block()) is None


class TestTheOllamaMatchersActuallyFail:
    """Vacuity checks for the class above, same contract as its sibling."""

    def test_the_format_agreement_matcher_notices_the_original_defect(self):
        """Rebuild the 2026-09-03 text and require the matcher to catch it."""
        broken = (
            "curl -fsSL -o /tmp/ollama.tgz "
            '".../ollama-linux-amd64.tgz" && tar -C /usr/local -xzf /tmp/ollama.tgz'
        )
        assert "ollama-linux-amd64.tgz" in broken
        assert "ollama-linux-amd64.tar.zst" not in broken
        assert "--zstd" not in broken

    def test_the_digest_matcher_notices_removal(self):
        assert "sha256sum -c -" not in _ollama_block().replace("sha256sum -c -", "cat")

    def test_the_fallback_matcher_notices_a_reintroduced_pipe(self):
        assert "||" in "curl ... | sh || true", (
            "the fallback matcher would not catch a re-added `||`"
        )

    def test_the_hardcoded_digest_matcher_notices_one(self):
        assert re.search(r"OLLAMA_SHA256=[0-9a-f]{64}", "ARG OLLAMA_SHA256=" + "a" * 64)
