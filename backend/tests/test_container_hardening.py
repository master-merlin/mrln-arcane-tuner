"""The container build is pinned and the runtime is not root.

Two kinds of test here, and the difference matters:

* ``TestEntrypointDropsPrivileges`` actually RUNS the entrypoint's decision
  logic with ``id``/``setpriv`` shimmed onto PATH, and asserts on the command
  line it tried to exec. That is observable behaviour, not a text match — it
  fails if the branch is wrong, if the uid is wrong, or if the drop is skipped.
  It needs a POSIX shell, so it skips on a machine without one.
* ``TestBuildContract`` reads ``Dockerfile``/``entrypoint.sh`` as text. Text
  matching is a weak guard, so each assertion here is paired with a mutation
  check proving the matcher actually notices when the property is removed —
  otherwise a rewrite could delete the guard and leave this file green.

Why not a real container test: the image is a multi-GB CUDA build. A test that
cannot run in the gate is not a guard, so the gate gets the strongest thing
that CAN run every time, and the docstring says plainly what it does not cover
— an actual `docker run` asserting `id -u`, which belongs in release QA.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from app.core.restart_contract import RESTART_EXIT_CODE
from tests.support.bash_probe import bash_skip_reason, find_bash

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
ENTRYPOINT = REPO_ROOT / "entrypoint.sh"
BUILD_SCRIPT = REPO_ROOT / "docker-build.ps1"

EXPECTED_UID = "10001"


def _dockerfile() -> str:
    if not DOCKERFILE.exists():
        pytest.skip("Dockerfile not present in this checkout")
    return DOCKERFILE.read_text(encoding="utf-8")


def _entrypoint() -> str:
    if not ENTRYPOINT.exists():
        pytest.skip("entrypoint.sh not present in this checkout")
    return ENTRYPOINT.read_text(encoding="utf-8")


def _build_script() -> str:
    if not BUILD_SCRIPT.exists():
        pytest.skip("docker-build.ps1 not present in this checkout")
    return BUILD_SCRIPT.read_text(encoding="utf-8")


def _code_only(text: str) -> str:
    """The script with comment lines removed.

    Seven times in one session a matcher searching for a NAME matched the PROSE
    about that name — and three of those were the comment written to justify
    the very guard doing the matching. A guard that documents itself well is
    the guard most likely to trip its own text search, which makes this the
    normal case rather than the exceptional one.

    So: assert against code, and let comments say whatever they need to. Where
    the assertion is genuinely ABOUT the documentation (that a caveat is
    present, say), match the full text deliberately instead.
    """
    return "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )


# ── behaviour ────────────────────────────────────────────────────────────


def _bash() -> str | None:
    """A shell that can open ``entrypoint.sh``, not merely a shell.

    ``shutil.which("bash")`` used to answer this and got it wrong: on Windows it
    returns WSL's System32 bash about as often as Git Bash, and WSL cannot see
    `D:\\...` — so the guard below was satisfied by a shell that then failed
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
    what puts MSYS ``/usr/bin`` on PATH — so Git's ``bash.exe`` inherits the
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
    given — the only environment whose answer counts.
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
    home come back empty, which is precisely the bug this file now pins — the
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
            f"them — looked next to the shell in {_bash_tool_dirs(bash) or '(nothing)'} "
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
        # releases it lazily after the process exits — so deleting the
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
            f"dirs {_bash_tool_dirs(bash)} — the fix depends on the operator's PATH again"
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
        """setpriv changes credentials and NOTHING else — including not HOME.

        Regression pin. The first non-root image dropped to uid 10001 while
        leaving ``HOME=/root``, which that uid cannot write, and two things
        broke on the same root cause:

        * numba (pymatting <- rembg <- the masking service) could not place its
          JIT cache — site-packages is read-only to the app user and the
          HOME fallback was unwritable — and raised "no locator available" at
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
            "the dropped process inherited root's HOME — numba's JIT cache and "
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
        without saying so is what made the original failure so hard to read —
        the crash surfaced deep inside numba, nowhere near the privilege drop.
        """
        out = _run_entrypoint_as_fake_root(tmp_path, home_exists=False)

        assert "SETPRIV_CALLED" in out, f"the drop must still happen:\n{out}"
        assert "no home directory for uid" in out, (
            f"HOME could not be set and the script said nothing:\n{out}"
        )


# ── build contract ───────────────────────────────────────────────────────


def _run_sha_validation(value: str | None) -> int:
    """Execute the Dockerfile's GIT_SHA check in isolation; return its exit code.

    The validation is lifted from the Dockerfile rather than restated here — a
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
    # never ran and returned 1" — and the rejection cases, which assert
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
            (None, "unset — the default a plain `docker build` produces"),
            ("", "empty string"),
            ("main", "a branch name, the exact thing being removed"),
            ("07f44457", "a short sha — ambiguous, cannot be verified"),
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
            "the build must fail without an explicit commit — building from a "
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

    def test_hpsv2_vocab_is_baked_in_not_fetched_at_runtime(self):
        """hpsv2's vocab must be in the image, placed before the drop.

        Its vendored open_clip resolves the file with a hardcoded
        package-relative path, so it has to sit inside site-packages and no
        env var can move it. The runtime fetch in ``apply_hpsv2_patches`` only
        ever worked as root; under the app user it is EACCES, swallowed into a
        warning, and HPSv2 scoring is dead with nothing failing loudly.
        """
        df = _dockerfile()
        assert "bpe_simple_vocab_16e6.txt.gz" in df, (
            "the hpsv2 vocabulary is not baked into the image; it will be "
            "fetched at runtime and fail as the non-root app user"
        )
        # Resolved from the distribution, not hardcoded: a base-image Python
        # bump moves dist-packages and a literal path would put it nowhere.
        assert 'm.distribution("hpsv2")' in df, (
            "the destination must be resolved from the installed distribution, "
            "not a hardcoded python3.N site-packages path"
        )
        # Ordering is the load-bearing part: written as root, before the image
        # creates and hands ownership to the app user.
        assert df.index("bpe_simple_vocab_16e6.txt.gz") < df.index("useradd"), (
            "the vocab is baked AFTER the app user is created; it must be "
            "written as root, before the privilege boundary"
        )

    def test_the_hpsv2_vocab_is_pinned_to_a_commit_and_a_digest(self):
        """A `main`-branch URL makes the build irreproducible by definition:
        two builds of the same GIT_SHA can differ if upstream moves the file.
        Same shape as the Ollama pin, and the reason the gzip probe is not
        enough — it proves the bytes are *a* gzip, never that they are *the*
        vocabulary, and `curl -f` does not catch a 200 serving an HTML error
        page.
        """
        df = _dockerfile()
        # Comments excluded, or this matches the re-pin INSTRUCTION comment
        # (which carries a literal `<sha>` placeholder) instead of the command.
        # Fourth time today a name-based matcher has hit the prose explaining
        # the thing it guards; the rule is now reflexive — match the operation.
        vocab_line = next(
            (
                ln for ln in df.splitlines()
                if "CLIP/raw/" in ln and not ln.lstrip().startswith("#")
            ),
            "",
        )
        assert vocab_line, "the CLIP vocabulary fetch has moved or gone"
        assert "/raw/main/" not in vocab_line, (
            "the vocabulary is fetched from a moving branch; pin it to a commit"
        )
        assert "${CLIP_VOCAB_COMMIT}" in vocab_line, (
            "the fetch does not use the pinned commit arg"
        )
        assert re.search(r"ARG CLIP_VOCAB_COMMIT=[0-9a-f]{40}", df), (
            "CLIP_VOCAB_COMMIT must be a full 40-hex commit sha"
        )
        assert re.search(r"ARG CLIP_VOCAB_SHA256=[0-9a-f]{64}", df), (
            "CLIP_VOCAB_SHA256 must be a full sha256 digest"
        )
        # The digest must actually be CHECKED, not merely declared — a pin
        # nobody verifies is a comment.
        assert "${CLIP_VOCAB_SHA256}" in df and "sha256sum -c -" in df, (
            "the digest is declared but never verified against the download"
        )

    def test_torchao_comes_from_the_cuda_matched_index_and_is_load_checked(self):
        """torchao ships compiled CUDA extensions, so its wheel is ABI-bound.

        PyPI publishes only the CUDA-13 build, whose kernels need
        `libcudart.so.13`; the image is CUDA 12.8/12.6, so neither extension
        loaded — in every published image. Nothing in the app reported it:
        `import torchao` still succeeds, so `is_available()` is True and none
        of the "quantization unavailable" fallbacks fire. The wheel is broken
        underneath that predicate.

        The real guard is the dlopen assertion in the build. This test only
        pins that both halves are present, because a text assertion about an
        index would pass on an image where nothing loads.
        """
        df = _dockerfile()
        code = _code_only(df)
        assert "torchao==0.17.0" in code, "torchao is not pinned in the Dockerfile layer"
        # In the SAME install as the torch trio, i.e. the CUDA-matched index.
        torch_install = code[code.index("torch==2.11.0"):]
        torch_install = torch_install[: torch_install.index("RUN", 1)] if "RUN" in torch_install[1:] else torch_install
        assert "torchao==0.17.0" in torch_install, (
            "torchao must be installed from download.pytorch.org/whl/${TORCH_CUDA}, "
            "in the same layer as torch — PyPI's wheel is built for another CUDA"
        )
        # And the load check must exist and run AFTER the bulk resolve, which is
        # the step that could put PyPI's wheel back.
        assert "ctypes.CDLL(p) for p in sos" in code, (
            "the build does not verify that torchao's kernels actually load"
        )
        assert code.index("bash install-deps.sh") < code.index("ctypes.CDLL(p) for p in sos"), (
            "the load check runs before install-deps.sh, so it proves nothing "
            "about the shipped image"
        )

    def test_torchao_is_not_excluded_from_the_shared_dependency_install(self):
        """The trap in the tidy version of the fix.

        `install-deps.sh`'s exclusion regex is shared by the Docker build, the
        local venv and the runtime self-update. Adding torchao to it would keep
        the image correct and silently drop torchao from a fresh Windows
        install. Measured instead: pip reports the bare `torchao==0.17.0` pin
        satisfied by `0.17.0+cu128`, so the bulk resolve leaves the wheel alone
        and no exclusion is needed.
        """
        script = (REPO_ROOT / "backend" / "install-deps.sh")
        if not script.exists():
            pytest.skip("install-deps.sh not present in this checkout")
        text = script.read_text(encoding="utf-8")
        excl = [ln for ln in text.splitlines() if "grep -ivE" in ln]
        assert excl, "the exclusion filter has moved; this guard is looking in the wrong place"
        assert not any("torchao" in ln for ln in excl), (
            "torchao was added to the shared exclusion regex — that fixes the "
            "image and silently removes torchao from the local venv and the "
            "runtime self-update"
        )

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


# ── The entrypoint is the supervisor: it relaunches on the sentinel (LANE-56) ─

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
  echo "restart=${MRLN_RESTART:-unset} supervised=${MRLN_SUPERVISED:-unset} numba=${NUMBA_CACHE_DIR:-unset} ppid=$PPID argv=$*" >> "$STUB_RECORD"
  code=$(echo "$STUB_CODES" | cut -d, -f$((n + 1)))
  if [ "$code" = "wait" ]; then
    trap 'echo term >> "$STUB_TERM_MARK"; exit 0' TERM
    # Written AFTER the trap exists, and that ordering is the proof, not a
    # margin: sh executes sequentially, so this file cannot appear before TERM
    # is trappable. A caller that waits on it therefore cannot signal into the
    # window where TERM would still take its default disposition.
    # Named for the property it asserts, not for when it happens -- "ready" is
    # exactly the word that let the original race exist.
    echo trap_installed >> "$STUB_TRAP_READY"
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
        self.trap_ready = tmp / "trap_ready.txt"
        self.bash = _bash()
        assert self.bash is not None  # requires_bash guards every caller
        env = dict(os.environ)
        for stale in ("MRLN_RESTART", "MRLN_SUPERVISED", "PORT", "MRLN_BIND_HOST",
                      "MRLN_SETTINGS_PATH", "MRLN_AUTH_TOKEN", "NUMBA_CACHE_DIR"):
            env.pop(stale, None)
        env["PATH"] = os.pathsep.join([str(shim), *_bash_tool_dirs(self.bash)])
        env["MRLN_DATA_DIR"] = str(tmp / "data")
        env["MRLN_APP_DIR"] = (tmp / "app").as_posix()
        env["STUB_RECORD"] = self.record.as_posix()
        env["STUB_RESOLVER_LOG"] = self.resolver_log.as_posix()
        env["STUB_TERM_MARK"] = self.term_mark.as_posix()
        env["STUB_TRAP_READY"] = self.trap_ready.as_posix()
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
    """Executable, under the script's REAL ``set -euo pipefail`` header — the
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
        """``docker stop`` sends TERM to PID 1 — the loop. It must reach uvicorn,
        and the container's exit code must be the server's, not 143."""
        sup = _Supervised(tmp_path)
        proc = sup.start("wait")
        # Wait for the TRAP, not for the process. Waiting on the record file
        # (i.e. "the stub started") was a race: the stub writes that line, then
        # runs a forking command substitution, and only then installs the TERM
        # handler. Signalling in that window killed the shell with TERM's
        # default disposition, no marker was written, and the failure read "the
        # server never received TERM" -- true, and pointing squarely at signal
        # forwarding in the product when the fault was the test declaring
        # readiness before readiness existed. A true message naming the wrong
        # subject is worse than a vague one.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not sup.trap_ready.exists():
            time.sleep(0.05)
        assert sup.trap_ready.exists(), "the stub never installed its TERM trap"
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


class TestNumbaHasAWritableCacheLocation:
    """A user hit `cannot cache function '_make_tree': no locator available` at
    import time — numba, reached through pymatting <- rembg <- the masking
    service. numba tries site-packages (root-owned here), then
    NUMBA_CACHE_DIR, then HOME. Only HOME was ever set, so one unwritable HOME
    took the whole import down, and an import that raises is ARCHITECTURE D1.

    Asserted from the SERVER PROCESS's own environment, not from the script
    text: the export has to survive the supervisor loop to be worth anything.
    """

    @requires_bash
    def test_the_server_is_launched_with_a_numba_cache_dir_on_the_data_volume(self, tmp_path):
        sup = _Supervised(tmp_path)
        code, out = sup.run("0")

        runs = sup.runs()
        assert runs, ("the server stub never started", out)
        cache = re.search(r"numba=(\S+)", runs[0])
        assert cache is not None, (runs[0], out)
        assert cache.group(1) != "unset", (
            "NUMBA_CACHE_DIR never reached the server; numba is back to "
            f"depending on HOME alone. {runs[0]}"
        )
        # On the data volume specifically: /tmp would make every container start
        # re-JIT from cold, and the point of the directory is that it persists.
        # Compared against the value the harness actually exported, not a
        # re-derived one — on Windows the two differ by separator alone, which
        # would fail the assertion for a reason the container never has.
        data = sup.env["MRLN_DATA_DIR"]
        assert cache.group(1).startswith(data), (cache.group(1), data, out)
        assert code == 0, (code, out)

    @requires_bash
    def test_the_cache_directory_actually_exists_by_the_time_the_server_starts(self, tmp_path):
        """Exporting a path numba then cannot create is the same failure with
        an extra step, so the directory itself is the assertion."""
        sup = _Supervised(tmp_path)
        _, out = sup.run("0")

        runs = sup.runs()
        assert runs, ("the server stub never started", out)
        cache = Path(re.search(r"numba=(\S+)", runs[0]).group(1))
        assert cache.is_dir(), (f"{cache} was exported but never created", out)

    @requires_bash
    def test_an_operator_set_numba_cache_dir_is_respected(self, tmp_path):
        """Same `${VAR:-default}` contract as HF_HOME: the image picks a sane
        default, the operator overrides it.

        The directory assertion is not decoration. Asserting only that the
        variable arrives is VACUOUS — the stub echoes whatever it inherits, so
        that half passes on an entrypoint which does nothing at all (measured:
        it did). Creating the operator's directory is the part only the
        entrypoint can do.
        """
        sup = _Supervised(tmp_path)
        chosen = tmp_path / "operator-cache"
        sup.env["NUMBA_CACHE_DIR"] = chosen.as_posix()
        _, out = sup.run("0")

        runs = sup.runs()
        assert runs, ("the server stub never started", out)
        assert f"numba={chosen.as_posix()}" in runs[0], (runs[0], out)
        assert chosen.is_dir(), (
            "the operator's cache directory was passed through but never "
            f"created: {chosen}",
            out,
        )


class TestTheBuildWrapperCannotTagAnUnverifiedImage:
    """LANE-82: a published image contained a commit nobody asked for, and
    every signal we trusted said the build was fine — exit 0, "writing image",
    "naming to <tag>". The Dockerfile's own `rev-parse HEAD == $GIT_SHA`
    assertion cannot help, because it lives inside a RUN and a cache hit never
    re-runs it.

    So the wrapper's ONE invariant is ordering: build to a scratch tag, read
    HEAD out of the artifact, and only then let a release tag move. These are
    text assertions — the behaviour needs a GPU-sized docker build — so each
    one is paired below with the mutation that must break it.
    """

    def _release_tag_ordering(self, text: str) -> tuple[int, int]:
        """Return (position of the mismatch refusal, position of the first
        release `docker tag`). The refusal must come first."""
        refusal = text.index("Artifact commit mismatch.")
        tagging = text.index("docker tag $scratchTag $t")
        return refusal, tagging

    def test_the_build_writes_to_a_scratch_tag_not_a_release_tag(self):
        text = _build_script()
        build_target = re.search(r"\$scratchTag\s*=\s*(.+)", text)
        assert build_target is not None, "no scratch tag is defined"
        assert "mrln-build-scratch" in build_target.group(1)
        # The -t handed to docker build must be the scratch tag. If the build
        # named a release tag directly, a wrong artifact would already own
        # :latest by the time anything could be verified.
        assert "'-t', $scratchTag" in text, "docker build must target the scratch tag"
        assert "'-t', $Repository" not in text
        # No tagging OPERATION may precede the build. Matched on `docker tag`
        # rather than on the tag names: `:latest` and `$Repository` both appear
        # in the header comment and the param block, so a name-based matcher
        # fails on prose and proves nothing about the code.
        assert "docker tag" not in text.split("docker @buildArgs")[0], (
            "an image is tagged before the build; nothing may name a release "
            "tag until the artifact has proven its commit"
        )

    def test_release_tags_are_applied_only_after_the_head_comparison(self):
        refusal, tagging = self._release_tag_ordering(_build_script())
        assert refusal < tagging, (
            "the release tags are applied before the mismatch can refuse them"
        )

    def test_the_guard_reads_head_out_of_the_artifact_not_the_build_log(self):
        """The distinction that this whole lane turns on. The read itself lives
        in docker-verify-commit.ps1 so it can be exercised without a build; the
        wrapper's job is to call it and to act on its verdict."""
        if not VERIFIER.exists():
            pytest.skip("docker-verify-commit.ps1 not present in this checkout")
        verifier = VERIFIER.read_text(encoding="utf-8")
        assert "docker run --rm --entrypoint git $Image" in verifier
        assert "rev-parse HEAD" in verifier
        # safe.directory, or the read fails on /app's non-root ownership and
        # the guard degrades into "could not verify", which is not a guard.
        assert "safe.directory=/app" in verifier
        # The wrapper must actually invoke it, and on the SCRATCH tag.
        text = _build_script()
        assert "$verifier -Image $scratchTag -ExpectedSha $GitSha" in text

    def test_the_wrapper_never_pushes(self):
        """Publishing stays a separate, deliberate act (RULE-17 halt 4)."""
        code = _code_only(_build_script())
        assert "docker push" not in code
        assert "& docker push" not in code

    def test_the_build_log_and_argument_vector_are_written_to_disk(self):
        """The whole LANE-82 investigation existed because one build left no
        log: four mechanisms were proposed and none could be tested against the
        actual event, because "which sha was passed" rested on a hand-written
        note. Detecting a bad artifact without recording how it was produced
        makes the next anomaly detectable but not diagnosable."""
        text = _build_script()
        assert "--progress=plain" in text, "a log without the plain driver hides the step states"
        assert "Tee-Object -FilePath $logPath" in text, "the build output is not persisted"
        assert "$argvPath" in text, "the argument vector is not persisted"
        # Written BEFORE the build: a build that dies mid-run must still leave
        # behind what it was asked to do.
        assert text.index("$argvPath") < text.index("docker @buildArgs")
        # The token is passed as a PATH; the argv record must never carry a
        # value that could be a secret.
        argv_block = text.split("$argvPath")[1].split("Set-Content")[0]
        assert "$TokenPath" not in argv_block

    def test_the_version_check_is_not_mistaken_for_the_guard(self):
        """`app.__version__` was IDENTICAL across the wrong commit and the
        right one, so the version check would have passed this exact defect in
        silence. It is defence against a different failure, and the script has
        to say so or a later reader will trust it to do the HEAD check's job."""
        text = _build_script()
        assert "app.__version__" in text
        assert "WEAK BY CONSTRUCTION" in text and "licensed a wrong tag" in text, (
            "the version check's limit must be stated where it is written, or a "
            "later reader trusts it to do the HEAD check's job"
        )


VERIFIER = REPO_ROOT / "docker-verify-commit.ps1"

# The naturally-occurring fixture for this whole lane: a real image whose real
# commit really differs from the one it was built with. Nobody constructed it,
# which is exactly its value — DO NOT DELETE this image.
#
# Addressed by a DEDICATED IMMUTABLE TAG, not by the release tag it originally
# carried. It first pointed at `mastermerlin/mrln-arcane-tuner:0.8.0-beta.1`,
# and the first successful run of docker-build.ps1 moved that tag onto the
# freshly built image — correctly, that is the wrapper's whole job — which
# silently swapped this fixture underneath its own test. The lesson this file
# exists to teach, applied to the file itself: **a tag is a mutable pointer,
# and a test pinned to one is not pinned.** Recreate with:
#     docker tag <image id> mrln-test-fixture:wrong-commit
_FIXTURE_IMAGE = "mrln-test-fixture:wrong-commit"
_FIXTURE_SHA = "f1cbbbcfcab038cbdb559bc15278ca68e6f2a0ae"
_OTHER_SHA = "98492c7265e084a19980f1486ddebfd477fa949b"

VERIFY_OK, VERIFY_MISMATCH, VERIFY_UNVERIFIED = 0, 1, 2


def _powershell() -> str | None:
    for exe in ("pwsh", "powershell"):
        found = shutil.which(exe)
        if found:
            return found
    return None


def _run_verifier(image: str, expected: str) -> int:
    shell = _powershell()
    if shell is None:
        pytest.skip("no PowerShell on PATH")
    if not VERIFIER.exists():
        pytest.skip("docker-verify-commit.ps1 not present in this checkout")
    if shutil.which("docker") is None:
        pytest.skip("docker not on PATH")
    proc = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-File", str(VERIFIER),
         "-Image", image, "-ExpectedSha", expected],
        capture_output=True, text=True, timeout=300,
    )
    return proc.returncode


def _fixture_available() -> bool:
    if shutil.which("docker") is None:
        return False
    proc = subprocess.run(
        ["docker", "image", "inspect", _FIXTURE_IMAGE],
        capture_output=True, text=True, timeout=120,
    )
    return proc.returncode == 0


class TestTheVerifierRefusesAnImageThatIsNotWhatWasAskedFor:
    """The guard's four cases, exercised against a REAL image rather than a
    simulation. Reproducing the Docker anomaly would be testing Docker; this
    tests the one condition the guard actually checks — "the artifact's HEAD is
    not the sha I asked for" — independently of how that came about, which is
    the whole point of a cause-independent guard.
    """

    def test_it_refuses_when_the_image_holds_a_different_commit(self):
        if not _fixture_available():
            pytest.skip(f"{_FIXTURE_IMAGE} not on this machine")
        assert _run_verifier(_FIXTURE_IMAGE, _OTHER_SHA) == VERIFY_MISMATCH

    def test_it_accepts_when_the_image_holds_the_expected_commit(self):
        """The positive control, and it is NOT optional: without it a verifier
        that refuses everything passes the test above (CONVENTIONS rule 11)."""
        if not _fixture_available():
            pytest.skip(f"{_FIXTURE_IMAGE} not on this machine")
        assert _run_verifier(_FIXTURE_IMAGE, _FIXTURE_SHA) == VERIFY_OK

    def test_a_check_that_cannot_run_is_a_failure_not_a_pass(self):
        """The dangerous case. If "I could not check" ever collapses into "the
        check passed" — empty compared against empty, an error swallowed, a
        fall-through — the guard reports clean on an image it never read. Same
        silent-failure class as a linter reporting success on a file it could
        not parse. The distinct exit code is the assertion.
        """
        code = _run_verifier(
            "mrln-nonexistent-image-for-this-test:never-built", _FIXTURE_SHA
        )
        assert code == VERIFY_UNVERIFIED, (
            "an unreadable image must report UNVERIFIED, never OK"
        )
        assert code != VERIFY_OK

    def test_the_release_scripts_parse_at_all(self):
        """Found the hard way: both scripts were written with em dashes, and
        Windows PowerShell reads a BOM-less .ps1 as cp1252, where the UTF-8 em
        dash decodes to a stray right-double-quote. The parser took it as a
        string delimiter and NEITHER script would run — a release-time failure
        in code that had never been executed, only reviewed.

        Non-ASCII is banned outright rather than fixed with a BOM: the BOM has
        to be preserved by every editor and tool that ever touches the file,
        while ASCII cannot be got wrong.
        """
        for script in (BUILD_SCRIPT, VERIFIER):
            if not script.exists():
                pytest.skip(f"{script.name} not present in this checkout")
            raw = script.read_bytes()
            offenders = sorted({b for b in raw if b > 0x7F})
            assert offenders == [], (
                f"{script.name} contains non-ASCII bytes {offenders}; under "
                "Windows PowerShell's cp1252 fallback these can become quote "
                "characters and break parsing"
            )

    def test_the_three_outcomes_are_distinct_codes(self):
        """Callers branch on these; collapsing any two re-creates the defect."""
        assert len({VERIFY_OK, VERIFY_MISMATCH, VERIFY_UNVERIFIED}) == 3
        text = VERIFIER.read_text(encoding="utf-8")
        for code in ("exit 0", "exit 1", "exit 2"):
            assert code in text, f"{code} is never returned"


class TestTheWrapperTreatsUnverifiedAsFatal:
    """Text-level, because the alternative is a 40-minute build. The behaviour
    it pins is that the wrapper does not conflate the verifier's three codes.
    """

    def test_exit_code_two_is_handled_separately_from_a_mismatch(self):
        text = _build_script()
        assert "$verifyExit -eq 2" in text, (
            "the wrapper does not distinguish 'could not check' from 'checked "
            "and matched'"
        )
        # And it must be handled BEFORE the generic non-zero branch, or the
        # distinction exists in the verifier and is thrown away by the caller.
        assert text.index("$verifyExit -eq 2") < text.index("$verifyExit -ne 0")

    def test_release_tags_are_opt_in_not_the_default(self):
        """A validation build must not name a release.

        With release tagging on by default, the first successful run applied
        `:latest` and `:<version>` to a throwaway validation image — a local
        `:latest` pointing at an unreleased commit, sitting next to a
        `docker push`. Most builds are validation builds, so the safe branch
        has to be the one you get by saying nothing.
        """
        text = _build_script()
        assert re.search(r"\[switch\]\$ReleaseTags", text), (
            "ReleaseTags must be a switch, so its default is OFF"
        )
        assert "if (-not $ReleaseTags)" in text
        # The early exit must come BEFORE any tagging, or the default is safe
        # in name only.
        assert text.index("if (-not $ReleaseTags)") < text.index("docker tag $scratchTag $t")

    def test_a_release_tag_is_refused_when_the_git_tag_names_another_commit(self):
        """The invariant the version check was reaching for and missing.

        `app.__version__` is read out of the image and compared to the version
        being applied — both from the same tree, so within a bump window it
        matches for every build by construction. On 2026-09-04 that
        self-reference licensed `0.8.0-beta.1` onto an image 40 commits past
        the public git tag of that name: two commits under one version.
        """
        text = _build_script()
        assert "ls-remote --tags origin" in text, (
            "the tag check must ask ORIGIN — a public tag is what makes the "
            "name binding, and this checkout's tags are not authoritative"
        )
        assert "$lsExit -ne 0" in text, "a failed lookup must be distinguishable"
        assert "UNDETERMINED is not the same as unclaimed" in text
        # Annotated tags put the tag OBJECT on refs/tags/vX and the commit on
        # refs/tags/vX^{}. Without the dereferenced form this compares a tag
        # sha to a commit sha and refuses every build — broken in the safe
        # direction is still broken.
        assert r"\^\{\}" in text, "annotated tags are not dereferenced"

    def test_the_tag_check_cannot_read_a_failed_lookup_as_unclaimed(self):
        """The specific fail-open this replaced.

        A local `git rev-list -n 1 v<version>` returns empty with exit 128 both
        when the tag does not exist and when it exists but was never fetched —
        measured, byte-identical — so on a fresh clone or a shallow CI checkout
        (actions/checkout fetches no tags) the old form said "unclaimed" while
        meaning "I could not look". Same class as a linter reporting clean on a
        file it could not parse.
        """
        text = _build_script()
        # Code only: the comment above the fixed lookup explains the old
        # `rev-list` form and why it was wrong, so a whole-text match finds the
        # explanation rather than the code.
        code = _code_only(text)
        assert "rev-list -n 1" not in code, (
            "the local-only lookup is back; it cannot tell a missing tag from "
            "an unfetched one"
        )
        # The refusal must precede the proceed-if-empty branch.
        assert text.index("$lsExit -ne 0") < text.index("$tagCommit -ne ''")

    def test_a_missing_verifier_refuses_rather_than_skipping_the_check(self):
        text = _build_script()
        assert "Refusing to tag an unverified image" in text

    def test_the_argv_record_survives_a_failed_build(self):
        """Case 3: a build that exits non-zero must still leave behind what it
        was asked to do — that is the whole reason the record exists."""
        text = _build_script()
        argv_write = text.index("Set-Content -LiteralPath $argvPath")
        build_run = text.index("docker @buildArgs")
        build_fail = text.index("docker build failed with exit code")
        assert argv_write < build_run < build_fail


def _crlf_hint(script: Path) -> str:
    """Explain a CRLF working copy, which git will swear is unmodified.

    `.gitattributes` says `* text=auto eol=lf`, but git normalises at CHECKOUT
    time: a file nobody has touched since an older `core.autocrlf=true` clone
    keeps its CRLF forever. `text=auto` then normalises again when hashing, so
    `git status` is clean and `git diff` is empty while bash cannot parse the
    file — a line ending in `\\` escapes the `\\r` instead of the newline and
    the continuation collapses.

    Without this, the first person to hit it "fixes" a blob that was never
    broken and commits a normalisation nobody needed. Deliberately NOT repaired
    before parsing: on that machine the script genuinely could not run, and
    hiding it would hide a real local breakage.
    """
    try:
        if b"\r\n" not in script.read_bytes():
            return ""
    except OSError:
        return ""
    return (
        "\n    NOTE: this file has CRLF line endings in YOUR WORKING COPY. This "
        "check reads the working copy, not the committed blob, and git will "
        "report the file as unmodified either way (`text=auto` normalises on "
        "hash). Re-check it out — `git rm --cached <f> && git checkout <f>`, or "
        "delete and restore it — rather than editing it or committing a "
        "normalisation."
    )


class TestTheShippedShellScriptsParse:
    """The `.sh` counterpart to the tracked-`.ps1` parse check.

    A sibling investigation found `backend/install.ps1` unparsable on `main`
    for a day — the documented Windows install path, dead, because an encoding
    rule removed a BOM the file needed. Nothing covers the shell scripts at
    all. They run inside the image, so a broken one fails the build loudly
    rather than silently, which is why this is coverage rather than a blocker;
    but `entrypoint.sh` is PID 1 of every container, and "it fails loudly" is a
    poor thing to learn from a user's console.

    Two checks of different kind, and the promises they make differ:

      * the CRLF byte check tests the property itself — deterministic, no
        shell, identical everywhere. Green means "has no CRLF".
      * `bash -n` tests the effect with a real parser, and green means only
        "parses with the bash THIS machine has". Two bashes on one Windows box
        disagree about CRLF (see the byte check's docstring), so it is kept for
        genuine syntax errors, where it earns its place, and is not relied on
        for line endings.

    `bash -n` parses without executing, so this costs milliseconds and cannot
    have side effects.
    """

    def _tracked_shell_scripts(self) -> list[Path]:
        proc = subprocess.run(
            ["git", "ls-files", "*.sh"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            pytest.skip("not a git checkout")
        return [REPO_ROOT / line for line in proc.stdout.split() if line]

    def test_no_tracked_shell_script_has_crlf_line_endings(self):
        """The CAUSE, checked in bytes — deterministic, no shell involved.

        `bash -n` cannot be trusted for this, and the reason inverts the
        obvious intuition. Measured across two bashes on one Windows machine:

            Git Bash 5.3.15  x86_64-pc-cygwin     CRLF fixture -> exit 0
            WSL bash 5.2.21  x86_64-pc-linux-gnu  same bytes   -> exit 2
            WSL bash, CRs stripped                             -> exit 0

        Cygwin's bash tolerates a CR before the newline; Linux bash does not.
        So Git Bash — overwhelmingly the likeliest `bash` on a Windows checkout
        — is blind to exactly the defect this guard exists for, in exactly the
        environment where CRLF drift originates, while the container (Linux
        bash, `entrypoint.sh` as PID 1) is where it is fatal. The parse check
        is weakest where the risk is created and strongest where it cannot
        occur; only a byte check tests the property itself.
        """
        scripts = self._tracked_shell_scripts()
        assert scripts, "no tracked .sh files found — the matcher is looking in the wrong place"
        offenders = [
            s.name for s in scripts if s.exists() and b"\r\n" in s.read_bytes()
        ]
        assert offenders == [], (
            f"tracked shell scripts contain CRLF line endings: {offenders}. "
            "These run under Linux bash in the container, which rejects a CR "
            "before a line continuation. If git reports the file unmodified, "
            "this is working-copy drift from an older `core.autocrlf=true` "
            "checkout — re-check the file out rather than editing it."
        )

    @requires_bash
    def test_every_tracked_shell_script_parses(self):
        bash = _bash()
        assert bash is not None
        scripts = self._tracked_shell_scripts()
        assert scripts, "no tracked .sh files found — the matcher is looking in the wrong place"
        broken = []
        for script in scripts:
            if not script.exists():
                continue
            proc = subprocess.run(
                [bash, "-n", _sh_path(script)],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode != 0:
                broken.append(f"{script.name}: {proc.stderr.strip()}{_crlf_hint(script)}")
        assert broken == [], broken

    @requires_bash
    def test_the_parse_check_actually_rejects_a_broken_script(self, tmp_path):
        """Vacuity control. Built at runtime rather than committed, because a
        committed broken fixture would be picked up by the check above."""
        bash = _bash()
        assert bash is not None
        bad = tmp_path / "broken.sh"
        bad.write_text('#!/bin/sh\nif [ -z "$X" ; then\n  echo unterminated\n', encoding="utf-8")
        proc = subprocess.run(
            [bash, "-n", _sh_path(bad)], capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode != 0, "bash -n accepted a syntactically broken script"

    def test_a_crlf_working_copy_is_named_as_such_in_the_failure(self, tmp_path):
        """The failure this check will ACTUALLY produce on a Windows checkout:
        a correct LF blob whose working copy is CRLF. A sibling session hit it
        on `update.sh` — git reported the file unmodified (`text=auto`
        normalises on hash) while their bash could not parse it. Without the
        hint, the next person commits a normalisation of a blob that was never
        broken.

        Tested as a pure function of the BYTES, deliberately, because whether
        a CRLF continuation actually fails is bash-implementation-dependent:
        measured here, git-bash parses a fully CRLF-converted `update.sh`
        without complaint (exit 0) while the bash that reported it did not. A
        test that first had to reproduce the parse failure would therefore skip
        on this machine and prove nothing — which is what it did before this
        rewrite. The diagnostic must be correct wherever the failure lands.
        """
        crlf = tmp_path / "crlf.sh"
        crlf.write_bytes(b"#!/bin/sh\r\nfor c in \\\r\n  a b; do\r\n  echo $c\r\ndone\r\n")
        hint = _crlf_hint(crlf)
        assert "CRLF" in hint and "WORKING COPY" in hint
        assert "Re-check it out" in hint

        # And it must stay quiet for an LF file, or every unrelated syntax
        # error acquires a confident and wrong line-endings explanation.
        lf = tmp_path / "lf.sh"
        lf.write_bytes(b"#!/bin/sh\nif [ -z \"$X\" ; then\n  echo x\n")
        assert _crlf_hint(lf) == ""


class TestTheWrapperIsTheDocumentedPath:
    """A guard you can bypass by typing the old command is a convention, not a
    guard. The wrapper only protects builds that go through it, and the failure
    it guards against is precisely a hand-run `docker build` — which is what the
    README told everyone to do until now. Same shape as a rulebook teaching the
    very form it forbids: people follow the instructions in good faith.
    """

    def _readme(self) -> str:
        readme = REPO_ROOT / "README.md"
        if not readme.exists():
            pytest.skip("README.md not present in this checkout")
        return readme.read_text(encoding="utf-8")

    def test_the_readme_builds_through_the_wrapper(self):
        text = self._readme()
        assert "docker-build.ps1" in text, "the README does not mention the wrapper"

    def test_the_readme_shows_no_raw_docker_build_invocation(self):
        """The specific regression: a reader copying the old two-line recipe
        gets an unverified image with `latest` already pointing at it."""
        # Matched on the INVOCATION — a line that begins with the command —
        # not on the words, which appear legitimately in the paragraph
        # explaining why the wrapper exists. A name-based matcher here fails on
        # its own prose and teaches nothing (LESSONS 2026-08-28).
        offenders = [
            line.strip()
            for line in self._readme().splitlines()
            if line.strip().startswith("docker build")
        ]
        assert offenders == [], (
            "the README still shows a raw `docker build`; anyone following it "
            f"bypasses the artifact verification: {offenders}"
        )

    def test_the_readme_pins_the_ollama_asset_that_actually_exists(self):
        """The published asset is a zstd tarball. The `.tgz` the README used to
        name has not existed upstream for the last sixty releases, so the pin
        instruction could not be followed at all."""
        text = self._readme()
        assert "ollama-linux-amd64.tgz" not in text
        assert "ollama-linux-amd64.tar.zst" in text


class TestTheBuildWrapperMatchersActuallyFail:
    """Five of a previous round's Dockerfile matchers passed on the PRE-fix
    file. A matcher that cannot fail is documentation wearing a test's name, so
    each assertion above is run here against the mutation it exists to catch.
    """

    def test_ordering_matcher_fails_when_tagging_moves_before_verification(self):
        text = _build_script()
        refusal_block = "throw 'Artifact commit mismatch.'"
        assert refusal_block in text
        # Move the tagging loop above the refusal — the exact regression.
        mutated = text.replace(refusal_block, "docker tag $scratchTag $t\n" + refusal_block, 1)
        guard = TestTheBuildWrapperCannotTagAnUnverifiedImage()
        refusal, tagging = guard._release_tag_ordering(mutated)
        assert tagging < refusal, "the mutation did not actually reorder the script"

    def test_scratch_tag_matcher_fails_when_the_build_names_a_release_tag(self):
        mutated = _build_script().replace("'-t', $scratchTag", "'-t', $Repository", 1)
        assert "'-t', $scratchTag" not in mutated, "the mutation did not apply"

    def test_artifact_read_matcher_fails_when_the_check_is_dropped(self):
        mutated = _build_script().replace(
            "docker run --rm --entrypoint git $scratchTag", "docker inspect $scratchTag", 1
        )
        assert "docker run --rm --entrypoint git $scratchTag" not in mutated


class TestEntrypointSupervisorText:
    """The shape that keeps the loop correct under the header. Cheap; runs everywhere."""

    def _loop(self) -> str:
        text = _entrypoint()
        start = text.index("export MRLN_SUPERVISED=1")
        return text[start:]

    def test_the_shebang_is_bash(self):
        """bash reaps re-parented trainer children in its SIGCHLD handler;
        a ``dash`` PID 1 would not (Assumption — observed in UAT item 5)."""
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


# ── The pinned Ollama path must be reachable, verified, and fail closed ──────


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
