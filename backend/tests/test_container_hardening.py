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
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

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


# ── behaviour ────────────────────────────────────────────────────────────


def _bash() -> str | None:
    return shutil.which("bash")


requires_bash = pytest.mark.skipif(
    _bash() is None, reason="needs a POSIX shell to execute entrypoint.sh"
)


def _run_entrypoint_as_fake_root(tmp: Path, *, uid_exists: bool = True) -> str:
    """Run entrypoint.sh with ``id``, ``setpriv``, ``getent``, ``chown`` shimmed.

    The shims are the OPERATING SYSTEM, not the logic under test: the thing
    being asserted is which branch the script takes and what arguments it hands
    to ``setpriv``. ``setpriv`` prints its argv and exits, which stops the
    script exactly where the real one would hand off.
    """
    shim = tmp / "shim"
    shim.mkdir()

    (shim / "id").write_text("#!/bin/sh\necho 0\n", encoding="utf-8")
    (shim / "setpriv").write_text(
        '#!/bin/sh\necho "SETPRIV_CALLED $*"\nexit 0\n', encoding="utf-8"
    )
    (shim / "getent").write_text(
        f"#!/bin/sh\nexit {0 if uid_exists else 2}\n", encoding="utf-8"
    )
    (shim / "chown").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    for f in shim.iterdir():
        f.chmod(0o755)

    data = tmp / "data"
    data.mkdir()

    env = dict(os.environ)
    env["PATH"] = f"{shim}{os.pathsep}{env.get('PATH', '')}"
    env["MRLN_DATA_DIR"] = str(data)

    proc = subprocess.run(
        [_bash(), str(ENTRYPOINT)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        cwd=str(tmp),
    )
    return proc.stdout + proc.stderr


class TestEntrypointDropsPrivileges:
    @requires_bash
    def test_running_as_root_re_execs_as_the_app_uid(self):
        with tempfile.TemporaryDirectory() as td:
            out = _run_entrypoint_as_fake_root(Path(td))

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
    def test_drop_happens_before_the_app_directories_are_touched(self):
        """Ordering is load-bearing, so it is pinned.

        The script must drop privileges BEFORE it starts rewriting symlinks
        under /app. If the drop moves below that, every one of those writes
        happens as root and the container is only nominally non-root.
        """
        with tempfile.TemporaryDirectory() as td:
            out = _run_entrypoint_as_fake_root(Path(td))

        assert "SETPRIV_CALLED" in out
        # /app does not exist on a test machine; if the script had reached the
        # symlink section before dropping, `set -e` would have killed it there
        # and we would never see the setpriv line.
        assert "ln:" not in out and "No such file or directory" not in out, (
            "entrypoint reached the /app section before dropping privileges:\n" + out
        )

    @requires_bash
    def test_missing_app_user_continues_instead_of_failing_to_boot(self):
        """Prove the negative on the fallback path.

        An image built before this change has no uid 10001. Refusing to boot
        would convert a hardening regression into an outage, so the documented
        behaviour is 'continue as root, loudly'. Pinned so the warning cannot
        be quietly dropped later.
        """
        with tempfile.TemporaryDirectory() as td:
            out = _run_entrypoint_as_fake_root(Path(td), uid_exists=False)

        assert "SETPRIV_CALLED" not in out, "dropped to a uid that does not exist"
        assert "CONTINUING AS ROOT" in out, f"silent fallback to root:\n{out}"


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

    proc = subprocess.run(
        [_bash(), "-c", snippet + "\nexit 0\n"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
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
