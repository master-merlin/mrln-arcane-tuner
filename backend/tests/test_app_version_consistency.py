"""The app's own version: one value, four files, two standards, one scheme.

THE LESSON THIS FILE EXISTS FOR. ``backend/tests/test_readme_version_claims.py``
pins the *dependency* versions the README announces — torch, CUDA, Angular —
thoroughly, and pins **nothing about the app's own version**. Meanwhile
``/bump-version`` (``.agent/scripts/sync_version.py``) writes the number into
three files by regex and no test looks afterwards, so a regex that stops
matching prints ``WARN … skipped`` and the release ships a README claiming the
previous version. A release audit on 2026-08-29 came within one command of
tagging a version that was not valid PEP 440 at all, and nothing in the repo
would have said so. **A value copied into N places without a guard is N places
that will disagree, and the disagreement surfaces at release time, when it is
the most expensive moment to find it.**

WHY BOTH STANDARDS. One number feeds ``npm`` (``frontend/package.json`` — semver
2.0.0) and, the day the backend is packaged, Python tooling (PEP 440). The two
grammars overlap but neither contains the other, measured here with the
installed ``packaging`` 26.0 and pinned as tests below:

* ``0.8.0-beta-rc0`` is **valid semver** (``beta-rc0`` is one legal
  alphanumeric pre-release identifier) and **invalid PEP 440** (two pre-release
  segments);
* ``0.8.0b1`` is **valid PEP 440** and **invalid semver** (nothing may follow
  the patch digits except ``-`` or ``+``).

So validity is asserted against both grammars independently. Passing one proves
nothing about the other.

WHY A SCHEME ON TOP OF BOTH. ``0.8.0-alpha`` is valid in *both* standards and is
still wrong here, because the ladder has to **order** correctly as well as
parse. The permitted pre-release set — empty, ``beta``, ``beta.<N>``,
``rc.<N>`` — is the set that is legal in both grammars and sorts
``0.8.0-beta.1 < 0.8.0-beta.2 < 0.8.0-rc.1 < 0.8.0`` in both, so a release
candidate can never outrank the release it precedes. That ladder is not
described here and trusted; it is asserted, in both orderings, below.
``-beta`` with no number stays legal because PEP 440 normalises it to
``0.8.0b0`` — beta zero — which is what today's ``0.7.9-beta`` relies on.

VACUITY (CONVENTIONS "Tests" rule 11). The README checks have the
collect-offenders shape: they scan prose for occurrences and assert every
occurrence agrees. An empty offender list is the same object whether the README
is correct or the scanner's regex has drifted off a reformatted line, so the
scan is required to find at least the occurrences that exist today
(``_assert_not_vacuous``), and that requirement is itself controlled — a
pattern that matches nothing must make the check FAIL, not pass quietly.

KNOWN LIMIT, inherited rather than rediscovered. These checks read prose, and
prose may one day legitimately name an old version (a changelog line, a note
about what an earlier release shipped). When that happens, narrow WHERE the
check looks — never exempt the file and never delete the assertion, which is
how a guard rots into decoration.

Anchored on ``__file__`` so it runs identically from any working directory
(ARCHITECTURE D10 invariant 9).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from packaging.version import InvalidVersion
from packaging.version import Version as Pep440Version

REPO_ROOT = Path(__file__).resolve().parents[2]

VERSION_SOURCE = REPO_ROOT / "backend" / "app" / "__init__.py"
README = REPO_ROOT / "README.md"
PACKAGE_JSON = REPO_ROOT / "frontend" / "package.json"
PACKAGE_LOCK = REPO_ROOT / "frontend" / "package-lock.json"

# ── the source of truth ────────────────────────────────────────────────
#: ``__version__ = "0.7.9-beta"`` in ``backend/app/__init__.py``. Read as text
#: rather than imported, because importing ``app`` runs the diffusers/hpsv2
#: compatibility patches, and because the literal in this file is the artifact
#: ``/bump-version`` rewrites. A separate test below imports the package and
#: proves the running value equals this literal.
_VERSION_ASSIGNMENT = re.compile(r"""^__version__\s*=\s*["'](?P<version>[^"']+)["']""", re.M)

# ── README occurrence patterns ─────────────────────────────────────────
#: The badge on line 5, written `` `v0.7.9-beta` ``.
_README_BADGE = re.compile(r"`v(\d+\.\d+\.\d+[^`]*)`")

#: Every Docker image tag, e.g. ``mastermerlin/mrln-arcane-tuner:0.7.9-beta``
#: and the CUDA-variant ``…:0.7.9-beta-cu126``. ``:latest`` matches too and is
#: dropped by the collector, on purpose: matching it and discarding it is how
#: the collector proves it is looking at the tags at all.
_README_IMAGE_TAG = re.compile(r"mrln-arcane-tuner:([\w.+-]+)")

#: The bare backtick shorthand `` `:0.7.9-beta-cu126` `` used in prose (README
#: ~line 262). ``sync_version.py``'s ``README_DOCKER_RE`` rewrites this form
#: too, so it is a real occurrence and must be checked like the others.
_README_BARE_TAG = re.compile(r"`:(\d[\w.+-]*)`")

#: A Docker build-variant suffix appended AFTER the version in an image tag.
#: Stripped before comparing; it is not part of the version.
_CUDA_VARIANT_SUFFIX = re.compile(r"-cu\d+$")

# Measured on this tree (2026-08-30, README.md at 0.7.9-beta): 1 badge,
# 6 version-bearing image tags (lines 194, 195, 222, 223, 229, 230 — line 222's
# second tag is `:latest` and is excluded), 1 bare shorthand (line 262).
# Floors, not equalities: the README may legitimately gain another tag block,
# but it must never silently lose the ones that exist.
_MIN_BADGES = 1
_MIN_IMAGE_TAGS = 6
_MIN_BARE_TAGS = 1

# ── the scheme ─────────────────────────────────────────────────────────
#: Permitted semver pre-release part: empty, ``beta``, ``beta.<N>``, ``rc.<N>``.
_ALLOWED_PRERELEASE = re.compile(r"^(?:beta(?:\.\d+)?|rc\.\d+)$")

#: Semver 2.0.0, the official regex from https://semver.org (Backus-Naur
#: grammar → "Is there a suggested regular expression (RegEx) to check a SemVer
#: string?"). Implemented here rather than shelled out to node: a test that
#: needs ``frontend/node_modules`` to exist is a test that fails for the wrong
#: reason on a fresh clone.
_SEMVER = re.compile(
    r"^(?P<major>0|[1-9]\d*)"
    r"\.(?P<minor>0|[1-9]\d*)"
    r"\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


# ── helpers ────────────────────────────────────────────────────────────
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def app_version() -> str:
    """The single source of truth: ``backend/app/__init__.py`` ``__version__``."""
    match = _VERSION_ASSIGNMENT.search(_read(VERSION_SOURCE))
    assert match, (
        f"{VERSION_SOURCE.relative_to(REPO_ROOT)} no longer assigns a literal "
        "__version__ -- the source of truth for every check in this file moved "
        "or became computed. Point this test at the new source; do not delete it."
    )
    return match.group("version")


def is_valid_semver(version: str) -> bool:
    return _SEMVER.match(version) is not None


def is_valid_pep440(version: str) -> bool:
    try:
        Pep440Version(version)
    except InvalidVersion:
        return False
    return True


def semver_prerelease(version: str) -> str:
    """The pre-release part of a semver string, ``""`` when there is none."""
    match = _SEMVER.match(version)
    assert match, f"{version!r} is not valid semver, so it has no pre-release part"
    return match.group("prerelease") or ""


def _semver_identifier_key(identifier: str) -> tuple[int, int, str]:
    """Semver 2.0.0 §11.4: numeric identifiers sort below alphanumeric ones,
    numerics compare numerically, alphanumerics compare in ASCII order."""
    if identifier.isdigit():
        return (0, int(identifier), "")
    return (1, 0, identifier)


def semver_precedence_key(version: str) -> tuple:
    """Sort key implementing semver 2.0.0 precedence (§11), build metadata
    ignored per §10. A version WITH a pre-release sorts below the same version
    without one (§11.3), which is the whole point of the ladder below."""
    match = _SEMVER.match(version)
    assert match, f"{version!r} is not valid semver"
    core = (int(match.group("major")), int(match.group("minor")), int(match.group("patch")))
    prerelease = match.group("prerelease")
    if prerelease is None:
        return (core, 1, ())
    return (core, 0, tuple(_semver_identifier_key(p) for p in prerelease.split(".")))


def collect_version_occurrences(
    *,
    readme: str,
    package_json: str,
    package_lock: str,
    badge_pattern: re.Pattern[str] = _README_BADGE,
    image_tag_pattern: re.Pattern[str] = _README_IMAGE_TAG,
    bare_tag_pattern: re.Pattern[str] = _README_BARE_TAG,
) -> list[tuple[str, str]]:
    """Every place the app version is written, as ``(label, value)`` pairs.

    Duplicates are kept — six README Docker tags are six chances to disagree,
    and the label carries the file so a failure names the offender.

    The patterns are parameters so the vacuity control can inject one that
    matches nothing and prove ``_assert_not_vacuous`` fails on it.
    """
    occurrences: list[tuple[str, str]] = []

    # Positive lookups: a missing key raises here rather than yielding an empty
    # scan, so these files cannot go unchecked without a loud failure.
    occurrences.append(("frontend/package.json version", json.loads(package_json)["version"]))
    lock = json.loads(package_lock)
    occurrences.append(("frontend/package-lock.json version", lock["version"]))
    occurrences.append(
        ("frontend/package-lock.json packages[''].version", lock["packages"][""]["version"])
    )

    for badge in badge_pattern.findall(readme):
        occurrences.append(("README.md `v<version>` badge", badge))
    for tag in image_tag_pattern.findall(readme):
        if tag == "latest":
            continue  # a floating tag, deliberately not version-pinned
        occurrences.append(
            ("README.md mrln-arcane-tuner:<tag> image tag", _CUDA_VARIANT_SUFFIX.sub("", tag))
        )
    for tag in bare_tag_pattern.findall(readme):
        occurrences.append(
            ("README.md `:<tag>` shorthand", _CUDA_VARIANT_SUFFIX.sub("", tag))
        )
    return occurrences


def _assert_not_vacuous(occurrences: list[tuple[str, str]]) -> None:
    """The scan found at least the occurrences that exist today.

    Without this, a drifted regex turns "every occurrence agrees" into "I found
    nothing, so nothing disagreed" — green forever, checking nothing.
    """
    badges = sum(1 for label, _ in occurrences if "badge" in label)
    image_tags = sum(1 for label, _ in occurrences if "mrln-arcane-tuner:<tag>" in label)
    bare_tags = sum(1 for label, _ in occurrences if "shorthand" in label)
    assert badges >= _MIN_BADGES, (
        f"README version-badge scan found {badges} occurrence(s), expected at least "
        f"{_MIN_BADGES}. The badge moved or the pattern drifted -- the check is blind, "
        "not satisfied."
    )
    assert image_tags >= _MIN_IMAGE_TAGS, (
        f"README Docker image-tag scan found {image_tags} occurrence(s), expected at "
        f"least {_MIN_IMAGE_TAGS}. Tags were removed or the pattern drifted."
    )
    assert bare_tags >= _MIN_BARE_TAGS, (
        f"README bare `:<tag>` scan found {bare_tags} occurrence(s), expected at least "
        f"{_MIN_BARE_TAGS}. The prose shorthand moved or the pattern drifted."
    )


def disagreements(occurrences: list[tuple[str, str]], version: str) -> list[str]:
    """Every occurrence that is not ``version``, rendered for a failure message."""
    return [
        f"{label}: {found!r} != {version!r}" for label, found in occurrences if found != version
    ]


@pytest.fixture(scope="module")
def occurrences() -> list[tuple[str, str]]:
    return collect_version_occurrences(
        readme=_read(README),
        package_json=_read(PACKAGE_JSON),
        package_lock=_read(PACKAGE_LOCK),
    )


class TestAgreement:
    """Every file that writes the app version writes the same one."""

    def test_the_scan_is_not_vacuous(self, occurrences):
        _assert_not_vacuous(occurrences)

    def test_the_vacuity_check_fails_on_a_pattern_that_matches_nothing(self):
        """Positive control for the control.

        A scanner that has stopped matching must FAIL, not report zero
        offenders and pass. Proven by pointing the README scan at a pattern
        no README will ever contain.
        """
        never_matches = re.compile(r"`v(THIS-PATTERN-MATCHES-NOTHING)`")
        blind = collect_version_occurrences(
            readme=_read(README),
            package_json=_read(PACKAGE_JSON),
            package_lock=_read(PACKAGE_LOCK),
            badge_pattern=never_matches,
        )
        assert not [label for label, _ in blind if "badge" in label]
        with pytest.raises(AssertionError, match="version-badge scan found 0"):
            _assert_not_vacuous(blind)

    def test_every_occurrence_equals_the_source_of_truth(self, occurrences):
        version = app_version()
        _assert_not_vacuous(occurrences)
        offenders = disagreements(occurrences, version)
        assert not offenders, (
            f"backend/app/__init__.py says __version__ = {version!r}, but "
            f"{len(offenders)} occurrence(s) disagree:\n  " + "\n  ".join(offenders) + "\n"
            "Re-run `/bump-version` (it syncs __init__.py, README.md and "
            "frontend/package.json) and `npm --prefix frontend install "
            "--package-lock-only` for the lockfile."
        )

    def test_the_agreement_check_catches_a_single_disagreeing_file(self):
        """Positive control: one drifted occurrence out of many is reported,
        and the message names the file that drifted."""
        drifted = [
            ("frontend/package.json version", "0.7.9-beta"),
            ("README.md `v<version>` badge", "0.7.8-beta"),
        ]
        offenders = disagreements(drifted, "0.7.9-beta")
        assert offenders == ["README.md `v<version>` badge: '0.7.8-beta' != '0.7.9-beta'"]

    def test_the_running_package_reports_the_literal_version(self):
        """The value the app serves (``/`` returns it) is the value on disk."""
        import app

        assert app.__version__ == app_version()


class TestBothStandards:
    """Valid semver AND valid PEP 440 — neither grammar contains the other."""

    def test_the_app_version_is_valid_semver(self):
        version = app_version()
        assert is_valid_semver(version), (
            f"__version__ = {version!r} is not valid semver 2.0.0, and it is written "
            "verbatim into frontend/package.json, which npm parses as semver."
        )

    def test_the_app_version_is_valid_pep440(self):
        version = app_version()
        assert is_valid_pep440(version), (
            f"__version__ = {version!r} is not valid PEP 440; Python packaging tooling "
            "would refuse it."
        )

    def test_semver_valid_but_pep440_invalid_is_a_real_case(self):
        """Positive control + the measured reason both checks exist.

        ``0.8.0-beta-rc0``: one legal semver pre-release identifier, two
        pre-release segments for PEP 440.
        """
        assert is_valid_semver("0.8.0-beta-rc0")
        assert not is_valid_pep440("0.8.0-beta-rc0")

    def test_pep440_valid_but_semver_invalid_is_a_real_case(self):
        """``0.8.0b1``: PEP 440's canonical beta spelling; semver allows
        nothing but ``-`` or ``+`` after the patch digits."""
        assert is_valid_pep440("0.8.0b1")
        assert not is_valid_semver("0.8.0b1")

    def test_the_whole_scheme_normalises_onto_pep440_prereleases(self):
        """Measured with ``packaging`` 26.0. ``-beta`` with no number is legal
        precisely because PEP 440 reads it as **beta 0** — the property today's
        ``0.7.9-beta`` depends on, and the reason the scheme need not force a
        number onto the first beta of a line."""
        assert str(Pep440Version("0.8.0-beta")) == "0.8.0b0"
        assert str(Pep440Version("0.8.0-beta.1")) == "0.8.0b1"
        assert str(Pep440Version("0.8.0-beta.2")) == "0.8.0b2"
        assert str(Pep440Version("0.8.0-rc.1")) == "0.8.0rc1"
        assert str(Pep440Version("0.8.0")) == "0.8.0"

    def test_todays_version_normalises_to_a_pep440_prerelease(self):
        """Version-agnostic on purpose: a check that hardcodes today's number
        would go red on every legitimate bump, and a guard that cries wolf at
        release time is the guard the release deletes."""
        version = app_version()
        assert bool(Pep440Version(version).pre) == bool(semver_prerelease(version)), (
            f"{version!r} is a pre-release in one standard and a final release in the "
            "other -- npm and Python packaging would disagree about what it is."
        )


class TestScheme:
    """Empty, ``beta``, ``beta.<N>`` or ``rc.<N>`` — nothing else."""

    def test_the_app_version_follows_the_scheme(self):
        version = app_version()
        prerelease = semver_prerelease(version)
        assert prerelease == "" or _ALLOWED_PRERELEASE.match(prerelease), (
            f"__version__ = {version!r} has pre-release part {prerelease!r}, which is "
            "outside the permitted set (empty, 'beta', 'beta.<N>', 'rc.<N>'). That set "
            "is the one that is valid in BOTH standards and orders correctly; see this "
            "file's docstring before widening it."
        )

    @pytest.mark.parametrize(
        "prerelease",
        ["", "beta", "beta.1", "beta.12", "rc.1", "rc.10"],
    )
    def test_the_scheme_accepts_the_whole_permitted_ladder(self, prerelease):
        version = f"0.8.0-{prerelease}" if prerelease else "0.8.0"
        assert is_valid_semver(version)
        assert is_valid_pep440(version)
        assert prerelease == "" or _ALLOWED_PRERELEASE.match(prerelease)

    @pytest.mark.parametrize(
        "prerelease, why",
        [
            ("alpha", "valid in both standards, but alpha is not part of this ladder"),
            ("beta1", "no separator: PEP 440 reads b1, semver reads one opaque identifier"),
            ("rc", "an unnumbered rc cannot be ordered against the next one"),
            ("beta.rc.1", "two kinds of pre-release in one version"),
            ("dev", "not a release-candidate stage this project publishes"),
        ],
    )
    def test_the_scheme_rejects_everything_else(self, prerelease, why):
        """Positive control for the scheme check: ``0.8.0-alpha`` parses in BOTH
        grammars, so only the scheme stops it."""
        assert not _ALLOWED_PRERELEASE.match(prerelease), why

    def test_alpha_is_rejected_by_the_scheme_alone(self):
        assert is_valid_semver("0.8.0-alpha")
        assert is_valid_pep440("0.8.0-alpha")
        assert not _ALLOWED_PRERELEASE.match("alpha")


class TestOrdering:
    """The reason the scheme is this set and not a wider one.

    A release candidate must never outrank the release it precedes, in either
    standard. Asserted, not described — a comment claiming an order the code
    does not check is worse than no comment.
    """

    LADDER = ["0.8.0-beta.1", "0.8.0-beta.2", "0.8.0-rc.1", "0.8.0"]

    def test_the_ladder_ascends_strictly_under_pep440(self):
        parsed = [Pep440Version(v) for v in self.LADDER]
        assert all(a < b for a, b in zip(parsed, parsed[1:])), [str(p) for p in parsed]

    def test_the_ladder_ascends_strictly_under_semver(self):
        keys = [semver_precedence_key(v) for v in self.LADDER]
        assert all(a < b for a, b in zip(keys, keys[1:])), self.LADDER

    def test_an_unnumbered_beta_is_beta_zero_and_sorts_below_beta_one(self):
        """Why ``-beta`` remains legal: PEP 440 places it at the bottom of the
        beta run rather than somewhere ambiguous."""
        assert Pep440Version("0.8.0-beta") < Pep440Version("0.8.0-beta.1")
        assert semver_precedence_key("0.8.0-beta") < semver_precedence_key("0.8.0-beta.1")
