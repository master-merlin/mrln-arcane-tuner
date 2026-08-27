"""The assertion-rewriter warning filter is narrow, and provably so.

WHY A SUPPRESSION NEEDS A TEST MORE THAN MOST CODE DOES: a `filterwarnings`
line is a guard-shaped thing that HIDES. When it is too broad it does not fail
— it silently stops reporting something real, and the failure is invisible
precisely because the mechanism's job is to make things invisible. So the
assertions that matter here are not "the filter works". They are the three that
prove it does not work on anything else.

WHAT IT SUPPRESSES, AND WHY (measured, not guessed): a full run on a cold
``__pycache__`` reported 125,429 warnings, of which 125,247 — **99.85%**, from
twelve lines of one third-party module — came from
``_pytest/assertion/rewrite.py``. pytest rewrites assertions at import and
caches the result, so a WARM cache never runs the rewriter and never emits
them: 6 warnings warm against 988 cold on an 84-test subset, the same tree and
the same command, a 165x swing reproducible on demand. That made the warning
total a property of the working directory's bytecode cache rather than of the
code — and a fresh worktree is cold by construction, so a clean fork's first
gate reported six figures and a 700x "divergence" between two branches was
chased as a defect for a day.

THIS IS A WORKAROUND. The warnings come from ``pytest==7.2.0``; modern pytest
does not emit them at all, so the real fix is the upgrade, filed as its own
lane because it is a dependency decision with version-coupled dependents. When
that lands, this filter and this file should go with it.
"""

from __future__ import annotations

import configparser
import warnings
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
PYTEST_INI = BACKEND / "pytest.ini"

#: The four texts the rewriter actually emitted in the measured cold run.
#: Enumerated from the output, not hand-listed — a hand-written list of three
#: missed ``Attribute s`` and would have left that share unfiltered.
REWRITER_MESSAGES = (
    "ast.Str is deprecated and will be removed in Python 3.14; use ast.Constant instead",
    "ast.Num is deprecated and will be removed in Python 3.14; use ast.Constant instead",
    "ast.NameConstant is deprecated and will be removed in Python 3.14; use ast.Constant instead",
    "Attribute s is deprecated and will be removed in Python 3.14; use value instead",
)

REWRITER_MODULE = "_pytest.assertion.rewrite"


def _ini_filters() -> list[str]:
    """The filter specs as the shipped ``pytest.ini`` actually declares them.

    Read from the file rather than restated here. A test that agrees with its
    own copy of the config proves nothing about what pytest loads.
    """
    if not PYTEST_INI.exists():
        pytest.skip("pytest.ini not in this checkout")
    parser = configparser.ConfigParser()
    parser.read(PYTEST_INI, encoding="utf-8")
    raw = parser.get("pytest", "filterwarnings", fallback="")
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _apply(specs: list[str]) -> None:
    """Install *specs* into the ambient filter list, innermost-wins order.

    Uses the real ``warnings`` machinery with the real ini strings, so what is
    exercised is Python's own matching — not a reimplementation of it, which
    would be the classic "test agrees with its own copy" vacuity.
    """
    for spec in specs:
        action, message, category, module = (spec.split(":") + ["", "", ""])[:4]
        warnings.filterwarnings(
            action,
            message=message,
            category=getattr(warnings, category, None) or eval(category),  # noqa: S307
            module=module,
        )


def _is_suppressed(message: str, module: str) -> bool:
    """Would a DeprecationWarning with this text, from this module, be shown?

    ``warn_explicit`` takes the module as a parameter, which is exactly how
    Python decides whether a filter's module axis matches. That makes this a
    test of the real matching rules rather than of a stand-in for them.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        warnings.simplefilter("always")
        _apply(_ini_filters())
        warnings.warn_explicit(
            message,
            DeprecationWarning,
            filename=f"{module.replace('.', '/')}.py",
            lineno=1,
            module=module,
            registry={},
        )
    return not caught


class TestItSuppressesTheNoise:
    def test_the_filter_is_declared_at_all(self):
        specs = _ini_filters()
        assert specs, (
            "pytest.ini declares no filterwarnings. Without it a cold "
            "__pycache__ makes the warning count six figures and meaningless."
        )

    @pytest.mark.parametrize("message", REWRITER_MESSAGES)
    def test_each_measured_rewriter_message_is_covered(self, message):
        """All four, because four is what the measured run contained."""
        assert _is_suppressed(message, REWRITER_MODULE), (
            f"the rewriter warning {message!r} is not suppressed; the warning "
            "count is cache-dependent again"
        )


class TestItIsNarrowOnBothAxes:
    """The assertions that actually matter.

    A filter that suppresses the noise is trivial to write; a filter that
    suppresses ONLY the noise is the whole requirement. Each of these fires
    against a broadening of one axis, and neither would fail if the filter
    merely worked.
    """

    @pytest.mark.parametrize("message", REWRITER_MESSAGES)
    def test_the_same_message_from_our_own_code_still_surfaces(self, message):
        """The module axis.

        ``ast.Str is deprecated`` occurs outside the rewriter too. A
        message-only ignore would hide it everywhere — including in our own
        code, where it would be a real finding about a real 3.14 incompatibility.
        """
        assert not _is_suppressed(message, "app.core.dataset_manager"), (
            "a rewriter message is suppressed even when it comes from OUR "
            "code. The module qualifier has been dropped or broadened, and a "
            "genuine 3.14 incompatibility in app/ would now be invisible."
        )

    def test_a_different_message_from_the_rewriter_still_surfaces(self):
        """The message axis.

        A module-only ignore would swallow the next genuinely different thing
        that module says — a real bug report from pytest, silenced because we
        were tired of its deprecation chatter. That is how a filter stops being
        a filter and becomes a blindfold.
        """
        assert not _is_suppressed(
            "assertion rewriting failed catastrophically", REWRITER_MODULE
        ), (
            "ANY message from the rewriter is suppressed, not just the ast "
            "deprecations. The filter is module-wide and is now hiding "
            "whatever else that module might need to tell us."
        )

    def test_an_ordinary_deprecation_from_our_code_still_surfaces(self):
        """The blunt one, and the one the Shepherd asked for by name.

        Without it, 'we filtered narrowly' is a claim about a regex nobody has
        fired. If this ever fails, the project has stopped hearing its own
        deprecation warnings entirely.
        """
        assert not _is_suppressed(
            "some_function is deprecated, use other_function",
            "app.engine.models.registry",
        )


class TestTheReasonSurvives:
    """A bare suppression reads as somebody hiding something and gets deleted.

    The measurement is what makes it obviously correct rather than merely
    present, so the measurement is pinned to the file.
    """

    def _ini_text(self) -> str:
        if not PYTEST_INI.exists():
            pytest.skip("pytest.ini not in this checkout")
        return PYTEST_INI.read_text(encoding="utf-8")

    def test_the_measurement_is_recorded_next_to_the_filter(self):
        text = self._ini_text()
        for token in ("125,247", "99.85%", "165x"):
            assert token in text, (
                f"the measurement {token!r} is gone from pytest.ini. Without "
                "the numbers this is an unexplained suppression, and the next "
                "person tidying will delete it."
            )

    def test_the_upgrade_is_named_as_the_real_fix(self):
        """So nobody reads the workaround as the solution."""
        text = self._ini_text()
        assert "WORKAROUND" in text and "pytest==7.2.0" in text, (
            "pytest.ini no longer says that the filter is a workaround and the "
            "version bump is the fix"
        )


class TestThisFileCanFail:
    """Vacuity checks.

    Every assertion above routes through ``_is_suppressed``. If that helper
    returned a constant — or silently failed to install the filters — the
    suppression tests and the narrowness tests would BOTH pass, in opposite
    directions, and this file would be decorative.
    """

    def test_the_helper_distinguishes_its_two_answers(self):
        assert _is_suppressed(REWRITER_MESSAGES[0], REWRITER_MODULE) is True
        assert _is_suppressed(REWRITER_MESSAGES[0], "app.whatever") is False

    def test_the_specs_are_actually_parsed_into_filters(self):
        """Guards the split-on-colon parsing, which is the fragile part."""
        specs = _ini_filters()
        with warnings.catch_warnings():
            warnings.resetwarnings()
            before = len(warnings.filters)
            _apply(specs)
            after = len(warnings.filters)
        assert after - before == len(specs), (
            f"{len(specs)} ini filter lines produced {after - before} real "
            "filters — the spec parsing is wrong, so every test here is "
            "measuring the wrong thing"
        )

    def test_both_axes_are_present_in_every_spec(self):
        """Structural backstop for the behavioural narrowness tests."""
        for spec in _ini_filters():
            parts = spec.split(":")
            assert len(parts) >= 4 and parts[1] and parts[3], (
                f"filter spec {spec!r} omits a message or module qualifier; "
                "one axis alone is too broad"
            )
