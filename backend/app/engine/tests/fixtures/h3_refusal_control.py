"""POSITIVE CONTROL for ``test_minimax_h3_driver.py::test_no_unlisted_pr0_refusal_remains``.

NOT production code; never imported by the engine. It carries the refusal
shapes the incremental guard's scanner must recognise — a direct
``raise NotImplementedError``, a raise through the ``_lands_in_pr1`` helper
shape, and one inside a method — so a scanner that finds nothing is caught
by ``test_refusal_scanner_flags_the_positive_control``. Plan row 2.1
(``_harness/plans/2026-09-12-minimax-h3-pr1.md``).
"""

from __future__ import annotations


def _lands_in_pr1(what: str) -> NotImplementedError:
    return NotImplementedError(what)


def direct_refusal() -> None:
    raise NotImplementedError("PR1")


def helper_refusal() -> None:
    raise _lands_in_pr1("PR1")


def not_a_refusal() -> None:
    # A docstring or comment naming NotImplementedError is NOT a site.
    """Mentions NotImplementedError only in prose."""
    raise ValueError("a real error, not a refusal")


class Control:
    def still_refuses(self) -> None:
        raise NotImplementedError("PR1")
