"""Every attribution README makes must be true of the file it points at.

WHY THIS EXISTS: ``flux2/trainer.py`` carried "Timestep sampling strategies
derived from ostris/ai-toolkit" in its module docstring. The file contains no
sampling code — it delegates to ``app.engine.strategies.timestep_sampling``,
which carries no ai-toolkit reference and shares 2 of its mode names with
ai-toolkit by convergence, not descent. So the claim was false.

It was also DOUBLE-ENTERED: README's Acknowledgments asserted the same
derivation and pointed at that very comment as its evidence. Deleting the
comment alone would have left README claiming a derivation and citing a file
that no longer said anything — a dangling attribution reads as verified and is
harder to notice than an obviously missing one.

A licence attribution is not a comment, so the guard is not "does the word
appear". It is: every "Credited in <file>" pointer resolves, and the file it
resolves to actually mentions that project. That is the property the pair broke,
and it generalises to the next one.

The ``ostris/ai-toolkit`` entry in ``NOTICE`` is untouched and stays untouched —
it is earned several times over by ``ideogram4/``, which reproduces
``get_qwen3_vl_features`` verbatim. Removing a false claim about ONE file must
not be mistaken for removing the obligation.

WHY THIS IS NOT A ONE-PRODUCER VIOLATION (RULE-21), stated because it looks
like one and the reasoning decides what this guard is for:

README's Acknowledgments restate projects that ``NOTICE`` also lists, and the
obvious reading is "two copies of one fact, tolerated". That reading is wrong on
the facts, and tolerating it would license the wrong thing. ``NOTICE`` carries
licence text and file pointers; it never asserts *what a dependency was used
for*. "Timestep sampling strategies are derived from ai-toolkit" was never in
``NOTICE`` and could not have been.

So README is not a second copy — it restates a list and adds a claim per item,
and that surplus is the part with no producer upstream of it. The general form:
**a document that restates another document's list and adds a claim per item has
not duplicated anything; it has authored something new, and the new part needs
its own verification.** This guard enforces exactly that surplus and deliberately
says nothing about the overlap with ``NOTICE``. Read as "duplication is fine
here", it would invite the next unverified sentence in under the same shelter.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"

#: `- **[Name](https://github.com/owner/repo)** — …`
_BULLET = re.compile(
    r"^- \*\*\[(?P<name>[^\]]+)\]"
    r"\(https://github\.com/(?P<owner>[^/)]+)/(?P<repo>[^/)\s]+)\)\*\*"
    r"(?P<rest>.*)$",
    re.M,
)
#: `Credited in [`x`](path)` — possibly several in one bullet.
_POINTER = re.compile(r"\]\((?!https?://)([^)]+)\)")


def _readme() -> str:
    if not README.exists():
        pytest.skip("README.md not in this checkout")
    return README.read_text(encoding="utf-8")


def _credited_pointers() -> list[tuple[str, str, str]]:
    """(project, owner, relative path) for every credit that names a file."""
    out = []
    for m in _BULLET.finditer(_readme()):
        rest = m.group("rest")
        idx = rest.find("Credited in")
        if idx == -1:
            continue  # an acknowledgment without a file claim; nothing to check
        for target in _POINTER.findall(rest[idx:]):
            out.append((m.group("repo"), m.group("owner"), target))
    return out


class TestEveryCreditedFileBacksItsClaim:
    def test_there_are_pointers_to_check(self):
        """Otherwise every test below passes by finding nothing.

        A README rewrite that changed the "Credited in" wording would silently
        empty this whole file, which is the failure mode a guard over
        hand-written prose is most prone to.
        """
        assert len(_credited_pointers()) >= 3, (
            "no 'Credited in <file>' pointers were parsed out of README's "
            "Acknowledgments — either they are gone, or the wording changed and "
            "this guard stopped guarding anything"
        )

    def test_each_pointer_resolves(self):
        missing = [
            (project, target)
            for project, _owner, target in _credited_pointers()
            if not (REPO_ROOT / target).exists()
        ]
        assert not missing, (
            f"README credits a file that does not exist: {missing}. A licence "
            "attribution pointing nowhere reads as verified and is worse than "
            "no pointer at all."
        )

    def test_each_credited_file_actually_mentions_the_project(self):
        """The half that the flux2 change would otherwise have broken.

        The file existing is not the claim — the claim is that THIS file is
        where that project's work was used.

        The REPOSITORY name is required, not the owner's. Mutation-tested and
        tightened for it: pointing the Ostris credit at ``core/naming.py``
        passed the owner check, because that file happens to contain the model
        id ``ostris/flux-dev``. An owner's name shows up in incidental places;
        the repo name is what a derivation cites.
        """
        for project, _owner, target in _credited_pointers():
            path = REPO_ROOT / target
            if not path.exists():
                continue  # reported by the test above
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            assert project.lower() in text, (
                f"README says {project} is 'Credited in {target}', but that "
                f"file never mentions {project}. Either the credit moved and "
                "README was not updated, or the claim was never true."
            )


class TestTheFalseClaimIsGone:
    """The specific regression, pinned in both places it lived."""

    def test_the_trainer_no_longer_claims_a_derivation_it_cannot_support(self):
        path = (
            REPO_ROOT / "backend" / "app" / "engine" / "models" / "families"
            / "flux2" / "trainer.py"
        )
        text = path.read_text(encoding="utf-8")
        assert "derived from ostris" not in text, (
            "the derivation claim is back in flux2/trainer.py. The file has no "
            "sampling code — it delegates to the shared strategies factory, "
            "which is not derived from ai-toolkit."
        )

    def test_the_readme_does_not_claim_it_either(self):
        assert "Timestep sampling strategies for flow-matching models are derived" \
            not in _readme(), (
            "README's Ostris entry claims the timestep derivation again. The "
            "acknowledgment itself is correct and should stay — ai-toolkit IS a "
            "real source here — but for ideogram4's Qwen3-VL port, which is what "
            "NOTICE's MIT entry covers."
        )

    def test_the_notice_obligation_survived(self):
        """Deleting a false claim must not delete a true one.

        The MIT terms are owed for ideogram4's verbatim port regardless of what
        any comment says, so the NOTICE entry is the thing that must NOT move.
        """
        notice = REPO_ROOT / "NOTICE"
        if not notice.exists():
            pytest.skip("NOTICE not in this checkout")
        text = notice.read_text(encoding="utf-8")
        assert "ostris/ai-toolkit" in text
        assert "Copyright (c) 2024 Ostris, LLC" in text


class TestTheseMatchersActuallyFail:
    """Vacuity checks — these assertions are all 'X in text', which passes
    identically against a file that never had the property."""

    def test_the_pointer_parser_finds_a_planted_credit(self):
        planted = (
            "- **[Someone](https://github.com/someone/somelib)** — words. "
            "Credited in [`f.py`](backend/f.py).\n"
        )
        m = _BULLET.search(planted)
        assert m is not None, "the bullet pattern no longer matches README's form"
        rest = m.group("rest")
        assert _POINTER.findall(rest[rest.find("Credited in"):]) == ["backend/f.py"]

    def test_the_parser_ignores_an_acknowledgment_with_no_file_claim(self):
        """Prove the negative: bullets without a pointer must not be flagged.

        Several entries thank a project for inspiration rather than for code.
        Demanding a file from those would push someone to invent one.
        """
        planted = "- **[Someone](https://github.com/someone/somelib)** — inspiration only.\n"
        m = _BULLET.search(planted)
        assert m is not None
        assert "Credited in" not in m.group("rest")

    def test_the_mention_check_would_catch_the_original_defect(self):
        """The exact pair that shipped: a claim about a file that does not
        support it. Reconstructed rather than described."""
        text = "FLUX.2 Trainer — family-specific hooks.\nNo attribution here.\n"
        assert "ai-toolkit" not in text.lower() and "ostris" not in text.lower()
