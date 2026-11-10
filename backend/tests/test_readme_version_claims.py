"""The README's version claims match the code, AND match each other.

THE LESSON THIS FILE EXISTS FOR: registering the README's claims in
a docs manifest (LANE-8) proved three of them false, and **two
of the three were the document contradicting itself**:

* line 5 announced ``PyTorch 2.10`` while line 82, twelve lines further down in
  the same file, told the reader to ``pip install torch==2.12.1``;
* the architecture diagram announced ``Angular 21 SPA`` while line 5 of the same
  file said ``Angular 22``.

A test that only compared the README against ``requirements.txt`` would have
caught the *first* half of each pair and left the second free to drift the other
way later. So the checks here are two-sided: every claim about a thing must
agree with every other claim about that thing, **and** the agreed value must
match the source of truth.

WHERE THE VERSIONS COME FROM. ``backend/torch-stack.cfg`` is the one place:

* the LOCAL dev venv runs ``backend/requirements.txt``'s pin, which a test holds
  equal to the cfg's default set (older machines get the cu126 set, which the
  README names as plain ``torch <version>``, never as ``PyTorch <version>``);
* the CONTAINER installs the set its ``Dockerfile`` names in ``ARG TORCH_CUDA``,
  by name, from the same cfg. The Dockerfile names no version of its own.

So the default image and the local venv run the same torch, and the README's
``PyTorch <version>`` claims are exactly that one version; the cu126 image's
torch is named once, as ``torch <version>``.

KNOWN LIMIT, stated so it is inherited rather than rediscovered. These checks
read PROSE, and prose can legitimately name an old version — a changelog line
"PyTorch 2.10 → 2.12.1", or a note about what an older release shipped, would
be flagged here as a false claim. That is the same wall a sibling guard hit
this week when a regex over source flagged the very comments explaining it.
When it happens the answer is to narrow WHERE the check looks (as
``readme_torch_install_pins`` already does, by reading only ``pip install``
lines), never to exempt a file or delete the assertion — exempting is how
guards rot into decoration.

Anchored on ``__file__`` so it runs identically from any working directory
(ARCHITECTURE D10 invariant 9).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

README = REPO_ROOT / "README.md"
REQUIREMENTS = REPO_ROOT / "backend" / "requirements.txt"
DOCKERFILE = REPO_ROOT / "Dockerfile"
PACKAGE_JSON = REPO_ROOT / "frontend" / "package.json"

#: ``PyTorch 2.12.1`` and the "one local, one container" shape
#: ``PyTorch 2.12.1 local / 2.11.0 container``. The version must follow the word
#: directly: prose like "Install PyTorch with CUDA 13.0" names no torch version
#: and must not be read as claiming 13.0.
_README_TORCH = re.compile(
    r"PyTorch\s+(\d+\.\d+(?:\.\d+)?)"          # first version
    r"(?:\s+\w+)?"                              # optional label, e.g. "local"
    r"(?:\s*/\s*(\d+\.\d+(?:\.\d+)?))?"        # optional "/ <version>"
)

#: ``torch==2.12.1`` in a shell block. ``torchaudio==`` / ``torchvision==`` do
#: not match, because ``torch==`` requires the ``==`` immediately after.
_TORCH_PIN = re.compile(r"\btorch==(\d+\.\d+(?:\.\d+)?)")

_README_ANGULAR = re.compile(r"\bAngular\s+(\d+)")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _local_torch_pin(requirements: str) -> str:
    match = _TORCH_PIN.search(requirements)
    assert match, "backend/requirements.txt no longer pins torch=="
    return match.group(1)


TORCH_STACK_CFG = REPO_ROOT / "backend" / "torch-stack.cfg"


def _cfg_values(cfg: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in cfg.splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def _set_torch(cfg: str, set_name: str) -> str:
    values = _cfg_values(cfg)
    key = f"{set_name.upper()}_TORCH"
    assert key in values, f"backend/torch-stack.cfg has no {key}"
    return values[key]


def _container_torch_pin(dockerfile: str, cfg: str) -> str:
    """The default image's torch: the cfg set the Dockerfile's ``ARG TORCH_CUDA`` names."""
    match = re.search(r"^ARG TORCH_CUDA=(\S+)", dockerfile, re.M)
    assert match, "the Dockerfile no longer names its torch set in ARG TORCH_CUDA"
    return _set_torch(cfg, match.group(1))


def _angular_major(package_json: str) -> str:
    spec = json.loads(package_json)["dependencies"]["@angular/core"]
    match = re.search(r"(\d+)", spec)
    assert match, f"cannot read an Angular major out of {spec!r}"
    return match.group(1)


def readme_torch_claims(readme: str) -> set[str]:
    """Every PyTorch version the README announces, as written."""
    claims: set[str] = set()
    for first, second in _README_TORCH.findall(readme):
        claims.add(first)
        if second:
            claims.add(second)
    return claims


def readme_torch_install_pins(readme: str) -> set[str]:
    """Every ``torch==`` the README tells the reader to **install**.

    Restricted to ``pip install`` lines on purpose. The README also writes
    ``torch==2.11.0`` while explaining that *torchaudio's own metadata* pins it,
    which is a true statement about a third party and not an instruction — and
    the same sentence appears as a comment in ``requirements.txt``. A check that
    flags it is wrong, and a check that is wrong on a correct tree gets deleted
    rather than fixed.
    """
    return {
        pin
        for line in readme.splitlines()
        if "pip install" in line
        for pin in _TORCH_PIN.findall(line)
    }


def readme_angular_claims(readme: str) -> set[str]:
    """Every Angular major the README names."""
    return set(_README_ANGULAR.findall(readme))


class TestPyTorch:
    def test_the_readme_announces_exactly_the_pins_that_exist(self):
        """Every ``PyTorch <version>`` is the local pin or the default image's,
        and both are named.

        Set equality rather than "every claim is one of them", because the
        weaker form passes a README that has quietly dropped one entirely.
        """
        local = _local_torch_pin(_read(REQUIREMENTS))
        container = _container_torch_pin(_read(DOCKERFILE), _read(TORCH_STACK_CFG))
        assert readme_torch_claims(_read(README)) == {local, container}, (
            "README.md's PyTorch versions no longer match the pins that exist: "
            f"local {local} (backend/requirements.txt), default image {container} "
            "(the torch-stack.cfg set the Dockerfile's ARG TORCH_CUDA names)."
        )

    def test_the_readme_names_the_cu126_images_torch(self):
        """The fallback image carries the cu126 set, whose torch the README states."""
        cu126 = _set_torch(_read(TORCH_STACK_CFG), "cu126")
        assert f"torch {cu126}" in _read(README), (
            f"README.md does not name the cu126 set's torch ({cu126}) as `torch {cu126}`"
        )

    def test_no_install_instruction_names_its_own_torch(self):
        """The README installs torch only through torch_stack.py.

        This half once caught the original defect from the other side: line 5
        said 2.10 while the install block said 2.12.1, so the file disagreed
        with itself before it disagreed with anything else. Since the selector
        (LANE-138) the install block names no version at all, so it cannot
        disagree; a `pip install torch==` reappearing is the drift to catch.
        """
        readme = _read(README)
        pins = readme_torch_install_pins(readme)
        assert pins == set(), (
            f"README.md tells the reader to install torch=={sorted(pins)} by hand; "
            "the torch stack comes from `python torch_stack.py install` (torch-stack.cfg)"
        )
        assert "torch_stack.py install" in readme, "README.md no longer shows how torch is installed"

    def test_the_container_pin_follows_the_dockerfiles_set(self):
        """Anti-vacuity: the image's torch is read through ARG TORCH_CUDA, so a
        Dockerfile naming the other set yields the other set's torch."""
        cfg = _read(TORCH_STACK_CFG)
        dockerfile = _read(DOCKERFILE)
        named = re.search(r"^ARG TORCH_CUDA=(\S+)", dockerfile, re.M).group(1)
        other = next(s for s in _cfg_values(cfg)["SETS"].split() if s != named)
        swapped = dockerfile.replace(f"ARG TORCH_CUDA={named}", f"ARG TORCH_CUDA={other}", 1)
        assert _container_torch_pin(swapped, cfg) == _set_torch(cfg, other)
        assert _set_torch(cfg, other) != _set_torch(cfg, named)


class TestAngular:
    def test_every_angular_major_named_agrees_with_package_json(self):
        """One assertion covering both directions.

        Every mention must equal the pinned major, so mentions also agree with
        each other — which is what the stale `Angular 21 SPA` in the
        architecture diagram needed, sitting as it did in a file whose own
        header line said 22.
        """
        major = _angular_major(_read(PACKAGE_JSON))
        claims = readme_angular_claims(_read(README))
        assert claims, "README.md no longer names an Angular version"
        assert claims == {major}, (
            f"README.md names Angular {sorted(claims)} while "
            f"frontend/package.json pins major {major}"
        )


class TestTheGuardCanActuallyFail:
    """Positive controls, driven at synthetic text.

    Without these, every assertion above is satisfied on the day the README is
    correct by an extractor that finds nothing at all — the failure mode that
    let ten non-existent paths through the docs manifest in this same lane.
    """

    def test_a_disagreeing_pair_inside_one_document_is_caught(self):
        readme = (
            "`v0.1` PyTorch 2.10 · Angular 22\n"
            "pip install torch==2.12.1\n"
            "the image bundles PyTorch 2.11.0\n"
        )
        assert readme_torch_claims(readme) == {"2.10", "2.11.0"}
        assert readme_torch_install_pins(readme) == {"2.12.1"}
        # The document contradicts itself: prose says 2.10, the install line
        # says 2.12.1, and neither check would pass against a 2.12.1 pin.
        assert readme_torch_claims(readme) != {"2.12.1", "2.11.0"}

    def test_the_local_container_shape_yields_both_versions(self):
        assert readme_torch_claims("PyTorch 2.12.1 local / 2.11.0 container") == {
            "2.12.1",
            "2.11.0",
        }

    def test_prose_without_a_version_claims_nothing(self):
        """`Install PyTorch with CUDA 13.0` must not be read as torch 13.0.

        A version-hunting regex that grabs the next number on the line turns
        every unrelated mention into a false claim, and a check that cries wolf
        gets deleted rather than fixed.
        """
        assert readme_torch_claims("Install PyTorch with CUDA 13.0 support") == set()
        assert readme_torch_claims("PyTorch · Diffusers · PEFT · SQLite") == set()

    def test_sibling_packages_are_not_read_as_the_torch_pin(self):
        text = "pip install torchaudio==2.11.0 torchvision==0.27.1"
        assert readme_torch_install_pins(text) == set()

    def test_prose_about_another_packages_pin_is_not_an_instruction(self):
        """The first draft of this check failed on a CORRECT README.

        `README.md:86` explains that *torchaudio's* metadata declares
        `torch==2.11.0` — a true statement about a third party, sitting eleven
        lines from the install line that says 2.12.1. Reading it as our own
        instruction made the guard report a defect that was not there, which is
        how guards get switched off.
        """
        readme = (
            "# metadata pins torch==2.11.0, so it must be installed --no-deps\n"
            "pip install torch==2.12.1 torchvision==0.27.1\n"
        )
        assert readme_torch_install_pins(readme) == {"2.12.1"}

    def test_a_stale_angular_major_is_caught(self):
        assert readme_angular_claims("Angular 21 SPA and Angular 22 elsewhere") == {
            "21",
            "22",
        }

    def test_the_angular_major_is_read_out_of_the_pin(self):
        assert _angular_major('{"dependencies": {"@angular/core": "22.1.3"}}') == "22"

    def test_a_missing_pin_is_an_error_not_a_pass(self):
        with pytest.raises(AssertionError):
            _local_torch_pin("numpy==2.0.0\n")
