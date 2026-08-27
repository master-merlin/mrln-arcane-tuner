"""Pins the scope claim that retires the Taylor cache's output-equivalence gate.

The clean-room Taylor feature cache in ``boogu_image/taylor_cache.py`` is not
reachable from any shipped code path: nothing sets ``enable_taylorseer``,
nothing calls ``cache_init``, and no configuration surface exposes either. Its
output-equivalence to the removed implementation is therefore **not a risk for
this release** — and, equally importantly, **has not been verified**.

That pair is the whole point. "Not a risk" alone reads as "checked and fine".

**This claim has a shelf life.** It stops being true the moment somebody wires
the feature up, and nothing else in the tree would notice. This test is what
notices. When it fails, the person who made it reachable inherits the
verification obligation instead of rediscovering it years later.

Nothing here bears on the licence work. The GPL-derived implementation was
**distributed**, and distribution is the trigger — reachability never was. The
removal was necessary regardless of whether the code ever executed.

--------------------------------------------------------------------------
If you are the person who made this test fail: how to run the oracle
--------------------------------------------------------------------------
The required check is a differential run of the replacement against the
pre-change implementation (a commit before ``16bb4079``, where the GPL
originals were removed): identical seeded inputs, a schedule that exercises
warm-up, the divided-difference ladder and the order ceiling, and a comparison
of every cached-step output. Fix the acceptance criterion in writing *before*
the numbers exist, so it cannot be adjusted to whatever comes out.

The replacement was written clean-room. Running the old code as a black-box
oracle does not breach that; two controls keep it that way, and both are
engineering rather than good intentions:

1. **Run it in a subprocess with ``sys.tracebacklimit = 0``, capturing only
   numeric output.** The real exposure is not opening the file — it is a
   traceback printing the old implementation's source lines into the reader's
   context. Engineer that out; do not rely on remembering it.
2. **A difference is adjudicated against the ICCV 2025 paper and the
   behavioural spec — never used to tune the implementation to match.** An
   oracle used to *derive* behaviour rather than to *check* it is a
   reverse-engineering channel, and the clean-room claim does not survive it.
"""

from __future__ import annotations

import ast
from pathlib import Path

BACKEND_APP = Path(__file__).resolve().parents[4] / "app"

FAMILY = BACKEND_APP / "engine" / "models" / "families" / "boogu_image"

#: The claim is about FIRST-PARTY code enabling the cache. Every family's
#: ``vendor/`` tree is third-party: boogu's own propagates the flag between its
#: layers (guarded by the flag itself), and another family's ``cache_init``
#: would be an unrelated function that happens to share a name. The vendored
#: caller's default is pinned separately below, which is the case that matters.
#: ``taylor_cache.py`` declares the names rather than using them.
EXEMPT_FILES = (FAMILY / "taylor_cache.py",)
EXEMPT_DIR_NAME = "vendor"

WIRED_UP = (
    "The Taylor feature cache is now reachable from shipped code. "
    "The differential oracle run against the pre-change implementation is now "
    "REQUIRED before release — see LANE-3 and this module's docstring for the "
    "method and the two clean-room controls. Do not delete this test to make "
    "the failure go away; it is the only thing carrying that obligation."
)


def _production_sources() -> list[Path]:
    """Every first-party backend module except the exempt paths above."""
    return [
        path
        for path in BACKEND_APP.rglob("*.py")
        if EXEMPT_DIR_NAME not in path.parts and path not in EXEMPT_FILES
    ]


def _parse(path: Path) -> ast.AST:
    """Parse a source file, tolerating a UTF-8 BOM.

    ``utf-8`` leaves a BOM in the string as U+FEFF and ``ast.parse`` rejects it;
    at least one vendored module in this tree carries one. Reading as
    ``utf-8-sig`` strips it. Without this the scan raises instead of reporting,
    which would look exactly like the cache having become reachable.
    """
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _enabling_assignments(tree: ast.AST) -> list[int]:
    """Line numbers assigning a non-``False`` value to ``enable_taylorseer``."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            name = (
                target.attr
                if isinstance(target, ast.Attribute)
                else target.id
                if isinstance(target, ast.Name)
                else None
            )
            if name != "enable_taylorseer":
                continue
            # `= False` is the disabled default and is what we want to see.
            if isinstance(node.value, ast.Constant) and node.value.value is False:
                continue
            lines.append(node.lineno)
    return lines


def _cache_init_calls(tree: ast.AST) -> list[int]:
    """Line numbers calling ``cache_init``.

    Matched on the AST, so ``taylor_cache_init`` (a different function) cannot
    trip it the way a substring search would, and an ``import`` of the name is
    not a call.
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else None
        )
        if name == "cache_init":
            lines.append(node.lineno)
    return lines


class TestTaylorCacheIsUnreachable:
    """The scope claim recorded on the release audit, asserted rather than asserted-to."""

    def test_no_shipped_module_enables_the_taylor_cache(self):
        offenders: list[str] = []
        for path in _production_sources():
            tree = _parse(path)
            for lineno in _enabling_assignments(tree):
                offenders.append(f"{path}:{lineno} sets enable_taylorseer")
            for lineno in _cache_init_calls(tree):
                offenders.append(f"{path}:{lineno} calls cache_init()")

        assert not offenders, WIRED_UP + "\n  " + "\n  ".join(offenders)

    def test_the_vendored_transformer_still_defaults_the_flag_off(self):
        # The exemption above covers upstream propagating an already-true flag
        # between its own layers. It must not become a way to flip the default:
        # every CONSTANT assignment to self.enable_taylorseer stays False.
        transformer = (
            FAMILY / "vendor" / "models" / "transformers" / "transformer_boogu.py"
        )
        assert transformer.is_file(), f"expected the vendored caller at {transformer}"

        tree = _parse(transformer)
        enabled: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(
                node.value, ast.Constant
            ):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "enable_taylorseer"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and node.value.value is not False
                ):
                    enabled.append(node.lineno)

        assert not enabled, (
            WIRED_UP + f"\n  {transformer} defaults the flag on at lines {enabled}"
        )

    def test_the_guard_can_actually_fail(self):
        # A guard nobody has seen fail is a comment. This runs the same two
        # detectors over source that DOES enable the cache, so the assertions
        # above are known to have teeth rather than assumed to.
        wired = ast.parse(
            "def go(pipe):\n"
            "    pipe.transformer.enable_taylorseer = True\n"
            "    cache_dic, current = cache_init(pipe, 20)\n"
        )
        assert _enabling_assignments(wired) == [2]
        assert _cache_init_calls(wired) == [3]

        # ...and does not fire on the disabled default or on the similarly
        # named helper, which is where a substring search would go wrong.
        benign = ast.parse(
            "class T:\n"
            "    def __init__(self):\n"
            "        self.enable_taylorseer = False\n"
            "    def step(self, c, n):\n"
            "        taylor_cache_init(c, n)\n"
        )
        assert _enabling_assignments(benign) == []
        assert _cache_init_calls(benign) == []

    def test_it_is_scanning_a_real_tree(self):
        # Guards the guard: an empty or mislocated file list would make every
        # assertion above pass vacuously.
        sources = _production_sources()
        assert len(sources) > 100, f"only {len(sources)} modules found under {BACKEND_APP}"
        assert all(EXEMPT_DIR_NAME not in p.parts for p in sources)
        # The module under discussion must actually be in the tree being scanned,
        # or "nothing enables it" would be true for the wrong reason.
        assert (FAMILY / "taylor_cache.py").is_file()

    def test_every_scanned_module_parses(self, tmp_path):
        # A parse error aborts the scan, which reads exactly like the cache
        # having become reachable. It happened: a vendored module carries a
        # UTF-8 BOM, which `encoding="utf-8"` leaves as U+FEFF for ast.parse to
        # reject. Both halves are pinned -- the whole tree parses, and the BOM
        # case specifically.
        for path in _production_sources():
            _parse(path)

        bom_file = tmp_path / "bom.py"
        bom_file.write_bytes(b"\xef\xbb\xbf# ruff: noqa\nx = 1\n")
        _parse(bom_file)
