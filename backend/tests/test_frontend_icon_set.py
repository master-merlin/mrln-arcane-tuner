"""The frontend must not import the whole Lucide icon barrel.

On 2026-09-03 an ordinary `@lucide/angular` minor bump (1.18.0 -> 1.37.0) took
the app's initial bundle from 1.68 MB to 6.34 MB raw and failed the production
build's 2.5 MB budget. Nothing in the app had changed. The cause was
``import { icons }`` -- the barrel holding every icon Lucide ships -- indexed
with a key built at runtime (``Lucide${name}``), which no bundler can narrow.
So every icon shipped, and each upstream release made the app bigger for free.

Replacing it with a named map of the icons actually used
(`frontend/src/app/icons/icon-set.ts`) took main to 538 kB, below even the
pre-bump figure, because the barrel had been costing more than a megabyte all
along.

Why this check is in the PYTHON suite. It has to read every frontend source
file, and the Angular test environment has no filesystem: its tsconfig carries
no node types, deliberately. The backend suite already walks the whole tree
cheaply (`test_source_encoding.py`, `test_container_hardening.py`), so this
runs in about a second on every gate rather than never. The type-level half of
the guard -- that `IconKey` has not been widened back to `string` -- lives
where the type does, in `icon-set.spec.ts`. Neither half covers the other.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"

#: `import { … icons … } from '@lucide/angular'` -- the barrel, not one icon.
#: Word-bounded so that `LucideDatabase`, `provideLucideIcons` and a local
#: `iconSet` do not match: a guard that fires on legitimate imports gets
#: deleted rather than obeyed.
BARREL_IMPORT = re.compile(
    r"import\s*\{[^}]*\bicons\b[^}]*\}\s*from\s*['\"]@lucide/angular['\"]",
    re.DOTALL,
)


def _frontend_sources() -> list[Path]:
    if not FRONTEND_SRC.is_dir():
        pytest.skip("frontend/src is not present in this checkout")
    return sorted(FRONTEND_SRC.rglob("*.ts"))


def test_no_source_file_imports_the_lucide_icons_barrel():
    files = _frontend_sources()
    assert files, "walked zero frontend sources -- the glob is wrong"
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in files
        if BARREL_IMPORT.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        f"these files import the `icons` barrel from @lucide/angular: {offenders}. "
        "It holds every icon Lucide ships and cannot be tree-shaken when indexed "
        "by a computed key -- that put 6.34 MB into main.js on 2026-09-03 and "
        "failed the production build. Add the specific icons to "
        "frontend/src/app/icons/icon-set.ts instead; its header says how, and "
        "`node tools/collect-icon-usage.mjs` will list what the tree uses."
    )


def test_the_curated_set_is_what_the_components_actually_use():
    """The positive: the replacement is in place, not merely the barrel absent.

    Without this, deleting `icon-set.ts` entirely would leave the test above
    green -- an absence of the wrong thing is not the presence of the right one.
    """
    icon_set = FRONTEND_SRC / "app" / "icons" / "icon-set.ts"
    assert icon_set.is_file(), "frontend/src/app/icons/icon-set.ts is gone"
    text = icon_set.read_text(encoding="utf-8")
    named = re.findall(r"^\s{4}Lucide[A-Za-z0-9]+,$", text, re.MULTILINE)
    assert len(named) >= 50, (
        f"icon-set.ts declares only {len(named)} named imports. It is generated "
        "from real usage and the app uses about a hundred; a number this small "
        "means the file was truncated or the format moved."
    )
    assert "export type IconKey = keyof typeof iconSet" in text, (
        "IconKey must stay `keyof typeof iconSet`. Widening it to `string` "
        "removes the compile-time proof that a requested icon exists, which is "
        "the only thing standing between a typo and a blank space in the UI."
    )


def test_the_matcher_catches_the_import_that_caused_the_regression():
    """Vacuity check: this is a substring-shaped guard, so prove it fires."""
    assert BARREL_IMPORT.search("import { icons } from '@lucide/angular';")
    assert BARREL_IMPORT.search(
        "import { LucideDynamicIcon, type LucideIcon, icons } from '@lucide/angular';"
    )
    assert BARREL_IMPORT.search("import {\n    icons,\n} from '@lucide/angular';")


def test_the_matcher_leaves_legitimate_named_imports_alone():
    """The inverse control, and the reason the pattern is word-bounded."""
    for legitimate in (
        "import { LucideDatabase } from '@lucide/angular';",
        "import { LucideDynamicIcon, type LucideIconData } from '@lucide/angular';",
        "import { provideLucideIcons } from '@lucide/angular';",
        "import { iconSet } from './icon-set';",
        "import { type IconKey, iconSet } from './icon-set';",
    ):
        assert not BARREL_IMPORT.search(legitimate), f"false positive on {legitimate!r}"
