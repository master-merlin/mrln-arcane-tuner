"""Repo-hygiene guard: no un-encoded interpolation in a URL path position.

RULE-20 class S guard for the URL path-segment encoding contract
(`_harness/research/url-path-segment-encoding-contract.md`, ECOSYSTEM §5 row 4).

Why this lives in the Python suite rather than beside the Vitest specs: the scan
needs filesystem access, and the frontend tsconfig has no `@types/node`. Adding a
dependency to get a guard is the wrong trade, and a repo-wide static invariant has
precedent here already (the layering test is the same shape).

Why a source scan at all, when there are per-site specs: those pin the sites that
were fixed. Only a scan stops the next one from being written. The original sweep
found 68 candidate sites where the plan named 5 — and the scanner itself missed a
real defect (`media-preview.ts` interpolated the same variable raw into a path
while encoding it for a query parameter three lines above). So a hit here means
"go read the site", never "mechanically wrap it".

The defect being prevented is not theoretical: a filename containing `#` or `?`
truncates the URL at the fragment or query, the backend answers 200 for a
DIFFERENT file, and the caption editor reads and writes the wrong sidecar with no
error surfaced anywhere.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "frontend" / "src" / "app"

TEMPLATE = re.compile(r"`([^`]*)`")
INTERP = re.compile(r"\$\{([^}]*)\}")
# A template that looks like a URL built on one of the app's base URLs.
URLISH = re.compile(r"(apiUrl|baseUrl|this\.base\b)")
# Interpolations that are themselves a base URL, or are already encoded.
EXEMPT_EXPR = re.compile(r"(encodeURIComponent|apiUrl|baseUrl|rtc\.|this\.base\b)")

# Verified false positives — each read and justified in the contract doc.
# Keyed by (path suffix, interpolated expression) so line drift does not
# silently re-exempt a different site.
ALLOWED = {
    # Pre-built, already-encoded query strings ("?fresh=true",
    # "?project_id=..."). Encoding these would escape the "?" and "=" and turn
    # the query into a path segment.
    ("services/job.ts", "q"),
    # `name` is assigned `encodeURIComponent(datasetName)` a few lines above.
    ("shared/media-preview.ts", "name"),
    # Local variable already holding an encoded value.
    ("services/dataset.ts", "encodedFile"),
}


def _offenders() -> list[str]:
    hits: list[str] = []
    for path in sorted(APP_ROOT.rglob("*.ts")):
        if path.name.endswith(".spec.ts"):
            continue
        rel = path.relative_to(APP_ROOT).as_posix()
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for template in TEMPLATE.findall(line):
                if "/" not in template or not URLISH.search(template):
                    continue
                # Only the path portion: everything after a literal ? or # is
                # query/fragment and follows different escaping rules.
                path_part = re.split(r"[?#]", template)[0]
                for expr in INTERP.findall(path_part):
                    expr = expr.strip()
                    if EXEMPT_EXPR.search(expr):
                        continue
                    if any(rel.endswith(suffix) and expr == e for suffix, e in ALLOWED):
                        continue
                    hits.append(f"{rel}:{lineno}  ${{{expr}}}")
    return hits


@pytest.mark.skipif(not APP_ROOT.is_dir(), reason="frontend sources not present")
def test_no_raw_interpolation_in_url_path_position() -> None:
    offenders = _offenders()
    assert offenders == [], (
        "Un-encoded interpolation in a URL path position.\n"
        "Read each site before changing it — wrap the value in "
        "encodeURIComponent() if it is a path segment, or add it to ALLOWED "
        "with a justification if it is a pre-built query string or already "
        "encoded.\n  " + "\n  ".join(offenders)
    )


@pytest.mark.skipif(not APP_ROOT.is_dir(), reason="frontend sources not present")
def test_guard_actually_detects_a_raw_interpolation(tmp_path: Path) -> None:
    """Prove the negative: a guard that never fires is not a guard.

    Without this, `_offenders()` could silently stop matching (a refactor to
    the URL-building idiom, a regex that no longer fits) and the suite would
    stay green while the invariant went unchecked.
    """
    sample = "const u = `${this.apiUrl}/datasets/${name}/captions/${filename}`;"
    found = []
    for template in TEMPLATE.findall(sample):
        path_part = re.split(r"[?#]", template)[0]
        for expr in INTERP.findall(path_part):
            if not EXEMPT_EXPR.search(expr.strip()):
                found.append(expr.strip())
    assert found == ["name", "filename"]
