"""Repo-hygiene guards for the frontend's public-release posture.

RULE-20 class S guards for Task 3 of the frontend public-release hardening plan
(F-06, F-08, NEW-1, NEW-2). Each pins a property that is invisible in normal use
and silently re-rots on the next edit:

* the app stays same-origin (no third-party font/CDN references),
* the shipped page title is the product's, not Angular's scaffold default,
* `window.open` never yields a live `opener` handle to the opened document.

These live in the Python suite for the same reason as
``test_frontend_url_encoding_guard.py``: the scans need filesystem access and the
frontend tsconfig has no ``@types/node``. Taking a dependency to host a guard
inverts the cost.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = REPO_ROOT / "frontend"
SRC = FRONTEND / "src"
INDEX_HTML = SRC / "index.html"

pytestmark = pytest.mark.skipif(
    not SRC.is_dir(), reason="frontend sources not present"
)

# Hosts that would re-introduce the cross-origin dependency Task 3 removed.
THIRD_PARTY_HOSTS = re.compile(
    r"https?://(?:fonts\.googleapis\.com|fonts\.gstatic\.com|[a-z0-9.-]*\.jsdelivr\.net"
    r"|cdnjs\.cloudflare\.com|unpkg\.com|cdn\.jsdelivr\.net)",
    re.IGNORECASE,
)


def _source_files() -> list[Path]:
    out: list[Path] = []
    for pattern in ("*.ts", "*.html", "*.css", "*.scss"):
        out.extend(p for p in SRC.rglob(pattern) if not p.name.endswith(".spec.ts"))
    return sorted(out)


def test_no_third_party_asset_origins() -> None:
    """The app must be same-origin.

    A third-party font or CDN link is three defects at once: an availability
    dependency (an offline or air-gapped install renders wrong or not at all), a
    privacy leak (every page load discloses the user's IP to that host), and a
    CSP that has to allow an origin nobody controls. Fonts are vendored under
    ``frontend/public/fonts`` instead.
    """
    offenders = []
    for path in _source_files():
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if THIRD_PARTY_HOSTS.search(line):
                rel = path.relative_to(FRONTEND).as_posix()
                offenders.append(f"{rel}:{lineno}  {line.strip()[:100]}")

    assert offenders == [], (
        "Third-party asset origin referenced from frontend sources. Vendor the "
        "asset under frontend/public/ instead — see "
        "_harness/research/csp-policy-for-request-2.md.\n  " + "\n  ".join(offenders)
    )


def test_guard_would_catch_a_third_party_host() -> None:
    """Prove the negative: a scan that matches nothing is not a guard."""
    sample = '<link href="https://fonts.googleapis.com/css2?family=Inter" rel="stylesheet">'
    assert THIRD_PARTY_HOSTS.search(sample) is not None


def test_page_title_is_the_product_not_the_scaffold() -> None:
    """`<title>Frontend</title>` shipped as the product's page title."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    match = re.search(r"<title>(.*?)</title>", html, re.S)
    assert match is not None, "index.html has no <title>"
    title = match.group(1).strip()
    assert title.lower() not in {"frontend", "app", "angular", ""}, (
        f"index.html still carries a scaffold page title: {title!r}"
    )
    assert "MRLN" in title, f"expected the product name in the page title, got {title!r}"


def test_vendored_fonts_are_present_and_licensed() -> None:
    """Self-hosting is only complete if the files and their licence ship too."""
    fonts_dir = FRONTEND / "public" / "fonts"
    assert fonts_dir.is_dir(), "frontend/public/fonts is missing"

    woff2 = sorted(p.name for p in fonts_dir.glob("*.woff2"))
    assert woff2, "no vendored woff2 files"
    assert (fonts_dir / "fonts.css").is_file(), "fonts.css is missing"

    # Redistributing OFL fonts requires shipping the licence with them.
    ofl = fonts_dir / "OFL.txt"
    assert ofl.is_file(), "OFL.txt missing beside the vendored fonts"
    text = ofl.read_text(encoding="utf-8")
    assert "SIL Open Font License" in text
    for family in ("Inter", "JetBrains Mono"):
        assert family in text, f"{family} not attributed in OFL.txt"

    # Every face referenced by the stylesheet must actually exist on disk.
    css = (fonts_dir / "fonts.css").read_text(encoding="utf-8")
    for referenced in re.findall(r"url\('/fonts/([^']+)'\)", css):
        assert (fonts_dir / referenced).is_file(), (
            f"fonts.css references {referenced}, which is not present"
        )


def test_window_open_always_severs_the_opener() -> None:
    """Without `noopener`, the opened document gets a live `window.opener`.

    That handle lets the target navigate this app's tab. These are export
    downloads pointed at backend URLs, so it is not currently exploitable — but
    it is one changed URL away from being so, and the fix costs nothing.
    """
    offenders = []
    for path in _source_files():
        if path.suffix != ".ts":
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"window\.open\(([^;]*?)\)\s*;", text, re.S):
            args = match.group(1)
            if "noopener" in args:
                continue
            lineno = text.count("\n", 0, match.start()) + 1
            rel = path.relative_to(FRONTEND).as_posix()
            offenders.append(f"{rel}:{lineno}  window.open({args.strip()[:80]})")

    assert offenders == [], (
        "window.open() without 'noopener'. Pass 'noopener,noreferrer' as the "
        "third argument.\n  " + "\n  ".join(offenders)
    )


def test_no_inline_style_attributes_in_index_html() -> None:
    """Inline `style=` attributes cannot be hashed by a CSP.

    A `<style>` block can be hashed; a style attribute needs 'unsafe-inline' (or
    'unsafe-hashes') in style-src-attr. Keeping index.html free of them is what
    lets the delivered policy stay tight — see
    _harness/research/csp-policy-for-request-2.md.
    """
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert re.search(r"\sstyle=\"", html) is None, (
        "index.html has an inline style attribute; move it into the hashed "
        "<style> block so the CSP does not need 'unsafe-inline' for it"
    )
