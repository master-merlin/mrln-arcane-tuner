"""The CSP permits exactly what the served page needs, and no more.

THE LESSON THIS FILE EXISTS FOR: the policy applies to what FastAPI *serves*,
which is the BUILT ``index.html``, not ``frontend/src/index.html``. Every
source-level check passed while the built file contained
``onload="this.media='all'"`` — an inline event handler that a strict
``script-src`` blocks, which would have left the entire application unstyled
because the deferred stylesheet never flips off ``media="print"``. Angular's
``inlineCritical`` created that construct at build time, so no amount of reading
the source could have found it.

So the coherence tests here prefer the built artifact and say so loudly when
they fall back to source — a skip that silently degrades to checking the wrong
file is how this got missed the first time.
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

import pytest

from app.api._security_headers import (
    CSP_DIRECTIVES,
    CSP_HEADER_VALUE,
    INLINE_BOOTSTRAP_SHA256,
    build_csp,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_INDEX = REPO_ROOT / "frontend" / "src" / "index.html"
BUILT_INDEX = REPO_ROOT / "frontend" / "dist" / "frontend" / "browser" / "index.html"


def _served_index() -> tuple[str, str]:
    """Return (html, which) preferring the BUILT file over source."""
    if BUILT_INDEX.exists():
        return BUILT_INDEX.read_text(encoding="utf-8"), "built"
    if SRC_INDEX.exists():
        return SRC_INDEX.read_text(encoding="utf-8"), "source"
    pytest.skip("no index.html in this checkout")


def _sha256_of(text: str) -> str:
    return "sha256-" + base64.b64encode(hashlib.sha256(text.encode()).digest()).decode()


def _inline_scripts(html: str) -> list[str]:
    """Inline <script> bodies — those WITHOUT a src attribute."""
    return [
        m.group(1)
        for m in re.finditer(
            r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S
        )
    ]


class TestNoInlineEventHandlers:
    """The defect that nearly shipped, pinned.

    ``script-src`` carries the real protection and deliberately has no
    ``'unsafe-inline'`` / ``'unsafe-hashes'``, so ANY inline event handler in
    the served page is dead on arrival. Angular's ``inlineCritical`` emits one;
    it is disabled in angular.json's production config for exactly this reason.
    """

    def test_served_page_has_no_inline_event_handlers(self):
        html, which = _served_index()
        handlers = sorted(set(re.findall(r"\s(on[a-z]+)\s*=", html)))
        assert not handlers, (
            f"the {which} index.html contains inline event handler(s) {handlers}, "
            "which this CSP blocks. If this is `onload` on a stylesheet link, "
            "Angular's optimization.styles.inlineCritical has been re-enabled — "
            "the app will render completely unstyled. A nonce does NOT fix this; "
            "nonces do not apply to event-handler attributes."
        )

    def test_this_check_would_have_caught_the_real_regression(self):
        """Vacuity guard: the matcher must fire on the actual markup."""
        real = (
            '<link rel="stylesheet" href="styles-SJQIASZF.css" '
            "media=\"print\" onload=\"this.media='all'\">"
        )
        assert re.findall(r"\s(on[a-z]+)\s*=", real) == ["onload"]

    def test_script_src_really_does_forbid_inline(self):
        """If either token appears, the test above stops meaning anything."""
        script_src = CSP_DIRECTIVES["script-src"]
        assert "'unsafe-inline'" not in script_src
        assert "'unsafe-hashes'" not in script_src


class TestInlineScriptHashMatchesTheServedPage:
    def test_the_bootstrap_hash_is_current(self):
        html, which = _served_index()
        scripts = _inline_scripts(html)
        assert scripts, f"no inline script found in the {which} index.html"

        hashes = [_sha256_of(s) for s in scripts]
        assert INLINE_BOOTSTRAP_SHA256 in hashes, (
            f"the hash in _security_headers.py matches no inline script in the "
            f"{which} index.html.\n  policy: {INLINE_BOOTSTRAP_SHA256}\n"
            f"  actual: {hashes}\n"
            "index.html changed without the policy being regenerated — the app "
            "will break at first paint with a flash of the wrong theme."
        )

    def test_every_inline_script_is_covered(self):
        """One hashed script is not enough if the build added a second."""
        html, which = _served_index()
        allowed = {t for t in CSP_DIRECTIVES["script-src"] if t.startswith("'sha256-")}
        for body in _inline_scripts(html):
            token = f"'{_sha256_of(body)}'"
            assert token in allowed, (
                f"an inline script in the {which} index.html is not hashed in the "
                f"policy and will be blocked:\n{body[:200]}"
            )


class TestExternalOriginsAreCoherent:
    """Every origin the page references must be permitted by the policy.

    This is the test that failed on the pre-merge branch (Google Fonts in
    index.html, `font-src 'self'` in the policy) and was the evidence for
    holding the header back. Keeping it permanent means the next reintroduction
    of a third-party asset fails the gate instead of the browser.
    """

    def test_no_external_origins_in_the_served_page(self):
        html, which = _served_index()
        origins = sorted(set(re.findall(r"https?://[^\"'\s>]+", html)))
        assert not origins, (
            f"the {which} index.html references external origin(s) {origins}, "
            "but the policy is same-origin only. Either self-host the asset "
            "(what the fonts change did) or add the origin to the right "
            "directive — do not leave the page loading something the policy "
            "blocks."
        )


class TestTheHeaderActuallyReachesResponses:
    """A policy that is never sent protects nothing.

    Also pins middleware ORDER. Starlette's ``add_middleware`` prepends, so the
    last registration is the outermost — meaning this middleware must be
    registered AFTER the auth gate to cover the 401 the gate short-circuits
    with. Getting that backwards is easy, silent, and leaves exactly the
    responses an injection wants uncovered, so it is asserted rather than
    reasoned about.
    """

    def test_api_responses_carry_the_policy(self, client):
        resp = client.get("/api/health")
        assert resp.headers.get("Content-Security-Policy") == CSP_HEADER_VALUE

    def test_csp_is_on_the_auth_gate_401(self):
        """The short-circuited response must be covered too."""
        from fastapi.testclient import TestClient

        from app.api._security_headers import SecurityHeadersMiddleware
        from app.core.auth import TokenAuthMiddleware

        from fastapi import FastAPI

        probe = FastAPI()

        @probe.get("/api/thing")
        async def _thing():  # pragma: no cover - body never runs when gated
            return {"ok": True}

        # Same registration order as main.py: gate first, headers last.
        probe.add_middleware(TokenAuthMiddleware, token="a-token")
        probe.add_middleware(SecurityHeadersMiddleware)

        resp = TestClient(probe).get("/api/thing")
        assert resp.status_code == 401, "expected the gate to short-circuit"
        assert resp.headers.get("Content-Security-Policy") == CSP_HEADER_VALUE, (
            "the auth gate's 401 has no CSP — SecurityHeadersMiddleware is "
            "registered INSIDE the gate. It must be added last so it is "
            "outermost."
        )

    def test_ordering_matters_and_this_test_can_tell(self):
        """Prove the negative: reversed order really does drop the header.

        Without this, the test above would pass just as happily if the
        middleware covered everything unconditionally, and would not be
        evidence about ordering at all.
        """
        from fastapi.testclient import TestClient

        from app.api._security_headers import SecurityHeadersMiddleware
        from app.core.auth import TokenAuthMiddleware

        from fastapi import FastAPI

        probe = FastAPI()

        @probe.get("/api/thing")
        async def _thing():  # pragma: no cover
            return {"ok": True}

        # Reversed: headers first (inner), gate last (outer).
        probe.add_middleware(SecurityHeadersMiddleware)
        probe.add_middleware(TokenAuthMiddleware, token="a-token")

        resp = TestClient(probe).get("/api/thing")
        assert resp.status_code == 401
        assert "Content-Security-Policy" not in resp.headers, (
            "reversed ordering still produced the header, so the ordering test "
            "above proves nothing"
        )


class TestPolicyShape:
    def test_serialization_round_trips(self):
        assert build_csp() == CSP_HEADER_VALUE
        assert CSP_HEADER_VALUE.startswith("default-src 'self'")

    def test_data_uris_are_images_only(self):
        """`data:` is a script-execution vector outside img-src."""
        for name, values in CSP_DIRECTIVES.items():
            if name == "img-src":
                continue
            assert "data:" not in values, f"data: leaked into {name}"

    def test_style_src_unsafe_inline_is_deliberate_and_documented(self):
        """Pinned so nobody 'tightens' it without reading why.

        Angular injects component styles as runtime <style> elements; a
        hash-only style-src breaks every component style in the app. Removing
        this token without the ngCspNonce work is a self-inflicted outage.
        """
        assert "'unsafe-inline'" in CSP_DIRECTIVES["style-src"]
        from app.api import _security_headers

        assert "ngCspNonce" in (_security_headers.__doc__ or ""), (
            "the reason for the unsafe-inline residual must stay next to it"
        )

    @pytest.mark.parametrize(
        "directive,expected",
        [
            ("object-src", "'none'"),
            ("frame-ancestors", "'none'"),
            ("base-uri", "'self'"),
            ("form-action", "'self'"),
        ],
    )
    def test_locked_directives(self, directive, expected):
        assert CSP_DIRECTIVES[directive] == (expected,)

    def test_websocket_schemes_are_present(self):
        """'self' does not reliably cover ws: — the app holds an always-on socket."""
        connect = CSP_DIRECTIVES["connect-src"]
        assert "ws:" in connect and "wss:" in connect
