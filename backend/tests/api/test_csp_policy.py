"""The CSP permits exactly what the served page needs, and no more.

THE LESSON THIS FILE EXISTS FOR: the policy applies to what FastAPI *serves*,
which is the BUILT ``index.html``, not ``frontend/src/index.html``. Every
source-level check passed while the built file contained
``onload="this.media='all'"`` — an inline event handler that a strict
``script-src`` blocks, which would have left the entire application unstyled
because the deferred stylesheet never flips off ``media="print"``. Angular's
``inlineCritical`` created that construct at build time, so no amount of reading
the source could have found it.

So the coherence tests here prefer the built artifact, and emit a
``SourceFallbackWarning`` when they fall back to source — a degradation that
silently checks the wrong file is how this got missed the first time.

THE SECOND LESSON, found while fixing the first: the fallback was announced
only inside assertion *failure* messages, so a green run said nothing at all.
A checkout with no ``frontend/dist/`` — which is every fresh clone, since the
build output is gitignored — passed these tests having verified precisely the
file the docstring above says cannot find the defect. That is the original bug's
exact shape living inside the test written to prevent it: correct behaviour,
degraded, announced in a line nobody reads.

The fallback deliberately still passes rather than failing or skipping. A fresh
clone cannot build the frontend before running the backend gate, and a gate that
is red out of the box gets suppressed rather than fixed. What it must not do is
look like full coverage.
"""

from __future__ import annotations

import base64
import hashlib
import re
import warnings
from pathlib import Path

import pytest

from app.api._security_headers import (
    CSP_DIRECTIVES,
    CSP_HEADER_VALUE,
    INLINE_BOOTSTRAP_SHA256,
    build_csp,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_REL = Path("frontend") / "src" / "index.html"
BUILT_REL = Path("frontend") / "dist" / "frontend" / "browser" / "index.html"

SRC_INDEX = REPO_ROOT / SRC_REL
BUILT_INDEX = REPO_ROOT / BUILT_REL

BUILD_COMMAND = "npm --prefix frontend run build"

SOURCE_FALLBACK_MESSAGE = (
    f"CSP coherence checked {SRC_REL.as_posix()} because {BUILT_REL.as_posix()} "
    "does not exist. That is NOT the page FastAPI serves, and the defect this "
    "file exists for (an inline event handler injected by Angular's "
    "inlineCritical) is created at build time and cannot appear in source. "
    f"These tests passing means less than it looks like. Run `{BUILD_COMMAND}` "
    "and run them again to get the coverage the names promise."
)


class SourceFallbackWarning(UserWarning):
    """Raised when the CSP checks run against source instead of the built page.

    A distinct category rather than a bare ``UserWarning`` so a caller that
    wants the strict behaviour can ask for it by name —
    ``-W error::...SourceFallbackWarning`` in CI, once CI builds the frontend
    before the backend gate. It is deliberately NOT escalated here: see the
    module docstring.
    """


def _resolve_index(root: Path) -> tuple[str, str] | None:
    """Return (html, which) preferring the BUILT file, or None if neither exists.

    Takes its root as an argument so the announcement below can be driven at a
    synthetic tree. A warning that only fires against the real repository is a
    claim about a line of code nobody executes.
    """
    built = root / BUILT_REL
    if built.exists():
        return built.read_text(encoding="utf-8"), "built"

    src = root / SRC_REL
    if src.exists():
        # stacklevel=3: past this helper and past _served_index, so the
        # warning is attributed to the test that degraded, not to this line.
        warnings.warn(SOURCE_FALLBACK_MESSAGE, SourceFallbackWarning, stacklevel=3)
        return src.read_text(encoding="utf-8"), "source"

    return None


def _served_index() -> tuple[str, str]:
    """Return (html, which) for this checkout, skipping if there is no page."""
    resolved = _resolve_index(REPO_ROOT)
    if resolved is None:
        pytest.skip("no index.html in this checkout")
    return resolved


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


def _tree(root: Path, *, built: str | None = None, source: str | None = None) -> Path:
    """Build a synthetic checkout containing whichever index.html files are given."""
    for rel, text in ((BUILT_REL, built), (SRC_REL, source)):
        if text is None:
            continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


class TestTheSourceFallbackIsAnnounced:
    """The guard on the guard: degrading must be visible on a PASS, not only on a fail.

    Every one of these drives ``_resolve_index`` at a synthetic tree rather than
    at this checkout, because which files exist here depends on whether someone
    has run a frontend build — so a test that read the real tree would assert
    different things on different machines and could not cover both branches.
    """

    def test_a_source_only_tree_warns(self, tmp_path):
        _tree(tmp_path, source="<html></html>")
        with pytest.warns(SourceFallbackWarning) as caught:
            _resolve_index(tmp_path)

        message = str(caught[0].message)
        assert BUILD_COMMAND in message, "must name the command that fixes it"
        assert SRC_REL.as_posix() in message
        assert BUILT_REL.as_posix() in message

    def test_the_fallback_still_returns_the_page(self, tmp_path):
        """Announced, NOT failed and NOT skipped.

        A fresh clone cannot build the frontend before running the backend gate.
        Turning this into a failure or a skip would make the checks disappear on
        exactly the machines that have never run them.
        """
        _tree(tmp_path, source="<html>source</html>")
        with pytest.warns(SourceFallbackWarning):
            html, which = _resolve_index(tmp_path)
        assert (html, which) == ("<html>source</html>", "source")

    def test_a_built_page_is_preferred_and_says_nothing(self, tmp_path):
        """The other half: no false alarm when coverage is real.

        A warning that fires either way carries no information, so the silence
        is asserted rather than assumed — ``simplefilter("error")`` turns any
        warning at all into a failure here.
        """
        _tree(tmp_path, built="<html>built</html>", source="<html>source</html>")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            html, which = _resolve_index(tmp_path)
        assert (html, which) == ("<html>built</html>", "built")

    def test_a_tree_with_neither_resolves_to_nothing(self, tmp_path):
        # This is what makes _served_index reach pytest.skip. Pinned so the
        # skip path cannot quietly become an exception.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _resolve_index(tmp_path) is None


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
        resp = client.get("/openapi.json")
        assert resp.status_code == 200, "probe endpoint must actually exist"
        assert resp.headers.get("Content-Security-Policy") == CSP_HEADER_VALUE

    def test_the_policy_is_on_a_404_too(self, client):
        """Every response, not just the ones that worked.

        Caught while fixing this file: the original probe hit a path that does
        not exist and passed anyway, which was true but was not the assertion
        it looked like.
        """
        resp = client.get("/definitely-not-a-route")
        assert resp.status_code == 404
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
