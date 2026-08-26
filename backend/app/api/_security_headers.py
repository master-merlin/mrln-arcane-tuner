"""Content-Security-Policy for the served SPA (REQUEST-2).

The policy is held as a mapping of directive -> sources rather than a string so
tests can ask real questions of it ("is this origin permitted by img-src?")
instead of substring-matching a blob. The serialized form is derived from the
mapping, so the two cannot drift apart.

WHERE THE PROTECTION ACTUALLY IS: ``script-src`` carries it and is genuinely
strict -- no ``'unsafe-inline'``, the one inline bootstrap is hashed, everything
else must be same-origin.

``style-src 'unsafe-inline'`` IS A KNOWN, DELIBERATE WEAKNESS -- not an
oversight, and please do not "fix" it by deleting the token. Angular injects
component styles as ``<style>`` elements at runtime, so a hash-only
``style-src`` breaks every component style in the application. The real fix is a
per-response nonce plus Angular's ``ngCspNonce``, which turns ``index.html``
from a static asset into a per-request rendered template -- a genuine
architectural change, deliberately deferred to its own decision rather than
smuggled in here.

The inline script hash covers the exact bytes between the ``<script>`` tags of
the SERVED ``index.html``. If that file changes and this constant does not, the
app breaks at first paint with a console error and a flash of the wrong theme --
which is why ``test_csp_policy.py`` recomputes it from the file rather than
trusting this value.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

#: sha256 of the inline theme-bootstrap script in ``frontend/src/index.html``.
#: It runs before first paint by design (a bundled script would not), so it
#: stays inline and is hashed instead of removed.
INLINE_BOOTSTRAP_SHA256 = "sha256-vbT0RDOhM7BAG5/XwAtJ1SNpPpMfoAUGkZnL9f8UWQ8="

CSP_DIRECTIVES: dict[str, tuple[str, ...]] = {
    "default-src": ("'self'",),
    "script-src": ("'self'", f"'{INLINE_BOOTSTRAP_SHA256}'"),
    # See the module docstring before touching this one.
    "style-src": ("'self'", "'unsafe-inline'"),
    "font-src": ("'self'",),
    # blob: for generated previews, data: for small inline raster assets.
    # data: is on img-src ONLY -- it is a script-execution vector elsewhere.
    "img-src": ("'self'", "data:", "blob:"),
    "media-src": ("'self'", "blob:"),
    # The ws:/wss: schemes are listed explicitly: 'self' does not reliably cover
    # the WebSocket scheme across browsers, and the app holds an always-on
    # socket for logs, metrics and task events.
    "connect-src": ("'self'", "ws:", "wss:"),
    "object-src": ("'none'",),
    "base-uri": ("'self'",),
    # The backend serves a real sign-in form at /login; that is the target
    # worth pinning.
    "form-action": ("'self'",),
    # Nothing embeds this app -- it is a local studio UI. Also supersedes
    # X-Frame-Options for every browser that reads CSP.
    "frame-ancestors": ("'none'",),
}


def build_csp(directives: dict[str, tuple[str, ...]] | None = None) -> str:
    """Serialize the directive mapping into a CSP header value."""
    src = CSP_DIRECTIVES if directives is None else directives
    return "; ".join(f"{name} {' '.join(values)}" for name, values in src.items())


CSP_HEADER_VALUE = build_csp()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach the CSP to every response.

    Every response, not only HTML documents: a policy applied per-route is one
    forgotten route away from a gap, and the directives are inert on a JSON
    body. An existing header is never overwritten -- a route that has computed
    a narrower policy for itself knows something this middleware does not.
    """

    header_name = "Content-Security-Policy"

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault(self.header_name, CSP_HEADER_VALUE)
        return response
