"""Optional shared-token access gate (pure-ASGI middleware).

When a token is configured, every HTTP and WebSocket request must carry it
via the ``mrln_auth`` cookie or the ``X-Auth-Token`` header. Unauthenticated
browsers get a minimal login page; the ``/login`` route (defined in
``main.py``) validates the token and sets the cookie. When no token is
configured the middleware is a no-op, so local dev is unchanged.

This is a pure-ASGI middleware (not ``@app.middleware("http")``) so it also
covers the WebSocket handshake — the log stream at ``/api/ws`` must be gated
too.
"""
from __future__ import annotations

import hmac

from starlette.requests import HTTPConnection
from starlette.responses import HTMLResponse, JSONResponse

from app.api.schemas.common_schemas import ErrorResponse

COOKIE_NAME = "mrln_auth"

LOGIN_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>MRLN Arcane Tuner — Sign in</title>
<style>
body{font-family:system-ui,Segoe UI,sans-serif;background:#0b0b0f;color:#eee;
display:flex;height:100vh;align-items:center;justify-content:center;margin:0}
form{background:#16161d;padding:2rem;border-radius:12px;min-width:280px;
box-shadow:0 10px 40px rgba(0,0,0,.5)}
h1{font-size:1.1rem;margin:0 0 1rem}
label{font-size:.85rem;color:#aaa}
input,button{width:100%;padding:.6rem;margin-top:.4rem;border-radius:8px;
border:1px solid #333;background:#0b0b0f;color:#eee;box-sizing:border-box}
button{background:oklch(0.68 0.13 55);color:#fff;border:none;cursor:pointer;
margin-top:1rem;font-weight:700;transition:all .15s;
box-shadow:0 4px 6px -1px oklch(0.70 0.18 55 / .2)}
button:hover{background:oklch(0.70 0.18 55 / .9)}
button:active{transform:scale(.97)}
</style></head>
<body><form method="post" action="/login">
<h1>MRLN Arcane Tuner</h1>
<label for="token">Access token</label>
<input id="token" type="password" name="token" autofocus autocomplete="off" />
<button type="submit">Sign in</button>
</form></body></html>"""


def supplied_token(conn: HTTPConnection) -> str:
    """Read the token from the cookie or the X-Auth-Token header."""
    return conn.cookies.get(COOKIE_NAME) or conn.headers.get("x-auth-token", "")


class TokenAuthMiddleware:
    """Gate HTTP + WebSocket scopes behind a shared token."""

    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if not self.token or scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # The login route must stay reachable so users can authenticate.
        if scope.get("path", "") == "/login":
            await self.app(scope, receive, send)
            return

        conn = HTTPConnection(scope)
        supplied = supplied_token(conn)
        if supplied and hmac.compare_digest(supplied, self.token):
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return

        path = scope.get("path", "")
        if path.startswith("/api") or path.startswith("/media"):
            # This middleware runs as raw ASGI, BEFORE FastAPI's routing/
            # exception-handling machinery — a 401 here never passes through
            # main.py's http_exception_handler, so the standard envelope
            # (_docs/API_CONVENTIONS.md) is built by hand to match it exactly.
            envelope = ErrorResponse(detail="unauthorized", error_code="UNAUTHORIZED")
            response = JSONResponse(envelope.model_dump(), status_code=401)
        else:
            response = HTMLResponse(LOGIN_HTML, status_code=401)
        await response(scope, receive, send)
