"""Tests for the optional shared-token ASGI gate."""
import pytest
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from app.core.auth import COOKIE_NAME, LOGIN_HTML, TokenAuthMiddleware


def _make_app(secret: str) -> FastAPI:
    app = FastAPI()

    @app.get("/api/ping")
    async def ping():
        return {"ok": True}

    @app.get("/login")
    async def login(token: str = ""):
        if secret and token == secret:
            resp = RedirectResponse("/", status_code=302)
            resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax")
            return resp
        return HTMLResponse(LOGIN_HTML, status_code=401)

    @app.get("/")
    async def root():
        return HTMLResponse("<h1>app</h1>")

    @app.websocket("/api/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text("hi")
        await websocket.close()

    if secret:
        app.add_middleware(TokenAuthMiddleware, token=secret)
    return app


def test_no_token_allows_everything():
    client = TestClient(_make_app(""))
    assert client.get("/api/ping").json() == {"ok": True}


def test_unauthenticated_api_returns_401_json():
    """W5.T10: the middleware's 401 matches the standard ErrorResponse
    envelope (docs/API_CONVENTIONS.md) — {"detail", "error_code", "context"} —
    the same shape every other error response gets via main.py's
    http_exception_handler, which this raw-ASGI middleware runs before."""
    client = TestClient(_make_app("s3cret"))
    r = client.get("/api/ping")
    assert r.status_code == 401
    body = r.json()
    assert body["detail"] == "unauthorized"
    assert body["error_code"] == "UNAUTHORIZED"
    assert body["context"] == {}


def test_unauthenticated_navigation_returns_login_page():
    client = TestClient(_make_app("s3cret"))
    r = client.get("/")
    assert r.status_code == 401
    assert "Access token" in r.text


def test_header_token_authorizes():
    client = TestClient(_make_app("s3cret"))
    r = client.get("/api/ping", headers={"X-Auth-Token": "s3cret"})
    assert r.status_code == 200


def test_login_route_is_reachable_without_auth():
    client = TestClient(_make_app("s3cret"))
    # Middleware must NOT block /login (returns login page for empty token).
    assert client.get("/login").status_code == 401


def test_login_sets_cookie_then_grants_access():
    client = TestClient(_make_app("s3cret"))
    r = client.get("/login", params={"token": "s3cret"}, follow_redirects=False)
    assert r.status_code == 302
    assert COOKIE_NAME in r.cookies
    # httpx stores the Set-Cookie; the next request now carries it.
    assert client.get("/api/ping").status_code == 200


def test_websocket_rejected_without_token():
    client = TestClient(_make_app("s3cret"))
    # The gate rejects the handshake before accept → the TestClient surfaces it
    # as a WebSocketDisconnect (not merely "some Exception").
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/ws"):
            pass


def test_websocket_allowed_with_cookie():
    client = TestClient(_make_app("s3cret"))
    client.cookies.set(COOKIE_NAME, "s3cret")
    with client.websocket_connect("/api/ws") as ws:
        assert ws.receive_text() == "hi"
