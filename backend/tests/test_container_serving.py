"""Integration tests for container-mode serving wired into the real app.

The app object is built at import time, so we reconfigure env and reload
the module to exercise both dev mode and container (SPA) mode.
"""
import importlib
import os

from fastapi.testclient import TestClient


def _reload_app(monkeypatch, *, dist_dir=None, token=""):
    if dist_dir:
        monkeypatch.setenv("MRLN_FRONTEND_DIST", str(dist_dir))
    else:
        # Force dev mode deterministically: an explicit but missing override
        # yields None from ``frontend_dist_dir()``, whereas merely unsetting
        # the var would let it fall back to the default build path — which may
        # actually exist on a machine that has run ``ng build`` (then "/" would
        # serve the SPA instead of the dev-mode JSON health endpoint).
        missing = os.path.join(os.path.dirname(__file__), "__no_such_dist__")
        monkeypatch.setenv("MRLN_FRONTEND_DIST", missing)
    if token:
        monkeypatch.setenv("MRLN_AUTH_TOKEN", token)
    else:
        monkeypatch.delenv("MRLN_AUTH_TOKEN", raising=False)
    import app.main as main
    importlib.reload(main)
    return main


def test_dev_mode_root_is_json(monkeypatch):
    main = _reload_app(monkeypatch, dist_dir=None)
    client = TestClient(main.app)
    r = client.get("/")
    assert r.status_code == 200
    assert "API is running" in r.json()["message"]


def test_container_mode_serves_spa_index(monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text("<html><body>SPA OK</body></html>")
    main = _reload_app(monkeypatch, dist_dir=tmp_path)
    client = TestClient(main.app)
    r = client.get("/datasets")
    assert r.status_code == 200
    assert "SPA OK" in r.text


def test_container_mode_login_route_redirects_with_token(monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text("<html><body>SPA OK</body></html>")
    main = _reload_app(monkeypatch, dist_dir=tmp_path, token="topsecret")
    client = TestClient(main.app)
    r = client.get("/login", params={"token": "topsecret"}, follow_redirects=False)
    assert r.status_code == 302
    from app.core.auth import COOKIE_NAME
    assert COOKIE_NAME in r.cookies


def test_container_mode_blocks_api_without_token(monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text("<html><body>SPA OK</body></html>")
    main = _reload_app(monkeypatch, dist_dir=tmp_path, token="topsecret")
    client = TestClient(main.app)
    assert client.get("/api/system/gpu").status_code == 401


def test_cleanup_reload_back_to_dev(monkeypatch):
    # Leave the module in dev state for the rest of the suite.
    _reload_app(monkeypatch, dist_dir=None)
