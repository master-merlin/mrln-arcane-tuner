"""Integration tests for container-mode serving wired into the real app.

The app object is built at import time, so we reconfigure env and reload
the module to exercise both dev mode and container (SPA) mode.
"""
import importlib
import os

import pytest
from fastapi.testclient import TestClient

# ``importlib.reload(app.main)`` rebinds the process-global ``app`` object that
# every later ``from app.main import app`` sees. Two consequences (LANE-63):
# the module stays on ONE xdist worker, in file order (load distribution had
# split it, so the token reload landed on a worker whose "cleanup" test lived
# elsewhere — 80 route tests answered 401), and every test here restores the
# dev app itself, in a fixture, instead of trusting a last-in-file test.
pytestmark = pytest.mark.xdist_group("app_reload")


@pytest.fixture(autouse=True)
def _dev_app_after_each_test():
    """Reload ``app.main`` under the RESTORED environment after the test.

    Deliberately takes no ``monkeypatch``: an autouse fixture without that
    dependency is set up before the test's own ``monkeypatch`` and therefore
    torn down AFTER it has undone the env, so the reload sees the real
    environment — no token, no dist override — and rebuilds the dev app.
    """
    yield
    import app.main as main

    importlib.reload(main)


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
    r = client.post("/login", data={"token": "topsecret"}, follow_redirects=False)
    assert r.status_code == 302
    from app.core.auth import COOKIE_NAME
    assert COOKIE_NAME in r.cookies


def test_container_mode_blocks_api_without_token(monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text("<html><body>SPA OK</body></html>")
    main = _reload_app(monkeypatch, dist_dir=tmp_path, token="topsecret")
    client = TestClient(main.app)
    assert client.get("/api/system/gpu").status_code == 401


def test_a_token_reload_does_not_outlive_its_test(monkeypatch):
    """Pins the fixture above: reload with a token, undo the env the way
    pytest will, run the fixture's restore, and the app answers the API
    without a token again. RED with the old last-in-file "cleanup test",
    which only worked when every test of this file ran on one worker in
    file order."""
    main = _reload_app(monkeypatch, dist_dir=None, token="topsecret")
    assert TestClient(main.app).get("/api/system/gpu").status_code == 401
    monkeypatch.undo()
    importlib.reload(main)
    assert TestClient(main.app).get("/api/system/gpu").status_code != 401


def test_the_app_is_open_when_this_module_is_done():
    # Runs last in file order: whatever the tests above did, the process-global
    # app must be the dev app now — the property the rest of the suite needs.
    import app.main as main

    assert TestClient(main.app).get("/api/system/gpu").status_code != 401
