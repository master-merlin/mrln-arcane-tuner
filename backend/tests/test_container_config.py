"""Unit tests for env-driven container configuration."""
from app.core import container_config as cc


def test_resolve_port_prefers_env(monkeypatch):
    monkeypatch.setenv("PORT", "9001")
    assert cc.resolve_port() == 9001


def test_resolve_port_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    assert cc.resolve_port(8000) == 8000


def test_resolve_port_ignores_garbage(monkeypatch):
    monkeypatch.setenv("PORT", "notaport")
    assert cc.resolve_port(8000) == 8000


def test_auth_token_default_empty(monkeypatch):
    monkeypatch.delenv("MRLN_AUTH_TOKEN", raising=False)
    assert cc.auth_token() == ""


def test_auth_token_strips_whitespace(monkeypatch):
    monkeypatch.setenv("MRLN_AUTH_TOKEN", "  secret  ")
    assert cc.auth_token() == "secret"


def test_is_container_flag(monkeypatch):
    monkeypatch.setenv("MRLN_CONTAINER", "1")
    assert cc.is_container() is True
    monkeypatch.delenv("MRLN_CONTAINER", raising=False)
    assert cc.is_container() is False


def test_frontend_dist_dir_explicit_present(monkeypatch, tmp_path):
    monkeypatch.setenv("MRLN_FRONTEND_DIST", str(tmp_path))
    assert cc.frontend_dist_dir() == str(tmp_path)


def test_frontend_dist_dir_explicit_missing(monkeypatch, tmp_path):
    missing = tmp_path / "nope"
    monkeypatch.setenv("MRLN_FRONTEND_DIST", str(missing))
    assert cc.frontend_dist_dir() is None


def test_frontend_dist_dir_default_present(monkeypatch, tmp_path):
    monkeypatch.delenv("MRLN_FRONTEND_DIST", raising=False)
    browser = tmp_path / "frontend" / "dist" / "frontend" / "browser"
    browser.mkdir(parents=True)
    monkeypatch.setattr(cc, "_PROJECT_ROOT", str(tmp_path))
    assert cc.frontend_dist_dir() == str(browser)


def test_frontend_dist_dir_default_absent(monkeypatch, tmp_path):
    monkeypatch.delenv("MRLN_FRONTEND_DIST", raising=False)
    monkeypatch.setattr(cc, "_PROJECT_ROOT", str(tmp_path))
    assert cc.frontend_dist_dir() is None
