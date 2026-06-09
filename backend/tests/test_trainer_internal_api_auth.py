"""The trainer subprocess must authenticate its own loopback API calls.

The trainer runs as a separate process with no browser cookie. When a shared
access token is configured (``MRLN_AUTH_TOKEN``), ``TokenAuthMiddleware`` 401s
every ``/api`` request without credentials — so the trainer's calls to
``http://localhost/api/datasets/...`` were rejected, the inventory came back
empty, and training failed with "No training data found in datasets."

The trainer inherits the token via the environment, so it forwards it as the
``X-Auth-Token`` header the middleware accepts.
"""
import pytest

from app.engine.core.pipeline.pipeline_data import _internal_api_headers


@pytest.fixture(autouse=True)
def _clean_token(monkeypatch):
    monkeypatch.delenv("MRLN_AUTH_TOKEN", raising=False)
    yield


def test_no_token_yields_no_headers(monkeypatch):
    """Auth disabled → no header (local dev / unprotected pods unaffected)."""
    assert _internal_api_headers() == {}


def test_token_present_sets_x_auth_token(monkeypatch):
    monkeypatch.setenv("MRLN_AUTH_TOKEN", "s3cret")
    assert _internal_api_headers() == {"X-Auth-Token": "s3cret"}


def test_token_is_stripped(monkeypatch):
    """auth_token() strips whitespace, so the header carries the clean value."""
    monkeypatch.setenv("MRLN_AUTH_TOKEN", "  s3cret  ")
    assert _internal_api_headers() == {"X-Auth-Token": "s3cret"}
