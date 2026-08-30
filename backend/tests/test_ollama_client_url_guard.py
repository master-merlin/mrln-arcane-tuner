# backend/tests/test_ollama_client_url_guard.py
"""The outbound URL guard belongs at the OllamaClient sink, not at its callers.

Release-audit finding (2026-08-30): ``app/core/url_guard.validate_base_url`` was
wired into exactly ONE caller (``app/core/llm/openai_compat.py:60``), while
``OllamaClient`` — whose base URL comes from the user-writable ``llm_refine``
setting — was constructed unguarded at two sites:
``app/api/llm_refine_routes.py:27`` and
``app/core/captioning/caption_refine_batch.py:104``. On a hosted install an
authenticated user could aim a refine at ``http://169.254.169.254`` and have the
server request the cloud metadata endpoint (blind SSRF).

These tests assert on the SINK, so a fourth caller added later is covered by
construction rather than by whoever remembers to call the guard.

Scope kept honest (``url_guard.py:34-44``): this pins the pre-connect check
only. DNS rebinding and redirect hops are NOT addressed here and are not
claimed to be.
"""

from __future__ import annotations

import pytest

from app.core.llm.ollama_client import OllamaClient
from app.core.url_guard import ALLOW_PRIVATE_ENV, OutboundUrlRejected

#: Literal IPs, so the guard's ``getaddrinfo`` resolves them without a network.
METADATA_URL = "http://169.254.169.254"
LOOPBACK_URL = "http://127.0.0.1:11434"


@pytest.fixture
def hosted(monkeypatch):
    """Run as the container image does: contained, no private-URL opt-in."""
    monkeypatch.setenv("MRLN_CONTAINER", "1")
    monkeypatch.delenv(ALLOW_PRIVATE_ENV, raising=False)


def test_hosted_rejects_cloud_metadata_base_url(hosted):
    """The sink itself refuses — no caller involved."""
    with pytest.raises(OutboundUrlRejected) as exc:
        OllamaClient(base_url=METADATA_URL)
    assert "link-local" in str(exc.value)


def test_hosted_rejects_loopback_base_url(hosted):
    with pytest.raises(OutboundUrlRejected):
        OllamaClient(base_url=LOOPBACK_URL)


def test_hosted_rejects_non_http_scheme(hosted):
    with pytest.raises(OutboundUrlRejected):
        OllamaClient(base_url="file:///etc/passwd")


def test_llm_refine_route_client_is_guarded(hosted, monkeypatch):
    """Caller 1 (``/api/llm-refine/*``) inherits the guard from the sink."""
    from app.api import llm_refine_routes as routes

    monkeypatch.setattr(routes, "_settings", lambda: {"base_url": METADATA_URL})
    with pytest.raises(OutboundUrlRejected):
        routes._make_client()


def test_refine_batch_is_guarded(hosted, monkeypatch, tmp_path):
    """Caller 2 (the background refine lane) inherits the guard from the sink."""
    from unittest.mock import MagicMock

    from app.core.captioning import caption_refine_batch as crb

    ds = MagicMock()
    ds.path = str(tmp_path)
    monkeypatch.setattr(crb, "dataset_manager", MagicMock(**{"get_dataset.return_value": ds}))
    monkeypatch.setattr(crb, "task_manager", MagicMock())

    with pytest.raises(OutboundUrlRejected):
        crb.run_caption_refine_batch(
            "t-guard",
            dataset_name="ds",
            image_rel_paths=["img1.png"],
            definition_id="flux1-schnell",
            preset="standardize",
            model="qwen2.5:7b-instruct",
            base_url=METADATA_URL,
        )


def test_local_install_still_reaches_a_local_provider(monkeypatch):
    """Prove the negative: the guard must NOT change the local column, where
    reaching ``localhost:11434`` is the documented, correct use."""
    monkeypatch.delenv("MRLN_CONTAINER", raising=False)
    monkeypatch.delenv(ALLOW_PRIVATE_ENV, raising=False)
    client = OllamaClient(base_url="http://localhost:11434/")
    assert client._base == "http://localhost:11434"


def test_hosted_opt_in_allows_a_private_provider(hosted, monkeypatch):
    """The documented escape hatch stays the ONLY way back in."""
    monkeypatch.setenv(ALLOW_PRIVATE_ENV, "1")
    assert OllamaClient(base_url=LOOPBACK_URL)._base == LOOPBACK_URL
