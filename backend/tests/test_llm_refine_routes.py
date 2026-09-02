# backend/tests/test_llm_refine_routes.py
from unittest.mock import AsyncMock, patch

import httpx

from app.core.url_guard import ALLOW_PRIVATE_ENV

_MOD = "app.api.llm_refine_routes"


@patch(f"{_MOD}._make_client")
def test_list_models(mock_make, client):
    fake = mock_make.return_value
    fake.available = AsyncMock(return_value=True)
    fake.list_models = AsyncMock(return_value=["qwen2.5:7b-instruct"])
    # LANE-76: the route serves the endpoint it probed, read off the client;
    # a bare MagicMock's ``base_url`` is a mock, not a str, and the response
    # model refuses it — the fake carries what a real client carries.
    fake.base_url = "http://127.0.0.1:11434"
    resp = client.get("/api/llm-refine/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert "qwen2.5:7b-instruct" in body["installed"]
    assert "qwen2.5:7b-instruct" in body["curated"]
    assert body["endpoint"] == "http://127.0.0.1:11434"


@patch(f"{_MOD}._make_client")
def test_list_models_when_unavailable(mock_make, client):
    fake = mock_make.return_value
    # Unreachable == the listing raises (``OllamaClient.available`` is exactly
    # ``list_models`` without the exception) — the guard probes once.
    fake.list_models = AsyncMock(side_effect=httpx.ConnectError("refused"))
    fake.base_url = "http://127.0.0.1:1"
    resp = client.get("/api/llm-refine/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False
    assert body["unavailable_reason"] and "http://127.0.0.1:1" in body["unavailable_reason"]


@patch(f"{_MOD}._make_client")
def test_pull(mock_make, client):
    mock_make.return_value.pull = AsyncMock(return_value=True)
    resp = client.post("/api/llm-refine/pull", json={"tag": "qwen2.5:3b-instruct"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_hosted_refuses_a_metadata_base_url_end_to_end(client, monkeypatch):
    """The whole route path, with NOTHING stubbed between the setting and the
    HTTP client: a hosted server pointed at the cloud metadata endpoint must
    answer 400 with the guard's reason, and must never issue the request."""
    monkeypatch.setenv("MRLN_CONTAINER", "1")
    monkeypatch.delenv(ALLOW_PRIVATE_ENV, raising=False)
    with patch(f"{_MOD}._settings", return_value={"base_url": "http://169.254.169.254"}):
        resp = client.get("/api/llm-refine/models")
    assert resp.status_code == 400
    assert "link-local" in resp.json()["detail"]


def test_hosted_refuses_on_pull_and_preview_too(client, monkeypatch):
    monkeypatch.setenv("MRLN_CONTAINER", "1")
    monkeypatch.delenv(ALLOW_PRIVATE_ENV, raising=False)
    with patch(f"{_MOD}._settings", return_value={"base_url": "http://169.254.169.254"}):
        pull = client.post("/api/llm-refine/pull", json={"tag": "qwen2.5:3b-instruct"})
        preview = client.post(
            "/api/llm-refine/refine-preview", json={"text": "x", "preset": "standardize"}
        )
    assert pull.status_code == 400
    assert preview.status_code == 400


@patch(f"{_MOD}.refine_caption", new_callable=AsyncMock)
@patch(f"{_MOD}._make_client")
def test_refine_preview(mock_make, mock_refine, client):
    mock_refine.return_value = "clean caption"
    resp = client.post("/api/llm-refine/refine-preview", json={"text": "messy", "preset": "standardize", "model": "qwen2.5:7b-instruct"})
    assert resp.status_code == 200
    assert resp.json()["refined"] == "clean caption"
