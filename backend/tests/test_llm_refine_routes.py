# backend/tests/test_llm_refine_routes.py
from unittest.mock import AsyncMock, patch

_MOD = "app.api.llm_refine_routes"


@patch(f"{_MOD}._make_client")
def test_list_models(mock_make, client):
    fake = mock_make.return_value
    fake.available = AsyncMock(return_value=True)
    fake.list_models = AsyncMock(return_value=["qwen2.5:7b-instruct"])
    resp = client.get("/api/llm-refine/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert "qwen2.5:7b-instruct" in body["installed"]
    assert "qwen2.5:7b-instruct" in body["curated"]


@patch(f"{_MOD}._make_client")
def test_list_models_when_unavailable(mock_make, client):
    fake = mock_make.return_value
    fake.available = AsyncMock(return_value=False)
    fake.list_models = AsyncMock(return_value=[])
    resp = client.get("/api/llm-refine/models")
    assert resp.status_code == 200
    assert resp.json()["available"] is False


@patch(f"{_MOD}._make_client")
def test_pull(mock_make, client):
    mock_make.return_value.pull = AsyncMock(return_value=True)
    resp = client.post("/api/llm-refine/pull", json={"tag": "qwen2.5:3b-instruct"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


@patch(f"{_MOD}.refine_caption", new_callable=AsyncMock)
@patch(f"{_MOD}._make_client")
def test_refine_preview(mock_make, mock_refine, client):
    mock_refine.return_value = "clean caption"
    resp = client.post("/api/llm-refine/refine-preview", json={"text": "messy", "preset": "standardize", "model": "qwen2.5:7b-instruct"})
    assert resp.status_code == 200
    assert resp.json()["refined"] == "clean caption"
