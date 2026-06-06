"""Verify curated /models/download emits the expected event sequence."""
from unittest.mock import patch, MagicMock
import pytest
from app.core import model_registry as mr


@pytest.mark.asyncio
async def test_download_model_emits_starting_downloading_complete(tmp_path, monkeypatch):
    # Seed MODEL_DB so the function recognizes the file
    # Category must be a valid DownloadProgress literal ("restore", "upscale", etc.)
    monkeypatch.setitem(mr.MODEL_DB, "restore", {
        "fake.bin": {"url": "http://example/fake.bin", "size_mb": 0.05},
    })

    # Build a fake httpx async streaming response that yields 2 chunks
    chunks = [b"x" * 32768, b"y" * 32768]  # 65536 bytes total
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.headers = {"content-length": str(sum(len(c) for c in chunks))}
    async def aiter_bytes(chunk_size: int):
        for c in chunks:
            yield c
    fake_resp.aiter_bytes = aiter_bytes

    class _FakeStreamCtx:
        async def __aenter__(self): return fake_resp
        async def __aexit__(self, *exc): return False
    class _FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *exc): return False
        def stream(self, *a, **kw): return _FakeStreamCtx()

    captured: list[dict] = []
    async def _fake_broadcast(event_type: str, payload: dict):
        captured.append({"type": event_type, **payload})

    monkeypatch.setattr(mr, "httpx", MagicMock(AsyncClient=lambda **kw: _FakeClient()))
    with patch("app.api.events.download_progress.event_manager.broadcast", new=_fake_broadcast):
        await mr.download_model("restore", "fake.bin", tmp_path)

    statuses = [e["status"] for e in captured if e["type"] == "model.download_progress"]
    assert statuses[0] == "starting"
    assert "complete" in statuses
    # All payloads carry source='curated'
    sources = {e["source"] for e in captured if e["type"] == "model.download_progress"}
    assert sources == {"curated"}
