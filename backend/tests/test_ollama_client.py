# backend/tests/test_ollama_client.py
"""Offline unit tests for OllamaClient via httpx.MockTransport."""

import asyncio
import json

import httpx

from app.core.llm.ollama_client import OllamaClient


def _client(handler) -> OllamaClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://test")
    return OllamaClient(base_url="http://test", client=http)


def test_chat_returns_message_content():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/chat/completions"
        body = json.loads(req.content)
        assert body["stream"] is False
        assert body["messages"][0]["role"] == "system"
        return httpx.Response(200, json={"choices": [{"message": {"content": "refined text"}}]})

    out = asyncio.run(_client(handler).chat("qwen2.5:7b-instruct", "sys", "user"))
    assert out == "refined text"


def test_list_models_extracts_names():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b-instruct"}, {"name": "llama3.1:8b"}]})

    out = asyncio.run(_client(handler).list_models())
    assert out == ["qwen2.5:7b-instruct", "llama3.1:8b"]


def test_available_true_and_false():
    def ok(req): return httpx.Response(200, json={"models": []})
    def boom(req): raise httpx.ConnectError("refused")
    assert asyncio.run(_client(ok).available()) is True
    assert asyncio.run(_client(boom).available()) is False


def test_pull_posts_tag():
    seen = {}
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/pull"
        seen["name"] = json.loads(req.content)["name"]
        return httpx.Response(200, json={"status": "success"})
    assert asyncio.run(_client(handler).pull("qwen2.5:3b-instruct")) is True
    assert seen["name"] == "qwen2.5:3b-instruct"
