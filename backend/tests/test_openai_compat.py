"""Tests for the OpenAI-compatible captioning client."""
import json

import httpx
import pytest

from app.core.llm import openai_compat


def _vision_kwargs(transport):
    return dict(
        base_url="https://api.test/v1",
        api_key="sk-test",
        model="gpt-4o",
        prompt="Describe this image.",
        image_jpeg=b"\xff\xd8fakejpeg",
        transport=transport,
    )


def test_chat_vision_success_returns_content_and_sends_data_url():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "  a red car  "}}],
        })

    result = openai_compat.chat_vision(
        **_vision_kwargs(httpx.MockTransport(handler)))

    assert result == "a red car"
    assert captured["url"] == "https://api.test/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    parts = captured["body"]["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "Describe this image."}
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert captured["body"]["model"] == "gpt-4o"


def test_chat_vision_retries_on_429_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(openai_compat, "_sleep", sleeps.append)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "3"})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}],
        })

    result = openai_compat.chat_vision(
        **_vision_kwargs(httpx.MockTransport(handler)))

    assert result == "ok"
    assert calls["n"] == 2
    # Backoff is sliced into ≤1s chunks (cancellation checks between chunks),
    # but the TOTAL still honours max(RETRY_DELAYS[0]=1, retry-after=3).
    assert sum(sleeps) == 3.0
    assert all(s <= 1.0 for s in sleeps)


def test_chat_vision_should_abort_stops_backoff_promptly(monkeypatch):
    """A cancelled task aborts between backoff slices instead of riding it out."""
    sleeps = []
    monkeypatch.setattr(openai_compat, "_sleep", sleeps.append)
    aborted = {"flag": False}

    def handler(request: httpx.Request) -> httpx.Response:
        aborted["flag"] = True  # cancel as soon as the first attempt fails
        return httpx.Response(429, headers={"retry-after": "30"})

    with pytest.raises(RuntimeError, match="aborted"):
        openai_compat.chat_vision(
            should_abort=lambda: aborted["flag"],
            **_vision_kwargs(httpx.MockTransport(handler)))
    assert sleeps == []  # aborted before the first backoff slice


def test_chat_vision_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr(openai_compat, "_sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    with pytest.raises(RuntimeError, match="failed after retries"):
        openai_compat.chat_vision(**_vision_kwargs(httpx.MockTransport(handler)))

    assert calls["n"] == len(openai_compat.RETRY_DELAYS) + 1


def test_chat_vision_401_fails_fast_without_retry(monkeypatch):
    monkeypatch.setattr(
        openai_compat, "_sleep",
        lambda s: pytest.fail("must not retry on 401"))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(RuntimeError, match="401"):
        openai_compat.chat_vision(**_vision_kwargs(httpx.MockTransport(handler)))
    assert calls["n"] == 1


def test_chat_vision_empty_content_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    with pytest.raises(RuntimeError, match="empty"):
        openai_compat.chat_vision(**_vision_kwargs(httpx.MockTransport(handler)))


def test_list_models_returns_ids():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.test/v1/models"
        return httpx.Response(200, json={
            "data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}, {"bogus": True}],
        })

    models = openai_compat.list_models(
        base_url="https://api.test/v1", api_key="sk-test",
        transport=httpx.MockTransport(handler))
    assert models == ["gpt-4o", "gpt-4o-mini"]


def test_provider_base_urls_cover_all_providers():
    assert set(openai_compat.PROVIDER_BASE_URLS) == {
        "openai", "anthropic", "gemini", "openrouter", "custom",
    }
    assert openai_compat.PROVIDER_BASE_URLS["custom"] is None
