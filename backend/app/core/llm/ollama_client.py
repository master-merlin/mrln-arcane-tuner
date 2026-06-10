# backend/app/core/llm/ollama_client.py
"""Async client for an OpenAI-compatible local LLM endpoint (Ollama / LM Studio).

Inference uses the portable ``/v1/chat/completions`` endpoint; model listing and
pulling use Ollama's native ``/api/*`` (LM Studio manages its own models, so the
caller hides pull/list when those endpoints are absent).
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_TIMEOUT = 120.0


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", client: httpx.AsyncClient | None = None) -> None:
        self._base = base_url.rstrip("/")
        self._injected = client

    def _acquire(self) -> tuple[httpx.AsyncClient, bool]:
        if self._injected is not None:
            return self._injected, False
        return httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT), True

    async def chat(self, model: str, system: str, user: str, options: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        if options:
            payload["options"] = options
        client, owned = self._acquire()
        try:
            resp = await client.post(f"{self._base}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        finally:
            if owned:
                await client.aclose()

    async def list_models(self) -> list[str]:
        client, owned = self._acquire()
        try:
            resp = await client.get(f"{self._base}/api/tags")
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        finally:
            if owned:
                await client.aclose()

    async def available(self) -> bool:
        try:
            await self.list_models()
            return True
        except Exception:
            return False

    async def pull(self, tag: str) -> bool:
        client, owned = self._acquire()
        try:
            resp = await client.post(f"{self._base}/api/pull", json={"name": tag, "stream": False})
            resp.raise_for_status()
            return True
        finally:
            if owned:
                await client.aclose()
