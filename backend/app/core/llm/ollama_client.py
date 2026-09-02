# backend/app/core/llm/ollama_client.py
"""Async client for an OpenAI-compatible local LLM endpoint (Ollama / LM Studio).

Inference uses the portable ``/v1/chat/completions`` endpoint; model listing and
pulling use Ollama's native ``/api/*`` (LM Studio manages its own models, so the
caller hides pull/list when those endpoints are absent).

The base URL is user-writable (``llm_refine.base_url``), so it is validated in
``__init__`` -- at the sink, not at the call sites. See the constructor.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.logger import get_logger
from app.core.llm.base_url_conventions import to_server_root
from app.core.url_guard import validate_base_url

logger = get_logger(__name__)

_DEFAULT_TIMEOUT = 120.0


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", client: httpx.AsyncClient | None = None) -> None:
        """Validate at construction, so no instance can exist with a base URL
        this server may not request.

        The guard sits HERE and not at the callers because every base URL that
        reaches this class comes from the user-writable ``llm_refine.base_url``
        setting, and a guard wired per-caller decays: the release audit found
        ``validate_base_url`` wired into ONE caller
        (``core/llm/openai_compat.py:60``) while this client was built
        unguarded at ``api/llm_refine_routes.py:27`` and
        ``core/captioning/caption_refine_batch.py:104``. A third caller is now
        covered by construction. There is deliberately NO ``validate=False``
        opt-out -- ``MRLN_ALLOW_PRIVATE_PROVIDER_URLS`` is the one documented
        way back in (``core/url_guard.py``), and it is a no-op on a local
        install, where reaching ``localhost:11434`` stays the correct use.

        Raises ``OutboundUrlRejected`` (a ``ValueError``). Known limits are the
        guard's own and are unchanged by placing it here: this checks the URL
        before connecting, so DNS rebinding and redirect hops remain open
        (``core/url_guard.py:34-44``). Note also that the check resolves the
        host synchronously; an async caller constructing a client with a
        hostile hostname pays that resolution on its event loop.
        """
        # SERVER_ROOT convention: every method here appends its own full path
        # (``/v1/chat/completions``, ``/api/tags``), so a value stored in the
        # OPENAI_API_BASE spelling would double-suffix -- LM Studio's own
        # documented endpoint is ``http://localhost:1234/v1``, and typing it
        # into the Server screen produced ``/v1/v1/chat/completions``.
        # ``to_server_root`` is idempotent (LANE-49).
        self._base = validate_base_url(to_server_root(base_url))
        self._injected = client

    @property
    def base_url(self) -> str:
        """The validated ``SERVER_ROOT`` this client talks to — the value a
        user-facing refusal names (``core/llm/refine_guard.py``)."""
        return self._base

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
