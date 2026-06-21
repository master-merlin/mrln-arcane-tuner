"""Sync OpenAI-compatible chat client for external captioning providers.

One httpx-based code path covers every provider — OpenAI, Anthropic, and
Gemini all expose official OpenAI-compatible endpoints, and "custom" covers
local servers (Ollama, LM Studio, vLLM). No provider SDKs.

API keys are never logged.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable

import httpx

from app.core.logger import get_logger

logger = get_logger(__name__)

#: model_id prefix marking external API providers (e.g. "api-openai").
API_MODEL_PREFIX = "api-"

#: Provider → OpenAI-compatible base URL. "custom" is user-configured.
PROVIDER_BASE_URLS: dict[str, str | None] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "openrouter": "https://openrouter.ai/api/v1",
    "custom": None,
}

#: Backoff schedule for retryable failures (429 / 5xx / transport errors).
RETRY_DELAYS = [1.0, 2.0, 4.0, 8.0, 16.0]

# Test seam — monkeypatched so retry tests don't actually sleep.
_sleep = time.sleep


class _RetryableHTTPError(Exception):
    def __init__(self, status_code: int, retry_after: str | None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.retry_after = retry_after


def provider_from_model_id(model_id: str) -> str | None:
    """Map an ``api-*`` caption model id to its provider name, else None."""
    if not model_id.startswith(API_MODEL_PREFIX):
        return None
    provider = model_id[len(API_MODEL_PREFIX) :]
    return provider if provider in PROVIDER_BASE_URLS else None


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def chat_vision(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    image_jpeg: bytes,
    extra_images_jpeg: list[bytes] | None = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 512,
    timeout: float = 120.0,
    transport: httpx.BaseTransport | None = None,
    should_abort: Callable[[], bool] | None = None,
    response_format: dict | None = None,
) -> str:
    """POST a text+image(s) chat completion and return the caption.

    ``extra_images_jpeg`` adds further image parts for multi-image (edit)
    captioning — OpenAI-compatible chat APIs accept multiple ``image_url``
    parts natively. Ordering matches the caller (control images first, target
    last). Retries 429/5xx/transport errors per RETRY_DELAYS (honouring a
    ``Retry-After`` header); other HTTP errors fail fast.  *should_abort* is
    polled before each attempt and between ≤1s backoff-sleep slices so a
    cancelled batch task stops promptly instead of riding out the backoff.
    """

    def _check_abort() -> None:
        if should_abort and should_abort():
            raise RuntimeError("API caption request aborted (task cancelled).")

    def _data_url(jpeg: bytes) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")

    url = f"{base_url.rstrip('/')}/chat/completions"
    content: list[dict] = [{"type": "text", "text": prompt}]
    for jpeg in [*(extra_images_jpeg or []), image_jpeg]:
        content.append({"type": "image_url", "image_url": {"url": _data_url(jpeg)}})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    last_error: Exception | None = None
    with httpx.Client(timeout=timeout, transport=transport) as client:
        for attempt in range(len(RETRY_DELAYS) + 1):
            _check_abort()
            try:
                resp = client.post(url, json=payload, headers=_headers(api_key))
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise _RetryableHTTPError(
                        resp.status_code, resp.headers.get("retry-after")
                    )
                if resp.status_code >= 400:
                    snippet = resp.text[:300]
                    raise RuntimeError(f"Provider error {resp.status_code}: {snippet}")
                body = resp.json()
                content = (
                    (body.get("choices") or [{}])[0].get("message", {}).get("content")
                )
                if not content or not content.strip():
                    raise RuntimeError("Provider returned an empty caption.")
                return content.strip()
            except (_RetryableHTTPError, httpx.TransportError) as exc:
                last_error = exc
                if attempt >= len(RETRY_DELAYS):
                    break
                delay = RETRY_DELAYS[attempt]
                retry_after = getattr(exc, "retry_after", None)
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                logger.warning(
                    "api_caption_retry",
                    attempt=attempt + 1,
                    delay=delay,
                    error=str(exc),
                )
                # Sleep in ≤1s slices so cancellation is honoured mid-backoff.
                remaining = delay
                while remaining > 0:
                    _check_abort()
                    step = min(1.0, remaining)
                    _sleep(step)
                    remaining -= step

    raise RuntimeError(f"API caption request failed after retries: {last_error}")


def list_models(
    *,
    base_url: str,
    api_key: str | None,
    timeout: float = 30.0,
    transport: httpx.BaseTransport | None = None,
) -> list[str]:
    """GET {base}/models and return the model id list."""
    url = f"{base_url.rstrip('/')}/models"
    with httpx.Client(timeout=timeout, transport=transport) as client:
        resp = client.get(url, headers=_headers(api_key))
        resp.raise_for_status()
        data = resp.json().get("data") or []
        return [m["id"] for m in data if isinstance(m, dict) and m.get("id")]
