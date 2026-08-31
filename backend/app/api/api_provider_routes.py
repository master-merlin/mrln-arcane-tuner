"""API-captioning provider endpoints — key management and model listing.

Separate from caption_routes so it stays importable without the torch-heavy
caption service. Raw API keys are write-only: every response carries only the
masked form.
"""

from __future__ import annotations

import asyncio

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.llm.openai_compat import list_models
from app.core.llm.provider_settings import (
    PROVIDERS,
    effective_base_url,
    get_provider_raw,
    mask_key,
    resolve_provider,
    set_provider,
)
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


class ProviderStatus(BaseModel):
    """Masked, browser-safe view of one provider's configuration."""

    provider: str
    configured: bool
    key_masked: str
    #: The EFFECTIVE endpoint, not the stored one — same value the request will
    #: use, because both come from ``effective_base_url`` (RULE-21).
    base_url: str
    #: Origin of ``base_url``: "provider" | "server_settings" | "builtin" | "none"
    #: (ECOSYSTEM §6). Named for the surface the user configured, not the settings
    #: module: they have never heard of ``llm_refine``, they used the Server screen.
    base_url_source: str


class ProviderUpdateRequest(BaseModel):
    """Partial credential update — omitted field = unchanged, '' = clear."""

    api_key: str | None = None
    base_url: str | None = None


class ProviderModelsResponse(BaseModel):
    models: list[str]


def _status(provider: str) -> ProviderStatus:
    raw = get_provider_raw(provider)
    # Same resolver the request uses, so "configured ✓" and where the call
    # actually goes cannot disagree (RULE-21). For custom that means an endpoint
    # inherited from Server settings reads as configured — because it is usable.
    effective = effective_base_url(provider)
    configured = (
        bool(effective.base_url) if provider == "custom" else bool(raw["api_key"]))
    return ProviderStatus(
        provider=provider,
        configured=configured,
        key_masked=mask_key(raw["api_key"]),
        base_url=effective.base_url,
        base_url_source=effective.source,
    )


@router.get("/api-providers", response_model=list[ProviderStatus])
async def list_provider_status():
    """Masked configuration status for every provider."""
    return [_status(p) for p in PROVIDERS]


@router.put("/api-providers/{provider}", response_model=ProviderStatus)
async def update_provider(provider: str, request: ProviderUpdateRequest):
    """Set/clear a provider's API key and (custom) base URL."""
    try:
        set_provider(provider, api_key=request.api_key, base_url=request.base_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    logger.info("api_provider_updated", provider=provider,
                key_set=request.api_key is not None)
    return _status(provider)


@router.get("/api-providers/{provider}/models",
            response_model=ProviderModelsResponse)
async def list_provider_models(provider: str):
    """Proxy the provider's /models listing using the stored credentials."""
    try:
        cfg = resolve_provider(provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        # list_models is a sync httpx call (up to 30s) — keep it off the loop.
        models = await asyncio.to_thread(
            lambda: list_models(base_url=cfg.base_url, api_key=cfg.api_key))
    except (httpx.HTTPError, ValueError) as e:
        raise HTTPException(
            status_code=502,
            detail=f"Could not fetch models from '{provider}': {e}")
    return ProviderModelsResponse(models=models)
