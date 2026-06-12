"""Persistence + resolution for external captioning provider credentials.

Stored as the ``api_captioning`` module in backend/settings.json (gitignored):

    {"providers": {"openai": {"api_key": "..."},
                   ...,
                   "custom": {"api_key": "", "base_url": "..."}}}

Raw keys never leave this module unmasked except to the HTTP client.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.llm.openai_compat import (
    API_MODEL_PREFIX,
    PROVIDER_BASE_URLS,
    provider_from_model_id,
)

MODULE = "api_captioning"
PROVIDERS = tuple(PROVIDER_BASE_URLS)  # ("openai", ..., "custom")


def _manager():
    """Indirection seam — patched in tests."""
    from app.core.settings_manager import get_settings_manager

    return get_settings_manager()


@dataclass
class ProviderConfig:
    provider: str
    base_url: str
    api_key: str


def get_provider_raw(provider: str) -> dict:
    """Return the stored {"api_key", "base_url"} dict (defaults to empty)."""
    providers = _manager().get_module_settings(MODULE).get("providers", {})
    raw = providers.get(provider, {})
    return {"api_key": raw.get("api_key", ""), "base_url": raw.get("base_url", "")}


def set_provider(
    provider: str, *, api_key: str | None = None, base_url: str | None = None
) -> None:
    """Update a provider's credentials. None = leave unchanged, "" = clear."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'")
    manager = _manager()
    providers = dict(manager.get_module_settings(MODULE).get("providers", {}))
    entry = dict(providers.get(provider, {}))
    if api_key is not None:
        entry["api_key"] = api_key
    if base_url is not None:
        entry["base_url"] = base_url
    providers[provider] = entry
    manager.update_module_settings(MODULE, {"providers": providers})


def mask_key(key: str) -> str:
    """Display form of a key: '' → '', short → '•••', else 'sk-…1234'."""
    if not key:
        return ""
    if len(key) < 8:
        return "•••"
    return f"{key[:3]}…{key[-4:]}"


def resolve_provider(provider: str) -> ProviderConfig:
    """Return a usable config or raise ValueError with a user-readable reason."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'")
    raw = get_provider_raw(provider)
    base_url = PROVIDER_BASE_URLS[provider] or raw["base_url"]
    if not base_url:
        raise ValueError(
            "Base URL is not configured for the Custom provider. "
            "Set it in the captioning API settings.")
    if provider != "custom" and not raw["api_key"]:
        raise ValueError(
            f"No API key configured for provider '{provider}'. "
            "Set it in the captioning API settings.")
    return ProviderConfig(provider=provider, base_url=base_url, api_key=raw["api_key"])


def validate_caption_model(model_id: str, params: dict | None = None) -> None:
    """Raise ValueError when an ``api-*`` model id is unusable. No-op otherwise.

    When *params* is given, additionally requires a non-empty provider model
    name (``params["model"]``) so empty selections fail at enqueue time
    instead of once per image inside the batch.
    """
    if not model_id.startswith(API_MODEL_PREFIX):
        return
    provider = provider_from_model_id(model_id)
    if provider is None:
        raise ValueError(f"Unknown API caption model '{model_id}'")
    resolve_provider(provider)
    if params is not None and not str(params.get("model") or "").strip():
        raise ValueError(
            f"No provider model selected for '{model_id}'. "
            "Pick one in the API captioning settings (e.g. gpt-4o).")


def lane_for_model(model_id: str) -> str:
    """API captioning runs on the non-GPU background lane."""
    return "background" if model_id.startswith(API_MODEL_PREFIX) else "gpu"
