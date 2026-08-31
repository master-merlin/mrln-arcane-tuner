"""Persistence + resolution for external captioning provider credentials.

Stored as the ``api_captioning`` module in backend/settings.json (gitignored):

    {"providers": {"openai": {"api_key": "..."},
                   ...,
                   "custom": {"api_key": "", "base_url": "..."}}}

Raw keys never leave this module unmasked except to the HTTP client.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.llm import refine_settings
from app.core.llm.base_url_conventions import is_usable_endpoint, to_openai_api_base
from app.core.llm.openai_compat import (
    API_MODEL_PREFIX,
    PROVIDER_BASE_URLS,
    provider_from_model_id,
)

MODULE = "api_captioning"
PROVIDERS = tuple(PROVIDER_BASE_URLS)  # ("openai", ..., "custom")

#: The Server screen's LLM endpoint (caption *refinement*). The custom captioning
#: provider inherits its ``base_url`` — see ``effective_base_url``.
SERVER_SETTINGS_MODULE = refine_settings.MODULE


def _manager():
    """Indirection seam — patched in tests."""
    from app.core.settings_manager import get_settings_manager

    return get_settings_manager()


@dataclass
class ProviderConfig:
    provider: str
    base_url: str
    api_key: str


@dataclass(frozen=True)
class EffectiveBaseUrl:
    """A provider's endpoint plus where it came from, for the UI to explain it."""

    base_url: str
    #: One of "provider" | "server_settings" | "builtin" | "none"
    #: (ECOSYSTEM §6 ``ProviderStatus.base_url_source``). "none" is a real value:
    #: "configured nowhere" must be distinguishable from a field a client could
    #: not parse, so this is never "" and never None.
    source: str


def effective_base_url(provider: str) -> EffectiveBaseUrl:
    """The ONE producer of a provider's endpoint (RULE-21).

    Precedence: the provider's own store → (``custom`` only) the Server screen's
    LLM endpoint → the builtin preset → nothing.

    ``resolve_provider`` and the ``/api-providers`` status route MUST both read
    this. The defect it closes (UAT-4.3): the badge said "configured" off one
    store while the request that followed resolved off another, so what the user
    was told and what actually happened could not be made to agree.

    **The return value is always in the OPENAI_API_BASE convention**
    (:mod:`app.core.llm.base_url_conventions`), because that is what this
    function's consumers -- ``openai_compat.list_models`` / ``chat_vision`` --
    append their resource path to. Inheriting from the Server screen is
    therefore a *transform*, not a copy: that store holds a SERVER_ROOT
    (``http://localhost:11434``), and LANE-46 copied it verbatim into a consumer
    that builds ``{base}/models``. Measured against a live Ollama: ``/models``
    answers 404, ``/v1/models`` answers 200 -- and because ``configured`` was
    ``bool(base_url)``, the badge went green off a value that could not answer
    (LANE-49). The transform is idempotent, so a user who has already typed a
    ``/v1``-suffixed URL into either field gets exactly one.

    **Only the base URL is inherited, never the model.** ``llm_refine.model`` is a
    TEXT model chosen for refining caption prose (default ``qwen2.5:7b-instruct``);
    captioning sends it an IMAGE. Prefilling it would hand the user a default that
    errors on first use, so the caption model stays empty and ``Fetch models``
    fills it from the now-known server (user's call, UAT-4.3; pinned by
    ``test_provider_settings.py::test_model_is_never_inherited_from_server_settings``).
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'")
    own = get_provider_raw(provider)["base_url"].strip()
    if own:
        return EffectiveBaseUrl(to_openai_api_base(own), "provider")
    if provider == "custom":
        # ``stored_base_url_of`` and NOT ``refine_settings.base_url()``: the
        # latter applies the localhost default, and inheriting a default nobody
        # chose would make a fresh install report "configured" off an endpoint
        # the user has never seen. Read through ``_manager()`` so this module
        # keeps ONE settings seam.
        inherited = refine_settings.stored_base_url_of(
            _manager().get_module_settings(SERVER_SETTINGS_MODULE) or {})
        if inherited:
            return EffectiveBaseUrl(to_openai_api_base(inherited), "server_settings")
    builtin = PROVIDER_BASE_URLS.get(provider) or ""
    if builtin:
        # Already in this convention (openai_compat.py:27-33); normalised anyway
        # so there is exactly one exit shape rather than one per branch.
        return EffectiveBaseUrl(to_openai_api_base(builtin), "builtin")
    return EffectiveBaseUrl("", "none")


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
    base_url = effective_base_url(provider).base_url
    if not base_url:
        raise ValueError(
            "Base URL is not configured for the Custom provider. "
            "Set it in the captioning API settings, or configure the LLM "
            "endpoint on the Server screen.")
    if not is_usable_endpoint(base_url):
        # Same predicate the status badge uses, so "not configured" and "the
        # request refused" cannot disagree (RULE-21 / LANE-49). Without this a
        # scheme-less host reaches ``openai_compat``, whose guard answers
        # "Provider base URL must start with http:// or https://" from three
        # layers down, as a 502.
        raise ValueError(
            f"Base URL {base_url!r} is not a usable endpoint for the Custom "
            "provider - it must be an absolute http:// or https:// URL "
            "(e.g. http://localhost:11434). Fix it in the captioning API "
            "settings, or on the Server screen if it was inherited from there.")
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
