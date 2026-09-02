"""LLM-endpoint readiness — the ONE predicate behind "can this start?"
(LANE-57 refine, LANE-65 api-* captioning).

The user, signing UAT round 5: the refine task *"started without any API Key
(not guarded)"* and *"can be used on unconfigured endpoints"*; signing round 6:
*"Refine is properly gated, Generate still clickable."* Until this module,
``POST /api/captions/refine-batch`` (and, until LANE-65, ``POST
/api/captions/batch`` for an api-* provider) enqueued first and the worker
discovered that the endpoint was dead or the model absent, then (since
LANE-52) reported the failure honestly — the product accepted work that
could not possibly succeed.

This is the single producer (RULE-21). One probe, ``endpoint_readiness``,
judges any OpenAI-compatible / Ollama endpoint; two thin entry points hand
it the listing the caller actually uses:

* ``refine_readiness`` — the refine endpoint (``OllamaClient.list_models``,
  ``/api/tags``), consumed by the refine-batch boundary (409 ``reason``) and
  ``GET /api/llm-refine/models`` → ``unavailable_reason``;
* ``caption_provider_readiness`` — an api-* captioning provider
  (``openai_compat.list_models_async``, ``{base}/models``), consumed by the
  caption-batch boundary (409 ``reason``) and
  ``GET /api/captions/api-providers/{provider}/readiness`` → ``unavailable_reason``.

In both pairs the status carries the SAME string the refusal uses, so the
button and the 409 cannot disagree. Neither the worker nor the frontend
re-derives it.

Note on "API key": the refine endpoint (``llm_refine`` settings, Ollama /
LM Studio) has no key field at all — ``OllamaClient`` sends none
(``core/llm/ollama_client.py:65``). The api-* providers' configuration half
(key, Base URL shape) is ``provider_settings.resolve_provider`` (``:144-169``);
``caption_provider_readiness`` reads it through that resolver and reports its
sentence as the reason rather than growing a second one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

from app.core.llm.ollama_client import OllamaClient
from app.core.url_guard import OutboundUrlRejected


@dataclass(frozen=True)
class RefineReadiness:
    """What one probe of an LLM endpoint found (name kept from LANE-57)."""

    base_url: str
    available: bool
    installed: list[str] = field(default_factory=list)
    #: ``None`` when a start may proceed; otherwise the user-facing sentence
    #: naming exactly what is missing (endpoint, or model on that endpoint).
    reason: str | None = None


@dataclass(frozen=True)
class ReadinessSurface:
    """Where the user fixes what the probe found — the only text that differs
    between the refine endpoint and a captioning provider."""

    #: "configure and test it on <where>"
    where: str
    #: the tail of the model-missing sentence
    model_hint: str


REFINE_SURFACE = ReadinessSurface(
    where="the Server screen (LLM Refine Endpoint)",
    model_hint="pull it on the Server screen or pick an installed model")
CAPTION_SURFACE = ReadinessSurface(
    where="the captioning API settings (Connection)",
    model_hint="pick one the provider lists (Fetch models in the captioning API settings)")

#: A UI probe, not a batch: a provider that has not answered a model listing
#: in this long is "unreachable" and the sentence says so. Applied in ONE
#: place, ``endpoint_readiness`` (LANE-70): before it, the refine probe ran
#: on ``OllamaClient``'s 120 s inference timeout and the caption probe on its
#: own 10 s, and a socket that accepts and never answers held the Generate /
#: Refine CTA for that long. A live local listing answers in ~0.2-0.5 s
#: (measured 2026-09-02), so a few seconds is generous, not tight.
PROBE_TIMEOUT_S = 5.0


def unreachable_reason(base_url: str, surface: ReadinessSurface = REFINE_SURFACE) -> str:
    return (f"LLM endpoint {base_url} is unreachable - start it, or configure "
            f"and test it on {surface.where}.")


def answered_error_reason(base_url: str, status: int,
                          surface: ReadinessSurface = REFINE_SURFACE) -> str:
    # Reachable but refusing to list models is not "unreachable": for a hosted
    # provider a 401 is a bad key, for a local server a 404 is the wrong path.
    return (f"LLM endpoint {base_url} answered HTTP {status} to a model listing - "
            f"check the endpoint (and the API key, if it needs one) on {surface.where}.")


def model_missing_reason(model: str, base_url: str,
                         surface: ReadinessSurface = REFINE_SURFACE) -> str:
    return f"Model '{model}' is not installed on {base_url} - {surface.model_hint}."


def _is_installed(model: str, installed: list[str]) -> bool:
    # Ollama resolves an untagged name to ``name:latest`` on the wire
    # (``ollama run gemma3`` == ``gemma3:latest``), while ``/api/tags`` lists
    # the tagged form; a user who typed the untagged form has a working model.
    return model in installed or (":" not in model and f"{model}:latest" in installed)


async def endpoint_readiness(
    base_url: str,
    list_installed: Callable[[], Awaitable[list[str]]],
    model: str | None,
    surface: ReadinessSurface,
) -> RefineReadiness:
    """Probe the endpoint once; judge the model against what it lists.

    An endpoint that answers with an EMPTY model list is not judged on the
    model: LM Studio's listing covers loaded models only, and refusing a
    request the server would have served is the wrong side to err on. A
    non-empty list that lacks the model is the user's own 5.1 test case
    ("a model that is not installed") and is refused by name.
    """
    try:
        # The bound lives HERE, not in each client: a listing is the probe's
        # whole job, and a client's own timeout is sized for inference.
        installed = await asyncio.wait_for(list_installed(), PROBE_TIMEOUT_S)
    except OutboundUrlRejected:
        raise  # the URL itself is refused (layer L0) — a 400, not a readiness verdict
    except TimeoutError:  # accepted the connection, never answered: unreachable
        return RefineReadiness(base_url=base_url, available=False,
                               reason=unreachable_reason(base_url, surface))
    except httpx.HTTPStatusError as e:
        return RefineReadiness(base_url=base_url, available=False,
                               reason=answered_error_reason(
                                   base_url, e.response.status_code, surface))
    except Exception:  # noqa: BLE001 - any other failure to answer IS "unreachable"
        return RefineReadiness(base_url=base_url, available=False,
                               reason=unreachable_reason(base_url, surface))
    if model and installed and not _is_installed(model, installed):
        return RefineReadiness(base_url=base_url, available=True, installed=installed,
                               reason=model_missing_reason(model, base_url, surface))
    return RefineReadiness(base_url=base_url, available=True, installed=installed)


async def refine_readiness(client: OllamaClient, model: str | None = None) -> RefineReadiness:
    """The refine endpoint (Ollama / LM Studio on the Server screen)."""
    return await endpoint_readiness(client.base_url, client.list_models, model, REFINE_SURFACE)


async def caption_provider_readiness(provider: str, model: str | None = None) -> RefineReadiness:
    """An api-* captioning provider, judged by the listing it will caption through.

    Configuration (unknown provider, no key, unusable Base URL) is the
    resolver's verdict and sentence — it never dials out. Raises
    ``OutboundUrlRejected`` (a ``ValueError``) when the layer-L0 guard refuses
    the URL itself, exactly as the listing call would; callers map that to 400.
    """
    from app.core.llm import provider_settings
    from app.core.llm.openai_compat import list_models_async

    try:
        cfg = provider_settings.resolve_provider(provider)
    except ValueError as e:
        return RefineReadiness(base_url="", available=False, reason=str(e))

    async def _list() -> list[str]:
        # The async listing, not ``to_thread(list_models)``: a thread cannot be
        # cancelled when the bound above fires, and the sync client's
        # sequential connect pays ~2 s on ``localhost`` (see the docstring).
        try:
            return await list_models_async(base_url=cfg.base_url, api_key=cfg.api_key,
                                           timeout=PROBE_TIMEOUT_S)
        except OutboundUrlRejected:
            raise
        except ValueError as e:  # a body that is not JSON: reachable, not a listing
            raise httpx.TransportError(str(e)) from e

    return await endpoint_readiness(cfg.base_url, _list, model, CAPTION_SURFACE)
