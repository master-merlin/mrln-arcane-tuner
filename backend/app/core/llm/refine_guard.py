"""Refine readiness — the ONE predicate behind "can a refine start?" (LANE-57).

The user, signing UAT round 5: the refine task *"started without any API Key
(not guarded)"* and *"can be used on unconfigured endpoints"*. Until this
module, ``POST /api/captions/refine-batch`` enqueued first and the worker
discovered that the endpoint was dead or the model absent, then (since
LANE-52) reported the failure honestly — the product accepted work that
could not possibly succeed.

This is the single producer (RULE-21) for two consumers:

* the request boundary (``api/caption_routes.py`` refine-batch) refuses with
  ``reason`` as the 409 detail and enqueues nothing;
* the status the UI disables its Start controls off
  (``GET /api/llm-refine/models`` → ``unavailable_reason``) carries the SAME
  string, so the button and the refusal cannot disagree.

Neither the worker nor the frontend re-derives it.

Note on "API key": the refine endpoint (``llm_refine`` settings, Ollama /
LM Studio) has no key field at all — ``OllamaClient`` sends none
(``core/llm/ollama_client.py:65``). What the user saw was the endpoint /
model half; the api-* captioning providers were already refused at their
boundary by ``provider_settings.resolve_provider`` (``:144-169``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.llm.ollama_client import OllamaClient


@dataclass(frozen=True)
class RefineReadiness:
    """What one probe of the refine endpoint found."""

    base_url: str
    available: bool
    installed: list[str] = field(default_factory=list)
    #: ``None`` when a refine may start; otherwise the user-facing sentence
    #: naming exactly what is missing (endpoint, or model on that endpoint).
    reason: str | None = None


def unreachable_reason(base_url: str) -> str:
    return (f"LLM endpoint {base_url} is unreachable - start it, or configure "
            "and test it on the Server screen (LLM Refine Endpoint).")


def model_missing_reason(model: str, base_url: str) -> str:
    return (f"Model '{model}' is not installed on {base_url} - pull it on the "
            "Server screen or pick an installed model.")


def _is_installed(model: str, installed: list[str]) -> bool:
    # Ollama resolves an untagged name to ``name:latest`` on the wire
    # (``ollama run gemma3`` == ``gemma3:latest``), while ``/api/tags`` lists
    # the tagged form; a user who typed the untagged form has a working model.
    return model in installed or (":" not in model and f"{model}:latest" in installed)


async def refine_readiness(client: OllamaClient, model: str | None = None) -> RefineReadiness:
    """Probe the endpoint once; judge the model against what it lists.

    An endpoint that answers with an EMPTY model list is not judged on the
    model: LM Studio's listing covers loaded models only, and refusing a
    request the server would have served is the wrong side to err on. A
    non-empty list that lacks the model is the user's own 5.1 test case
    ("a model that is not installed") and is refused by name.
    """
    base_url = client.base_url
    try:
        installed = await client.list_models()
    except Exception:  # noqa: BLE001 - any failure to answer IS "unreachable"
        return RefineReadiness(base_url=base_url, available=False,
                               reason=unreachable_reason(base_url))
    if model and installed and not _is_installed(model, installed):
        return RefineReadiness(base_url=base_url, available=True, installed=installed,
                               reason=model_missing_reason(model, base_url))
    return RefineReadiness(base_url=base_url, available=True, installed=installed)
