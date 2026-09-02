# backend/app/api/llm_refine_routes.py
"""Routes for local-LLM caption refinement: model listing/pull + single-caption preview."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.llm import refine_settings
from app.core.llm.caption_refine import refine_caption
from app.core.llm.ollama_client import OllamaClient
from app.core.llm.refine_guard import refine_readiness
from app.core.logger import get_logger
from app.core.url_guard import OutboundUrlRejected

router = APIRouter(prefix="/api/llm-refine", tags=["llm-refine"])
logger = get_logger(__name__)

#: Re-exported from the settings accessor so the route and the fallback model
#: cannot drift apart (``refine_settings.DEFAULT_MODEL`` is ``CURATED_MODELS[0]``).
CURATED_MODELS = refine_settings.CURATED_MODELS


def _settings() -> dict:
    # Through the accessor the refine-batch boundary reads (RULE-21): a second
    # reader of the store here is how this status probed one endpoint while
    # the refusal in ``caption_routes`` judged another (LANE-57).
    return refine_settings.raw_settings()


def _make_client() -> OllamaClient:
    """Build the client. The base URL is validated by ``OllamaClient`` itself
    (the sink), which raises ``OutboundUrlRejected`` for a destination this
    server may not request while hosted. This function does NOT re-check —
    a second copy of the rule here is how the first one drifts."""
    return OllamaClient(base_url=refine_settings.base_url_of(_settings()))


def _client_or_400() -> OllamaClient:
    """Turn a refused base URL into an actionable 400 rather than a 500.

    The refusal is the user's own configuration talking back to them, so it
    carries the guard's message verbatim. Hosted installs default to
    ``localhost:11434``, which the guard blocks — without this the caption
    settings page would answer 500 on a perfectly ordinary hosted deployment.
    """
    try:
        return _make_client()
    except OutboundUrlRejected as e:
        logger.warning("llm_refine_base_url_rejected", error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e


def _default_model() -> str:
    """The configured refine model, or the curated fallback.

    Reads through ``refine_settings`` because a PRESENT-but-empty ``model`` key
    is what ``dict.get(key, default)`` hands straight to Ollama (LANE-49).
    """
    return refine_settings.model_of(_settings())


class ModelsResponse(BaseModel):
    curated: list[str]
    installed: list[str]
    #: The endpoint answered a listing. A surface that picks its own model
    #: from ``installed`` (the mass-caption Refine tab) gates on this.
    available: bool
    #: APPENDED (LANE-57): the sentence ``POST /captions/refine-batch`` refuses
    #: with — the UI disables Start with THIS text, never a re-derived one
    #: (RULE-21). ``None`` when a refine that names NO model may start:
    #: judged against the configured default model, because that is the model
    #: a model-less request (the detail sidebar's Refine) is served with
    #: (LANE-70; ``caption_routes.py`` ``refine-batch`` resolves the same
    #: way). So ``available`` can be ``True`` with a reason: endpoint up,
    #: default model not installed there — the sidebar is blocked, the
    #: Refine tab with an installed model chosen is not.
    unavailable_reason: str | None = None


class PullRequest(BaseModel):
    tag: str


class RefinePreviewRequest(BaseModel):
    text: str
    preset: str
    model: str | None = None


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    # One probe through the same predicate the refine boundary refuses on,
    # judging the model a model-less refine request is served with.
    ready = await refine_readiness(_client_or_400(), _default_model())
    return ModelsResponse(curated=CURATED_MODELS, installed=ready.installed,
                          available=ready.available, unavailable_reason=ready.reason)


@router.post("/pull")
async def pull(req: PullRequest) -> dict:
    ok = await _client_or_400().pull(req.tag)
    return {"ok": ok}


@router.post("/refine-preview")
async def refine_preview(req: RefinePreviewRequest) -> dict:
    model = req.model or _default_model()
    refined = await refine_caption(_client_or_400(), model, req.text, req.preset)
    return {"refined": refined}
