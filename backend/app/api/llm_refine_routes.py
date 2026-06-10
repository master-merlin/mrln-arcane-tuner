# backend/app/api/llm_refine_routes.py
"""Routes for local-LLM caption refinement: model listing/pull + single-caption preview."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.llm.caption_refine import refine_caption
from app.core.llm.ollama_client import OllamaClient
from app.core.logger import get_logger
from app.core.settings_manager import SettingsManager

router = APIRouter(prefix="/api/llm-refine", tags=["llm-refine"])
logger = get_logger(__name__)

CURATED_MODELS = ["qwen2.5:7b-instruct", "llama3.1:8b-instruct-q4_K_M", "qwen2.5:3b-instruct"]
_DEFAULT_BASE_URL = "http://localhost:11434"


def _settings() -> dict:
    return SettingsManager.get_instance().get_module_settings("llm_refine") or {}


def _make_client() -> OllamaClient:
    base = _settings().get("base_url", _DEFAULT_BASE_URL)
    return OllamaClient(base_url=base)


def _default_model() -> str:
    return _settings().get("model", CURATED_MODELS[0])


class ModelsResponse(BaseModel):
    curated: list[str]
    installed: list[str]
    available: bool


class PullRequest(BaseModel):
    tag: str


class RefinePreviewRequest(BaseModel):
    text: str
    preset: str
    model: str | None = None


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    client = _make_client()
    available = await client.available()
    installed = await client.list_models() if available else []
    return ModelsResponse(curated=CURATED_MODELS, installed=installed, available=available)


@router.post("/pull")
async def pull(req: PullRequest) -> dict:
    ok = await _make_client().pull(req.tag)
    return {"ok": ok}


@router.post("/refine-preview")
async def refine_preview(req: RefinePreviewRequest) -> dict:
    model = req.model or _default_model()
    refined = await refine_caption(_make_client(), model, req.text, req.preset)
    return {"refined": refined}
