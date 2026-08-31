"""Caption generation API — single-image captioning and model lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api._path_guard import validate_path_within
from app.api.schemas.common_schemas import TaskEnqueuedResponse
from app.core.captioning.caption_batch import run_caption_batch
from app.core.captioning.caption_refine_batch import run_caption_refine_batch
from app.core.captioning.caption_service import CaptionService
from app.core.llm import provider_settings
from app.core.logger import get_logger
from app.core.tasks.task_manager import task_manager

router = APIRouter()
logger = get_logger(__name__)


class GenerateCaptionResponse(BaseModel):
    """Response body for single-image caption generation."""

    caption: str


class UnloadModelsResponse(BaseModel):
    """Response body for the caption-model unload endpoint."""

    status: str = "success"
    message: str = "All models unloaded and VRAM cleared."


class GenerateCaptionRequest(BaseModel):
    """Request body for single-image caption generation."""

    dataset_name: str
    image_rel_path: str
    model_id: str
    params: dict[str, Any]
    system_prompt: str | None = None
    target: str = "original"  # "original" or "masked"
    # Extra (e.g. control/"before") image rel-paths for two-image edit
    # captioning. Only honoured by multi-image-capable models.
    extra_image_paths: list[str] = []
    # When set, resolves the definition's CaptionFormat and injects its
    # generation prompt + overrides before captioning, then normalizes the
    # returned string to the format's canonical representation.
    definition_id: str | None = None


@router.post("/generate", response_model=GenerateCaptionResponse)
async def generate_caption_api(request: GenerateCaptionRequest):
    """Generate a caption for a single image using the specified model."""
    logger.info(
        "caption_request_received",
        dataset=request.dataset_name,
        image=request.image_rel_path,
        model=request.model_id,
    )

    try:
        provider_settings.validate_caption_model(request.model_id, request.params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    service = CaptionService.get_instance()

    from app.core.dataset_manager import dataset_manager as manager

    dataset = await asyncio.to_thread(manager.get_dataset, request.dataset_name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_root = Path(dataset.path)
    full_path = validate_path_within(
        dataset_root / request.image_rel_path,
        dataset_root,
    )

    # When target is "masked", remap to masked/{stem}.jpg
    if request.target == "masked":
        stem = Path(request.image_rel_path).stem
        masked_path = dataset_root / "masked" / f"{stem}.jpg"
        if not masked_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Masked image not found: masked/{stem}.jpg",
            )
        full_path = masked_path

    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found at {full_path}")

    # Resolve + validate any extra (control) images against the dataset root.
    extra_paths: list[str] = []
    if request.extra_image_paths:
        if not service.supports_multi_image(request.model_id):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_id}' does not support multi-image "
                "captioning.",
            )
        for rel in request.extra_image_paths:
            p = validate_path_within(dataset_root / rel, dataset_root)
            if not p.exists():
                raise HTTPException(
                    status_code=404, detail=f"Control image not found: {rel}"
                )
            extra_paths.append(str(p))

    try:
        params = request.params.copy()
        if request.system_prompt:
            params["system_prompt"] = request.system_prompt

        # Resolve the definition's CaptionFormat (if provided) and inject
        # its generation prompt + overrides before calling the captioner.
        from app.core.captioning.formats import (
            apply_generation_seam,
            get_caption_format_for_definition,
        )

        caption_format = (
            get_caption_format_for_definition(request.definition_id)
            if request.definition_id
            else None
        )
        if caption_format:
            apply_generation_seam(params, caption_format, request.model_id)

        caption = await asyncio.to_thread(
            service.generate_caption,
            image_path=str(full_path),
            model_id=request.model_id,
            params=params,
            extra_image_paths=extra_paths or None,
        )

        # Normalize the raw caption through the format's generation ingest
        # (parse + bbox x->y swap + serialize) so the returned string is always
        # in canonical JSON form with y-first bboxes.
        if caption_format and caption_format.is_structured:
            caption = caption_format.serialize(caption_format.ingest_generated(caption))

        # When target is "masked", auto-save caption alongside masked image
        if request.target == "masked":
            stem = Path(request.image_rel_path).stem
            masked_dir = dataset_root / "masked"
            await asyncio.to_thread(masked_dir.mkdir, parents=True, exist_ok=True)
            caption_path = masked_dir / f"{stem}.txt"

            def _write_caption():
                caption_path.write_text(caption, encoding="utf-8")

            await asyncio.to_thread(_write_caption)

            # Update has_masked_caption in DB
            lookup_key = request.image_rel_path.replace("\\", "/")
            if lookup_key in dataset.media_metadata:
                await manager.update_media_flags_async(
                    request.dataset_name, request.image_rel_path,
                    has_masked_caption=True,
                )

        # Single captions had no unload path at all — only a batch freed its
        # model, so a one-off caption held VRAM until the user switched models.
        # Arming rather than unloading keeps the interactive loop fast; each
        # further caption pushes the deadline out.
        CaptionService.arm_idle_unload()

        return {"caption": caption}
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("caption_generation_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/unload", response_model=UnloadModelsResponse)
async def unload_models_api():
    """Unload all caption models and free VRAM.

    No-ops while a caption batch is pending/running: the batch worker loads
    the model once and frees it in its own ``finally`` after the last image,
    and an eager unload from the UI (model/variant/tab change) would force a
    per-image reload — or crash a generate in flight on the GPU lane.

    The active-batch check and the unload itself run under ONE lock inside
    ``CaptionService.unload_models(skip_if_batch_active=True)`` — closing the
    check-then-act race a separate ``task_manager.list()`` check here used to
    have (a batch could start in the window between the check and the
    ``asyncio.to_thread``-dispatched unload actually running).
    """
    try:
        logger.info("unloading_caption_models")
        service = CaptionService.get_instance()
        did_unload = await asyncio.to_thread(
            service.unload_models, skip_if_batch_active=True
        )
        if not did_unload:
            logger.info("unload_skipped_caption_batch_active")
            return {
                "status": "success",
                "message": "Unload skipped — a captioning batch is in progress; "
                "models are freed when it finishes.",
            }
        return {"status": "success", "message": "All models unloaded and VRAM cleared."}
    except (OSError, RuntimeError) as e:
        logger.error("unload_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class BatchCaptionRequest(BaseModel):
    """Request body for batch caption generation."""

    dataset_name: str
    image_rel_paths: list[str]
    model_id: str
    params: dict[str, Any]
    system_prompt: str | None = None
    target: str = "original"
    # Two-image edit captioning: feed each target's control image(s) to the VLM
    # so the caption is an edit instruction. Requires a multi-image model.
    include_control: bool = False


@router.post("/batch", response_model=TaskEnqueuedResponse)
async def batch_caption_api(request: BatchCaptionRequest):
    """Start a backend-owned captioning task. Always creates the task (queued if
    the GPU lane is busy) and returns its id immediately."""
    try:
        provider_settings.validate_caption_model(request.model_id, request.params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if (
        request.include_control
        and not CaptionService.get_instance().supports_multi_image(request.model_id)
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_id}' does not support multi-image "
            "captioning; disable 'include control' or pick a multi-image model.",
        )

    # Distinguish masked-caption runs in the Task Center — they target the
    # masked/<stem>.txt sidecar, not the plain caption, and can run alongside
    # an original-caption task for the same dataset.
    kind = "Captioning (masked)" if request.target == "masked" else "Captioning"
    title = f"{kind} · {request.dataset_name}"
    task = task_manager.create(
        type="caption_batch",
        title=title,
        total=len(request.image_rel_paths),
        dataset_name=request.dataset_name,
        target=request.target,
    )
    task_manager.enqueue(
        task.id,
        lambda tid: run_caption_batch(
            tid,
            dataset_name=request.dataset_name,
            image_rel_paths=request.image_rel_paths,
            model_id=request.model_id,
            params=request.params,
            system_prompt=request.system_prompt,
            target=request.target,
            include_control=request.include_control,
        ),
        lane=provider_settings.lane_for_model(request.model_id),
    )
    return {"task_id": task.id}


class RefineBatchRequest(BaseModel):
    """Request body for batch LLM caption refinement."""

    dataset_name: str
    image_rel_paths: list[str]
    definition_id: str
    preset: str
    model: str | None = None
    target: str = "original"
    # Caption style: "auto" (derive from the model's text encoder), "tags", or
    # "natural_language" (explicit user override of the refinement template).
    style: str = "auto"
    # When true, promote each refined caption straight to the live variant
    # instead of staging a suggestion for per-image review.
    auto_accept: bool = False


@router.post("/refine-batch", response_model=TaskEnqueuedResponse)
async def refine_batch_api(request: RefineBatchRequest):
    """Start a backend-owned LLM caption-refinement task on the background lane.

    Inference is offloaded to the Ollama server, so this runs on the non-GPU
    ``background`` lane and can proceed alongside a GPU captioning task. Each
    image's caption is refined and written as a pending suggestion (the user
    accepts via the suggestion routes — the live variant is never overwritten)."""
    from app.core.llm import refine_settings

    # Through the accessor, not `.get(key, default)`: `llm_refine.model` is a
    # present-but-EMPTY key in the wild, and `dict.get` returns "" for a present
    # key, which Ollama answers with 400 "model is required" (LANE-49).
    settings = refine_settings.raw_settings()
    base_url = refine_settings.base_url_of(settings)
    model = (request.model or "").strip() or refine_settings.model_of(settings)
    task = task_manager.create(
        type="caption_refine_batch",
        title=f"Refine captions ({request.definition_id})",
        total=len(request.image_rel_paths),
        dataset_name=request.dataset_name,
        target=request.target,
    )
    task_manager.enqueue(
        task.id,
        lambda tid: run_caption_refine_batch(
            tid,
            dataset_name=request.dataset_name,
            image_rel_paths=request.image_rel_paths,
            definition_id=request.definition_id,
            preset=request.preset,
            model=model,
            base_url=base_url,
            target=request.target,
            style=request.style,
            auto_accept=request.auto_accept,
        ),
        lane="background",
    )
    return {"task_id": task.id}
