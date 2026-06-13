"""Pair-health and role-ordering routes for paired edit datasets.

Control slot *files* are managed through the existing upload/delete
routes (``crud_routes``); this module owns the pair-level views and
mutations: the on-demand health report and the logical role ordering
(``control_info.role_order`` — metadata only, files never move).
"""

from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, HTTPException

from app.api.schemas.common_schemas import TaskEnqueuedResponse
from app.api.schemas.dataset_schemas import (
    ControlGenerateBatchRequest,
    OrphansDeletedResponse,
    PairHealthResponse,
    PairOrderApplyAllRequest,
    PairOrderApplyAllResponse,
    PairOrderRequest,
    PairOrderResponse,
)
from app.core.dataset.control_batch import run_control_batch
from app.core.dataset.control_helpers import compute_pair_health
from app.core.dataset_manager import dataset_manager
from app.core.logger import get_logger
from app.core.tasks.task_manager import task_manager

router = APIRouter()
logger = get_logger(__name__)


def _value_error_status(exc: ValueError) -> int:
    """Map manager ValueErrors: missing entities → 404, bad input → 400."""
    return 404 if "not found" in str(exc).lower() else 400


@router.get(
    "/datasets/{name}/control/health", response_model=PairHealthResponse,
)
async def get_pair_health(name: str):
    """On-demand pair-health report (disk walk; warnings never block)."""
    dataset = await asyncio.to_thread(dataset_manager.get_dataset, name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return await asyncio.to_thread(compute_pair_health, dataset)


@router.delete(
    "/datasets/{name}/control/orphans", response_model=OrphansDeletedResponse,
)
async def delete_orphan_controls(name: str):
    """Delete every control file whose stem has no target image.

    Orphans have no ``media_items`` row, so this is pure filesystem
    cleanup — the health report is the only thing that changes.
    """
    import os

    dataset = await asyncio.to_thread(dataset_manager.get_dataset, name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    def _delete() -> int:
        health = compute_pair_health(dataset)
        deleted = 0
        for orphan in health["orphans"]:
            try:
                os.remove(os.path.join(dataset.path, orphan["rel_path"]))
                deleted += 1
            except OSError:
                logger.warning(
                    "orphan_control_delete_failed",
                    dataset_name=name, rel_path=orphan["rel_path"],
                )
        return deleted

    deleted = await asyncio.to_thread(_delete)
    logger.info("orphan_controls_deleted", dataset_name=name, deleted=deleted)
    return {"deleted": deleted}


@router.post(
    "/datasets/{name}/control/generate-batch",
    response_model=TaskEnqueuedResponse,
)
async def generate_control_batch(name: str, request: ControlGenerateBatchRequest):
    """Enqueue a PIL-only batch that degrades each target into a control slot.

    Runs on the non-GPU ``background`` lane (never blocks training/caption);
    skips targets that already have a control in the slot unless ``overwrite``.
    """
    dataset = await asyncio.to_thread(dataset_manager.get_dataset, name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    wanted = set(request.stems) if request.stems else None
    total = sum(
        1
        for k in (dataset.media_metadata or {})
        if wanted is None or os.path.splitext(os.path.basename(k))[0] in wanted
    )

    ops = [o.model_dump() for o in request.ops]
    task = task_manager.create(
        type="control_generate",
        title=f"Controls · {name}",
        total=total,
        dataset_name=name,
    )
    task_manager.enqueue(
        task.id,
        lambda tid: run_control_batch(
            tid, dataset_name=name, slot=request.slot, ops=ops,
            overwrite=request.overwrite, stems=request.stems,
        ),
        lane="background",
    )
    logger.info(
        "control_generate_enqueued",
        dataset_name=name, slot=request.slot, total=total,
        overwrite=request.overwrite,
    )
    return {"task_id": task.id}


@router.patch(
    "/datasets/{name}/images/{media_file:path}/pair-order",
    response_model=PairOrderResponse,
)
async def set_pair_order(name: str, media_file: str, request: PairOrderRequest):
    """Set or clear (null) the logical role order for one pair group."""
    try:
        logger.info(
            "setting_pair_order", dataset_name=name,
            media_file=media_file, role_order=request.role_order,
        )
        return await asyncio.to_thread(
            dataset_manager.set_pair_order, name, media_file, request.role_order,
        )
    except ValueError as e:
        raise HTTPException(status_code=_value_error_status(e), detail=str(e))


@router.post(
    "/datasets/{name}/pair-order/apply-all",
    response_model=PairOrderApplyAllResponse,
)
async def apply_pair_order_all(name: str, request: PairOrderApplyAllRequest):
    """Apply one role order to every pair group that has the named slots."""
    try:
        logger.info(
            "applying_pair_order_all", dataset_name=name,
            role_order=request.role_order,
        )
        return await asyncio.to_thread(
            dataset_manager.apply_pair_order_all, name, request.role_order,
        )
    except ValueError as e:
        raise HTTPException(status_code=_value_error_status(e), detail=str(e))
