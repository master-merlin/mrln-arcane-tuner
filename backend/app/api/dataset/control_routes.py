"""Pair-health and role-ordering routes for paired edit datasets.

Control slot *files* are managed through the existing upload/delete
routes (``crud_routes``); this module owns the pair-level views and
mutations: the on-demand health report and the logical role ordering
(``control_info.role_order`` — metadata only, files never move).
"""

from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, Depends, HTTPException

from app.api._deps import dataset_or_404
from app.api._path_guard import sanitize_filename, validate_path_within
from app.api.schemas.common_schemas import TaskEnqueuedResponse
from app.api.schemas.dataset_schemas import (
    ControlAssignRequest,
    ControlAssignResponse,
    ControlGenerateBatchRequest,
    OrphansDeletedResponse,
    PairHealthResponse,
    PairOrderApplyAllRequest,
    PairOrderApplyAllResponse,
    PairOrderRequest,
    PairOrderResponse,
)
from app.core.dataset.control_batch import run_control_batch
from app.core.dataset.control_helpers import (
    CONTROL_IMAGE_EXTS,
    CONTROL_SLOTS,
    compute_pair_health,
    control_slot_dir_name,
    prepare_control_slot_path,
)
from app.core.dataset_manager import Dataset, dataset_manager
from app.core.logger import get_logger
from app.core.tasks.task_manager import task_manager

router = APIRouter()
logger = get_logger(__name__)


def get_dataset_or_404(name: str) -> Dataset:
    """Path-operation dependency: resolve a dataset by name or 404."""
    return dataset_or_404(dataset_manager.get_dataset(name))


def _value_error_status(exc: ValueError) -> int:
    """Map manager ValueErrors: missing entities → 404, bad input → 400."""
    return 404 if "not found" in str(exc).lower() else 400


@router.get(
    "/datasets/{name}/control/health", response_model=PairHealthResponse,
)
async def get_pair_health(name: str, dataset: Dataset = Depends(get_dataset_or_404)):
    """On-demand pair-health report (disk walk; warnings never block)."""
    return await asyncio.to_thread(compute_pair_health, dataset)


@router.delete(
    "/datasets/{name}/control/orphans", response_model=OrphansDeletedResponse,
)
async def delete_orphan_controls(name: str, dataset: Dataset = Depends(get_dataset_or_404)):
    """Delete every control file whose stem has no target image.

    Orphans have no ``media_items`` row, so this is pure filesystem
    cleanup — the health report is the only thing that changes.
    """
    import os

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
    "/datasets/{name}/control/assign", response_model=ControlAssignResponse,
)
async def assign_control(
    name: str, request: ControlAssignRequest, dataset: Dataset = Depends(get_dataset_or_404),
):
    """Re-match an existing on-disk control file to a target stem/slot.

    Moves/renames ``src_rel_path`` (a file under a control slot dir) to
    ``control{slot}/{target_stem}{ext}`` so it pairs with an existing
    target, then refreshes that target's control metadata (emits the
    ``entity.changed`` event). Powers the Pairs-manager "re-match orphan"
    action — no re-upload needed. Files only move; no logical role order
    is touched.
    """
    slot = request.slot
    if not 1 <= slot <= len(CONTROL_SLOTS):
        raise HTTPException(
            status_code=400, detail=f"slot must be 1..{len(CONTROL_SLOTS)}",
        )

    target_stem = sanitize_filename((request.target_stem or "").strip())
    if not target_stem:
        raise HTTPException(status_code=400, detail="target_stem is required")
    stems = {os.path.splitext(k)[0] for k in (dataset.media_metadata or {})}
    if target_stem not in stems:
        raise HTTPException(
            status_code=400,
            detail=f"No target image with stem '{target_stem}' in '{name}'",
        )

    dataset_root = dataset.path
    src_rel = (request.src_rel_path or "").replace("\\", "/").strip()
    # Containment first (escapes → 403), then a structural slot/ext check.
    validate_path_within(os.path.join(dataset_root, src_rel), dataset_root)
    src_parts = src_rel.split("/")
    if len(src_parts) != 2 or src_parts[0] not in CONTROL_SLOTS:
        raise HTTPException(
            status_code=400,
            detail="src_rel_path must be a control-slot file (e.g. control/foo.jpg)",
        )
    ext = os.path.splitext(src_parts[1])[1].lower()
    if ext not in CONTROL_IMAGE_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Control images must be one of {list(CONTROL_IMAGE_EXTS)}",
        )
    src_abs = os.path.join(dataset_root, src_rel)
    if not os.path.isfile(src_abs):
        raise HTTPException(
            status_code=400, detail=f"Control file '{src_rel}' not found",
        )

    def _move() -> str:
        # prepare_control_slot_path makes the slot dir and purges same-stem
        # siblings of OTHER extensions in the destination so detection stays
        # deterministic. Skip the move when src is already the destination.
        dest_abs = prepare_control_slot_path(dataset_root, slot, target_stem, ext)
        if os.path.abspath(src_abs) != os.path.abspath(dest_abs):
            os.replace(src_abs, dest_abs)
        return f"{control_slot_dir_name(slot)}/{target_stem}{ext}"

    rel_path = await asyncio.to_thread(_move)
    await asyncio.to_thread(
        dataset_manager.refresh_control_metadata, name, target_stem,
    )
    logger.info(
        "control_assigned",
        dataset_name=name, src=src_rel, dest=rel_path, slot=slot,
    )
    return {"rel_path": rel_path, "target_stem": target_stem}


@router.post(
    "/datasets/{name}/control/generate-batch",
    response_model=TaskEnqueuedResponse,
)
async def generate_control_batch(
    name: str, request: ControlGenerateBatchRequest,
    dataset: Dataset = Depends(get_dataset_or_404),
):
    """Enqueue a PIL-only batch that degrades each target into a control slot.

    Runs on the non-GPU ``background`` lane (never blocks training/caption);
    skips targets that already have a control in the slot unless ``overwrite``.
    """
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
