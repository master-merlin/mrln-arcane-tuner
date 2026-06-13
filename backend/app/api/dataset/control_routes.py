"""Pair-health and role-ordering routes for paired edit datasets.

Control slot *files* are managed through the existing upload/delete
routes (``crud_routes``); this module owns the pair-level views and
mutations: the on-demand health report and the logical role ordering
(``control_info.role_order`` — metadata only, files never move).
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from app.api.schemas.dataset_schemas import (
    PairHealthResponse,
    PairOrderApplyAllRequest,
    PairOrderApplyAllResponse,
    PairOrderRequest,
    PairOrderResponse,
)
from app.core.dataset.control_helpers import compute_pair_health
from app.core.dataset_manager import dataset_manager
from app.core.logger import get_logger

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
