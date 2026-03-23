"""Dataset domain API routes — re-exports all sub-routers."""

from fastapi import APIRouter

from app.api.dataset.crud_routes import router as crud_router
from app.api.dataset.adjustment_routes import router as adjustment_router
from app.api.dataset.crop_routes import router as crop_router
from app.api.dataset.analysis_routes import router as analysis_router
from app.api.dataset.upscale_routes import router as upscale_router
from app.api.dataset.overlay_routes import router as overlay_router

router = APIRouter()
router.include_router(crud_router)
router.include_router(adjustment_router)
router.include_router(crop_router)
router.include_router(analysis_router)
router.include_router(upscale_router)
router.include_router(overlay_router)

