"""Training domain API routes — re-exports all sub-routers."""

from fastapi import APIRouter

from app.api.training.plugin_routes import router as plugin_router
from app.api.training.checkpoint_routes import router as checkpoint_router
from app.api.training.definition_routes import router as definition_router
from app.api.training.job_routes import router as job_router
from app.api.training.lora_routes import router as lora_router
from app.api.training.history_routes import router as history_router
from app.api.training.template_routes import router as template_router

router = APIRouter()
router.include_router(plugin_router)
router.include_router(checkpoint_router)
router.include_router(definition_router)
router.include_router(job_router)
router.include_router(lora_router)
router.include_router(history_router)
router.include_router(template_router)
