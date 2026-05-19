"""Model Override Manager — read/write per-model source overrides.

Thin service layer over ``SettingsManager`` that serialises
:class:`ModelSettings` to/from the ``models`` module in
``settings.json``.
"""

from __future__ import annotations

import asyncio

import structlog

from app.core.schemas.model_overrides import (
    ModelOverride,
    ModelSettings,
    ModelSourceType,
)
from app.core.settings_manager import get_settings_manager

logger = structlog.get_logger(__name__)


class ModelOverrideManager:
    """Static helper that persists per-model source overrides in settings.json."""

    # ── Internal I/O ────────────────────────────────────────────────────

    @staticmethod
    def _load() -> ModelSettings:
        raw = get_settings_manager().get_module_settings("models")
        if not raw:
            return ModelSettings()
        return ModelSettings.model_validate(raw)

    @staticmethod
    def _save(settings: ModelSettings) -> None:
        get_settings_manager().update_module_settings(
            "models", settings.model_dump(),
        )

    # ── CRUD ────────────────────────────────────────────────────────────

    @staticmethod
    def get_all() -> ModelSettings:
        """Return full model settings (global flags + all overrides)."""
        return ModelOverrideManager._load()

    @staticmethod
    def get_override(definition_id: str) -> ModelOverride | None:
        """Return the override for *definition_id*, or ``None``."""
        return ModelOverrideManager._load().overrides.get(definition_id)

    @staticmethod
    def set_override(definition_id: str, override: ModelOverride) -> None:
        """Create or update the override for *definition_id*."""
        settings = ModelOverrideManager._load()
        settings.overrides[definition_id] = override
        ModelOverrideManager._save(settings)
        logger.info(
            "model_override_saved",
            id=definition_id,
            source_type=override.source_type,
        )

    @staticmethod
    def delete_override(definition_id: str) -> None:
        """Remove the override for *definition_id* (revert to YAML default)."""
        settings = ModelOverrideManager._load()
        if settings.overrides.pop(definition_id, None) is not None:
            ModelOverrideManager._save(settings)
            logger.info("model_override_removed", id=definition_id)

    # ── Async variants (R-API-07) ───────────────────────────────────────
    # FastAPI routes use these; engine code (is_offline,
    # resolve_effective_source) keeps the sync API since those are called
    # from sync trainer-subprocess contexts.
    #
    # NOTE: set_override_async / delete_override_async widen the pre-existing
    # load->mutate->save race window because each `await` releases the event
    # loop. Two concurrent calls for different definition_ids can interleave
    # load-A, load-B, save-A, save-B and the second save will clobber the
    # first override. This is the same hazard the sync versions have under
    # threading, just easier to trigger from a single event loop. Adding a
    # lock is deferred per the R-API-07 spec; revisit if telemetry shows
    # actual override clobbering.

    @staticmethod
    async def _load_async() -> ModelSettings:
        """Async variant of :meth:`_load` -- offloaded settings.json read."""
        return await asyncio.to_thread(ModelOverrideManager._load)

    @staticmethod
    async def _save_async(settings: ModelSettings) -> None:
        """Async variant of :meth:`_save` -- offloaded settings.json write."""
        await asyncio.to_thread(ModelOverrideManager._save, settings)

    @staticmethod
    async def get_all_async() -> ModelSettings:
        """Async variant of :meth:`get_all`."""
        return await ModelOverrideManager._load_async()

    @staticmethod
    async def get_override_async(definition_id: str) -> ModelOverride | None:
        """Async variant of :meth:`get_override`."""
        settings = await ModelOverrideManager._load_async()
        return settings.overrides.get(definition_id)

    @staticmethod
    async def set_override_async(
        definition_id: str, override: ModelOverride,
    ) -> None:
        """Async variant of :meth:`set_override`."""
        settings = await ModelOverrideManager._load_async()
        settings.overrides[definition_id] = override
        await ModelOverrideManager._save_async(settings)
        logger.info(
            "model_override_saved",
            id=definition_id,
            source_type=override.source_type,
        )

    @staticmethod
    async def delete_override_async(definition_id: str) -> None:
        """Async variant of :meth:`delete_override`."""
        settings = await ModelOverrideManager._load_async()
        if settings.overrides.pop(definition_id, None) is not None:
            await ModelOverrideManager._save_async(settings)
            logger.info("model_override_removed", id=definition_id)

    # ── Global offline mode ─────────────────────────────────────────────

    @staticmethod
    def set_global_offline(enabled: bool) -> None:
        settings = ModelOverrideManager._load()
        settings.global_offline_mode = enabled
        ModelOverrideManager._save(settings)
        logger.info("global_offline_mode_set", enabled=enabled)

    # ── Query helpers ───────────────────────────────────────────────────

    @staticmethod
    def is_offline(definition_id: str) -> bool:
        """``True`` when either global offline or per-model skip_update is active."""
        settings = ModelOverrideManager._load()
        if settings.global_offline_mode:
            return True
        override = settings.overrides.get(definition_id)
        return override.skip_update if override else False

    @staticmethod
    def resolve_effective_source(
        definition_id: str,
    ) -> tuple[ModelSourceType, str | None, bool]:
        """Determine the effective source for *definition_id*.

        Returns:
            ``(source_type, local_path_or_None, local_files_only)``

            * HF_HUB → ``(HF_HUB, None, is_offline)``
            * LOCAL_DIFFUSERS → ``(LOCAL_DIFFUSERS, local_path, False)``
            * LOCAL_SAFETENSORS → ``(LOCAL_SAFETENSORS, local_path, False)``
        """
        settings = ModelOverrideManager._load()
        override = settings.overrides.get(definition_id)

        if not override or override.source_type == ModelSourceType.HF_HUB:
            offline = settings.global_offline_mode or (
                override.skip_update if override else False
            )
            return ModelSourceType.HF_HUB, None, offline

        return override.source_type, override.local_path, False
