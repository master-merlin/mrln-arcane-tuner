
import os
import json
import asyncio
import threading
from typing import Any
import structlog

logger = structlog.get_logger(__name__)

# Sentinel distinct from any real mtime (int) or "file absent" (None) so the
# very first stale-check always triggers a load.
_NEVER_LOADED = object()


class SettingsManager:
    """
    Singleton service for managing global application settings.

    As of V4, this only handles application-level config (ports, log level).
    Training, captioning, and masking templates are in SQLite via the
    project/template system.
    """
    _instance = None

    # Class-level (not per-instance) so it protects the load-merge-save
    # critical section across every SettingsManager the process creates —
    # this class is a process-wide singleton in production, and tests that
    # bypass __init__ via __new__ still get correct mutual exclusion.
    # Reentrant: save()/load() take it too, and update_module_settings()
    # wraps a load()+save() pair from the same thread.
    _io_lock = threading.RLock()

    def __init__(self, storage_file: str = "settings.json"):
        # Resolve absolute paths relative to this file
        # This file is in backend/app/core/
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # An explicit MRLN_SETTINGS_PATH wins. The container entrypoint points it
        # at the persistent data volume so UI-set settings — notably the Hugging
        # Face token — survive a pod restart (the default location lives in the
        # ephemeral /app checkout, which a fresh container wipes). Local dev sets
        # no env var and keeps settings beside the backend, unchanged.
        env_path = os.environ.get("MRLN_SETTINGS_PATH")
        if env_path:
            self.storage_file = env_path
            parent = os.path.dirname(self.storage_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
        else:
            self.storage_file = os.path.join(self.root_dir, storage_file)

        self.settings: dict[str, Any] = {}
        self.load()

        # Ensure default application settings
        defaults = {
            "backend_port": 8000,
            "frontend_port": 4200,
            "log_level": "INFO"
        }
        
        if "application" not in self.settings:
            self.settings["application"] = defaults
            self.save()
        else:
            # Merge checks: If any key is missing in existing settings, add it
            changed = False
            for k, v in defaults.items():
                if k not in self.settings["application"]:
                    self.settings["application"][k] = v
                    changed = True
            if changed:
                self.save()

    @classmethod
    def get_instance(cls) -> "SettingsManager":
        if cls._instance is None:
            cls._instance = SettingsManager()
        return cls._instance

    def load(self):
        """Load settings from disk, tracking the file's mtime so
        get_module_settings() can tell a fresh read apart from a stale one."""
        with self._io_lock:
            try:
                stat = os.stat(self.storage_file)
            except OSError:
                self.settings = {}
                self._loaded_mtime_ns = None
                return
            try:
                with open(self.storage_file, "r") as f:
                    self.settings = json.load(f)
                self._loaded_mtime_ns = stat.st_mtime_ns
            except Exception as e:
                logger.error(
                    "settings_load_failed", path=self.storage_file, error=str(e)
                )
                self.settings = {}
                self._loaded_mtime_ns = None

    def _is_cache_stale(self) -> bool:
        """True if the on-disk file's mtime doesn't match what we last
        loaded — either we've never loaded, or something else (an external
        process, or a restore from backup) wrote the file since."""
        try:
            current = os.stat(self.storage_file).st_mtime_ns
        except OSError:
            current = None
        return current != getattr(self, "_loaded_mtime_ns", _NEVER_LOADED)

    def save(self):
        """Save settings to disk atomically.

        Writes the full document to a sibling ``.tmp`` file and
        ``os.replace``s it onto the real path (same pattern as
        dataset/thumbnails.py:88-89) so a crash/exception mid-write can
        never leave settings.json truncated or corrupt — it holds the HF
        token and the jobs.auto_queue/auto_resume prefs the queue depends
        on for its behavior.
        """
        with self._io_lock:
            tmp_path = self.storage_file + ".tmp"
            try:
                # Defensive: Ensure application settings exist before saving
                if "application" not in self.settings:
                    logger.warning("missing_app_settings_restoring_default")
                    self.settings["application"] = {
                        "backend_port": 8000,
                        "frontend_port": 4200,
                        "log_level": "INFO"
                    }

                # Enforce 'application' key is first for readability
                ordered_settings = {}
                if "application" in self.settings:
                    ordered_settings["application"] = self.settings["application"]

                for k, v in self.settings.items():
                    if k != "application":
                        ordered_settings[k] = v

                with open(tmp_path, "w") as f:
                    json.dump(ordered_settings, f, indent=2)
                os.replace(tmp_path, self.storage_file)
                try:
                    self._loaded_mtime_ns = os.stat(self.storage_file).st_mtime_ns
                except OSError:
                    self._loaded_mtime_ns = None
            except Exception as e:
                logger.error(
                    "settings_save_failed", path=self.storage_file, error=str(e)
                )
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass

    def get_module_settings(self, module: str) -> dict[str, Any]:
        """Get settings for a specific module.

        Only re-reads disk when the file's mtime has moved since our last
        load (this manager's own writes, or an external edit) — previously
        this hit disk on every single call regardless.
        """
        with self._io_lock:
            if self._is_cache_stale():
                self.load()
            return self.settings.get(module, {})

    def update_module_settings(self, module: str, settings: dict[str, Any]):
        """
        Update settings for a specific module.

        The whole load-merge-save sequence runs under _io_lock so two
        threads updating different modules concurrently can't race each
        other's read-modify-write and silently drop one side's update.
        """
        with self._io_lock:
            self.load()  # Merge strategy: Load latest
            if module not in self.settings:
                self.settings[module] = {}

            # Merge if dict, otherwise replace
            if isinstance(self.settings[module], dict) and isinstance(settings, dict):
                self.settings[module].update(settings)
            else:
                self.settings[module] = settings

            self.save()

    # ── Async variants (R-API-07) ───────────────────────────────────────
    # Offload disk I/O to a worker thread so FastAPI route handlers
    # don't block the event loop. Sync versions stay for engine callers
    # (engine/utils/model_override_manager.py, engine/core/pipeline/pipeline_data.py)
    # which run in trainer subprocess contexts without an event loop.

    async def load_async(self) -> None:
        """Async wrapper around load()."""
        await asyncio.to_thread(self.load)

    async def save_async(self) -> None:
        """Async wrapper around save()."""
        await asyncio.to_thread(self.save)

    async def get_module_settings_async(self, module: str) -> dict[str, Any]:
        """Async wrapper around get_module_settings()."""
        return await asyncio.to_thread(self.get_module_settings, module)

    async def update_module_settings_async(
        self, module: str, settings: dict[str, Any],
    ) -> None:
        """Async wrapper around update_module_settings().

        After the disk write completes, broadcasts an `entity.changed:updated`
        event so the frontend SettingsStore can reconcile optimistic
        mutations. We're already on the event loop here, so a direct await
        is fine — no run_coroutine_threadsafe needed.

        The sync variant deliberately does NOT emit: it's only called from
        trainer subprocesses (engine/utils/model_override_manager.py,
        engine/core/pipeline/pipeline_data.py) where there is no loop or
        broadcast to reach.
        """
        # Lazy import to avoid a circular dependency (events -> logger ->
        # ... -> settings_manager) at module import time.
        from app.core.events import emit_entity_change, event_manager

        await asyncio.to_thread(self.update_module_settings, module, settings)

        # Re-read the merged module dict so the broadcast reflects the
        # actual on-disk state (the sync update merges into existing keys
        # — callers may have passed a partial delta).
        merged = self.settings.get(module, {})
        merged_dict = (
            dict(merged) if isinstance(merged, dict) else {"value": merged}
        )
        # Shape the payload so the frontend EntityStore can upsert it
        # directly: `id` keys the row, and `settings` carries the actual
        # module dict (mirrors the ModuleSettings TS interface).
        await emit_entity_change(
            event_manager.broadcast,
            entity="settings",
            op="updated",
            id=module,
            payload={"id": module, "module": module, "settings": merged_dict},
        )


def get_settings_manager() -> SettingsManager:
    return SettingsManager.get_instance()


