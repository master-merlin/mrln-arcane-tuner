"""R-API-07: ModelOverrideManager async variants must offload to a worker
thread and produce results identical to their sync counterparts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.schemas.model_overrides import ModelOverride, ModelSourceType
from app.core.settings_manager import SettingsManager
from app.engine.utils.model_override_manager import ModelOverrideManager


@pytest.fixture
def isolated_singleton(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Repoint SettingsManager singleton at a tmp_path settings.json."""
    storage = tmp_path / "settings.json"
    storage.write_text(json.dumps({"application": {"backend_port": 9000}}))
    mgr = SettingsManager.__new__(SettingsManager)
    mgr.root_dir = str(tmp_path)
    mgr.storage_file = str(storage)
    mgr.settings = {}
    mgr.load()
    monkeypatch.setattr(SettingsManager, "_instance", mgr)
    yield
    monkeypatch.setattr(SettingsManager, "_instance", None)


@pytest.mark.asyncio
async def test_get_all_async_matches_sync(isolated_singleton: None) -> None:
    """R-API-07: get_all_async returns same ModelSettings as get_all."""
    sync_result = ModelOverrideManager.get_all()
    async_result = await ModelOverrideManager.get_all_async()
    assert sync_result.global_offline_mode == async_result.global_offline_mode
    assert sync_result.overrides == async_result.overrides


@pytest.mark.asyncio
async def test_get_override_async_returns_none_when_absent(
    isolated_singleton: None,
) -> None:
    """R-API-07: get_override_async returns None for unknown definition_id."""
    result = await ModelOverrideManager.get_override_async("not-a-real-id")
    assert result is None


@pytest.mark.asyncio
async def test_set_override_async_persists(isolated_singleton: None) -> None:
    """R-API-07: set_override_async writes through to settings.json."""
    override = ModelOverride(
        source_type=ModelSourceType.LOCAL_DIFFUSERS,
        local_path="C:/some/local/path",
        skip_update=False,
    )
    await ModelOverrideManager.set_override_async("test-def", override)
    fetched = await ModelOverrideManager.get_override_async("test-def")
    assert fetched is not None
    assert fetched.source_type == ModelSourceType.LOCAL_DIFFUSERS
    assert fetched.local_path == "C:/some/local/path"


@pytest.mark.asyncio
async def test_delete_override_async_removes(isolated_singleton: None) -> None:
    """R-API-07: delete_override_async clears the override."""
    override = ModelOverride(
        source_type=ModelSourceType.LOCAL_DIFFUSERS,
        local_path="C:/path",
        skip_update=False,
    )
    await ModelOverrideManager.set_override_async("test-def-2", override)
    await ModelOverrideManager.delete_override_async("test-def-2")
    assert await ModelOverrideManager.get_override_async("test-def-2") is None


@pytest.mark.asyncio
async def test_save_async_round_trip(isolated_singleton: None) -> None:
    """R-API-07: _save_async persists a ModelSettings instance."""
    settings = ModelOverrideManager.get_all()
    settings.global_offline_mode = True
    await ModelOverrideManager._save_async(settings)
    reloaded = await ModelOverrideManager.get_all_async()
    assert reloaded.global_offline_mode is True
