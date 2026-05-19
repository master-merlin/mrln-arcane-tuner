"""R-API-07: SettingsManager async variants must offload to a worker thread
and produce results identical to their sync counterparts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.settings_manager import SettingsManager


@pytest.fixture
def isolated_settings_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SettingsManager:
    """Build a SettingsManager whose storage_file is under tmp_path so tests
    do not touch the real backend/settings.json. Avoid the singleton."""
    storage = tmp_path / "settings.json"
    storage.write_text(json.dumps({"application": {"backend_port": 9000}}))
    monkeypatch.setattr(SettingsManager, "_instance", None)
    mgr = SettingsManager.__new__(SettingsManager)
    mgr.root_dir = str(tmp_path)
    mgr.storage_file = str(storage)
    mgr.settings = {}
    mgr.load()
    return mgr


@pytest.mark.asyncio
async def test_load_async_matches_sync(isolated_settings_manager: SettingsManager) -> None:
    """R-API-07: load_async must produce the same in-memory state as load."""
    expected = {"application": {"backend_port": 9000}}
    isolated_settings_manager.settings = {}
    await isolated_settings_manager.load_async()
    assert isolated_settings_manager.settings == expected


@pytest.mark.asyncio
async def test_get_module_settings_async_matches_sync(
    isolated_settings_manager: SettingsManager,
) -> None:
    """R-API-07: async getter must return identical value to sync."""
    sync_result = isolated_settings_manager.get_module_settings("application")
    async_result = await isolated_settings_manager.get_module_settings_async("application")
    assert sync_result == async_result == {"backend_port": 9000}


@pytest.mark.asyncio
async def test_save_async_writes_to_disk(
    isolated_settings_manager: SettingsManager, tmp_path: Path,
) -> None:
    """R-API-07: save_async must persist to disk identically to save."""
    isolated_settings_manager.settings["application"]["backend_port"] = 7777
    await isolated_settings_manager.save_async()
    on_disk = json.loads((tmp_path / "settings.json").read_text())
    assert on_disk["application"]["backend_port"] == 7777


@pytest.mark.asyncio
async def test_update_module_settings_async_merges(
    isolated_settings_manager: SettingsManager,
) -> None:
    """R-API-07: async updater must merge into the module dict like sync."""
    await isolated_settings_manager.update_module_settings_async(
        "models", {"global_offline_mode": True},
    )
    assert isolated_settings_manager.settings["models"]["global_offline_mode"] is True
