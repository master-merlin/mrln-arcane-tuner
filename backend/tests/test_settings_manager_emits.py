"""SettingsManager.update_module_settings_async broadcasts entity.changed
with entity='settings' so the frontend SettingsStore can reconcile
optimistic mutations.

Emission lives in the async variant (no `_loop` field exists; the async
method is already on the event loop, so a direct await is safe). The
sync variant is intentionally NOT instrumented — it's only used by
trainer subprocesses that have no loop/broadcast to begin with.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.core.settings_manager import SettingsManager


@pytest.fixture
def isolated_settings_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> SettingsManager:
    """Build a SettingsManager whose storage_file is under tmp_path so the
    test doesn't touch the real backend/settings.json. Bypasses the
    singleton via __new__."""
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
async def test_update_module_settings_async_broadcasts_updated(
    isolated_settings_manager: SettingsManager,
) -> None:
    """update_module_settings_async must emit entity.changed:updated with
    entity='settings', id=<module>, payload=<settings dict>.
    """
    mock_broadcast = AsyncMock()

    # The settings_manager imports event_manager lazily inside the
    # async method (to avoid a circular dep), so patch the source.
    with patch("app.core.events.event_manager.broadcast", mock_broadcast):
        await isolated_settings_manager.update_module_settings_async(
            "models", {"global_offline_mode": True},
        )

    entity_calls = [
        c for c in mock_broadcast.await_args_list
        if c.args and c.args[0] == "entity.changed"
    ]
    updates = [
        c for c in entity_calls
        if c.args[1]["op"] == "updated"
        and c.args[1]["entity"] == "settings"
    ]
    assert len(updates) == 1, (
        f"expected one updated event, got {len(updates)} from {entity_calls}"
    )
    env = updates[0].args[1]
    assert env["id"] == "models"
    assert env["payload"] is not None
    # Payload shape matches the ModuleSettings TS interface so the
    # frontend EntityStore can upsert directly.
    assert env["payload"]["id"] == "models"
    assert env["payload"]["module"] == "models"
    # The nested `settings` reflects the merged module dict, not the delta.
    assert env["payload"]["settings"]["global_offline_mode"] is True
