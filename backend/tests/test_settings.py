from unittest.mock import AsyncMock, MagicMock, patch

@patch("app.api.settings_routes.SettingsManager")
def test_get_settings(mock_settings_manager_cls, client):
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.get_module_settings.return_value = {"theme": "dark", "notifications": True}
    mock_settings_manager_cls.get_instance.return_value = mock_instance

    response = client.get("/api/settings/ui")

    assert response.status_code == 200
    assert response.json() == {"theme": "dark", "notifications": True}
    mock_instance.get_module_settings.assert_called_once_with("ui")

@patch("app.api.settings_routes.SettingsManager")
def test_update_settings(mock_settings_manager_cls, client):
    # Setup mock — the route now calls update_module_settings_async so the
    # SettingsStore receives an entity.changed broadcast.
    mock_instance = MagicMock()
    mock_instance.update_module_settings_async = AsyncMock(return_value=None)
    mock_settings_manager_cls.get_instance.return_value = mock_instance

    payload = {"theme": "light", "notifications": False}
    response = client.put("/api/settings/ui", json=payload)

    assert response.status_code == 200
    assert response.json() == payload
    mock_instance.update_module_settings_async.assert_awaited_once_with("ui", payload)


# ── B-CLEAN-5: the 410 migration-guard is gone; former + unknown modules ──
# flow through to the generic (schemaless, per-module) settings store.


@patch("app.api.settings_routes.SettingsManager")
def test_get_settings_no_longer_410s_formerly_migrated_modules(mock_settings_manager_cls, client):
    """`training`/`captioning`/`masking` used to 410 — the guard is removed
    now that the frontend no longer calls them; they behave like any other
    module (empty dict if never written)."""
    mock_instance = MagicMock()
    mock_instance.get_module_settings.return_value = {}
    mock_settings_manager_cls.get_instance.return_value = mock_instance

    for module in ("training", "captioning", "masking"):
        response = client.get(f"/api/settings/{module}")
        assert response.status_code == 200
        assert response.json() == {}


@patch("app.api.settings_routes.SettingsManager")
def test_put_settings_no_longer_410s_formerly_migrated_modules(mock_settings_manager_cls, client):
    mock_instance = MagicMock()
    mock_instance.update_module_settings_async = AsyncMock(return_value=None)
    mock_settings_manager_cls.get_instance.return_value = mock_instance

    for module in ("training", "captioning", "masking"):
        response = client.put(f"/api/settings/{module}", json={"x": 1})
        assert response.status_code == 200
        assert response.json() == {"x": 1}


@patch("app.api.settings_routes.SettingsManager")
def test_get_settings_unknown_module_is_sane_empty_dict_not_500(
    mock_settings_manager_cls, client,
):
    """The settings store is a schemaless per-module key-value blob (see
    module docstring / SettingsService) — a module nobody has written to
    yet is not an "unknown resource" 404 candidate, it is a legitimate
    empty read, mirroring `dict.get(key, {})`. Pin that this stays a sane
    200 + `{}` and never a 500."""
    mock_instance = MagicMock()
    mock_instance.get_module_settings.return_value = {}
    mock_settings_manager_cls.get_instance.return_value = mock_instance

    response = client.get("/api/settings/__never_configured_module__")

    assert response.status_code == 200
    assert response.json() == {}
