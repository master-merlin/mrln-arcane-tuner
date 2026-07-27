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
    payload = {"theme": "light", "notifications": False}
    # No pre-existing keys in this fixture, so the merged read-back happens
    # to equal the raw payload — see test_update_settings_returns_merged_dict
    # below for the case where it does NOT.
    mock_instance.get_module_settings.return_value = payload
    mock_settings_manager_cls.get_instance.return_value = mock_instance

    response = client.put("/api/settings/ui", json=payload)

    assert response.status_code == 200
    assert response.json() == payload
    mock_instance.update_module_settings_async.assert_awaited_once_with("ui", payload)


@patch("app.api.settings_routes.SettingsManager")
def test_update_settings_returns_merged_dict(mock_settings_manager_cls, client):
    """W5.T10: PUT returns the ACTUAL persisted state, not an echo of the
    request body — SettingsManager.update_module_settings MERGES the payload
    into the module's existing dict, so a partial update's response must
    reflect the full merged result (pre-existing keys included), not just
    what this request sent."""
    mock_instance = MagicMock()
    mock_instance.update_module_settings_async = AsyncMock(return_value=None)
    # The module already had `theme` persisted; this PUT only sends
    # `notifications`, but the merged (persisted) state carries BOTH.
    merged = {"theme": "dark", "notifications": False}
    mock_instance.get_module_settings.return_value = merged
    mock_settings_manager_cls.get_instance.return_value = mock_instance

    response = client.put("/api/settings/ui", json={"notifications": False})

    assert response.status_code == 200
    assert response.json() == merged
    mock_instance.get_module_settings.assert_called_with("ui")


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
    mock_instance.get_module_settings.return_value = {"x": 1}
    mock_settings_manager_cls.get_instance.return_value = mock_instance

    for module in ("training", "captioning", "masking"):
        response = client.put(f"/api/settings/{module}", json={"x": 1})
        assert response.status_code == 200
        assert response.json() == {"x": 1}


@patch("app.api.settings_routes.SettingsManager")
def test_get_settings_nested_shape_survives_open_response_model(
    mock_settings_manager_cls, client,
):
    """P3c: the `dict[str, Any]` response_model must not filter/coerce a
    module blob with nested objects, lists, and null — proving the open
    contract doesn't silently drop keys the way a named schema could."""
    mock_instance = MagicMock()
    payload = {
        "nested": {"a": 1, "b": [1, 2, 3]},
        "list_of_objects": [{"x": True}, {"y": None}],
        "null_field": None,
        "bool_field": False,
    }
    mock_instance.get_module_settings.return_value = payload
    mock_settings_manager_cls.get_instance.return_value = mock_instance

    response = client.get("/api/settings/some_module")
    assert response.status_code == 200
    assert response.json() == payload


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
