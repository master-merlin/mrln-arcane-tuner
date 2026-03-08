from unittest.mock import MagicMock, patch

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
    # Setup mock
    mock_instance = MagicMock()
    mock_settings_manager_cls.get_instance.return_value = mock_instance

    payload = {"theme": "light", "notifications": False}
    response = client.put("/api/settings/ui", json=payload)
    
    assert response.status_code == 200
    assert response.json() == payload
    mock_instance.update_module_settings.assert_called_once_with("ui", payload)
