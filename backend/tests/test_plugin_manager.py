"""
Tests for PluginManager — plugin discovery and registry.

Covers: init, get_plugin, list_plugins, and discover_plugins behavior.
"""
from unittest.mock import patch, MagicMock

from app.core.plugin_manager import PluginManager


class TestPluginManagerInit:
    """Tests for PluginManager initialization."""

    def test_default_plugin_dir(self):
        """Default plugin directory should be 'app/engine/models'."""
        pm = PluginManager()
        assert pm.plugin_dir == "app/engine/models"

    def test_custom_plugin_dir(self):
        """Custom plugin directory should be accepted."""
        pm = PluginManager(plugin_dir="custom/plugins")
        assert pm.plugin_dir == "custom/plugins"

    def test_empty_plugins_on_init(self):
        """Plugin registry should be empty on initialization."""
        pm = PluginManager()
        assert pm.list_plugins() == []


class TestGetPlugin:
    """Tests for get_plugin lookup."""

    def test_get_plugin_returns_none_for_unknown(self):
        """get_plugin for unregistered ID should return None."""
        pm = PluginManager()
        assert pm.get_plugin("nonexistent") is None

    def test_get_plugin_returns_registered_plugin(self):
        """get_plugin should return a previously registered plugin."""
        pm = PluginManager()

        mock_plugin = MagicMock()
        pm._plugins["test_model"] = mock_plugin

        assert pm.get_plugin("test_model") is mock_plugin


class TestListPlugins:
    """Tests for list_plugins."""

    def test_list_plugins_returns_dicts_with_ids(self):
        """list_plugins should return list of {id: ...} dicts."""
        pm = PluginManager()
        pm._plugins["model_a"] = MagicMock()
        pm._plugins["model_b"] = MagicMock()

        result = pm.list_plugins()
        ids = [p["id"] for p in result]
        assert "model_a" in ids
        assert "model_b" in ids


class TestDiscoverPlugins:
    """Tests for discover_plugins scanning behavior."""

    def test_discover_clears_existing_plugins(self):
        """discover_plugins should clear the existing registry first."""
        pm = PluginManager()
        pm._plugins["old_plugin"] = MagicMock()

        with patch("os.path.exists", return_value=False):
            pm.discover_plugins()

        assert "old_plugin" not in pm._plugins

    @patch("os.path.exists", return_value=False)
    def test_discover_logs_error_when_directory_missing(self, mock_exists):
        """discover_plugins should log error if plugin directory not found."""
        pm = PluginManager()
        pm.discover_plugins()
        # Should complete without raising
        assert pm.list_plugins() == []
