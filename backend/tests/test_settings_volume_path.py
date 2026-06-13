"""SettingsManager storage location.

In the container, app settings (notably the HF token set via the Server UI)
must live on the persistent data volume, not in the ephemeral ``/app`` checkout
that a pod restart wipes. The entrypoint points ``MRLN_SETTINGS_PATH`` at the
volume; these tests pin that the env override is honoured (and that local dev,
with no env var, is unchanged).
"""
import json
import os

from app.core.settings_manager import SettingsManager


def test_env_override_directs_storage_to_the_given_path(tmp_path, monkeypatch):
    target = tmp_path / "workspace" / "settings.json"  # parent dir absent on purpose
    monkeypatch.setenv("MRLN_SETTINGS_PATH", str(target))

    sm = SettingsManager()
    sm.update_module_settings("models", {"hf_token": "secret-token"})

    assert sm.storage_file == str(target)
    assert target.exists()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["models"]["hf_token"] == "secret-token"


def test_env_override_creates_missing_parent_dirs(tmp_path, monkeypatch):
    target = tmp_path / "deep" / "nested" / "settings.json"
    monkeypatch.setenv("MRLN_SETTINGS_PATH", str(target))

    SettingsManager()  # constructor seeds defaults → writes the file

    assert target.exists()


def test_defaults_to_backend_path_without_env(monkeypatch):
    monkeypatch.delenv("MRLN_SETTINGS_PATH", raising=False)
    # Don't touch the real on-disk settings.json while probing the default path.
    monkeypatch.setattr(SettingsManager, "save", lambda self: None)

    sm = SettingsManager()

    assert sm.storage_file.replace("\\", "/").endswith("/backend/settings.json")
