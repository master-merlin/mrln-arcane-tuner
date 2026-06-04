"""The DB engine should honor MRLN_DB_PATH when no explicit path is given."""
from app.core.db.engine import DatabaseEngine


def test_db_path_env_override(monkeypatch, tmp_path):
    target = tmp_path / "custom_arcane.db"
    monkeypatch.setenv("MRLN_DB_PATH", str(target))
    # Construct directly (not the singleton) so we read the env at __init__.
    eng = DatabaseEngine()
    assert eng.db_path == str(target)


def test_db_path_default_when_env_absent(monkeypatch):
    monkeypatch.delenv("MRLN_DB_PATH", raising=False)
    eng = DatabaseEngine()
    assert eng.db_path.endswith("arcane_tuner.db")
