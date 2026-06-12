"""Tests for API captioning param schema + seeded default templates."""
import json
import sqlite3

from app.core.schemas.captioning_settings import CAPTION_PARAM_MODELS, ApiCaptionParams


def test_api_param_schema_registered_for_all_providers():
    for provider in ("openai", "anthropic", "gemini", "openrouter", "custom"):
        assert CAPTION_PARAM_MODELS[f"api-{provider}"] is ApiCaptionParams


def test_api_param_defaults():
    p = ApiCaptionParams()
    assert p.model == ""
    assert p.temperature == 0.7
    assert p.max_tokens == 512
    assert p.max_long_side == 1024


def test_migrate_v13_seeds_api_defaults(tmp_path):
    from app.core.db.migrations import _migrate_v13

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE captioning_templates (
            id TEXT PRIMARY KEY, project_id TEXT, model_id TEXT, name TEXT,
            is_default INTEGER, readonly INTEGER, system_prompt TEXT,
            config TEXT, created_at REAL, updated_at REAL,
            used_count INTEGER DEFAULT 0, last_used_at REAL, branched_from TEXT,
            wildcard TEXT DEFAULT ''
        )
    """)
    _migrate_v13(conn)
    rows = conn.execute(
        "SELECT * FROM captioning_templates WHERE model_id LIKE 'api-%'"
    ).fetchall()
    assert {r["model_id"] for r in rows} == {
        "api-openai", "api-anthropic", "api-gemini", "api-openrouter", "api-custom",
    }
    for r in rows:
        assert r["is_default"] == 1 and r["readonly"] == 1
        cfg = json.loads(r["config"])
        assert {"model", "temperature", "top_p", "max_tokens", "max_long_side"} <= set(cfg)
    # idempotent (INSERT OR IGNORE)
    _migrate_v13(conn)
    n = conn.execute(
        "SELECT COUNT(*) FROM captioning_templates WHERE model_id LIKE 'api-%'"
    ).fetchone()[0]
    assert n == 5
