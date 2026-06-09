"""Route tests for the masked HF token in /api/models/settings.

The raw token must never appear in any response; GET exposes only the
``hf_token_set`` flag; PUT persists/clears it and applies HF auth.
"""
import os

import pytest

import app.core.hf_auth as hf_auth
from app.core.schemas.model_overrides import ModelSettings
from app.engine.utils.model_override_manager import ModelOverrideManager


@pytest.fixture
def mem_settings(monkeypatch):
    """In-memory model settings so the test never writes real settings.json."""
    state = {"s": ModelSettings()}

    async def fake_get_all_async():
        return state["s"]

    async def fake_save_async(s):
        state["s"] = s

    monkeypatch.setattr(
        ModelOverrideManager, "get_all_async", staticmethod(fake_get_all_async)
    )
    monkeypatch.setattr(
        ModelOverrideManager, "_save_async", staticmethod(fake_save_async)
    )
    # No external env token in these tests → settings token is effective.
    monkeypatch.setattr(hf_auth, "_EXTERNAL_HF_TOKEN", "")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    return state


def test_put_sets_token_masked_and_applies_env(client, mem_settings):
    r = client.put("/api/models/settings", json={"hf_token": "hf_secret123"})
    assert r.status_code == 200
    body = r.json()
    assert body["hf_token_set"] is True
    assert "hf_token" not in body  # masked: no raw field
    assert "hf_secret123" not in r.text  # raw value never echoed
    assert mem_settings["s"].hf_token == "hf_secret123"  # persisted
    assert os.environ["HF_TOKEN"] == "hf_secret123"  # applied


def test_get_masks_token(client, mem_settings):
    mem_settings["s"].hf_token = "hf_secret123"
    r = client.get("/api/models/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["hf_token_set"] is True
    assert "hf_secret123" not in r.text


def test_put_clears_token(client, mem_settings):
    mem_settings["s"].hf_token = "hf_old"
    os.environ["HF_TOKEN"] = "hf_old"
    r = client.put("/api/models/settings", json={"hf_token": ""})
    assert r.json()["hf_token_set"] is False
    assert mem_settings["s"].hf_token == ""
    assert "HF_TOKEN" not in os.environ


def test_put_other_field_leaves_token_untouched(client, mem_settings):
    mem_settings["s"].hf_token = "hf_keep"
    r = client.put("/api/models/settings", json={"global_offline_mode": True})
    assert r.json()["hf_token_set"] is True
    assert mem_settings["s"].hf_token == "hf_keep"
