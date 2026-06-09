"""Tests for Hugging Face auth application.

Precedence: an external env token (e.g. a RunPod pod's HF_TOKEN) always wins;
the in-app settings token is the fallback used only when no env token exists.
"""
import os

import app.core.hf_auth as hf_auth


def test_settings_token_used_when_no_external(monkeypatch):
    monkeypatch.setattr(hf_auth, "_EXTERNAL_HF_TOKEN", "")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    eff = hf_auth.apply_hf_auth("hf_settings")
    assert eff == "hf_settings"
    assert os.environ["HF_TOKEN"] == "hf_settings"
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_settings"


def test_external_env_token_wins(monkeypatch):
    monkeypatch.setattr(hf_auth, "_EXTERNAL_HF_TOKEN", "hf_env")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    eff = hf_auth.apply_hf_auth("hf_settings")
    assert eff == "hf_env"
    assert os.environ["HF_TOKEN"] == "hf_env"


def test_cleared_when_neither_present(monkeypatch):
    monkeypatch.setattr(hf_auth, "_EXTERNAL_HF_TOKEN", "")
    monkeypatch.setenv("HF_TOKEN", "stale")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "stale")
    eff = hf_auth.apply_hf_auth("")
    assert eff == ""
    assert "HF_TOKEN" not in os.environ
    assert "HUGGING_FACE_HUB_TOKEN" not in os.environ


def test_whitespace_settings_token_is_empty(monkeypatch):
    monkeypatch.setattr(hf_auth, "_EXTERNAL_HF_TOKEN", "")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert hf_auth.apply_hf_auth("   ") == ""
    assert "HF_TOKEN" not in os.environ


def test_none_settings_token_is_safe(monkeypatch):
    monkeypatch.setattr(hf_auth, "_EXTERNAL_HF_TOKEN", "")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert hf_auth.apply_hf_auth(None) == ""
