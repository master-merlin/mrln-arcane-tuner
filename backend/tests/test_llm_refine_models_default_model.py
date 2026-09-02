# backend/tests/test_llm_refine_models_default_model.py
"""LANE-70 — ``GET /api/llm-refine/models`` judges the configured DEFAULT model.

The user, signing UAT round 7 (7.4): *"Refine is available the whole time
(not guarded properly)."* The detail sidebar's Refine sends no model, so
``POST /captions/refine-batch`` serves it with ``refine_settings.model_of``
(the configured default) and refuses when that model is not installed —
while the status the sidebar gated on never asked about that model, so the
button stayed enabled with a reason it could not know. The status now judges
the same model the model-less request is served with, so the sentence the
sidebar disables on IS the sentence the refusal would carry (RULE-21).

Real socket (``fake_ollama``), the same one the boundary guard uses.
Mutation that turns this red: probe without ``_default_model()`` in
``llm_refine_routes.list_models`` (the first test's reason comes back null).
"""

from __future__ import annotations

from app.core.llm import refine_settings

_DEFAULT = "qwen2.5:7b-instruct"


def _point(monkeypatch, base_url: str) -> None:
    monkeypatch.setattr(refine_settings, "raw_settings",
                        lambda: {"base_url": base_url, "model": _DEFAULT})


def test_status_names_the_missing_default_model_while_the_endpoint_is_up(
        client, monkeypatch, fake_ollama) -> None:
    fake_ollama.models[:] = ["gemma3:12b"]
    _point(monkeypatch, fake_ollama.url)

    body = client.get("/api/llm-refine/models").json()

    assert body["available"] is True            # the endpoint answered
    assert body["installed"] == ["gemma3:12b"]
    assert body["unavailable_reason"] and _DEFAULT in body["unavailable_reason"]
    assert fake_ollama.url in body["unavailable_reason"]


def test_status_reason_is_the_refusal_sentence_verbatim(
        client, monkeypatch, fake_ollama) -> None:
    """One producer: the sidebar's disabled text == the 409 the click would get."""
    fake_ollama.models[:] = ["gemma3:12b"]
    _point(monkeypatch, fake_ollama.url)
    from app.api import caption_routes
    monkeypatch.setattr(caption_routes, "run_caption_refine_batch", lambda *a, **kw: None)

    status_reason = client.get("/api/llm-refine/models").json()["unavailable_reason"]
    refusal = client.post("/api/captions/refine-batch", json={
        "dataset_name": "ds", "image_rel_paths": ["a.png"],
        "definition_id": "flux1-schnell", "preset": "standardize"})

    assert refusal.status_code == 409, refusal.text
    assert refusal.json()["detail"] == status_reason


def test_positive_control_installed_default_clears_the_reason(
        client, monkeypatch, fake_ollama) -> None:
    fake_ollama.models[:] = ["gemma3:12b", _DEFAULT]
    _point(monkeypatch, fake_ollama.url)

    body = client.get("/api/llm-refine/models").json()

    assert body["available"] is True
    assert body["unavailable_reason"] is None
