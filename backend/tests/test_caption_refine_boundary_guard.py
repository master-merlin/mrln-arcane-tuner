# backend/tests/test_caption_refine_boundary_guard.py
"""LANE-57 — a refine that cannot succeed is REFUSED at the request boundary.

The user, signing UAT round 5: *"even so the refine task started without any
API Key (not guarded)"* and *"it can be used on unconfigured endpoints."*
``POST /api/captions/refine-batch`` enqueued first and let the worker discover
that the endpoint was dead or the model absent (``caption_routes.py:311-338``
before this lane). The guard belongs at the boundary: refuse with a message
that names what is missing, and enqueue NOTHING.

The endpoint under test is a REAL socket (``fake_ollama`` in conftest, or a
port nobody listens on) — the predicate is exercised end to end, not stubbed.
The worker is replaced because it is downstream of the seam under test; the
``task_manager`` is the real one and its ``list()`` is the observable.
"""

from __future__ import annotations

from app.api import caption_routes
from app.core.llm import refine_settings
from app.core.tasks.task_manager import task_manager

_BODY = {
    "dataset_name": "ds",
    "image_rel_paths": ["a.png"],
    "definition_id": "flux1-schnell",
    "preset": "standardize",
}


def _task_ids() -> set[str]:
    return {t.id for t in task_manager.list()}


def _no_worker(monkeypatch) -> None:
    monkeypatch.setattr(caption_routes, "run_caption_refine_batch",
                        lambda *a, **kw: None)


def test_refuses_an_unreachable_endpoint_and_enqueues_nothing(
        client, monkeypatch, closed_port) -> None:
    """(b) an endpoint that was never validated: nobody listens there."""
    _no_worker(monkeypatch)
    dead = f"http://127.0.0.1:{closed_port}"
    monkeypatch.setattr(refine_settings, "raw_settings",
                        lambda: {"base_url": dead, "model": "qwen2.5:7b-instruct"})
    before = _task_ids()

    resp = client.post("/api/captions/refine-batch", json=_BODY)

    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert dead in detail, detail                      # names the endpoint
    assert "Server" in detail and "unreachable" in detail, detail
    assert _task_ids() == before, "a refused refine must enqueue nothing"


def test_refuses_a_model_the_endpoint_does_not_have(
        client, monkeypatch, fake_ollama) -> None:
    """The user's own 5.1 scenario: a model that is not installed."""
    _no_worker(monkeypatch)
    fake_ollama.models[:] = ["llama3.1:8b-instruct-q4_K_M"]
    monkeypatch.setattr(refine_settings, "raw_settings",
                        lambda: {"base_url": fake_ollama.url, "model": "qwen2.5:7b-instruct"})
    before = _task_ids()

    resp = client.post("/api/captions/refine-batch", json=_BODY)

    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert "qwen2.5:7b-instruct" in detail and "not installed" in detail, detail
    assert fake_ollama.url in detail, detail
    assert _task_ids() == before


def test_the_request_model_is_the_one_checked(client, monkeypatch, fake_ollama) -> None:
    """An explicit ``model`` in the body overrides the stored default — the
    guard must check THAT one, or it refuses a working request (and vice versa)."""
    _no_worker(monkeypatch)
    fake_ollama.models[:] = ["qwen2.5:7b-instruct"]
    monkeypatch.setattr(refine_settings, "raw_settings",
                        lambda: {"base_url": fake_ollama.url, "model": "qwen2.5:7b-instruct"})
    before = _task_ids()

    resp = client.post("/api/captions/refine-batch", json={**_BODY, "model": "gemma3:12b"})

    assert resp.status_code == 409, resp.text
    assert "gemma3:12b" in resp.json()["detail"]
    assert _task_ids() == before


def test_positive_control_configured_and_reachable_is_accepted(
        client, monkeypatch, fake_ollama) -> None:
    """(c) configured + validated -> accepted and enqueued exactly once."""
    _no_worker(monkeypatch)
    fake_ollama.models[:] = ["qwen2.5:7b-instruct"]
    monkeypatch.setattr(refine_settings, "raw_settings",
                        lambda: {"base_url": fake_ollama.url, "model": "qwen2.5:7b-instruct"})
    before = _task_ids()

    resp = client.post("/api/captions/refine-batch", json=_BODY)

    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]
    assert _task_ids() - before == {task_id}


def test_an_untagged_name_matches_its_latest_tag(client, monkeypatch, fake_ollama) -> None:
    """Ollama resolves ``name`` to ``name:latest`` on the wire; a user who typed
    the untagged form must not be refused for a model that IS installed."""
    _no_worker(monkeypatch)
    fake_ollama.models[:] = ["gemma3:latest"]
    monkeypatch.setattr(refine_settings, "raw_settings",
                        lambda: {"base_url": fake_ollama.url, "model": "gemma3"})

    resp = client.post("/api/captions/refine-batch", json=_BODY)

    assert resp.status_code == 200, resp.text


def test_models_status_carries_the_same_reason_the_refusal_uses(
        client, monkeypatch, closed_port) -> None:
    """RULE-21: the UI disables Start off ``GET /api/llm-refine/models`` — its
    ``unavailable_reason`` must be the very string the 409 carries, so the
    button and the refusal cannot say two different things."""
    dead = f"http://127.0.0.1:{closed_port}"
    monkeypatch.setattr(refine_settings, "raw_settings", lambda: {"base_url": dead})
    monkeypatch.setattr(caption_routes, "run_caption_refine_batch", lambda *a, **kw: None)

    status = client.get("/api/llm-refine/models").json()
    refusal = client.post("/api/captions/refine-batch", json=_BODY)

    assert status["available"] is False
    assert refusal.status_code == 409
    assert status["unavailable_reason"] == refusal.json()["detail"]


def test_models_status_reason_is_null_when_reachable(client, monkeypatch, fake_ollama) -> None:
    monkeypatch.setattr(refine_settings, "raw_settings", lambda: {"base_url": fake_ollama.url})
    status = client.get("/api/llm-refine/models").json()
    assert status["available"] is True
    assert status["unavailable_reason"] is None
