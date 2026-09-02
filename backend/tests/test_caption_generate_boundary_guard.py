# backend/tests/test_caption_generate_boundary_guard.py
"""LANE-65 — an api-* caption batch that cannot succeed is REFUSED at the boundary.

The user, signing UAT round 6 (6.3): *"Refine is properly gated, Generate
still clickable."* LANE-57 closed this class for the Refine tab; the Generate
tab's ``POST /api/captions/batch`` still enqueued an api-* batch against a
provider whose endpoint was dead or whose model it did not list, and let the
worker fail honestly (LANE-52). Same class, same cure, same seams: ONE
readiness producer (``core/llm/refine_guard`` — RULE-21), refused at the
request boundary with a sentence that names what is missing, surfaced
verbatim on ``GET /api/captions/api-providers/{provider}/readiness`` for the
CTA to disable off.

The provider endpoint under test is a REAL socket (``fake_ollama`` serving the
OpenAI-compatible ``/v1/models``, or a port nobody listens on) — the
predicate is exercised end to end, never stubbed. The worker is replaced
because it is downstream of the seam; ``task_manager.list()`` is the
observable. Local captioning is the negative space: it must make NO probe.
"""

from __future__ import annotations

import pytest

from app.api import caption_routes
from app.core.llm import provider_settings
from app.core.tasks.task_manager import task_manager


class _FakeSettingsManager:
    def __init__(self, modules: dict[str, dict]) -> None:
        self.modules = modules

    def get_module_settings(self, module):
        return self.modules.get(module, {})

    def update_module_settings(self, module, settings):
        self.modules.setdefault(module, {}).update(settings)


def _task_ids() -> set[str]:
    return {t.id for t in task_manager.list()}


def _body(model_id: str, model: str | None = "llava:13b") -> dict:
    return {
        "dataset_name": "ds",
        "image_rel_paths": ["a.png"],
        "model_id": model_id,
        "params": {"model": model} if model is not None else {},
    }


@pytest.fixture
def custom_provider(monkeypatch):
    """Point the ``custom`` captioning provider at *base_url* (no key needed)."""
    def _point(base_url: str) -> None:
        mgr = _FakeSettingsManager(
            {provider_settings.MODULE: {"providers": {"custom": {"base_url": base_url}}}})
        monkeypatch.setattr(provider_settings, "_manager", lambda: mgr)
    return _point


@pytest.fixture
def no_worker(monkeypatch):
    monkeypatch.setattr(caption_routes, "run_caption_batch", lambda *a, **kw: None)


def test_refuses_an_unreachable_provider_and_enqueues_nothing(
        client, no_worker, custom_provider, closed_port) -> None:
    """(a) the Generate tab against a port nobody listens on -> 409 naming the
    endpoint, and NOTHING in the task list."""
    dead = f"http://127.0.0.1:{closed_port}"
    custom_provider(dead)
    before = _task_ids()

    resp = client.post("/api/captions/batch", json=_body("api-custom"))

    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert dead in detail and "unreachable" in detail, detail
    assert "captioning API settings" in detail, detail   # where to fix it, not the Server screen
    assert _task_ids() == before, "a refused caption batch must enqueue nothing"


def test_refuses_a_model_the_provider_does_not_list(
        client, no_worker, custom_provider, fake_ollama) -> None:
    fake_ollama.models[:] = ["llama3.2-vision:11b"]
    custom_provider(fake_ollama.url)
    before = _task_ids()

    resp = client.post("/api/captions/batch", json=_body("api-custom", "llava:13b"))

    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert "llava:13b" in detail and "not installed" in detail, detail
    assert _task_ids() == before


def test_positive_control_reachable_with_the_model_is_accepted(
        client, no_worker, custom_provider, fake_ollama) -> None:
    """(b) reachable + model listed -> accepted, enqueued exactly once, and the
    probe went to the OpenAI-compatible listing the provider actually uses."""
    fake_ollama.models[:] = ["llava:13b"]
    custom_provider(fake_ollama.url)
    before = _task_ids()

    resp = client.post("/api/captions/batch", json=_body("api-custom", "llava:13b"))

    assert resp.status_code == 200, resp.text
    assert _task_ids() - before == {resp.json()["task_id"]}
    assert "/v1/models" in fake_ollama.hits, fake_ollama.hits


def test_an_untagged_name_matches_its_latest_tag(
        client, no_worker, custom_provider, fake_ollama) -> None:
    fake_ollama.models[:] = ["llava:latest"]
    custom_provider(fake_ollama.url)
    resp = client.post("/api/captions/batch", json=_body("api-custom", "llava"))
    assert resp.status_code == 200, resp.text


def test_local_captioning_makes_no_probe_and_still_starts(
        client, no_worker, custom_provider, fake_ollama) -> None:
    """Local captioning (the default tab) is NOT touched: a local model id goes
    through untouched even when the api-* provider would be refused."""
    custom_provider(fake_ollama.url)          # reachable, but lists nothing
    before = _task_ids()

    resp = client.post("/api/captions/batch", json=_body("florence-2", "llava:13b"))

    assert resp.status_code == 200, resp.text
    assert _task_ids() - before == {resp.json()["task_id"]}
    assert fake_ollama.hits == [], "local captioning must not dial the api-* provider"


def test_readiness_status_carries_the_same_reason_the_refusal_uses(
        client, no_worker, custom_provider, closed_port) -> None:
    """RULE-21: the Generate CTA disables off the readiness route — its
    ``unavailable_reason`` must be the very string the 409 carries."""
    dead = f"http://127.0.0.1:{closed_port}"
    custom_provider(dead)

    status = client.get("/api/captions/api-providers/custom/readiness",
                        params={"model": "llava:13b"}).json()
    refusal = client.post("/api/captions/batch", json=_body("api-custom", "llava:13b"))

    assert status["available"] is False
    assert refusal.status_code == 409
    assert status["unavailable_reason"] == refusal.json()["detail"]


def test_readiness_status_names_the_missing_model_and_clears_when_present(
        client, custom_provider, fake_ollama) -> None:
    fake_ollama.models[:] = ["llama3.2-vision:11b"]
    custom_provider(fake_ollama.url)

    absent = client.get("/api/captions/api-providers/custom/readiness",
                        params={"model": "llava:13b"}).json()
    present = client.get("/api/captions/api-providers/custom/readiness",
                         params={"model": "llama3.2-vision:11b"}).json()

    assert absent["available"] is False and "llava:13b" in absent["unavailable_reason"]
    assert present["available"] is True and present["unavailable_reason"] is None


def test_readiness_status_reports_a_key_less_hosted_provider_as_unconfigured(
        client, custom_provider, fake_ollama) -> None:
    """A hosted provider without a key never dials out: the resolver's own
    sentence (``provider_settings.resolve_provider``) is the reason."""
    custom_provider(fake_ollama.url)
    status = client.get("/api/captions/api-providers/openai/readiness").json()
    assert status["available"] is False
    assert "No API key configured for provider 'openai'" in status["unavailable_reason"]
    assert fake_ollama.hits == []
