"""LANE-49 (a): a PRESENT-but-EMPTY settings key is not a missing key.

``backend/settings.json`` held ``"llm_refine": {"model": ""}``. Every read was
``settings.get("model", DEFAULT)``, and ``dict.get`` returns the stored ``""``
because the key is PRESENT -- the default only applies when it is absent. So
caption refine POSTed ``{"model": ""}`` and Ollama answered
``400 {"error":{"message":"model is required"}}`` (``server.log``,
2026-08-31T17:37:17Z, ``caption_refine_batch.py:138``).

Every test below feeds the empty string specifically. Feeding a MISSING key
would pass against the broken code, which is the shape that let this ship.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.core.llm import refine_settings
from app.core.llm.refine_guard import RefineReadiness

# The exact stored dict, copied from the machine that reproduced the defect.
BROKEN_STORE = {"base_url": "http://localhost:11434", "provider": "ollama", "model": ""}


async def _endpoint_is_ready(client, model=None) -> RefineReadiness:
    """Stand-in for ``refine_guard.refine_readiness`` at the batch route.

    The route gates on a live probe of the LLM endpoint (LANE-57) before it
    resolves anything about the model. That probe is below the seam under test
    here -- what is asserted is the ``model`` keyword the route computed -- and
    on a CI runner with nothing on :11434 the real probe answers 409 before the
    resolution ever runs (gate.yml run 33687356291).
    """
    return RefineReadiness(base_url=BROKEN_STORE["base_url"], available=True, installed=[])


# --------------------------------------------------------------------------
# The accessor
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stored", [{}, {"model": ""}, {"model": "   "}, {"model": None}, BROKEN_STORE])
def test_model_never_resolves_empty(stored: dict) -> None:
    resolved = refine_settings.model_of(stored)
    assert resolved == refine_settings.DEFAULT_MODEL
    assert resolved.strip(), "an empty model is what Ollama refuses with 400"


def test_a_chosen_model_is_returned_verbatim() -> None:
    """The negative control: the accessor must not always answer the default."""
    assert refine_settings.model_of({"model": "gemma3:12b"}) == "gemma3:12b"
    assert refine_settings.model_of({"model": "  gemma3:12b  "}) == "gemma3:12b"


@pytest.mark.parametrize("stored", [{}, {"base_url": ""}, {"base_url": "  "}, {"base_url": None}])
def test_base_url_never_resolves_empty(stored: dict) -> None:
    """``base_url`` had the same ``.get(key, default)`` shape, so it gets the same pin.

    An empty base URL is not merely a 400: ``OllamaClient`` would build
    ``/v1/chat/completions`` -- a relative path httpx refuses outright.
    """
    assert refine_settings.base_url_of(stored) == refine_settings.DEFAULT_BASE_URL


def test_base_url_is_returned_in_the_server_root_convention() -> None:
    assert refine_settings.base_url_of({"base_url": "http://host:1234/v1"}) == "http://host:1234"
    assert refine_settings.base_url_of({"base_url": "http://host:1234/"}) == "http://host:1234"


# --------------------------------------------------------------------------
# The two consumers, at their routes -- the observable output
# --------------------------------------------------------------------------


def test_refine_preview_route_never_sends_an_empty_model(client) -> None:
    """Asserts the model handed to the LLM client, not the settings dict.

    ``refine_caption`` is the seam BELOW the resolution, not the seam under
    test, so stubbing it is legitimate: what is asserted is the argument the
    route computed.
    """
    with patch("app.api.llm_refine_routes._settings", return_value=BROKEN_STORE), \
            patch("app.api.llm_refine_routes._make_client"), \
            patch("app.api.llm_refine_routes.refine_caption",
                  new_callable=AsyncMock, return_value="refined") as refine:
        resp = client.post("/api/llm-refine/refine-preview",
                           json={"text": "a cat", "preset": "standardize"})

    assert resp.status_code == 200
    model_arg = refine.call_args.args[1]
    assert model_arg == refine_settings.DEFAULT_MODEL
    assert model_arg, "the route posted the empty model straight through"


def test_refine_preview_still_honours_an_explicit_model(client) -> None:
    with patch("app.api.llm_refine_routes._settings", return_value=BROKEN_STORE), \
            patch("app.api.llm_refine_routes._make_client"), \
            patch("app.api.llm_refine_routes.refine_caption",
                  new_callable=AsyncMock, return_value="refined") as refine:
        resp = client.post("/api/llm-refine/refine-preview",
                           json={"text": "a cat", "preset": "standardize", "model": "gemma3:12b"})

    assert resp.status_code == 200
    assert refine.call_args.args[1] == "gemma3:12b"


def test_refine_batch_route_never_enqueues_an_empty_model(client, tmp_path, monkeypatch) -> None:
    """The route in the traceback (``caption_routes.py``), asserted on the
    keyword the batch will actually run with."""
    import app.api.caption_routes as caption_routes

    captured: dict = {}

    def fake_run(tid, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(caption_routes, "run_caption_refine_batch", fake_run)
    monkeypatch.setattr(caption_routes, "refine_readiness", _endpoint_is_ready)
    monkeypatch.setattr(refine_settings, "raw_settings", lambda: BROKEN_STORE)

    class _Task:
        id = "t1"

    monkeypatch.setattr(caption_routes.task_manager, "create", lambda **kw: _Task())
    monkeypatch.setattr(caption_routes.task_manager, "enqueue",
                        lambda tid, fn, lane=None: fn(tid))

    resp = client.post("/api/captions/refine-batch", json={
        "dataset_name": "ds", "image_rel_paths": ["a.jpg"],
        "definition_id": "sdxl", "preset": "standardize",
    })

    assert resp.status_code == 200, resp.text
    assert captured["model"] == refine_settings.DEFAULT_MODEL
    assert captured["model"], "the batch would POST {'model': ''} -> Ollama 400"
    assert captured["base_url"] == "http://localhost:11434"


def test_refine_batch_still_honours_an_explicit_model(client, monkeypatch) -> None:
    """Negative control: the route must not always answer the default."""
    import app.api.caption_routes as caption_routes

    captured: dict = {}
    monkeypatch.setattr(caption_routes, "run_caption_refine_batch",
                        lambda tid, **kw: captured.update(kw))
    monkeypatch.setattr(caption_routes, "refine_readiness", _endpoint_is_ready)
    monkeypatch.setattr(refine_settings, "raw_settings", lambda: BROKEN_STORE)

    class _Task:
        id = "t1"

    monkeypatch.setattr(caption_routes.task_manager, "create", lambda **kw: _Task())
    monkeypatch.setattr(caption_routes.task_manager, "enqueue",
                        lambda tid, fn, lane=None: fn(tid))

    resp = client.post("/api/captions/refine-batch", json={
        "dataset_name": "ds", "image_rel_paths": ["a.jpg"],
        "definition_id": "sdxl", "preset": "standardize", "model": "gemma3:12b",
    })
    assert resp.status_code == 200, resp.text
    assert captured["model"] == "gemma3:12b"


def test_the_curated_default_is_the_first_curated_model() -> None:
    """The Server screen shows ``curated[0]`` by name as "what will be used when
    you choose nothing". If the fallback and that list ever disagree, the hint
    is a lie."""
    from app.api.llm_refine_routes import CURATED_MODELS

    assert refine_settings.DEFAULT_MODEL == CURATED_MODELS[0]
