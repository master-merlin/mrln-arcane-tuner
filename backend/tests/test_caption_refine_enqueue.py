# backend/tests/test_caption_refine_enqueue.py
"""The refine-batch route ENQUEUES; endpoint readiness is a gate in front of it.

The seam under test is the enqueue. The route first asks
``refine_guard.refine_readiness`` (LANE-57) whether the configured LLM endpoint
can serve the batch, and answers 409 without creating anything when it cannot.
That producer is patched here, at the route's import site, because it probes a
live process: on a developer box with Ollama on :11434 it answers "ready" and
these tests pass by accident; on a CI runner with nothing listening it answers
"unreachable" and they fail with ``409 != 200`` (gate.yml run 33687356291).
``test_refine_batch_refuses_when_the_endpoint_is_unreachable`` is the negative
control that proves the gate is real and that nothing here depends on a socket.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from app.core.llm.refine_guard import RefineReadiness, unreachable_reason

_MOD = "app.api.caption_routes"
_BASE_URL = "http://localhost:11434"

READY = RefineReadiness(base_url=_BASE_URL, available=True, installed=["gemma3:12b"])
UNREACHABLE = RefineReadiness(
    base_url=_BASE_URL, available=False, reason=unreachable_reason(_BASE_URL)
)


@patch(f"{_MOD}.refine_readiness", new_callable=AsyncMock, return_value=READY)
@patch(f"{_MOD}.task_manager")
def test_refine_batch_enqueues(mock_tm, _ready, client):
    task = MagicMock()
    task.id = "task-123"
    mock_tm.create.return_value = task
    resp = client.post("/api/captions/refine-batch", json={
        "dataset_name": "ds",
        "image_rel_paths": ["a.png", "b.png"],
        "definition_id": "flux1-schnell",
        "preset": "standardize",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["task_id"] == "task-123"
    mock_tm.create.assert_called_once()
    mock_tm.enqueue.assert_called_once()
    # enqueued on the background/cpu lane, not gpu
    _, kwargs = mock_tm.enqueue.call_args
    lane = kwargs.get("lane") if "lane" in kwargs else mock_tm.enqueue.call_args.args[-1]
    assert lane in ("background", "cpu")


@patch(f"{_MOD}.refine_readiness", new_callable=AsyncMock, return_value=READY)
@patch(f"{_MOD}.run_caption_refine_batch")
@patch(f"{_MOD}.task_manager")
def test_refine_batch_forwards_masked_target(mock_tm, mock_worker, _ready, client):
    task = MagicMock()
    task.id = "task-456"
    mock_tm.create.return_value = task
    resp = client.post("/api/captions/refine-batch", json={
        "dataset_name": "ds",
        "image_rel_paths": ["a.png"],
        "definition_id": "flux1-schnell",
        "preset": "standardize",
        "target": "masked",
    })
    assert resp.status_code == 200, resp.text
    # the created task carries target="masked"
    create_kwargs = mock_tm.create.call_args.kwargs
    assert create_kwargs["target"] == "masked"
    # the enqueued worker is invoked with target="masked"
    worker_fn = mock_tm.enqueue.call_args.args[1]
    worker_fn("task-456")
    assert mock_worker.call_args.kwargs["target"] == "masked"


@patch(f"{_MOD}.refine_readiness", new_callable=AsyncMock, return_value=UNREACHABLE)
@patch(f"{_MOD}.task_manager")
def test_refine_batch_refuses_when_the_endpoint_is_unreachable(mock_tm, ready, client):
    """Negative control: the gate is real, and it is the producer that decides.

    With the SAME request the tests above enqueue, an "unreachable" verdict must
    answer 409 with the producer's sentence and create/enqueue nothing. Were the
    route still probing the socket itself, this test would flip with the
    machine — a live Ollama would turn the 409 into a 200.
    """
    resp = client.post("/api/captions/refine-batch", json={
        "dataset_name": "ds",
        "image_rel_paths": ["a.png"],
        "definition_id": "flux1-schnell",
        "preset": "standardize",
    })
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == UNREACHABLE.reason
    assert "unreachable" in resp.json()["detail"]
    ready.assert_awaited_once()
    mock_tm.create.assert_not_called()
    mock_tm.enqueue.assert_not_called()
