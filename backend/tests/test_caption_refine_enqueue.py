# backend/tests/test_caption_refine_enqueue.py
from unittest.mock import MagicMock, patch

_MOD = "app.api.caption_routes"


@patch(f"{_MOD}.task_manager")
def test_refine_batch_enqueues(mock_tm, client):
    task = MagicMock()
    task.id = "task-123"
    mock_tm.create.return_value = task
    resp = client.post("/api/captions/refine-batch", json={
        "dataset_name": "ds",
        "image_rel_paths": ["a.png", "b.png"],
        "definition_id": "flux1-schnell",
        "preset": "standardize",
    })
    assert resp.status_code == 200
    assert resp.json()["task_id"] == "task-123"
    mock_tm.create.assert_called_once()
    mock_tm.enqueue.assert_called_once()
    # enqueued on the background/cpu lane, not gpu
    _, kwargs = mock_tm.enqueue.call_args
    lane = kwargs.get("lane") if "lane" in kwargs else mock_tm.enqueue.call_args.args[-1]
    assert lane in ("background", "cpu")


@patch(f"{_MOD}.run_caption_refine_batch")
@patch(f"{_MOD}.task_manager")
def test_refine_batch_forwards_masked_target(mock_tm, mock_worker, client):
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
    assert resp.status_code == 200
    # the created task carries target="masked"
    create_kwargs = mock_tm.create.call_args.kwargs
    assert create_kwargs["target"] == "masked"
    # the enqueued worker is invoked with target="masked"
    worker_fn = mock_tm.enqueue.call_args.args[1]
    worker_fn("task-456")
    assert mock_worker.call_args.kwargs["target"] == "masked"
