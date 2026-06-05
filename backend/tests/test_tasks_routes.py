from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.tasks_routes import router
from app.core.tasks.task_manager import task_manager


def _client():
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_list_tasks():
    task_manager.set_loop(None)
    t = task_manager.create(type="caption_batch", title="x", total=2)
    client = _client()
    res = client.get("/api/tasks")
    assert res.status_code == 200
    ids = [row["id"] for row in res.json()]
    assert t.id in ids


def test_cancel_running_task():
    task_manager.set_loop(None)
    t = task_manager.create(type="caption_batch", title="x", total=2)
    task_manager.start(t.id)
    client = _client()
    res = client.post(f"/api/tasks/{t.id}/cancel")
    assert res.status_code == 200
    assert task_manager.is_cancelled(t.id) is True


def test_cancel_unknown_404():
    client = _client()
    res = client.post("/api/tasks/does-not-exist/cancel")
    assert res.status_code == 404
