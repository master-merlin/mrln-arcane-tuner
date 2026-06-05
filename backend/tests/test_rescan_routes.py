import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dataset import crud_routes
from app.core.tasks.task_manager import task_manager


def _client():
    app = FastAPI()
    app.include_router(crud_routes.router, prefix="/api")
    return TestClient(app)


def test_scan_batch_returns_task_id(monkeypatch):
    task_manager.set_loop(None)
    monkeypatch.setattr(crud_routes, "count_multimedia", lambda names: 3)
    monkeypatch.setattr(crud_routes.task_manager, "enqueue", lambda *a, **k: None)
    monkeypatch.setattr(crud_routes.dataset_manager, "get_dataset",
                        lambda name: types.SimpleNamespace(path="/x"))

    res = _client().post("/api/datasets/ds1/scan/batch?force_full=false")
    assert res.status_code == 200
    assert "task_id" in res.json()


def test_scan_batch_404_when_missing(monkeypatch):
    task_manager.set_loop(None)
    monkeypatch.setattr(crud_routes.dataset_manager, "get_dataset", lambda name: None)

    res = _client().post("/api/datasets/ghost/scan/batch")
    assert res.status_code == 404


def test_scan_all_batch_returns_task_id(monkeypatch):
    task_manager.set_loop(None)
    monkeypatch.setattr(crud_routes.dataset_manager, "discover_and_list_dataset_names",
                        lambda: ["a", "b"])
    monkeypatch.setattr(crud_routes, "count_multimedia", lambda names: 7)
    monkeypatch.setattr(crud_routes.task_manager, "enqueue", lambda *a, **k: None)

    res = _client().post("/api/datasets/scan-all/batch?force_full=true")
    assert res.status_code == 200
    assert "task_id" in res.json()
