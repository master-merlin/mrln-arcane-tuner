import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import masking_routes
from app.core.tasks.task_manager import task_manager


def _client():
    app = FastAPI()
    app.include_router(masking_routes.router, prefix="/api")
    return TestClient(app)


def test_generate_batch_returns_task_id(monkeypatch):
    task_manager.set_loop(None)
    monkeypatch.setattr(masking_routes.task_manager, "enqueue", lambda *a, **k: None)
    monkeypatch.setattr(masking_routes.dataset_manager, "get_dataset",
                        lambda name: types.SimpleNamespace(path="/x"))

    res = _client().post("/api/datasets/ds1/masking/generate/batch", json={
        "image_rel_paths": ["a.png", "b.png"], "model_id": "rembg", "params": {},
    })
    assert res.status_code == 200
    assert "task_id" in res.json()


def test_generate_batch_404_when_missing(monkeypatch):
    task_manager.set_loop(None)
    monkeypatch.setattr(masking_routes.dataset_manager, "get_dataset", lambda name: None)

    res = _client().post("/api/datasets/ghost/masking/generate/batch", json={
        "image_rel_paths": ["a.png"], "model_id": "rembg", "params": {},
    })
    assert res.status_code == 404


def test_apply_batch_returns_task_id(monkeypatch, tmp_path):
    task_manager.set_loop(None)
    (tmp_path / "masks").mkdir()
    (tmp_path / "masks" / "a.png").write_bytes(b"x")
    monkeypatch.setattr(masking_routes.task_manager, "enqueue", lambda *a, **k: None)
    monkeypatch.setattr(masking_routes.dataset_manager, "get_dataset",
                        lambda name: types.SimpleNamespace(path=str(tmp_path)))

    res = _client().post("/api/datasets/ds1/masking/apply/batch?opacity=0.2&overwrite=true")
    assert res.status_code == 200
    assert "task_id" in res.json()


def test_apply_batch_404_when_missing(monkeypatch):
    task_manager.set_loop(None)
    monkeypatch.setattr(masking_routes.dataset_manager, "get_dataset", lambda name: None)

    res = _client().post("/api/datasets/ghost/masking/apply/batch")
    assert res.status_code == 404
