"""
E2E tests for api/training/template_routes.py — domain-scoped CRUD + usage counter.

Routes are organized per domain (captioning, masking, training):
    GET    /api/templates/{domain}           list
    POST   /api/templates/{domain}           create
    GET    /api/templates/{domain}/{id}      detail
    PUT    /api/templates/{domain}/{id}      update
    DELETE /api/templates/{domain}/{id}      delete
    POST   /api/templates/{domain}/{id}/use  bump usage
"""

from unittest.mock import patch


_TRAIN_REPO = "app.core.db.repositories.training_template_repo.TrainingTemplateRepository"
_CAP_REPO = "app.core.db.repositories.captioning_template_repo.CaptioningTemplateRepository"
_MASK_REPO = "app.core.db.repositories.masking_template_repo.MaskingTemplateRepository"


# ── List ─────────────────────────────────────────────────────────────────


@patch(_TRAIN_REPO)
def test_list_training_templates(MockRepo, client):
    # Realistic training-template row (definition_id/created_at are NOT-NULL
    # columns the response_model requires).
    MockRepo.return_value.list_for_project.return_value = [
        {"id": "t1", "definition_id": "sdxl_base_1.0", "name": "Default", "created_at": 0.0}
    ]
    response = client.get("/api/templates/training")
    assert response.status_code == 200
    assert len(response.json()) == 1


@patch(_CAP_REPO)
def test_list_captioning_templates(MockRepo, client):
    MockRepo.return_value.list_for_project.return_value = []
    response = client.get("/api/templates/captioning")
    assert response.status_code == 200
    assert response.json() == []


@patch(_MASK_REPO)
def test_list_masking_templates(MockRepo, client):
    MockRepo.return_value.list_for_project.return_value = []
    response = client.get("/api/templates/masking")
    assert response.status_code == 200
    assert response.json() == []


# ── Detail ───────────────────────────────────────────────────────────────


@patch(_TRAIN_REPO)
def test_get_template_found(MockRepo, client):
    MockRepo.return_value.get_by_id.return_value = {"id": "t1", "name": "Default"}
    response = client.get("/api/templates/training/t1")
    assert response.status_code == 200
    assert response.json()["id"] == "t1"


@patch(_TRAIN_REPO)
def test_get_template_not_found(MockRepo, client):
    MockRepo.return_value.get_by_id.return_value = None
    response = client.get("/api/templates/training/ghost")
    assert response.status_code == 404


# ── Create ───────────────────────────────────────────────────────────────


@patch(_TRAIN_REPO)
def test_create_training_template(MockRepo, client):
    MockRepo.return_value.create.return_value = {
        "id": "new-id", "definition_id": "sdxl_base_1.0",
        "name": "My Template", "created_at": 0.0,
    }
    response = client.post("/api/templates/training", json={
        "definition_id": "sdxl_base_1.0",
        "name": "My Template",
        "config": {"lr": 1e-4},
    })
    assert response.status_code == 201
    assert response.json()["id"] == "new-id"


# ── Update ───────────────────────────────────────────────────────────────


@patch(_TRAIN_REPO)
def test_update_template(MockRepo, client):
    MockRepo.return_value.get_by_id.return_value = {"id": "t1", "name": "Old", "readonly": False}
    MockRepo.return_value.update.return_value = {"id": "t1", "name": "Updated"}
    response = client.put("/api/templates/training/t1", json={"name": "Updated"})
    assert response.status_code == 200
    assert response.json()["name"] == "Updated"


@patch(_TRAIN_REPO)
def test_update_template_not_found(MockRepo, client):
    MockRepo.return_value.get_by_id.return_value = None
    response = client.put("/api/templates/training/ghost", json={"name": "x"})
    assert response.status_code == 404


@patch(_TRAIN_REPO)
def test_update_readonly_forbidden(MockRepo, client):
    MockRepo.return_value.get_by_id.return_value = {"id": "default", "readonly": True}
    response = client.put("/api/templates/training/default", json={"name": "x"})
    assert response.status_code == 403


# ── Delete ───────────────────────────────────────────────────────────────


@patch(_TRAIN_REPO)
def test_delete_template(MockRepo, client):
    MockRepo.return_value.get_by_id.return_value = {"id": "t1", "readonly": False}
    response = client.delete("/api/templates/training/t1")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"


@patch(_TRAIN_REPO)
def test_delete_template_not_found(MockRepo, client):
    MockRepo.return_value.get_by_id.return_value = None
    response = client.delete("/api/templates/training/ghost")
    assert response.status_code == 404


# ── Usage counter ────────────────────────────────────────────────────────


@patch(_TRAIN_REPO)
def test_use_template(MockRepo, client):
    response = client.post("/api/templates/training/t1/use")
    assert response.status_code == 200
    assert response.json()["status"] == "recorded"
