"""
E2E tests for api/training/template_routes.py — full CRUD + usage counter.
"""

from unittest.mock import patch


_TPL_REPO = "app.core.db.repositories.template_repo.TemplateRepository"


@patch(_TPL_REPO)
def test_list_templates(MockRepo, client):
    MockRepo.return_value.list_by_category.return_value = [{"id": "t1", "name": "Default"}]
    response = client.get("/api/templates?category=training")
    assert response.status_code == 200
    assert len(response.json()) == 1


@patch(_TPL_REPO)
def test_get_template_found(MockRepo, client):
    MockRepo.return_value.get_by_id.return_value = {"id": "t1", "name": "Default"}
    response = client.get("/api/templates/t1")
    assert response.status_code == 200
    assert response.json()["id"] == "t1"


@patch(_TPL_REPO)
def test_get_template_not_found(MockRepo, client):
    MockRepo.return_value.get_by_id.return_value = None
    response = client.get("/api/templates/ghost")
    assert response.status_code == 404


@patch(_TPL_REPO)
def test_create_template(MockRepo, client):
    MockRepo.return_value.create.return_value = {"id": "new-id"}
    response = client.post("/api/templates", json={
        "name": "My Template",
        "category": "training",
        "config": {"lr": 1e-4},
    })
    assert response.status_code == 200


@patch(_TPL_REPO)
def test_update_template(MockRepo, client):
    MockRepo.return_value.get_by_id.return_value = {"id": "t1", "name": "Updated"}
    response = client.put("/api/templates/t1", json={
        "name": "Updated",
    })
    assert response.status_code == 200


@patch(_TPL_REPO)
def test_delete_template(MockRepo, client):
    response = client.delete("/api/templates/t1")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"


@patch(_TPL_REPO)
def test_use_template(MockRepo, client):
    response = client.post("/api/templates/t1/use")
    assert response.status_code == 200
    assert response.json()["status"] == "recorded"
