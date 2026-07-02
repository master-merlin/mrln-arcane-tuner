"""Template CRUD endpoints (all three domains) broadcast entity.changed
with entity='template' (B-ARCH-4).

Mirrors test_template_routes.py's repo-mocking style (one shared repo class
per domain, patched wholesale) plus the house event-emission assertion
pattern from test_definition_routes_registry_emits.py /
test_project_entity_emits.py.
"""
from __future__ import annotations

from unittest.mock import patch

from app.api.training.template_routes import event_manager


_TRAIN_REPO = "app.core.db.repositories.training_template_repo.TrainingTemplateRepository"
_CAP_REPO = "app.core.db.repositories.captioning_template_repo.CaptioningTemplateRepository"
_MASK_REPO = "app.core.db.repositories.masking_template_repo.MaskingTemplateRepository"


def _entity_calls(mock_broadcast, *, op: str | None = None):
    calls = [
        c for c in mock_broadcast.call_args_list
        if c.args and c.args[0] == "entity.changed" and c.args[1]["entity"] == "template"
    ]
    if op is not None:
        calls = [c for c in calls if c.args[1]["op"] == op]
    return calls


@patch.object(event_manager, "broadcast")
@patch(_TRAIN_REPO)
def test_create_training_template_broadcasts_created(MockRepo, mock_broadcast, client):
    MockRepo.return_value.create.return_value = {
        "id": "new-id", "definition_id": "sdxl_base_1.0",
        "name": "My Template", "created_at": 0.0,
    }
    response = client.post("/api/templates/training", json={
        "definition_id": "sdxl_base_1.0", "name": "My Template", "config": {},
    })
    assert response.status_code == 201, response.text

    created = _entity_calls(mock_broadcast, op="created")
    assert len(created) == 1, created
    env = created[0].args[1]
    assert env["id"] == "new-id"
    assert env["payload"]["name"] == "My Template"


@patch.object(event_manager, "broadcast")
@patch(_CAP_REPO)
def test_create_captioning_template_broadcasts_created(MockRepo, mock_broadcast, client):
    MockRepo.return_value.create.return_value = {
        "id": "cap-1", "model_id": "m", "name": "Cap Tpl",
        "system_prompt": "Describe.", "wildcard": "", "config": {}, "created_at": 0.0,
    }
    response = client.post("/api/templates/captioning", json={
        "model_id": "m", "name": "Cap Tpl",
    })
    assert response.status_code == 201, response.text

    created = _entity_calls(mock_broadcast, op="created")
    assert len(created) == 1, created
    assert created[0].args[1]["id"] == "cap-1"


@patch.object(event_manager, "broadcast")
@patch(_MASK_REPO)
def test_create_masking_template_broadcasts_created(MockRepo, mock_broadcast, client):
    MockRepo.return_value.create.return_value = {
        "id": "mask-1", "model_id": "m", "name": "Mask Tpl", "config": {}, "created_at": 0.0,
    }
    response = client.post("/api/templates/masking", json={"model_id": "m", "name": "Mask Tpl"})
    assert response.status_code == 201, response.text

    created = _entity_calls(mock_broadcast, op="created")
    assert len(created) == 1, created
    assert created[0].args[1]["id"] == "mask-1"


@patch.object(event_manager, "broadcast")
@patch(_TRAIN_REPO)
def test_create_training_template_from_job_broadcasts_created(MockRepo, mock_broadcast, client):
    from unittest.mock import patch as _patch

    MockRepo.return_value.create_from_job.return_value = {
        "id": "from-job-1", "definition_id": "sdxl_base_1.0",
        "name": "From Job", "created_at": 0.0,
    }
    with _patch(
        "app.core.db.repositories.job_repo.JobHistoryRepository.get_by_id",
        return_value={"config": {}},
    ):
        response = client.post("/api/templates/training/from-job", json={
            "job_id": "job-1", "name": "From Job",
        })
    assert response.status_code == 201, response.text

    created = _entity_calls(mock_broadcast, op="created")
    assert len(created) == 1, created
    assert created[0].args[1]["id"] == "from-job-1"


@patch.object(event_manager, "broadcast")
@patch(_TRAIN_REPO)
def test_update_template_broadcasts_updated(MockRepo, mock_broadcast, client):
    MockRepo.return_value.get_by_id.return_value = {"id": "t1", "name": "Old", "readonly": False}
    MockRepo.return_value.update.return_value = {"id": "t1", "name": "Updated"}
    response = client.put("/api/templates/training/t1", json={"name": "Updated"})
    assert response.status_code == 200, response.text

    updated = _entity_calls(mock_broadcast, op="updated")
    assert len(updated) == 1, updated
    assert updated[0].args[1]["id"] == "t1"
    assert updated[0].args[1]["payload"]["name"] == "Updated"


@patch.object(event_manager, "broadcast")
@patch(_TRAIN_REPO)
def test_delete_template_broadcasts_deleted(MockRepo, mock_broadcast, client):
    MockRepo.return_value.get_by_id.return_value = {"id": "t1", "readonly": False}
    response = client.delete("/api/templates/training/t1")
    assert response.status_code == 200, response.text

    deleted = _entity_calls(mock_broadcast, op="deleted")
    assert len(deleted) == 1, deleted
    assert deleted[0].args[1]["id"] == "t1"
    assert deleted[0].args[1]["payload"] is None
