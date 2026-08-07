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

import pytest

from app.core.db.engine import DatabaseEngine
from app.core.db.migrations import run_migrations
from app.core.db.repositories.adaptive_preset_repo import AdaptivePresetRepository
from app.core.db.repositories.captioning_template_repo import CaptioningTemplateRepository
from app.core.db.repositories.masking_template_repo import MaskingTemplateRepository
from app.core.db.repositories.training_template_repo import TrainingTemplateRepository


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


@patch(_CAP_REPO)
def test_captioning_template_response_includes_wildcard(MockRepo, client):
    """Regression: wildcard must survive the response model. Without the field
    on CaptioningTemplate, FastAPI strips it and a saved wildcard never reads
    back — so the UI looks like it didn't autosave (system_prompt did)."""
    MockRepo.return_value.list_for_project.return_value = [{
        "id": "c1", "model_id": "m", "name": "T",
        "system_prompt": "Describe {wildcard}.", "wildcard": "Alice",
        "config": {}, "created_at": 0.0,
    }]
    response = client.get("/api/templates/captioning")
    assert response.status_code == 200
    body = response.json()
    assert body[0]["wildcard"] == "Alice"
    assert body[0]["system_prompt"] == "Describe {wildcard}."


@patch(_MASK_REPO)
def test_list_masking_templates(MockRepo, client):
    MockRepo.return_value.list_for_project.return_value = []
    response = client.get("/api/templates/masking")
    assert response.status_code == 200
    assert response.json() == []


# ── Detail ───────────────────────────────────────────────────────────────


@patch(_TRAIN_REPO)
def test_get_template_found(MockRepo, client):
    # created_at is a NOT-NULL column the TemplateRow response_model requires.
    MockRepo.return_value.get_by_id.return_value = {
        "id": "t1", "definition_id": "sdxl_base_1.0", "name": "Default", "created_at": 0.0,
    }
    response = client.get("/api/templates/training/t1")
    assert response.status_code == 200
    assert response.json()["id"] == "t1"


@patch(_TRAIN_REPO)
def test_get_template_full_payload(MockRepo, client):
    """P3c pin: TemplateRow (open, extra=allow) must not strip a domain-
    specific field (definition_id) it doesn't declare as a named field."""
    row = {
        "id": "t1", "project_id": None, "definition_id": "sdxl_base_1.0",
        "name": "Default", "is_default": True, "readonly": False,
        "config": {"lr": 1e-4}, "created_at": 0.0, "updated_at": None,
        "used_count": 3, "last_used_at": 5.0, "branched_from": None,
    }
    MockRepo.return_value.get_by_id.return_value = row
    response = client.get("/api/templates/training/t1")
    assert response.status_code == 200
    assert response.json() == row


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
    MockRepo.return_value.get_by_id.return_value = {
        "id": "t1", "definition_id": "sdxl_base_1.0", "name": "Old",
        "readonly": False, "created_at": 0.0,
    }
    MockRepo.return_value.update.return_value = {
        "id": "t1", "definition_id": "sdxl_base_1.0", "name": "Updated", "created_at": 0.0,
    }
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


# ── Branch ───────────────────────────────────────────────────────────────


@patch(_TRAIN_REPO)
def test_branch_template_full_payload(MockRepo, client):
    """P3c pin: branch returns the new (branched) row via TemplateRow."""
    branched = {
        "id": "t2", "project_id": "proj-1", "definition_id": "sdxl_base_1.0",
        "name": "Default (branched)", "is_default": False, "readonly": False,
        "config": {}, "created_at": 1.0, "updated_at": None,
        "used_count": 0, "last_used_at": None, "branched_from": "t1",
    }
    MockRepo.return_value.branch.return_value = branched
    response = client.post(
        "/api/templates/training/t1/branch", json={"target_project_id": "proj-1"}
    )
    assert response.status_code == 200
    assert response.json() == branched


@patch(_TRAIN_REPO)
def test_branch_template_not_found(MockRepo, client):
    MockRepo.return_value.branch.side_effect = ValueError("not found")
    response = client.post(
        "/api/templates/training/ghost/branch", json={"target_project_id": "proj-1"}
    )
    assert response.status_code == 404


# ── Import: plan + apply ────────────────────────────────────────────────


def test_plan_template_import_full_payload(client):
    """P3c pin: TemplateImportPlanResponse — entries stay open (extra=allow)
    so domain-specific plan fields (model_id/model_available for
    captioning/masking) survive alongside the common fields."""
    from app.core.portable.archive import write_manifest_zip
    from app.core.template import portable

    entry = {
        "domain": "captioning", "name": "My Template", "model_id": "florence-2",
        "config": {}, "system_prompt": "Describe.", "wildcard": "",
    }
    manifest = portable.build_template_manifest([entry], "0.0.0-test")
    zb = write_manifest_zip(manifest).getvalue()
    with patch(
        "app.core.template.import_service.model_available", return_value=True
    ), patch(
        "app.core.template.import_service.validate_config", return_value=None
    ):
        response = client.post(
            "/api/templates/import/plan",
            files={"file": ("t.zip", zb, "application/zip")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] is None
    assert body["importable_count"] == 1
    assert len(body["entries"]) == 1
    entry = body["entries"][0]
    assert entry["index"] == 0
    assert entry["domain"] == "captioning"
    assert entry["name"] == "My Template"
    assert entry["duplicate_name"] is False
    assert entry["blocker"] is False
    # domain-specific fields pass through untouched
    assert entry["model_id"] == "florence-2"
    assert entry["model_available"] is True


@patch(_CAP_REPO)
def test_apply_template_import_full_payload(MockRepo, client):
    """P3c pin: TemplateImportApplyResponse — created/skipped/installed_definitions."""
    from app.core.portable.archive import write_manifest_zip
    from app.core.template import portable

    entry = {
        "domain": "captioning", "name": "My Template", "model_id": "florence-2",
        "config": {},
    }
    manifest = portable.build_template_manifest([entry], "0.0.0-test")
    zb = write_manifest_zip(manifest).getvalue()
    MockRepo.return_value.create.return_value = {"id": "new-id", "name": "My Template"}
    with patch(
        "app.core.template.import_service.model_available", return_value=True
    ):
        response = client.post(
            "/api/templates/import/apply",
            files={"file": ("t.zip", zb, "application/zip")},
        )
    assert response.status_code == 200
    assert response.json() == {
        "created": [{"index": 0, "id": "new-id", "name": "My Template"}],
        "skipped": [],
        "installed_definitions": [],
    }


# ── Usage counter ────────────────────────────────────────────────────────


@patch(_TRAIN_REPO)
def test_use_template(MockRepo, client):
    response = client.post("/api/templates/training/t1/use")
    assert response.status_code == 200
    assert response.json()["status"] == "recorded"


# ── Usage counter: the real column, every domain ─────────────────────────


@pytest.fixture()
def usage_engine(tmp_path):
    """Isolated engine with the full migration chain applied."""
    engine = DatabaseEngine(db_path=str(tmp_path / "templates.db"))
    run_migrations(engine)
    yield engine
    engine.close()


_USAGE_DOMAINS = [
    (
        "captioning",
        "app.core.db.repositories.captioning_template_repo",
        CaptioningTemplateRepository,
        {"name": "C", "model_id": "florence-2"},
    ),
    (
        "masking",
        "app.core.db.repositories.masking_template_repo",
        MaskingTemplateRepository,
        {"name": "M", "model_id": "sam3"},
    ),
    (
        "training",
        "app.core.db.repositories.training_template_repo",
        TrainingTemplateRepository,
        {"name": "T", "definition_id": "sdxl_base_1.0"},
    ),
    (
        "adaptive",
        "app.core.db.repositories.adaptive_preset_repo",
        AdaptivePresetRepository,
        {"name": "A"},
    ),
]


@pytest.mark.parametrize(
    "domain,module,repo_cls,payload",
    _USAGE_DOMAINS,
    ids=[d[0] for d in _USAGE_DOMAINS],
)
def test_increment_usage_moves_the_real_column(
    usage_engine, domain, module, repo_cls, payload
):
    """`used_count`/`last_used_at` are the Templates library's only signal for
    ranking by real use, and every domain's frontend now records one. The route
    test above proves the wiring against a mock; this proves the column
    actually moves, on the table THIS domain's repo owns — a copy-pasted
    `increment_usage` pointing at a sibling table would pass the route test.
    """
    with patch(f"{module}.get_db", return_value=usage_engine):
        repo = repo_cls()
        created = repo.create(dict(payload))
        assert created["used_count"] == 0
        assert created.get("last_used_at") is None

        repo.increment_usage(created["id"])

        after = repo.get_by_id(created["id"])
        assert after["used_count"] == 1
        assert after["last_used_at"] is not None

    with usage_engine.connection() as conn:
        row = conn.execute(
            f"SELECT used_count FROM {repo_cls.TABLE} WHERE id = ?", (created["id"],)
        ).fetchone()
    assert row["used_count"] == 1  # …on this domain's own table
