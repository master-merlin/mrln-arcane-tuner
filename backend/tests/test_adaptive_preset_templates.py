"""Adaptive preset template domain: seeding, CRUD, readonly, portability (spec §4).

The `adaptive` domain mirrors `masking` minus `model_id` — an adaptive preset is
a bag of tuning knobs, not a model-scoped template. Three FACTORY presets ship
readonly; editing one branches a user preset (the branch itself is a frontend
flow, so the backend only has to accept `branched_from` at create time).
"""

import io
import json
import zipfile
from unittest.mock import patch

import pytest

from app.core.db.engine import DatabaseEngine
from app.core.db.migrations import run_migrations
from app.core.db.repositories.adaptive_preset_repo import AdaptivePresetRepository
from app.core.portable.archive import write_manifest_zip
from app.core.template import portable
from app.engine.models.adaptive import FACTORY_PRESETS, AdaptiveTargetingConfig

_ADAPT_REPO = "app.core.db.repositories.adaptive_preset_repo.AdaptivePresetRepository"
_MASK_REPO = "app.core.db.repositories.masking_template_repo.MaskingTemplateRepository"

_FACTORY_IDS = {"factory-conservative", "factory-balanced", "factory-aggressive"}
_TABLE = "adaptive_preset_templates"


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def migrated_engine(tmp_path):
    """Isolated engine with the full migration chain applied."""
    engine = DatabaseEngine(db_path=str(tmp_path / "adaptive.db"))
    run_migrations(engine)
    yield engine
    engine.close()


@pytest.fixture()
def repo(migrated_engine):
    """AdaptivePresetRepository bound to the isolated engine."""
    instance = AdaptivePresetRepository()
    with patch(
        "app.core.db.repositories.adaptive_preset_repo.get_db",
        return_value=migrated_engine,
    ):
        yield instance


def _rows(engine):
    with engine.connection() as conn:
        return conn.execute(f"SELECT * FROM {_TABLE} ORDER BY id").fetchall()


def _make_project(engine, project_id: str) -> str:
    """Insert a project row — ``project_id`` is a real FK on the preset table."""
    with engine.write() as conn:
        conn.execute(
            "INSERT INTO projects (id, name, created_at, updated_at) "
            "VALUES (?, ?, 0.0, 0.0)",
            (project_id, f"proj_{project_id}"),
        )
    return project_id


# ── Migration + factory seeding ──────────────────────────────────────────


def test_migration_creates_table_and_seeds_factory_presets(migrated_engine):
    """The schema migration both creates the table and seeds the presets, so a
    fresh install has them without any extra startup hook."""
    ids = {r["id"] for r in _rows(migrated_engine)}
    assert ids == _FACTORY_IDS


def test_factory_presets_seeded_idempotently(repo, migrated_engine):
    """Seeding is INSERT OR IGNORE: a second call must not duplicate rows."""
    repo.seed_factory_presets()
    repo.seed_factory_presets()  # second call must not duplicate

    rows = repo.list_for_project()
    factory = [r for r in rows if r["readonly"]]
    assert {r["id"] for r in factory} == _FACTORY_IDS
    assert len(rows) == 3

    by_id = {r["id"]: r for r in factory}
    for name, knobs in FACTORY_PRESETS.items():
        cfg = by_id[f"factory-{name}"]["config"]
        assert cfg["preset"] == f"factory:{name}"
        # Real knob values, not placeholders.
        for key, value in knobs.items():
            assert cfg[key] == value
        assert cfg["interval_steps"] > 0


def test_seeded_factory_configs_are_valid_knob_sets(repo):
    """A seeded preset must survive AdaptiveTargetingConfig — otherwise a user
    could select it and only discover at job-submit time that it is unusable."""
    repo.seed_factory_presets()
    for row in repo.list_for_project():
        AdaptiveTargetingConfig.model_validate(row["config"])


def test_reseeding_never_overwrites_an_existing_row(repo, migrated_engine):
    """A row that already exists is left completely alone on re-seed — the
    factory ids are stable, so an overwrite would silently revert live state."""
    with migrated_engine.write() as conn:
        conn.execute(
            f"UPDATE {_TABLE} SET name = ?, config = ? WHERE id = ?",
            ("Tampered", json.dumps({"preset": "factory:balanced",
                                     "interval_steps": 999}), "factory-balanced"),
        )

    repo.seed_factory_presets()

    row = repo.get_by_id("factory-balanced")
    assert row["name"] == "Tampered"
    assert row["config"]["interval_steps"] == 999


# ── Repository CRUD ──────────────────────────────────────────────────────


def test_repo_crud_roundtrip(repo):
    created = repo.create({"name": "My Fast",
                           "config": {"preset": "factory:aggressive",
                                      "interval_steps": 120}})
    assert created["readonly"] is False
    assert created["config"]["interval_steps"] == 120

    fetched = repo.get_by_id(created["id"])
    assert fetched["name"] == "My Fast"

    updated = repo.update(created["id"], {"name": "Renamed"})
    assert updated["name"] == "Renamed"

    repo.increment_usage(created["id"])
    assert repo.get_by_id(created["id"])["used_count"] == 1

    repo.delete(created["id"])
    assert repo.get_by_id(created["id"]) is None


def test_repo_update_drops_other_domains_scoping_keys(repo):
    """The shared PUT body spans all four domains. ``model_id`` has no column
    here, so it must be dropped — reaching the SET clause would turn a valid
    rename into an unknown-column 500."""
    created = repo.create({"name": "Mine", "config": {}})
    updated = repo.update(created["id"], {"name": "Renamed",
                                          "model_id": "sam3",
                                          "definition_id": "flux2-x"})
    assert updated["name"] == "Renamed"


def test_repo_update_cannot_flip_readonly(repo):
    """readonly is minted by seeding only — a client must not be able to forge
    an undeletable preset by writing the flag through a rename."""
    created = repo.create({"name": "Mine", "config": {}})
    assert repo.update(created["id"], {"readonly": True})["readonly"] is False


def test_repo_create_never_honours_a_client_supplied_readonly(repo):
    """Import entries are untrusted: a carried ``readonly`` must not survive."""
    created = repo.create({"name": "Forged", "config": {}, "readonly": True})
    assert created["readonly"] is False


def test_repo_delete_refuses_a_readonly_row(repo):
    """Defense in depth behind the route guard: the DELETE is scoped to
    ``readonly = 0`` so a factory preset survives even a direct repo call."""
    repo.delete("factory-balanced")
    assert repo.get_by_id("factory-balanced") is not None


def test_repo_branch_yields_an_editable_copy(repo, migrated_engine):
    """Branching a readonly factory preset must produce an EDITABLE row —
    a readonly copy would wedge the frontend's auto-branch-then-edit flow."""
    _make_project(migrated_engine, "proj-1")
    branched = repo.branch("factory-balanced", "proj-1", "Balanced (custom)")
    assert branched["readonly"] is False
    assert branched["branched_from"] == "factory-balanced"
    assert branched["project_id"] == "proj-1"
    assert branched["config"] == repo.get_by_id("factory-balanced")["config"]


def test_repo_list_merges_general_and_project_scope(repo, migrated_engine):
    _make_project(migrated_engine, "proj-1")
    _make_project(migrated_engine, "proj-2")
    repo.create({"name": "Proj", "project_id": "proj-1", "config": {}})
    repo.create({"name": "Other", "project_id": "proj-2", "config": {}})

    scoped = {r["name"] for r in repo.list_for_project(project_id="proj-1")}
    assert "Proj" in scoped and "Other" not in scoped
    assert "Balanced" in {r["name"] for r in repo.list_for_project(project_id="proj-1")}

    general = {r["name"] for r in repo.list_for_project()}
    assert "Proj" not in general and "Other" not in general


# ── Routes ───────────────────────────────────────────────────────────────


@patch(_ADAPT_REPO)
def test_list_adaptive_presets(MockRepo, client):
    MockRepo.return_value.list_for_project.return_value = [
        {"id": "factory-balanced", "name": "Balanced", "readonly": True,
         "config": {"preset": "factory:balanced"}, "created_at": 0.0},
    ]
    response = client.get("/api/templates/adaptive")
    assert response.status_code == 200
    body = response.json()
    assert body[0]["id"] == "factory-balanced"
    assert body[0]["readonly"] is True
    # Not model-scoped: the response model must not invent a model_id.
    assert "model_id" not in body[0]


@patch(_ADAPT_REPO)
def test_crud_roundtrip_and_delete(MockRepo, client):
    MockRepo.return_value.create.return_value = {
        "id": "adaptive_1", "name": "My Fast", "created_at": 0.0,
        "config": {"preset": "factory:aggressive", "interval_steps": 120},
    }
    created = client.post("/api/templates/adaptive", json={
        "name": "My Fast",
        "config": {"preset": "factory:aggressive", "interval_steps": 120},
    })
    assert created.status_code == 201
    tid = created.json()["id"]

    MockRepo.return_value.get_by_id.return_value = {
        "id": tid, "name": "My Fast", "readonly": False, "created_at": 0.0,
    }
    assert client.get(f"/api/templates/adaptive/{tid}").json()["name"] == "My Fast"

    MockRepo.return_value.update.return_value = {
        "id": tid, "name": "Renamed", "readonly": False, "created_at": 0.0,
    }
    updated = client.put(f"/api/templates/adaptive/{tid}", json={"name": "Renamed"})
    assert updated.status_code == 200
    assert updated.json()["name"] == "Renamed"

    assert client.delete(f"/api/templates/adaptive/{tid}").status_code == 200


@patch(_ADAPT_REPO)
def test_create_records_branch_lineage(MockRepo, client):
    """The card's auto-branch: create with branched_from → lineage persisted.

    Without ``branched_from`` on the create request the frontend cannot nest a
    user preset under the factory preset it was derived from.
    """
    def _echo(data):
        return {"id": "adaptive_2", "created_at": 0.0, "readonly": False, **data}

    MockRepo.return_value.create.side_effect = _echo
    created = client.post("/api/templates/adaptive", json={
        "name": "Balanced (custom)",
        "branched_from": "factory-balanced",
        "config": {"preset": "factory:balanced", "interval_steps": 175},
    })
    assert created.status_code == 201
    assert created.json()["branched_from"] == "factory-balanced"
    assert MockRepo.return_value.create.call_args[0][0]["branched_from"] == (
        "factory-balanced")


@patch(_ADAPT_REPO)
def test_update_on_factory_preset_rejected(MockRepo, client):
    """A factory preset is immutable — the client is expected to branch."""
    MockRepo.return_value.get_by_id.return_value = {
        "id": "factory-balanced", "readonly": True,
    }
    r = client.put("/api/templates/adaptive/factory-balanced", json={"name": "hack"})
    assert r.status_code == 409
    MockRepo.return_value.update.assert_not_called()


@patch(_ADAPT_REPO)
def test_delete_on_factory_preset_rejected(MockRepo, client):
    MockRepo.return_value.get_by_id.return_value = {
        "id": "factory-balanced", "readonly": True,
    }
    r = client.delete("/api/templates/adaptive/factory-balanced")
    assert r.status_code == 409
    MockRepo.return_value.delete.assert_not_called()


@patch(_MASK_REPO)
def test_other_domains_keep_their_403_readonly_status(MockRepo, client):
    """Prove the negative: adding adaptive's 409 must NOT retrofit the other
    three domains, whose clients are pinned to 403."""
    MockRepo.return_value.get_by_id.return_value = {"id": "mask_default_sam3",
                                                    "readonly": True}
    assert client.put("/api/templates/masking/mask_default_sam3",
                      json={"name": "hack"}).status_code == 403


@patch(_ADAPT_REPO)
def test_branch_and_use_routes(MockRepo, client):
    MockRepo.return_value.branch.return_value = {
        "id": "adaptive_3", "name": "Balanced (Project)", "created_at": 0.0,
        "readonly": False, "branched_from": "factory-balanced",
    }
    r = client.post("/api/templates/adaptive/factory-balanced/branch",
                    json={"target_project_id": "proj-1"})
    assert r.status_code == 200
    assert r.json()["branched_from"] == "factory-balanced"

    r = client.post("/api/templates/adaptive/adaptive_3/use")
    assert r.status_code == 200
    MockRepo.return_value.increment_usage.assert_called_once_with("adaptive_3")


# ── Routes against the REAL repo (no mocks) ──────────────────────────────
# The mocked route tests above cannot catch a column/row-shape mismatch —
# they never touch SQL. These drive the real repository over the session's
# migrated DB, which is the only way the seeding + INSERT columns are proven
# to actually line up with the table the migration created.


def test_end_to_end_create_update_delete_against_real_db(client):
    created = client.post("/api/templates/adaptive", json={
        "name": "E2E Preset",
        "branched_from": "factory-balanced",
        "config": {"preset": "factory:balanced", "interval_steps": 175},
    })
    assert created.status_code == 201, created.text
    row = created.json()
    tid = row["id"]
    try:
        assert row["branched_from"] == "factory-balanced"
        assert row["readonly"] is False
        assert row["config"]["interval_steps"] == 175

        listed = client.get("/api/templates/adaptive").json()
        assert tid in {r["id"] for r in listed}
        assert _FACTORY_IDS <= {r["id"] for r in listed}

        renamed = client.put(f"/api/templates/adaptive/{tid}",
                             json={"name": "E2E Renamed"})
        assert renamed.status_code == 200, renamed.text
        assert renamed.json()["name"] == "E2E Renamed"

        assert client.post(f"/api/templates/adaptive/{tid}/use").status_code == 200
    finally:
        assert client.delete(f"/api/templates/adaptive/{tid}").status_code == 200
    assert client.get(f"/api/templates/adaptive/{tid}").status_code == 404


def test_end_to_end_factory_preset_is_immutable(client):
    """The seeded factory row really is readonly on the live DB — not just in
    a mock that was told to say so."""
    detail = client.get("/api/templates/adaptive/factory-balanced")
    assert detail.status_code == 200, detail.text
    assert detail.json()["readonly"] is True

    assert client.put("/api/templates/adaptive/factory-balanced",
                      json={"name": "hack"}).status_code == 409
    assert client.delete("/api/templates/adaptive/factory-balanced").status_code == 409
    # Unchanged after both attempts.
    assert client.get("/api/templates/adaptive/factory-balanced").json()["name"] == (
        "Balanced")


# ── Portability ──────────────────────────────────────────────────────────


def _zip_bytes(entries) -> bytes:
    return write_manifest_zip(
        portable.build_template_manifest(entries, "test")
    ).getvalue()


def _upload(client, path, zip_bytes, **form):
    return client.post(
        path, files={"file": ("t.template.zip", zip_bytes, "application/zip")},
        data=form,
    )


@patch(_ADAPT_REPO)
def test_export_single_adaptive_preset(MockRepo, client):
    MockRepo.return_value.get_by_id.return_value = {
        "id": "adaptive_1", "project_id": "p1", "name": "My Fast",
        "config": {"preset": "factory:aggressive", "interval_steps": 120},
        "created_at": 1.0,
    }
    resp = client.get("/api/templates/adaptive/adaptive_1/export")
    assert resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    entry = manifest["templates"][0]
    assert entry["domain"] == "adaptive"
    assert entry["name"] == "My Fast"
    assert entry["config"]["interval_steps"] == 120
    # Machine-specific / model-scoping fields must not travel.
    assert "id" not in entry and "project_id" not in entry and "model_id" not in entry


@patch(_ADAPT_REPO)
def test_import_plan_valid_preset_has_no_warning(MockRepo, client):
    MockRepo.return_value.list_for_project.return_value = []
    entry = portable.build_template_entry(
        "adaptive",
        {"name": "Fast", "config": {"preset": "factory:aggressive",
                                    "interval_steps": 150}}, None)
    resp = _upload(client, "/api/templates/import/plan", _zip_bytes([entry]))
    assert resp.status_code == 200
    item = resp.json()["entries"][0]
    assert item["domain"] == "adaptive"
    assert item["config_warning"] is None
    assert item["blocker"] is False


@patch(_ADAPT_REPO)
def test_import_plan_invalid_preset_surfaces_config_warning(MockRepo, client):
    """An archive is untrusted: a knob set that fails AdaptiveTargetingConfig
    must be surfaced, never silently imported to become a job config later."""
    MockRepo.return_value.list_for_project.return_value = []
    entry = portable.build_template_entry(
        "adaptive",
        # probe_steps >= interval_steps violates the cross-field rule.
        {"name": "Bad", "config": {"preset": "custom", "interval_steps": 20,
                                   "probe_steps": 400}}, None)
    resp = _upload(client, "/api/templates/import/plan", _zip_bytes([entry]))
    assert resp.status_code == 200
    item = resp.json()["entries"][0]
    assert item["config_warning"]
    assert "probe_steps" in item["config_warning"]
    # Non-blocking, exactly like the other domains' config sanity checks.
    assert item["blocker"] is False


@patch(_ADAPT_REPO)
def test_import_apply_creates_a_preset_row(MockRepo, client):
    MockRepo.return_value.list_for_project.return_value = []
    MockRepo.return_value.create.side_effect = lambda data: {"id": "adaptive_9", **data}
    entry = portable.build_template_entry(
        "adaptive",
        {"name": "Fast", "config": {"preset": "factory:aggressive",
                                    "interval_steps": 150}}, None)
    resp = _upload(client, "/api/templates/import/apply", _zip_bytes([entry]))
    assert resp.status_code == 200
    body = resp.json()
    assert body["skipped"] == []
    assert body["created"][0]["name"] == "Fast"
    # An adaptive preset is not model-scoped — no phantom model_id column write.
    assert "model_id" not in MockRepo.return_value.create.call_args[0][0]
