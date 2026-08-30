"""Route-level tests for project CRUD, dataset-association and preference
endpoints in ``app/api/project_routes.py``.

These handlers previously ran their repo calls directly on the event loop and
had no route-level coverage; they were moved onto ``asyncio.to_thread``. The
tests pin response shapes/status codes so the wrapping is behaviour-preserving
(including HTTPException propagation out of the worker thread).
"""

from __future__ import annotations

import contextlib


@contextlib.contextmanager
def _isolated_db(tmp_path):
    from app.core.db.engine import DatabaseEngine

    prev = DatabaseEngine._instance
    eng = DatabaseEngine(db_path=str(tmp_path / "projects.db"))
    eng.initialize()
    DatabaseEngine._instance = eng
    try:
        yield eng
    finally:
        eng.close()
        DatabaseEngine._instance = prev


_STATS_KEYS = {
    "captioning_templates",
    "masking_templates",
    "training_templates",
    "adaptive_preset_templates",
    "datasets",
    "jobs",
}


def test_project_crud_roundtrip(client, tmp_path):
    with _isolated_db(tmp_path):
        # create
        resp = client.post("/api/projects", json={"name": "Alpha", "color": "#123456"})
        assert resp.status_code == 201
        proj = resp.json()
        pid = proj["id"]
        assert proj["name"] == "Alpha"
        assert proj["color"] == "#123456"

        # duplicate name → 409
        assert client.post("/api/projects", json={"name": "Alpha"}).status_code == 409

        # list (with stats)
        resp = client.get("/api/projects")
        assert resp.status_code == 200
        listing = resp.json()
        assert len(listing) == 1
        assert set(listing[0]["stats"]) == _STATS_KEYS

        # get one (with stats)
        resp = client.get(f"/api/projects/{pid}")
        assert resp.status_code == 200
        assert set(resp.json()["stats"]) == _STATS_KEYS

        # get missing → 404
        assert client.get("/api/projects/ghost").status_code == 404

        # patch
        resp = client.patch(f"/api/projects/{pid}", json={"description": "hello"})
        assert resp.status_code == 200
        assert resp.json()["description"] == "hello"

        # patch with no fields → 400
        assert client.patch(f"/api/projects/{pid}", json={}).status_code == 400
        # patch missing project → 404
        assert client.patch("/api/projects/ghost", json={"name": "x"}).status_code == 404

        # delete
        assert client.delete(f"/api/projects/{pid}").status_code == 204
        assert client.delete(f"/api/projects/{pid}").status_code == 404


def test_project_full_payload_create_list_get_update(client, tmp_path):
    """P3c pin: ProjectRow/ProjectWithStats — exact key set on create/list/
    get/update, including the `stats` sub-object only list/get inject."""
    with _isolated_db(tmp_path):
        resp = client.post(
            "/api/projects", json={"name": "Gamma", "description": "d", "color": "#111111"}
        )
        assert resp.status_code == 201
        proj = resp.json()
        pid = proj["id"]
        assert set(proj) == {"id", "name", "description", "color", "created_at", "updated_at"}
        assert proj["name"] == "Gamma"
        assert proj["description"] == "d"
        assert proj["color"] == "#111111"

        listing = client.get("/api/projects").json()
        assert set(listing[0]) == {
            "id", "name", "description", "color", "created_at", "updated_at", "stats",
        }
        assert listing[0]["stats"] == {
            "captioning_templates": 0, "masking_templates": 0,
            "training_templates": 0, "adaptive_preset_templates": 0,
            "datasets": 0, "jobs": 0,
        }

        got = client.get(f"/api/projects/{pid}").json()
        assert set(got) == {
            "id", "name", "description", "color", "created_at", "updated_at", "stats",
        }

        updated = client.patch(f"/api/projects/{pid}", json={"description": "e"}).json()
        assert set(updated) == {"id", "name", "description", "color", "created_at", "updated_at"}
        assert updated["description"] == "e"


def test_branched_adaptive_preset_counts_toward_the_project_template_stat(
    client, tmp_path
):
    """Adaptive presets are project-scopable exactly like the other three
    domains — the detail screen lists, branches and deletes them. A stat that
    skips the domain disagrees with the list right beside it."""
    with _isolated_db(tmp_path):
        pid = client.post("/api/projects", json={"name": "Delta"}).json()["id"]
        assert client.get(f"/api/projects/{pid}").json()["stats"][
            "adaptive_preset_templates"
        ] == 0

        factory = client.get("/api/templates/adaptive").json()
        assert factory, "the three factory presets are seeded by migration"
        resp = client.post(
            f"/api/templates/adaptive/{factory[0]['id']}/branch",
            json={"target_project_id": pid},
        )
        assert resp.status_code == 200

        stats = client.get(f"/api/projects/{pid}").json()["stats"]
        assert stats["adaptive_preset_templates"] == 1
        # The global factory rows stay out of it — only the project's own.
        assert stats["training_templates"] == 0


def test_project_datasets_association(client, tmp_path):
    with _isolated_db(tmp_path) as eng:
        proj = client.post("/api/projects", json={"name": "Beta"}).json()
        pid = proj["id"]

        # seed a dataset row so the association FK resolves
        with eng.write() as conn:
            conn.execute(
                "INSERT INTO datasets (id, name, path, created_at) "
                "VALUES ('ds-1', 'myds', '/tmp/myds', 1.0)"
            )

        assert client.get(f"/api/projects/{pid}/datasets").json() == []

        resp = client.post(
            f"/api/projects/{pid}/datasets", json={"dataset_id": "ds-1"}
        )
        assert resp.status_code == 201
        assert resp.json() == {"status": "added"}

        datasets = client.get(f"/api/projects/{pid}/datasets").json()
        assert [d["id"] for d in datasets] == ["ds-1"]
        # P3c pin: ProjectDatasetRow is open (extra=allow) — every column of
        # the `datasets` table (not just id/name) must survive untouched.
        assert set(datasets[0]) == {
            "id", "name", "path", "description", "created_at", "last_scanned_at",
            "file_count", "total_size_bytes", "multimedia_count", "caption_count",
            "mask_count", "caption_coverage", "missing", "preview_image",
            "preview_pinned",
            "majority_ar", "harmonization_score", "classifier", "version",
            "has_cache", "source_type", "license", "updated_at",
            "trigger_word", "tags", "notes", "kind",
        }

        assert (
            client.delete(f"/api/projects/{pid}/datasets/ds-1").status_code == 204
        )
        assert client.get(f"/api/projects/{pid}/datasets").json() == []

        # associating against a missing project → 404
        assert (
            client.post(
                "/api/projects/ghost/datasets", json={"dataset_id": "ds-1"}
            ).status_code
            == 404
        )


def test_project_preferences(client, tmp_path):
    with _isolated_db(tmp_path):
        # general preferences (seeded by the v4 migration)
        resp = client.get("/api/projects/general/preferences")
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict)

        resp = client.put(
            "/api/projects/general/preferences",
            json={"selected_caption_model": "florence-2"},
        )
        assert resp.status_code == 200
        assert resp.json()["selected_caption_model"] == "florence-2"


def test_project_preferences_full_payload(client, tmp_path):
    """P3c pin: ProjectPreferencesRow — exact key set, training_selections
    stays a decoded dict (not a raw JSON string)."""
    with _isolated_db(tmp_path):
        resp = client.get("/api/projects/general/preferences")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {
            "id", "project_id", "selected_caption_model", "active_caption_template",
            "qwen3_variant", "selected_mask_model", "active_mask_template",
            "training_selections",
        }
        assert body["project_id"] is None
        assert body["training_selections"] == {}

        resp = client.put(
            "/api/projects/general/preferences",
            json={"training_selections": {"lr": 1e-4, "steps": 100}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {
            "id", "project_id", "selected_caption_model", "active_caption_template",
            "qwen3_variant", "selected_mask_model", "active_mask_template",
            "training_selections",
        }
        assert body["training_selections"] == {"lr": 1e-4, "steps": 100}
