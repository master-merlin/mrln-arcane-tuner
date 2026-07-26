"""Traversal-guard regression tests for overlay / adjustment / masking routes.

Each of these three route modules accepts a client-supplied path relative to
the dataset root and (pre-fix) joined it directly onto disk without going
through the shared ``validate_path_within`` containment check
(``app/api/_path_guard.py``). Overlay commit is a WRITE primitive
(``shutil.copy2`` into ``dataset_root / request.image_path``); the
adjustment and masking sites are arbitrary-image READ primitives. A
``"../"`` segment in the client payload escapes the dataset directory.

Mirrors the local-router TestClient pattern used by ``test_control_routes.py``
/ ``test_mask_routes.py``: mount just the module's router, monkeypatch the
module's ``dataset_manager`` singleton's ``get_dataset`` to return a fake
dataset rooted at ``tmp_path``.
"""

from __future__ import annotations

import types

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(module):
    app = FastAPI()
    app.include_router(module.router, prefix="/api")
    return TestClient(app)


def _fake_dataset(tmp_path):
    return types.SimpleNamespace(path=str(tmp_path), media_metadata={})


# ── overlay_routes ────────────────────────────────────────────────────────


class TestOverlayTraversal:
    def test_render_pipeline_rejects_traversal(self, tmp_path, monkeypatch):
        from app.api.dataset import overlay_routes

        client = _client(overlay_routes)
        monkeypatch.setattr(
            overlay_routes.dataset_manager, "get_dataset",
            lambda name: _fake_dataset(tmp_path),
        )

        res = client.post(
            "/api/datasets/ds/render-pipeline",
            json={"image_path": "../../outside.png", "blocks": []},
        )
        assert res.status_code == 403

    def test_commit_overlay_rejects_traversal(self, tmp_path, monkeypatch):
        from app.api.dataset import overlay_routes

        client = _client(overlay_routes)
        monkeypatch.setattr(
            overlay_routes.dataset_manager, "get_dataset",
            lambda name: _fake_dataset(tmp_path),
        )

        res = client.post(
            "/api/datasets/ds/overlay/commit",
            json={"image_path": "../../outside.png"},
        )
        assert res.status_code == 403


# ── adjustment_routes ─────────────────────────────────────────────────────


class TestAdjustmentTraversal:
    def test_adjust_media_rejects_traversal(self, tmp_path, monkeypatch):
        """``/adjust`` is a WRITE primitive (opens + overwrites in place via
        dataset_manager.apply_adjustments) — the most severe adjustment-side
        gap, even though it wasn't in the brief's line list."""
        from app.api.dataset import adjustment_routes

        client = _client(adjustment_routes)
        monkeypatch.setattr(
            adjustment_routes.dataset_manager, "get_dataset",
            lambda name: _fake_dataset(tmp_path),
        )

        res = client.post(
            "/api/datasets/ds/adjust",
            json={"path": "../../outside.png"},
        )
        assert res.status_code == 403

    def test_adjust_media_color_match_reference_rejects_traversal(self, tmp_path, monkeypatch):
        """The embedded ``color_match.reference_path`` on /adjust is a second,
        independent client-supplied path — must be guarded even when the
        primary ``path`` is legitimate."""
        from app.api.dataset import adjustment_routes

        client = _client(adjustment_routes)
        monkeypatch.setattr(
            adjustment_routes.dataset_manager, "get_dataset",
            lambda name: _fake_dataset(tmp_path),
        )

        res = client.post(
            "/api/datasets/ds/adjust",
            json={
                "path": "a.png",
                "color_match": {"reference_path": "../../ref.png"},
            },
        )
        assert res.status_code == 403

    def test_adjust_media_batch_rejects_traversal_before_apply(self, tmp_path, monkeypatch):
        """/adjust-batch streams SSE progress; a guard failure can't become a
        raw HTTP 403 mid-stream (the 200 + stream is already committed), so
        it must surface as a per-item ``status: error`` event instead of
        crashing the whole batch.

        ``dataset_manager.apply_adjustments`` is spied rather than left real:
        both a pre-fix ``ValueError`` (dataset not registered in the real
        singleton) and a post-fix ``HTTPException(403)`` would otherwise
        collapse to the same observable ``status: error``, so the only
        assertion that actually discriminates "guard blocked it" from
        "failed for an unrelated reason" is that the escaping path never
        reaches ``apply_adjustments`` at all, while a sibling legitimate path
        in the same batch still does.
        """
        import json

        from app.api.dataset import adjustment_routes

        calls: list[str] = []

        def _spy_apply(name, path, adjustments):
            calls.append(path)
            return True

        client = _client(adjustment_routes)
        monkeypatch.setattr(
            adjustment_routes.dataset_manager, "get_dataset",
            lambda name: _fake_dataset(tmp_path),
        )
        monkeypatch.setattr(adjustment_routes.dataset_manager, "apply_adjustments", _spy_apply)

        res = client.post(
            "/api/datasets/ds/adjust-batch",
            json={"paths": ["../../outside.png", "legit.png"]},
        )
        assert res.status_code == 200
        events = [
            json.loads(line.replace("data: ", ""))
            for line in res.text.strip().split("\n")
            if line.startswith("data:")
        ]
        assert events[0]["status"] == "error"
        assert events[1]["status"] == "ok"
        assert calls == ["legit.png"]

    def test_color_match_preview_rejects_traversal_source(self, tmp_path, monkeypatch):
        from app.api.dataset import adjustment_routes

        client = _client(adjustment_routes)
        monkeypatch.setattr(
            adjustment_routes.dataset_manager, "get_dataset",
            lambda name: _fake_dataset(tmp_path),
        )

        res = client.post(
            "/api/datasets/ds/color-match",
            json={"source_path": "../../a.png", "reference_path": "b.png"},
        )
        assert res.status_code == 403

    def test_color_match_preview_rejects_traversal_reference(self, tmp_path, monkeypatch):
        from app.api.dataset import adjustment_routes

        client = _client(adjustment_routes)
        monkeypatch.setattr(
            adjustment_routes.dataset_manager, "get_dataset",
            lambda name: _fake_dataset(tmp_path),
        )

        res = client.post(
            "/api/datasets/ds/color-match",
            json={"source_path": "a.png", "reference_path": "../../b.png"},
        )
        assert res.status_code == 403

    def test_histogram_rejects_traversal(self, tmp_path, monkeypatch):
        from app.api.dataset import adjustment_routes

        client = _client(adjustment_routes)
        monkeypatch.setattr(
            adjustment_routes.dataset_manager, "get_dataset",
            lambda name: _fake_dataset(tmp_path),
        )

        res = client.get(
            "/api/datasets/ds/histogram",
            params={"image_path": "../../outside.png"},
        )
        assert res.status_code == 403


# ── masking_routes ────────────────────────────────────────────────────────


class TestMaskingTraversal:
    def test_generate_mask_rejects_traversal(self, tmp_path, monkeypatch):
        from app.api import masking_routes

        client = _client(masking_routes)
        monkeypatch.setattr(
            masking_routes.dataset_manager, "get_dataset",
            lambda name: _fake_dataset(tmp_path),
        )

        res = client.post(
            "/api/datasets/ds/masking/generate",
            json={
                "dataset_name": "ds",
                "image_rel_path": "../../outside.png",
                "model_id": "rembg",
            },
        )
        assert res.status_code == 403

    def test_apply_mask_rejects_traversal(self, tmp_path, monkeypatch):
        from app.api import masking_routes

        client = _client(masking_routes)
        monkeypatch.setattr(
            masking_routes.dataset_manager, "get_dataset",
            lambda name: _fake_dataset(tmp_path),
        )

        res = client.post(
            "/api/datasets/ds/masking/apply",
            json={"image_rel_path": "../../outside.png"},
        )
        assert res.status_code == 403

    def test_preview_mask_rejects_traversal(self, tmp_path, monkeypatch):
        from app.api import masking_routes

        client = _client(masking_routes)
        monkeypatch.setattr(
            masking_routes.dataset_manager, "get_dataset",
            lambda name: _fake_dataset(tmp_path),
        )

        res = client.get(
            "/api/datasets/ds/masking/preview",
            params={"image_rel_path": "../../outside.png"},
        )
        assert res.status_code == 403
