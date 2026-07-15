"""Tests for pair health, role-ordering API, and control slot upload — PR2.

Covers:
- ``compute_pair_health``: missing slots, orphans, dim/staleness/role
  warnings, fully_paired
- ``DatasetManager.set_pair_order`` / ``apply_pair_order_all``
- ``control_routes``: health + pair-order endpoints (route layer)
- Control slot upload through ``POST /datasets/{name}/upload``
- Thumbnail namespacing for subdirectory rel-paths (control/ images must
  not collide with the root image of the same stem)
"""

import os
import time

import av
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from PIL import Image

from app.core.dataset.control_helpers import compute_pair_health
from app.core.dataset.thumbnails import ensure_thumbnail, thumbnail_path_for
from app.core.dataset_manager import DatasetManager


# ── Fixtures (mirror test_edit_dataset_pairs) ────────────────────────────


@pytest.fixture()
def mock_settings():
    mock_instance = MagicMock()
    mock_instance.get_module_settings.return_value = {}
    mock_instance.update_module_settings = MagicMock()
    with patch(
        "app.core.dataset_manager.get_settings_manager",
        return_value=mock_instance,
    ):
        yield mock_instance


@pytest.fixture()
def manager(tmp_path, mock_settings):
    default_root = str(tmp_path / "datasets")
    os.makedirs(default_root, exist_ok=True)

    with patch.object(DatasetManager, "__init__", lambda self, **kw: None):
        mgr = DatasetManager()

    mgr.root_dir = str(tmp_path)
    mgr.storage_file = str(tmp_path / "dataset_locations.json")
    mgr.default_root = default_root
    mgr.settings_manager = mock_settings
    mgr.datasets = {}
    mgr._loop = None
    mgr._db = MagicMock()
    mgr._dataset_repo = MagicMock()
    mgr._media_repo = MagicMock()
    return mgr


def _create_image(path: str, width: int = 64, height: int = 64, color: str = "red"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (width, height), color).save(path)


def _create_video(
    path: str, *, n_frames: int = 8, fps: int = 24, width: int = 32, height: int = 24
):
    """Tiny h264 mp4 — mirrors ``test_probe._write_clip`` (BR0 video controls)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with av.open(path, mode="w") as container:
        vstream = container.add_stream("libx264", rate=fps)
        vstream.width = width
        vstream.height = height
        vstream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            arr = np.full((height, width, 3), (i * 20) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in vstream.encode(frame):
                container.mux(packet)
        for packet in vstream.encode():
            container.mux(packet)


def _make_edit_dataset(manager, name: str = "editds", *, controls=("img1",),
                       targets=("img1", "img2")):
    ds = manager.create_dataset(name, kind="edit")
    for stem in targets:
        _create_image(os.path.join(ds.path, f"{stem}.png"))
    for stem in controls:
        _create_image(os.path.join(ds.path, "control", f"{stem}.jpg"), color="blue")
    manager.scan_dataset(name)
    return ds


# ── Pair health ──────────────────────────────────────────────────────────


class TestPairHealth:
    def test_missing_and_paired_counts(self, manager):
        ds = _make_edit_dataset(manager, controls=("img1",), targets=("img1", "img2"))
        health = compute_pair_health(ds)
        assert health["kind"] == "edit"
        assert health["target_count"] == 2
        assert health["paired_count"] == 1
        assert health["fully_paired"] is False
        assert health["missing_by_slot"] == {"control": ["img2"]}
        assert health["orphans"] == []

    def test_fully_paired(self, manager):
        ds = _make_edit_dataset(manager, controls=("img1", "img2"),
                                targets=("img1", "img2"))
        health = compute_pair_health(ds)
        assert health["fully_paired"] is True
        assert health["missing_by_slot"] == {}
        assert health["paired_count"] == 2

    def test_orphan_controls_reported(self, manager):
        ds = _make_edit_dataset(manager, controls=("img1", "ghost"),
                                targets=("img1",))
        health = compute_pair_health(ds)
        assert {"slot": "control", "rel_path": "control/ghost.jpg"} in health["orphans"]

    def test_edit_dataset_without_any_controls(self, manager):
        ds = _make_edit_dataset(manager, controls=(), targets=("img1", "img2"))
        health = compute_pair_health(ds)
        assert health["fully_paired"] is False
        assert health["missing_by_slot"] == {"control": ["img1", "img2"]}

    def test_dim_mismatch_warning(self, manager):
        ds = manager.create_dataset("dimwarn", kind="edit")
        _create_image(os.path.join(ds.path, "img1.png"), 64, 64)
        # Control with a clearly different aspect (2:1 vs 1:1).
        _create_image(os.path.join(ds.path, "control", "img1.jpg"), 128, 64)
        manager.scan_dataset("dimwarn")
        health = compute_pair_health(manager.datasets["dimwarn"])
        assert {"stem": "img1", "type": "dim_mismatch"} in health["warnings"]

    def test_no_dim_warning_for_same_aspect(self, manager):
        ds = manager.create_dataset("dimok", kind="edit")
        _create_image(os.path.join(ds.path, "img1.png"), 64, 64)
        _create_image(os.path.join(ds.path, "control", "img1.jpg"), 32, 32)
        manager.scan_dataset("dimok")
        health = compute_pair_health(manager.datasets["dimok"])
        assert all(w["type"] != "dim_mismatch" for w in health["warnings"])

    def test_role_order_invalid_warning(self, manager):
        ds = _make_edit_dataset(manager, controls=("img1",), targets=("img1",))
        ds.media_metadata["img1.png"]["control_info"]["role_order"] = [
            "control_3", "root",
        ]
        health = compute_pair_health(ds)
        assert {"stem": "img1", "type": "role_order_invalid"} in health["warnings"]

    def test_target_edited_after_control_warning(self, manager):
        ds = _make_edit_dataset(manager, controls=("img1",), targets=("img1",))
        # Stamp the target edit far in the future relative to control mtime.
        ds.media_metadata["img1.png"]["control_info"]["target_edited_at"] = (
            time.time() + 9999
        )
        health = compute_pair_health(ds)
        assert {"stem": "img1", "type": "target_edited_after_control"} in (
            health["warnings"]
        )

    def test_video_control_pair_counted_in_mixed_dataset(self, manager):
        """Task BR0: root/clip1.mp4 <-> control/clip1.mp4 pairs by stem just
        like images, and a mixed image+video edit dataset is legal — both
        pairs count toward paired_count."""
        ds = manager.create_dataset("mixedset", kind="edit")
        _create_image(os.path.join(ds.path, "img1.png"))
        _create_image(os.path.join(ds.path, "control", "img1.jpg"), color="blue")
        _create_video(os.path.join(ds.path, "clip1.mp4"))
        _create_video(os.path.join(ds.path, "control", "clip1.mp4"))
        manager.scan_dataset("mixedset")

        health = compute_pair_health(manager.datasets["mixedset"])
        assert health["target_count"] == 2
        assert health["paired_count"] == 2
        assert health["fully_paired"] is True

        slot = ds.media_metadata["clip1.mp4"]["control_info"]["slots"]["control"]
        assert slot["rel_path"] == "control/clip1.mp4"
        assert slot["num_frames"] > 0
        assert slot["fps"] > 0


# ── Pair-order mutations ─────────────────────────────────────────────────


class TestPairOrderMutations:
    def test_set_pair_order(self, manager):
        ds = _make_edit_dataset(manager, controls=("img1",), targets=("img1",))
        result = manager.set_pair_order("editds", "img1.png", ["control", "root"])
        assert result["role_order"] == ["control", "root"]
        assert ds.media_metadata["img1.png"]["control_info"]["role_order"] == [
            "control", "root",
        ]

    def test_set_pair_order_clears_with_none(self, manager):
        ds = _make_edit_dataset(manager, controls=("img1",), targets=("img1",))
        manager.set_pair_order("editds", "img1.png", ["control", "root"])
        result = manager.set_pair_order("editds", "img1.png", None)
        assert result["role_order"] is None
        assert "role_order" not in ds.media_metadata["img1.png"]["control_info"]

    def test_set_pair_order_rejects_unavailable_slot(self, manager):
        _make_edit_dataset(manager, controls=("img1",), targets=("img1",))
        with pytest.raises(ValueError, match="control_3"):
            manager.set_pair_order("editds", "img1.png", ["control_3", "root"])

    def test_set_pair_order_unknown_media_raises(self, manager):
        _make_edit_dataset(manager, controls=("img1",), targets=("img1",))
        with pytest.raises(ValueError, match="not found"):
            manager.set_pair_order("editds", "ghost.png", ["root"])

    def test_apply_pair_order_all_counts(self, manager):
        # img1 has a control → applies; img2 has none → skipped.
        _make_edit_dataset(manager, controls=("img1",), targets=("img1", "img2"))
        result = manager.apply_pair_order_all("editds", ["control", "root"])
        assert result == {"applied": 1, "skipped": 1}
        meta = manager.datasets["editds"].media_metadata
        assert meta["img1.png"]["control_info"]["role_order"] == ["control", "root"]
        assert (meta["img2.png"].get("control_info") or {}).get("role_order") is None


# ── Route layer ──────────────────────────────────────────────────────────


def _client(manager):
    from app.api.dataset import control_routes

    app = FastAPI()
    app.include_router(control_routes.router, prefix="/api")
    return TestClient(app), control_routes


class TestControlRoutes:
    def test_health_endpoint(self, manager, monkeypatch):
        _make_edit_dataset(manager, controls=("img1",), targets=("img1", "img2"))
        client, mod = _client(manager)
        monkeypatch.setattr(mod, "dataset_manager", manager)

        res = client.get("/api/datasets/editds/control/health")
        assert res.status_code == 200
        body = res.json()
        assert body["target_count"] == 2
        assert body["missing_by_slot"] == {"control": ["img2"]}

    def test_health_404(self, manager, monkeypatch):
        client, mod = _client(manager)
        monkeypatch.setattr(mod, "dataset_manager", manager)
        assert client.get("/api/datasets/ghost/control/health").status_code == 404

    def test_pair_order_patch(self, manager, monkeypatch):
        _make_edit_dataset(manager, controls=("img1",), targets=("img1",))
        client, mod = _client(manager)
        monkeypatch.setattr(mod, "dataset_manager", manager)

        res = client.patch(
            "/api/datasets/editds/images/img1.png/pair-order",
            json={"role_order": ["control", "root"]},
        )
        assert res.status_code == 200
        assert res.json()["role_order"] == ["control", "root"]

    def test_pair_order_patch_invalid_slot_400(self, manager, monkeypatch):
        _make_edit_dataset(manager, controls=("img1",), targets=("img1",))
        client, mod = _client(manager)
        monkeypatch.setattr(mod, "dataset_manager", manager)

        res = client.patch(
            "/api/datasets/editds/images/img1.png/pair-order",
            json={"role_order": ["control_3", "root"]},
        )
        assert res.status_code == 400

    def test_delete_orphans_route(self, manager, monkeypatch):
        ds = _make_edit_dataset(manager, controls=("img1", "ghost"),
                                targets=("img1",))
        client, mod = _client(manager)
        monkeypatch.setattr(mod, "dataset_manager", manager)

        res = client.delete("/api/datasets/editds/control/orphans")
        assert res.status_code == 200
        assert res.json() == {"deleted": 1}
        assert not os.path.exists(os.path.join(ds.path, "control", "ghost.jpg"))
        assert os.path.exists(os.path.join(ds.path, "control", "img1.jpg"))

    def test_apply_all_route(self, manager, monkeypatch):
        _make_edit_dataset(manager, controls=("img1",), targets=("img1", "img2"))
        client, mod = _client(manager)
        monkeypatch.setattr(mod, "dataset_manager", manager)

        res = client.post(
            "/api/datasets/editds/pair-order/apply-all",
            json={"role_order": ["control", "root"]},
        )
        assert res.status_code == 200
        assert res.json() == {"applied": 1, "skipped": 1}


# ── Slot upload ──────────────────────────────────────────────────────────


class TestSlotUpload:
    def _upload_client(self, manager, monkeypatch):
        from app.api.dataset import crud_routes

        app = FastAPI()
        app.include_router(crud_routes.router, prefix="/api")
        monkeypatch.setattr(crud_routes, "dataset_manager", manager)
        return TestClient(app)

    def test_upload_into_slot_renames_to_target_stem(self, manager, monkeypatch):
        ds = _make_edit_dataset(manager, controls=(), targets=("img1",))
        client = self._upload_client(manager, monkeypatch)

        res = client.post(
            "/api/datasets/editds/upload",
            files={"file": ("before_pic.jpg", b"\xff\xd8\xff\xdb fake", "image/jpeg")},
            data={"slot": "1", "target_stem": "img1"},
        )
        assert res.status_code == 200
        assert res.json()["filename"] == "control/img1.jpg"
        assert os.path.exists(os.path.join(ds.path, "control", "img1.jpg"))

    def test_upload_slot_requires_existing_target(self, manager, monkeypatch):
        _make_edit_dataset(manager, controls=(), targets=("img1",))
        client = self._upload_client(manager, monkeypatch)

        res = client.post(
            "/api/datasets/editds/upload",
            files={"file": ("x.jpg", b"data", "image/jpeg")},
            data={"slot": "1", "target_stem": "nope"},
        )
        assert res.status_code == 400

    def test_upload_slot_updates_control_metadata(self, manager, monkeypatch):
        ds = _make_edit_dataset(manager, controls=(), targets=("img1",))
        client = self._upload_client(manager, monkeypatch)

        # Real (decodable) image so dims land in control_info.
        import io
        buf = io.BytesIO()
        Image.new("RGB", (32, 48), "blue").save(buf, format="JPEG")
        client.post(
            "/api/datasets/editds/upload",
            files={"file": ("any.jpg", buf.getvalue(), "image/jpeg")},
            data={"slot": "1", "target_stem": "img1"},
        )
        meta = ds.media_metadata["img1.png"]
        assert meta["control_count"] == 1
        assert meta["control_info"]["slots"]["control"]["rel_path"] == "control/img1.jpg"

    def test_upload_slot_replaces_other_extension(self, manager, monkeypatch):
        ds = _make_edit_dataset(manager, controls=(), targets=("img1",))
        _create_image(os.path.join(ds.path, "control", "img1.png"))
        manager.scan_dataset("editds")
        client = self._upload_client(manager, monkeypatch)

        client.post(
            "/api/datasets/editds/upload",
            files={"file": ("new.jpg", b"jpgdata", "image/jpeg")},
            data={"slot": "1", "target_stem": "img1"},
        )
        assert os.path.exists(os.path.join(ds.path, "control", "img1.jpg"))
        assert not os.path.exists(os.path.join(ds.path, "control", "img1.png"))

    def test_plain_upload_unchanged(self, manager, monkeypatch):
        ds = _make_edit_dataset(manager, controls=(), targets=("img1",))
        client = self._upload_client(manager, monkeypatch)

        res = client.post(
            "/api/datasets/editds/upload",
            files={"file": ("img9.png", b"data", "image/png")},
        )
        assert res.status_code == 200
        assert res.json()["filename"] == "img9.png"
        assert os.path.exists(os.path.join(ds.path, "img9.png"))


# ── Control reassignment (re-match an on-disk orphan to a target) ─────────


class TestControlAssign:
    def test_assign_orphan_renames_and_pairs(self, manager, monkeypatch):
        # control/ghost.jpg has no target stem "ghost" → orphan.
        ds = _make_edit_dataset(manager, controls=("ghost",), targets=("img1",))
        client, mod = _client(manager)
        monkeypatch.setattr(mod, "dataset_manager", manager)

        res = client.post(
            "/api/datasets/editds/control/assign",
            json={
                "slot": 1,
                "src_rel_path": "control/ghost.jpg",
                "target_stem": "img1",
            },
        )
        assert res.status_code == 200
        assert res.json()["rel_path"] == "control/img1.jpg"
        assert os.path.exists(os.path.join(ds.path, "control", "img1.jpg"))
        assert not os.path.exists(os.path.join(ds.path, "control", "ghost.jpg"))
        # Targeted metadata refresh ran for the new target stem.
        assert ds.media_metadata["img1.png"]["control_count"] == 1

    def test_assign_can_move_between_slots(self, manager, monkeypatch):
        ds = _make_edit_dataset(manager, controls=("ghost",), targets=("img1",))
        client, mod = _client(manager)
        monkeypatch.setattr(mod, "dataset_manager", manager)

        res = client.post(
            "/api/datasets/editds/control/assign",
            json={
                "slot": 2,
                "src_rel_path": "control/ghost.jpg",
                "target_stem": "img1",
            },
        )
        assert res.status_code == 200
        assert res.json()["rel_path"] == "control_2/img1.jpg"
        assert os.path.exists(os.path.join(ds.path, "control_2", "img1.jpg"))

    def test_assign_unknown_target_400(self, manager, monkeypatch):
        _make_edit_dataset(manager, controls=("ghost",), targets=("img1",))
        client, mod = _client(manager)
        monkeypatch.setattr(mod, "dataset_manager", manager)

        res = client.post(
            "/api/datasets/editds/control/assign",
            json={
                "slot": 1,
                "src_rel_path": "control/ghost.jpg",
                "target_stem": "nope",
            },
        )
        assert res.status_code == 400

    def test_assign_missing_source_400(self, manager, monkeypatch):
        _make_edit_dataset(manager, controls=(), targets=("img1",))
        client, mod = _client(manager)
        monkeypatch.setattr(mod, "dataset_manager", manager)

        res = client.post(
            "/api/datasets/editds/control/assign",
            json={
                "slot": 1,
                "src_rel_path": "control/missing.jpg",
                "target_stem": "img1",
            },
        )
        assert res.status_code == 400

    def test_assign_non_control_source_400(self, manager, monkeypatch):
        # Root target image is not a control file → cannot be reassigned.
        _make_edit_dataset(manager, controls=(), targets=("img1",))
        client, mod = _client(manager)
        monkeypatch.setattr(mod, "dataset_manager", manager)

        res = client.post(
            "/api/datasets/editds/control/assign",
            json={"slot": 1, "src_rel_path": "img1.png", "target_stem": "img1"},
        )
        assert res.status_code == 400

    def test_assign_rejects_traversal_403(self, manager, monkeypatch):
        _make_edit_dataset(manager, controls=("ghost",), targets=("img1",))
        client, mod = _client(manager)
        monkeypatch.setattr(mod, "dataset_manager", manager)

        res = client.post(
            "/api/datasets/editds/control/assign",
            json={
                "slot": 1,
                "src_rel_path": "../secret.jpg",
                "target_stem": "img1",
            },
        )
        assert res.status_code == 403

    def test_assign_404_unknown_dataset(self, manager, monkeypatch):
        client, mod = _client(manager)
        monkeypatch.setattr(mod, "dataset_manager", manager)

        res = client.post(
            "/api/datasets/ghostds/control/assign",
            json={
                "slot": 1,
                "src_rel_path": "control/ghost.jpg",
                "target_stem": "img1",
            },
        )
        assert res.status_code == 404


# ── Thumbnail namespacing ────────────────────────────────────────────────


class TestThumbnailNamespacing:
    def test_subdir_thumbnail_does_not_collide(self, tmp_path):
        ds = str(tmp_path)
        root_thumb = thumbnail_path_for(ds, "img1.png")
        ctl_thumb = thumbnail_path_for(ds, "control/img1.jpg")
        assert root_thumb != ctl_thumb
        # Root naming is unchanged (existing thumbnails stay valid).
        assert root_thumb.name == "img1.webp"

    def test_ensure_thumbnail_for_control_image(self, tmp_path):
        ds = str(tmp_path)
        _create_image(os.path.join(ds, "img1.png"), color="red")
        _create_image(os.path.join(ds, "control", "img1.jpg"), color="blue")

        root_thumb = ensure_thumbnail(ds, "img1.png")
        ctl_thumb = ensure_thumbnail(ds, "control/img1.jpg")
        assert root_thumb is not None and ctl_thumb is not None
        assert root_thumb != ctl_thumb
        assert ctl_thumb.exists()
