"""PR7 — control-image pair production: degradation batch + overlay→control.

Three layers, tested bottom-up:
- ``apply_degradations`` — pure PIL op pipeline (grayscale/blur/downscale/noise)
- ``generate_controls`` — disk core: write controls into a slot, skip/overwrite
- the ``/control/generate-batch`` route + ``run_control_batch`` worker (lane,
  metadata refresh, summary event)
- ``/overlay/commit`` with ``target`` = a control slot (non-destructive save)
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.core.dataset import control_batch
from app.core.dataset.control_batch import apply_degradations, generate_controls
from app.core.dataset.control_helpers import prepare_control_slot_path
from app.core.dataset_manager import DatasetManager


def _img(path: str, w: int = 32, h: int = 32, color="red"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (w, h), color).save(path)


def _checkerboard(path: str, w: int = 32, h: int = 32):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (w, h), "black")
    px = img.load()
    for y in range(h):
        for x in range(w):
            if (x + y) % 2 == 0:
                px[x, y] = (255, 255, 255)
    img.save(path)


# ── Pure degradation ops ─────────────────────────────────────────────────


class TestApplyDegradations:
    def test_grayscale_equalizes_channels(self):
        img = Image.new("RGB", (8, 8), (200, 50, 10))
        out = apply_degradations(img, [{"type": "grayscale"}])
        r, g, b = out.getpixel((0, 0))
        assert r == g == b

    def test_blur_changes_an_edge(self):
        img = Image.new("RGB", (16, 16), "black")
        for y in range(16):  # vertical white half → sharp edge
            for x in range(8, 16):
                img.putpixel((x, y), (255, 255, 255))
        out = apply_degradations(img, [{"type": "blur", "params": {"radius": 2.0}}])
        # The edge pixel is no longer pure black/white after blurring.
        assert out.getpixel((7, 8)) != (0, 0, 0)

    def test_downscale_preserves_size_but_loses_detail(self, tmp_path):
        p = str(tmp_path / "cb.png")
        _checkerboard(p, 32, 32)
        with Image.open(p) as img:
            src = img.convert("RGB")
            out = apply_degradations(src, [{"type": "downscale", "params": {"factor": 4}}])
        assert out.size == src.size  # buckets with the target
        assert list(out.getdata()) != list(src.getdata())  # detail destroyed

    def test_noise_is_seed_deterministic(self):
        img = Image.new("RGB", (8, 8), (128, 128, 128))
        a = apply_degradations(img, [{"type": "noise", "params": {"sigma": 0.1, "seed": 7}}])
        b = apply_degradations(img, [{"type": "noise", "params": {"sigma": 0.1, "seed": 7}}])
        assert list(a.getdata()) == list(b.getdata())
        assert list(a.getdata()) != [(128, 128, 128)] * 64

    def test_op_chain_applies_in_order(self):
        img = Image.new("RGB", (8, 8), (200, 50, 10))
        out = apply_degradations(
            img, [{"type": "grayscale"}, {"type": "blur", "params": {"radius": 1}}]
        )
        r, g, b = out.getpixel((0, 0))
        assert r == g == b  # grayscale survived the subsequent blur

    def test_unknown_op_raises(self):
        with pytest.raises(ValueError, match="unknown degradation op"):
            apply_degradations(Image.new("RGB", (4, 4)), [{"type": "bogus"}])


# ── Disk core: generate_controls ─────────────────────────────────────────


class TestGenerateControls:
    def test_writes_control_into_slot(self, tmp_path):
        ds = str(tmp_path)
        _img(os.path.join(ds, "img1.png"))
        summary = generate_controls(
            ds, [("img1.png", "img1")],
            slot_index=1, ops=[{"type": "grayscale"}],
        )
        assert summary["ok"] == 1
        assert summary["written"] == ["control/img1.png"]
        assert os.path.exists(os.path.join(ds, "control", "img1.png"))

    def test_slot_2_targets_control_2_dir(self, tmp_path):
        ds = str(tmp_path)
        _img(os.path.join(ds, "img1.png"))
        generate_controls(ds, [("img1.png", "img1")], slot_index=2,
                          ops=[{"type": "grayscale"}])
        assert os.path.exists(os.path.join(ds, "control_2", "img1.png"))

    def test_skips_existing_without_overwrite(self, tmp_path):
        ds = str(tmp_path)
        _img(os.path.join(ds, "img1.png"))
        first = generate_controls(ds, [("img1.png", "img1")], slot_index=1,
                                  ops=[{"type": "grayscale"}])
        assert first["ok"] == 1
        second = generate_controls(ds, [("img1.png", "img1")], slot_index=1,
                                   ops=[{"type": "grayscale"}])
        assert second == {"ok": 0, "skipped": 1, "failed": 0, "written": []}

    def test_overwrite_regenerates(self, tmp_path):
        ds = str(tmp_path)
        _img(os.path.join(ds, "img1.png"))
        generate_controls(ds, [("img1.png", "img1")], slot_index=1,
                          ops=[{"type": "grayscale"}])
        again = generate_controls(ds, [("img1.png", "img1")], slot_index=1,
                                  ops=[{"type": "grayscale"}], overwrite=True)
        assert again["ok"] == 1 and again["skipped"] == 0

    def test_missing_source_counts_as_failed(self, tmp_path):
        ds = str(tmp_path)
        summary = generate_controls(ds, [("ghost.png", "ghost")], slot_index=1,
                                    ops=[{"type": "grayscale"}])
        assert summary == {"ok": 0, "skipped": 0, "failed": 1, "written": []}

    def test_cancellation_stops_the_loop(self, tmp_path):
        ds = str(tmp_path)
        for s in ("a", "b", "c"):
            _img(os.path.join(ds, f"{s}.png"))
        calls = {"n": 0}

        def cancel_after_first():
            done = calls["n"] >= 1
            calls["n"] += 1
            return done

        summary = generate_controls(
            ds, [("a.png", "a"), ("b.png", "b"), ("c.png", "c")],
            slot_index=1, ops=[{"type": "grayscale"}],
            is_cancelled=cancel_after_first,
        )
        assert summary["ok"] == 1  # only the first item ran


def test_prepare_control_slot_path_purges_sibling_ext(tmp_path):
    ds = str(tmp_path)
    os.makedirs(os.path.join(ds, "control"))
    _img(os.path.join(ds, "control", "img1.jpg"))
    dest = prepare_control_slot_path(ds, 1, "img1", ".png")
    assert dest.endswith(os.path.join("control", "img1.png"))
    assert not os.path.exists(os.path.join(ds, "control", "img1.jpg"))


# ── Route + worker ─────────────────────────────────────────────────────────


class _StubDataset:
    def __init__(self, path, media_metadata):
        self.path = path
        self.media_metadata = media_metadata


class _StubManager:
    def __init__(self, ds):
        self._ds = ds
        self.refreshed: list[str] = []

    def get_dataset(self, name):
        return self._ds

    def refresh_control_metadata(self, name, stem):
        self.refreshed.append(stem)


class TestGenerateBatchRoute:
    def _setup(self, tmp_path, monkeypatch, stems=("img1", "img2")):
        ds_root = str(tmp_path / "ds")
        os.makedirs(ds_root)
        for s in stems:
            _img(os.path.join(ds_root, f"{s}.png"))
        stub = _StubManager(_StubDataset(ds_root, {f"{s}.png": {} for s in stems}))

        from app.api.dataset import control_routes

        monkeypatch.setattr(control_routes, "dataset_manager", stub)
        monkeypatch.setattr("app.core.dataset_manager.dataset_manager", stub)

        captured: dict = {}

        def fake_enqueue(task_id, worker, *, lane="gpu"):
            captured.update(task_id=task_id, worker=worker, lane=lane)

        monkeypatch.setattr(control_routes.task_manager, "enqueue", fake_enqueue)

        app = FastAPI()
        app.include_router(control_routes.router, prefix="/api")
        return TestClient(app), stub, ds_root, captured

    def test_route_enqueues_on_background_lane_and_runs(self, tmp_path, monkeypatch):
        client, stub, ds_root, captured = self._setup(tmp_path, monkeypatch)
        emitted: dict = {}
        monkeypatch.setattr(
            control_batch, "_emit_control_summary",
            lambda **kw: emitted.update(kw),
        )

        res = client.post(
            "/api/datasets/ds/control/generate-batch",
            json={"slot": 1, "ops": [{"type": "grayscale"}]},
        )
        assert res.status_code == 200
        assert "task_id" in res.json()
        assert captured["lane"] == "background"

        captured["worker"](captured["task_id"])  # run worker inline (deterministic)

        assert os.path.exists(os.path.join(ds_root, "control", "img1.png"))
        assert os.path.exists(os.path.join(ds_root, "control", "img2.png"))
        assert set(stub.refreshed) == {"img1", "img2"}
        assert emitted["ok"] == 2 and emitted["slot"] == 1

    def test_route_honors_stems_subset(self, tmp_path, monkeypatch):
        client, stub, ds_root, captured = self._setup(tmp_path, monkeypatch)
        monkeypatch.setattr(control_batch, "_emit_control_summary", lambda **kw: None)

        res = client.post(
            "/api/datasets/ds/control/generate-batch",
            json={"slot": 1, "ops": [{"type": "grayscale"}], "stems": ["img1"]},
        )
        assert res.status_code == 200
        captured["worker"](captured["task_id"])
        assert os.path.exists(os.path.join(ds_root, "control", "img1.png"))
        assert not os.path.exists(os.path.join(ds_root, "control", "img2.png"))

    def test_empty_ops_rejected(self, tmp_path, monkeypatch):
        client, *_ = self._setup(tmp_path, monkeypatch)
        res = client.post(
            "/api/datasets/ds/control/generate-batch",
            json={"slot": 1, "ops": []},
        )
        assert res.status_code == 422

    def test_bad_slot_rejected(self, tmp_path, monkeypatch):
        client, *_ = self._setup(tmp_path, monkeypatch)
        res = client.post(
            "/api/datasets/ds/control/generate-batch",
            json={"slot": 9, "ops": [{"type": "grayscale"}]},
        )
        assert res.status_code == 422


# ── Overlay → control slot (commit target) ──────────────────────────────────


@pytest.fixture()
def mock_settings():
    inst = MagicMock()
    inst.get_module_settings.return_value = {}
    inst.update_module_settings = MagicMock()
    with patch(
        "app.core.dataset_manager.get_settings_manager", return_value=inst,
    ):
        yield inst


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


class TestOverlayCommitToControl:
    def _overlay_client(self, manager, monkeypatch):
        from app.api.dataset import overlay_routes

        monkeypatch.setattr(overlay_routes, "dataset_manager", manager)
        app = FastAPI()
        app.include_router(overlay_routes.router, prefix="/api")
        return TestClient(app)

    def _edit_ds_with_overlay(self, manager):
        ds = manager.create_dataset("editds", kind="edit")
        _img(os.path.join(ds.path, "img1.png"), color="red")
        manager.scan_dataset("editds")
        os.makedirs(os.path.join(ds.path, "overlays"), exist_ok=True)
        _img(os.path.join(ds.path, "overlays", "img1.png"), color="blue")
        return ds

    def test_commit_to_control_is_non_destructive(self, manager, monkeypatch):
        ds = self._edit_ds_with_overlay(manager)
        ds.media_metadata["img1.png"]["has_mask"] = True
        client = self._overlay_client(manager, monkeypatch)

        res = client.post(
            "/api/datasets/editds/overlay/commit",
            json={"image_path": "img1.png", "target": "control"},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["status"] == "saved_to_control"
        assert body["file"] == "control/img1.png"

        # Control written (blue overlay), original untouched (still red).
        ctl = os.path.join(ds.path, "control", "img1.png")
        assert os.path.exists(ctl)
        assert Image.open(ctl).convert("RGB").getpixel((0, 0)) == (0, 0, 255)
        assert Image.open(os.path.join(ds.path, "img1.png")).convert("RGB").getpixel(
            (0, 0)
        ) == (255, 0, 0)
        # Overlay is NOT consumed (unlike a bake-to-original).
        assert os.path.exists(os.path.join(ds.path, "overlays", "img1.png"))
        # Masks NOT invalidated; control metadata refreshed on the pair.
        assert ds.media_metadata["img1.png"]["has_mask"] is True
        assert ds.media_metadata["img1.png"]["control_count"] == 1

    def test_commit_to_control_2_slot(self, manager, monkeypatch):
        ds = self._edit_ds_with_overlay(manager)
        client = self._overlay_client(manager, monkeypatch)
        res = client.post(
            "/api/datasets/editds/overlay/commit",
            json={"image_path": "img1.png", "target": "control_2"},
        )
        assert res.status_code == 200
        assert res.json()["file"] == "control_2/img1.png"
        assert os.path.exists(os.path.join(ds.path, "control_2", "img1.png"))

    def test_commit_missing_overlay_404(self, manager, monkeypatch):
        manager.create_dataset("editds", kind="edit")
        _img(os.path.join(manager.datasets["editds"].path, "img1.png"))
        manager.scan_dataset("editds")
        client = self._overlay_client(manager, monkeypatch)
        res = client.post(
            "/api/datasets/editds/overlay/commit",
            json={"image_path": "img1.png", "target": "control"},
        )
        assert res.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
