import pytest
from PIL import Image

from app.core.tasks.task_manager import task_manager
from app.core.dataset import harmonize_batch


@pytest.fixture(autouse=True)
def no_loop():
    task_manager.set_loop(None)


def test_harmonize_completes_and_emits_summary(monkeypatch):
    summaries = []
    monkeypatch.setattr(harmonize_batch, "_harmonize",
                        lambda name, cb: {"processed": 3, "converted": 1, "renamed": 3})
    monkeypatch.setattr(harmonize_batch, "_emit_harmonize_summary",
                        lambda **kw: summaries.append(kw))
    t = task_manager.create(type="harmonize", title="x", total=3, dataset_name="ds")
    harmonize_batch.run_harmonize_batch(t.id, dataset_name="ds")
    task = task_manager.get(t.id)
    assert task.status.value == "completed"
    # The Task Center's "<n> done" reads task.ok — it must reflect the processed
    # count, not stay at 0 (regression: harmonize only set `current`).
    assert task.ok == 3
    assert summaries == [{"dataset_name": "ds", "processed": 3, "converted": 1, "renamed": 3}]


def test_harmonize_progress_cb_sets_ok(monkeypatch):
    """Per-file progress mirrors `current` into `ok` so "done" tracks live, and
    the final count reconciles to `processed` (which can be < pairs visited)."""
    def fake_harmonize(name, cb):
        cb(1, 3, "a.jpg")
        cb(2, 3, "b.jpg")
        cb(3, 3, "c.jpg")  # a visited pair errored out → processed (2) < visited (3)
        return {"processed": 2, "converted": 0, "renamed": 2}

    monkeypatch.setattr(harmonize_batch, "_harmonize", fake_harmonize)
    monkeypatch.setattr(harmonize_batch, "_emit_harmonize_summary", lambda **kw: None)
    t = task_manager.create(type="harmonize", title="x", total=3, dataset_name="ds")
    harmonize_batch.run_harmonize_batch(t.id, dataset_name="ds")
    task = task_manager.get(t.id)
    assert task.current == 3          # bar reached 100%
    assert task.ok == 2               # reconciled to processed, not the 3 visited


def test_harmonize_failure_marks_failed(monkeypatch):
    def boom(name, cb):
        raise RuntimeError("disk full")
    monkeypatch.setattr(harmonize_batch, "_harmonize", boom)
    monkeypatch.setattr(harmonize_batch, "_emit_harmonize_summary", lambda **kw: None)
    t = task_manager.create(type="harmonize", title="x", total=1, dataset_name="ds")
    harmonize_batch.run_harmonize_batch(t.id, dataset_name="ds")
    task = task_manager.get(t.id)
    assert task.status.value == "failed"
    assert task.error == "disk full"


def test_harmonize_files_progress_cb_fires_per_pair(monkeypatch, tmp_path):
    """Integration: real harmonize_files on a tiny dataset (1 png + 1 jpg)."""
    from app.core.dataset_manager import Dataset, dataset_manager

    ds_name = "hz-test"
    ds_root = tmp_path / "hz"
    ds_root.mkdir()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(ds_root / "a.png")
    Image.new("RGB", (8, 8), (4, 5, 6)).save(ds_root / "b.jpg")
    ds = Dataset(id="i", name=ds_name, path=str(ds_root), created_at=0.0,
                 media_metadata={"a.png": {}, "b.jpg": {}})
    dataset_manager.datasets[ds_name] = ds
    monkeypatch.setattr(dataset_manager, "scan_dataset", lambda name, *a, **k: None)
    calls = []
    try:
        res = dataset_manager.harmonize_files(ds_name, progress_cb=lambda c, t, f: calls.append((c, t, f)))
        assert [c[0] for c in calls[:2]] == [1, 2]
        assert any(str(f).endswith(".png") for _, _, f in calls)
        assert res["processed"] == 2
        assert res["converted"] == 1
        files = {p.name for p in ds_root.iterdir() if p.suffix == ".jpg"}
        assert len(files) == 2
        assert not (ds_root / "a.png").exists()
    finally:
        dataset_manager.datasets.pop(ds_name, None)


def test_harmonize_renames_control_slot_files(monkeypatch, tmp_path):
    """Edit dataset: harmonize must rename the stem-matched control files in
    control/ control_2/ control_3/ in lockstep with their target so the
    target<->control pairing survives (target stem N -> control stem N).

    Regression: harmonize only renamed root image + caption + masks/masked,
    leaving control slots on their old stems -> every pair broke."""
    from app.core.dataset_manager import Dataset, dataset_manager

    ds_name = "hz-edit"
    ds_root = tmp_path / "hzedit"
    ds_root.mkdir()
    # Two targets (a.png converts to jpg, b.jpg already jpg).
    Image.new("RGB", (8, 8), (1, 2, 3)).save(ds_root / "a.png")
    Image.new("RGB", (8, 8), (4, 5, 6)).save(ds_root / "b.jpg")
    # Control slots, stem-matched, mixed extensions across slots.
    (ds_root / "control").mkdir()
    (ds_root / "control_2").mkdir()
    Image.new("RGB", (8, 8), (9, 9, 9)).save(ds_root / "control" / "a.png")
    Image.new("RGB", (8, 8), (7, 7, 7)).save(ds_root / "control" / "b.jpg")
    Image.new("RGB", (8, 8), (5, 5, 5)).save(ds_root / "control_2" / "a.webp")

    ds = Dataset(id="i", name=ds_name, path=str(ds_root), created_at=0.0,
                 kind="edit", media_metadata={"a.png": {}, "b.jpg": {}})
    dataset_manager.datasets[ds_name] = ds
    monkeypatch.setattr(dataset_manager, "scan_dataset", lambda name, *a, **k: None)
    try:
        dataset_manager.harmonize_files(ds_name)
        base = "hz_edit"  # "hz-edit" -> snake
        # Pairs are sorted by media_file: a -> _00001, b -> _00002.
        # Controls follow the same new stem, each preserving its extension.
        assert (ds_root / "control" / f"{base}_00001.png").exists()
        assert (ds_root / "control" / f"{base}_00002.jpg").exists()
        assert (ds_root / "control_2" / f"{base}_00001.webp").exists()
        # Old-stem control files are gone (no orphans, pairing intact).
        assert not (ds_root / "control" / "a.png").exists()
        assert not (ds_root / "control" / "b.jpg").exists()
        assert not (ds_root / "control_2" / "a.webp").exists()
    finally:
        dataset_manager.datasets.pop(ds_name, None)


def test_harmonize_task_route_enqueues(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api.dataset import analysis_routes

    captured = {}
    monkeypatch.setattr(analysis_routes.dataset_manager, "get_dataset_pairs",
                        lambda name: [{"media_file": "a.jpg"}, {"media_file": "b.jpg"}])

    def fake_enqueue(task_id, worker_fn, *, lane="gpu"):
        captured["task_id"] = task_id
        captured["lane"] = lane
    monkeypatch.setattr(analysis_routes.task_manager, "enqueue", fake_enqueue)

    created = {}
    orig = analysis_routes.task_manager.create
    monkeypatch.setattr(analysis_routes.task_manager, "create",
                        lambda **kw: created.update(kw) or orig(**kw))

    resp = TestClient(app).post("/api/datasets/ds1/harmonize/task")
    assert resp.status_code == 200
    assert resp.json()["task_id"] == captured["task_id"]
    assert captured["lane"] == "gpu"
    assert created["type"] == "harmonize"
    assert created["total"] == 2
