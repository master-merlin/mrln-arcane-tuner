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
    assert task_manager.get(t.id).status.value == "completed"
    assert summaries == [{"dataset_name": "ds", "processed": 3, "converted": 1, "renamed": 3}]


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
