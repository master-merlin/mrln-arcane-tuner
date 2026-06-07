import pytest

from app.core.tasks.task_manager import task_manager
from app.core.masking import mask_apply_batch


@pytest.fixture(autouse=True)
def no_loop():
    task_manager.set_loop(None)


def test_apply_runs_mass_apply_reconciles_and_summarizes(monkeypatch):
    calls = {}

    def fake_mass_apply(path, opacity, overwrite, progress_callback):
        assert path == "/ds" and overwrite is True
        progress_callback(1, 2, "a")
        progress_callback(2, 2, "b")
        return {"applied": 2, "skipped": 0, "missing_masks": ["x.jpg"]}

    monkeypatch.setattr(mask_apply_batch, "_dataset_path", lambda n: "/ds")
    monkeypatch.setattr(mask_apply_batch, "_mass_apply", fake_mass_apply)
    monkeypatch.setattr(mask_apply_batch, "_reconcile_has_masked",
                        lambda n: calls.__setitem__("reconciled", n))
    monkeypatch.setattr(mask_apply_batch, "_emit_apply_summary",
                        lambda **kw: calls.__setitem__("summary", kw))

    t = task_manager.create(type="mask_apply_batch", title="x", total=2, dataset_name="ds")
    mask_apply_batch.run_mask_apply_batch(
        t.id, dataset_name="ds", opacity=0.0, overwrite=True,
    )

    task = task_manager.get(t.id)
    assert task.status.value == "completed"
    assert task.current == 2
    # "<n> done" reads task.ok — must reflect applied, not stay 0 (regression).
    assert task.ok == 2
    assert calls["reconciled"] == "ds"
    assert calls["summary"]["applied"] == 2
    assert calls["summary"]["missing_masks_count"] == 1


def test_apply_ok_reconciles_to_applied_not_visited(monkeypatch):
    """When some pairs are skipped, "done" shows the applied count, not the
    number of pairs visited by the progress bar."""
    def fake_mass_apply(path, opacity, overwrite, progress_callback):
        progress_callback(1, 3, "a")
        progress_callback(2, 3, "b")
        progress_callback(3, 3, "c")            # visited 3...
        return {"applied": 2, "skipped": 1, "missing_masks": []}  # ...applied 2

    monkeypatch.setattr(mask_apply_batch, "_dataset_path", lambda n: "/ds")
    monkeypatch.setattr(mask_apply_batch, "_mass_apply", fake_mass_apply)
    monkeypatch.setattr(mask_apply_batch, "_reconcile_has_masked", lambda n: None)
    monkeypatch.setattr(mask_apply_batch, "_emit_apply_summary", lambda **kw: None)

    t = task_manager.create(type="mask_apply_batch", title="x", total=3, dataset_name="ds")
    mask_apply_batch.run_mask_apply_batch(
        t.id, dataset_name="ds", opacity=0.0, overwrite=False,
    )
    task = task_manager.get(t.id)
    assert task.current == 3
    assert task.ok == 2


def test_apply_setup_error_fails(monkeypatch):
    def boom(n):
        raise ValueError("no dataset")
    monkeypatch.setattr(mask_apply_batch, "_dataset_path", boom)

    t = task_manager.create(type="mask_apply_batch", title="x", total=0, dataset_name="ds")
    mask_apply_batch.run_mask_apply_batch(
        t.id, dataset_name="ds", opacity=0.0, overwrite=False,
    )

    task = task_manager.get(t.id)
    assert task.status.value == "failed"
    assert "no dataset" in (task.error or "")


def test_reconcile_has_masked_sets_and_clears(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from app.core.dataset_manager import dataset_manager as dm

    masked = tmp_path / "masked"
    masked.mkdir()
    (masked / "a.jpg").write_bytes(b"x")          # a has a masked composite

    ds = SimpleNamespace(
        path=str(tmp_path),
        media_metadata={
            "a.png": {"has_masked": False},        # should flip True
            "b.png": {"has_masked": True},         # should flip False (no file)
        },
    )
    persisted = {"v": False}
    monkeypatch.setattr(dm, "get_dataset", lambda n: ds)
    monkeypatch.setattr(dm, "_persist_dataset", lambda d: persisted.__setitem__("v", True))

    mask_apply_batch._reconcile_has_masked("ds")

    assert ds.media_metadata["a.png"]["has_masked"] is True
    assert ds.media_metadata["b.png"]["has_masked"] is False
    assert persisted["v"] is True
