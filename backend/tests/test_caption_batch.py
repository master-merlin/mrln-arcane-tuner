import pytest

from app.core.tasks.task_manager import task_manager
from app.core.captioning import caption_batch


class StubService:
    unloaded = False

    def generate_caption(self, image_path, model_id, params):
        return f"caption for {image_path}"

    @classmethod
    def unload_models(cls):
        cls.unloaded = True


@pytest.fixture(autouse=True)
def no_loop():
    task_manager.set_loop(None)


def test_worker_writes_and_unloads(tmp_path, monkeypatch):
    StubService.unloaded = False
    writes = []
    emits = []

    monkeypatch.setattr(caption_batch, "_get_service", lambda: StubService())
    monkeypatch.setattr(caption_batch.CaptionService, "unload_models", StubService.unload_models)
    monkeypatch.setattr(caption_batch, "_full_path", lambda ds, rel: f"/fake/{ds}/{rel}")
    monkeypatch.setattr(caption_batch, "_write_caption",
                        lambda ds, rel, text, target: writes.append((rel, text)))
    monkeypatch.setattr(caption_batch, "_emit_caption_written", lambda **kw: emits.append(kw))

    t = task_manager.create(type="caption_batch", title="x", total=2, dataset_name="ds")
    caption_batch.run_caption_batch(
        t.id, dataset_name="ds", image_rel_paths=["a.png", "b.png"],
        model_id="m", params={}, system_prompt=None, target="original",
    )

    assert [w[0] for w in writes] == ["a.png", "b.png"]
    assert task_manager.get(t.id).status.value == "completed"
    assert task_manager.get(t.id).ok == 2
    assert StubService.unloaded is True            # finally ran
    assert len(emits) == 2


def test_worker_cancel_midway(tmp_path, monkeypatch):
    StubService.unloaded = False
    writes = []

    monkeypatch.setattr(caption_batch, "_get_service", lambda: StubService())
    monkeypatch.setattr(caption_batch.CaptionService, "unload_models", StubService.unload_models)
    monkeypatch.setattr(caption_batch, "_full_path", lambda ds, rel: f"/fake/{ds}/{rel}")
    monkeypatch.setattr(caption_batch, "_emit_caption_written", lambda **kw: None)

    t = task_manager.create(type="caption_batch", title="x", total=3, dataset_name="ds")

    def write_then_cancel(ds, rel, text, target):
        writes.append(rel)
        task_manager.cancel(t.id)      # cancel after first write

    monkeypatch.setattr(caption_batch, "_write_caption", write_then_cancel)

    caption_batch.run_caption_batch(
        t.id, dataset_name="ds", image_rel_paths=["a.png", "b.png", "c.png"],
        model_id="m", params={}, system_prompt=None, target="original",
    )

    assert writes == ["a.png"]                      # stopped after first
    assert task_manager.get(t.id).status.value == "cancelled"
    assert StubService.unloaded is True


def test_worker_masked_uses_masked_source(tmp_path, monkeypatch):
    StubService.unloaded = False
    masked_calls = []
    full_calls = []

    monkeypatch.setattr(caption_batch, "_get_service", lambda: StubService())
    monkeypatch.setattr(caption_batch.CaptionService, "unload_models", StubService.unload_models)
    monkeypatch.setattr(caption_batch, "_masked_path",
                        lambda ds, rel: masked_calls.append(rel) or f"/masked/{rel}")
    monkeypatch.setattr(caption_batch, "_full_path",
                        lambda ds, rel: full_calls.append(rel) or f"/orig/{rel}")
    monkeypatch.setattr(caption_batch, "_write_caption", lambda ds, rel, text, target: None)
    monkeypatch.setattr(caption_batch, "_emit_caption_written", lambda **kw: None)

    t = task_manager.create(type="caption_batch", title="x", total=1, dataset_name="ds")
    caption_batch.run_caption_batch(
        t.id, dataset_name="ds", image_rel_paths=["a.png"],
        model_id="m", params={}, system_prompt=None, target="masked",
    )

    assert masked_calls == ["a.png"]      # masked composite used as source
    assert full_calls == []               # original NOT used for masked target
    assert task_manager.get(t.id).status.value == "completed"
