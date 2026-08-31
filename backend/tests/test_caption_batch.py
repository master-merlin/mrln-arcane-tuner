import pytest

from app.core.tasks.task_manager import task_manager
from app.core.captioning import caption_batch


class StubService:
    unloaded = False

    def generate_caption(self, image_path, model_id, params, extra_image_paths=None):
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


def test_worker_api_model_skips_unload_and_injects_abort(monkeypatch):
    """api-* batches must never trigger the global unload (lane safety) and
    must hand the HTTP client a cancellation probe."""
    StubService.unloaded = False
    seen_params = []

    class ApiStub(StubService):
        def generate_caption(self, image_path, model_id, params, extra_image_paths=None):
            seen_params.append(params)
            return "cap"

    monkeypatch.setattr(caption_batch, "_get_service", lambda: ApiStub())
    monkeypatch.setattr(caption_batch.CaptionService, "unload_models", StubService.unload_models)
    monkeypatch.setattr(caption_batch, "_full_path", lambda ds, rel: f"/fake/{ds}/{rel}")
    monkeypatch.setattr(caption_batch, "_write_caption", lambda ds, rel, text, target: None)
    monkeypatch.setattr(caption_batch, "_emit_caption_written", lambda **kw: None)

    t = task_manager.create(type="caption_batch", title="x", total=1, dataset_name="ds")
    caption_batch.run_caption_batch(
        t.id, dataset_name="ds", image_rel_paths=["a.png"],
        model_id="api-openai", params={"model": "gpt-4o"}, system_prompt=None,
        target="original",
    )

    assert task_manager.get(t.id).status.value == "completed"
    assert StubService.unloaded is False           # finally must NOT unload
    assert callable(seen_params[0]["_should_abort"])
    assert seen_params[0]["_should_abort"]() is False  # task not cancelled


def test_worker_api_model_fails_fast_after_consecutive_failures(monkeypatch):
    StubService.unloaded = False

    class FailingStub(StubService):
        def generate_caption(self, image_path, model_id, params, extra_image_paths=None):
            raise RuntimeError("401 bad key")

    monkeypatch.setattr(caption_batch, "_get_service", lambda: FailingStub())
    monkeypatch.setattr(caption_batch.CaptionService, "unload_models", StubService.unload_models)
    monkeypatch.setattr(caption_batch, "_full_path", lambda ds, rel: f"/fake/{ds}/{rel}")
    monkeypatch.setattr(caption_batch, "_write_caption", lambda ds, rel, text, target: None)
    monkeypatch.setattr(caption_batch, "_emit_caption_written", lambda **kw: None)

    rels = [f"{i}.png" for i in range(20)]
    t = task_manager.create(type="caption_batch", title="x", total=len(rels), dataset_name="ds")
    caption_batch.run_caption_batch(
        t.id, dataset_name="ds", image_rel_paths=rels,
        model_id="api-openai", params={"model": "gpt-4o"}, system_prompt=None,
        target="original",
    )

    task = task_manager.get(t.id)
    assert task.status.value == "failed"
    assert "consecutive" in (task.error or "")
    assert task.failed == 5                        # stopped at the 5th failure
    assert StubService.unloaded is False           # api path: still no unload


def test_worker_local_model_failures_do_not_fast_fail_and_still_unload(monkeypatch):
    """The fast-fail and unload-skip are api-only — local batches keep the
    old semantics: ride out failures, always unload in finally."""
    StubService.unloaded = False

    class FailingStub(StubService):
        def generate_caption(self, image_path, model_id, params, extra_image_paths=None):
            raise RuntimeError("CUDA OOM")

    monkeypatch.setattr(caption_batch, "_get_service", lambda: FailingStub())
    monkeypatch.setattr(caption_batch.CaptionService, "unload_models", StubService.unload_models)
    monkeypatch.setattr(caption_batch, "_full_path", lambda ds, rel: f"/fake/{ds}/{rel}")
    monkeypatch.setattr(caption_batch, "_write_caption", lambda ds, rel, text, target: None)
    monkeypatch.setattr(caption_batch, "_emit_caption_written", lambda **kw: None)

    rels = [f"{i}.png" for i in range(7)]
    t = task_manager.create(type="caption_batch", title="x", total=len(rels), dataset_name="ds")
    caption_batch.run_caption_batch(
        t.id, dataset_name="ds", image_rel_paths=rels,
        model_id="florence-2", params={}, system_prompt=None, target="original",
    )

    task = task_manager.get(t.id)
    # No FAST-fail for local: every image is attempted, the run is not aborted
    # after N consecutive failures the way an api-* batch is.
    assert task.failed == 7                        # processed every image
    assert StubService.unloaded is True            # finally still unloads
    # ...but "not fast-failed" is not "succeeded". This assertion read
    # `== "completed"` until LANE-52: a batch in which all 7 items raised
    # reported success, which is what the user hit on the refine batch. The
    # tally decides the outcome (TaskManager.finish_batch).
    assert task.status.value == "failed"
    assert task.error is not None and "CUDA OOM" in task.error
