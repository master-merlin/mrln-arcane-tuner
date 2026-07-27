"""Tests for DatasetManager.update_media_flags (W4.T14).

Replaces ~10 route-side copies of "hand-edit dataset.media_metadata[key] in
place, then call _persist_media_item_async" — each site mutated the shared
in-memory dict on the event loop with no lock, so two concurrent requests on
the SAME item (e.g. a mask delete racing an upscale) could interleave
field-by-field and persist a torn snapshot. update_media_flags takes the
manager's mutation lock across the whole mutate+persist sequence instead.
"""

import threading

import pytest

from app.core.dataset_manager import Dataset, dataset_manager


def _make_dataset(name: str) -> Dataset:
    return Dataset(
        id=f"{name}-id",
        name=name,
        path=f"/tmp/{name}",
        created_at=0.0,
        media_metadata={"a.jpg": {"field_a": 0, "field_b": 0}},
    )


@pytest.fixture
def seeded_dataset():
    ds = _make_dataset("umf-test")
    dataset_manager.datasets["umf-test"] = ds
    yield ds
    dataset_manager.datasets.pop("umf-test", None)


def test_update_media_flags_unknown_dataset_raises_value_error():
    with pytest.raises(ValueError):
        dataset_manager.update_media_flags("does-not-exist-ds", "a.jpg", field_a=1)


def test_update_media_flags_unknown_item_raises_value_error(seeded_dataset):
    with pytest.raises(ValueError):
        dataset_manager.update_media_flags("umf-test", "missing.jpg", field_a=1)


def test_update_media_flags_sets_fields_and_persists(seeded_dataset, monkeypatch):
    persisted = []
    monkeypatch.setattr(
        dataset_manager,
        "_persist_media_item",
        lambda dataset, rel_path: persisted.append(
            dict(dataset.media_metadata[rel_path])
        ),
    )

    dataset_manager.update_media_flags("umf-test", "a.jpg", field_a=5, field_c="x")

    assert seeded_dataset.media_metadata["a.jpg"]["field_a"] == 5
    assert seeded_dataset.media_metadata["a.jpg"]["field_c"] == "x"
    assert len(persisted) == 1
    assert persisted[0]["field_a"] == 5
    assert persisted[0]["field_c"] == "x"


def test_update_media_flags_normalizes_backslash_lookup_key(seeded_dataset, monkeypatch):
    monkeypatch.setattr(dataset_manager, "_persist_media_item", lambda dataset, rel_path: None)

    dataset_manager.update_media_flags("umf-test", "a.jpg".replace("/", "\\"), field_a=9)

    assert seeded_dataset.media_metadata["a.jpg"]["field_a"] == 9


def test_update_media_flags_remove_field_pops_key(seeded_dataset, monkeypatch):
    monkeypatch.setattr(dataset_manager, "_persist_media_item", lambda dataset, rel_path: None)
    seeded_dataset.media_metadata["a.jpg"]["mask_info"] = {"x": 1}

    dataset_manager.update_media_flags(
        "umf-test", "a.jpg",
        mask_info=dataset_manager.REMOVE_FIELD, has_mask=False,
    )

    assert "mask_info" not in seeded_dataset.media_metadata["a.jpg"]
    assert seeded_dataset.media_metadata["a.jpg"]["has_mask"] is False


@pytest.mark.asyncio
async def test_update_media_flags_async_delegates_to_sync(seeded_dataset, monkeypatch):
    monkeypatch.setattr(dataset_manager, "_persist_media_item", lambda dataset, rel_path: None)

    await dataset_manager.update_media_flags_async("umf-test", "a.jpg", field_a=42)

    assert seeded_dataset.media_metadata["a.jpg"]["field_a"] == 42


def test_update_media_flags_derive_reads_meta_under_the_lock(seeded_dataset, monkeypatch):
    """``derive`` (W4 Finding 2 fix) must be invoked with the item's LIVE
    meta dict, evaluated after the lock is acquired — a caller that needs
    to seed a field's value from the item's own current state (e.g.
    commit_overlay carrying overlay_dimensions into width/height) must not
    have to read that state itself before calling this method, since that
    reopens the exact race this method exists to close.
    """
    monkeypatch.setattr(dataset_manager, "_persist_media_item", lambda dataset, rel_path: None)
    seeded_dataset.media_metadata["a.jpg"]["field_a"] = 7

    seen_meta = {}

    def _derive(meta):
        seen_meta.update(meta)
        return {"field_b": meta["field_a"] * 10}

    dataset_manager.update_media_flags("umf-test", "a.jpg", derive=_derive)

    assert seen_meta["field_a"] == 7
    assert seeded_dataset.media_metadata["a.jpg"]["field_b"] == 70


def test_update_media_flags_explicit_changes_win_over_derived(seeded_dataset, monkeypatch):
    """Explicit kwargs are the caller's non-derived intent — they must not
    be silently overridden by a same-named field the derive callable also
    produces."""
    monkeypatch.setattr(dataset_manager, "_persist_media_item", lambda dataset, rel_path: None)

    dataset_manager.update_media_flags(
        "umf-test", "a.jpg",
        derive=lambda meta: {"field_a": 999},
        field_a=5,
    )

    assert seeded_dataset.media_metadata["a.jpg"]["field_a"] == 5


def test_update_media_flags_derive_returning_empty_dict_is_a_noop(seeded_dataset, monkeypatch):
    """A derive callable that finds nothing to contribute (e.g.
    commit_overlay's ``overlay_dimensions`` already cleared) must not
    clobber unrelated explicit changes or error."""
    monkeypatch.setattr(dataset_manager, "_persist_media_item", lambda dataset, rel_path: None)

    dataset_manager.update_media_flags(
        "umf-test", "a.jpg",
        derive=lambda meta: {},
        field_a=3,
    )

    assert seeded_dataset.media_metadata["a.jpg"]["field_a"] == 3


def test_update_media_flags_second_caller_blocks_until_first_fully_persists(
    seeded_dataset, monkeypatch,
):
    """RED (pre-fix, hand-mutation + persist with no lock around the pair):
    a second concurrent update on the SAME item could run — mutate AND
    persist — to completion while a first (slow) update was still mid-flight,
    so their SQLite writes race and the slower one can finish last with a
    stale snapshot, silently reverting the faster one's fields. GREEN
    (post-fix): mutate+persist is one atomic critical section under
    ``_media_mutation_lock``, so a second caller cannot even START its own
    mutate+persist until the first's is fully done — deterministically
    proven here via explicit gating (Event), not timing luck."""
    thread_a: dict[str, threading.Thread] = {}
    persist_a_started = threading.Event()
    release_persist_a = threading.Event()

    def _gated_persist(dataset, rel_path):
        if threading.current_thread() is thread_a.get("thread"):
            persist_a_started.set()
            release_persist_a.wait(timeout=5)

    monkeypatch.setattr(dataset_manager, "_persist_media_item", _gated_persist)

    def _call_a():
        dataset_manager.update_media_flags("umf-test", "a.jpg", field_a="A")

    ta = threading.Thread(target=_call_a)
    thread_a["thread"] = ta
    ta.start()
    assert persist_a_started.wait(timeout=5), "thread A never reached persist"

    # While A is parked inside its persist call (and, if the fix works,
    # still holding the mutation lock), fire a second update from another
    # thread and see whether it manages to run to completion regardless.
    b_done = threading.Event()

    def _call_b():
        dataset_manager.update_media_flags("umf-test", "a.jpg", field_b="B")
        b_done.set()

    tb = threading.Thread(target=_call_b)
    tb.start()

    b_finished_while_a_still_blocked = b_done.wait(timeout=0.3)

    release_persist_a.set()
    ta.join(timeout=5)
    tb.join(timeout=5)
    assert not ta.is_alive() and not tb.is_alive(), "threads never finished"

    assert not b_finished_while_a_still_blocked, (
        "a second update_media_flags call completed WHILE the first was "
        "still mid-persist — mutate+persist is not held as one atomic "
        "critical section, so the two calls' writes can interleave/race"
    )
    # Nothing lost: both fields survive regardless of scheduling order.
    assert seeded_dataset.media_metadata["a.jpg"]["field_a"] == "A"
    assert seeded_dataset.media_metadata["a.jpg"]["field_b"] == "B"
