"""`with_progress` must not flash the download bar on a cache hit.

Loading already-downloaded model shards from disk into VRAM is NOT a download,
so emitting starting/complete download-progress events is misleading. When a
caller passes `repo_id` and that repo is already cached, with_progress yields
without emitting. Without `repo_id` (or on a cache miss) it emits as before.
"""

from unittest.mock import patch

from app.api.events import download_progress as dp


def test_with_progress_suppresses_emits_when_repo_cached():
    emitted = []
    with patch.object(dp, "schedule_emit_from_thread", lambda p: emitted.append(p)), \
         patch.object(dp, "_is_repo_cached", lambda repo, revision=None: True):
        with dp.with_progress(model_id="Qwen/Qwen3-VL", category="caption", repo_id="Qwen/Qwen3-VL"):
            pass
    assert emitted == [], "cache hit must not emit download-progress events"


def test_with_progress_emits_when_repo_not_cached():
    emitted = []
    with patch.object(dp, "schedule_emit_from_thread", lambda p: emitted.append(p)), \
         patch.object(dp, "_is_repo_cached", lambda repo, revision=None: False):
        with dp.with_progress(model_id="Qwen/Qwen3-VL", category="caption", repo_id="Qwen/Qwen3-VL"):
            pass
    statuses = [p.status for p in emitted]
    assert statuses == ["starting", "complete"]


def test_a_pinned_snapshot_counts_as_cached_without_refs_main(empty_hub_cache):
    # A loader pinned to a commit reads that snapshot whatever refs/main says,
    # so the cache check must ask for the same revision.
    pinned, other = "a" * 40, "b" * 40
    repo = empty_hub_cache / "models--org--name"
    (repo / "snapshots" / pinned).mkdir(parents=True)
    (repo / "snapshots" / pinned / "config.json").write_text("{}", encoding="utf-8")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(other, encoding="utf-8")

    assert dp._is_repo_cached("org/name", pinned) is True
    assert dp._is_repo_cached("org/name") is False

    emitted = []
    with patch.object(dp, "schedule_emit_from_thread", lambda p: emitted.append(p)):
        with dp.with_progress(model_id="name", category="caption", repo_id="org/name", revision=pinned):
            pass
    assert emitted == [], "a cached pinned snapshot must not emit download-progress events"


def test_with_progress_emits_when_repo_id_omitted():
    # No repo_id → no cache check → always emit (single-file fetches etc.).
    emitted = []
    with patch.object(dp, "schedule_emit_from_thread", lambda p: emitted.append(p)):
        with dp.with_progress(model_id="facebook/sam3/merges.txt", category="mask"):
            pass
    statuses = [p.status for p in emitted]
    assert statuses == ["starting", "complete"]
