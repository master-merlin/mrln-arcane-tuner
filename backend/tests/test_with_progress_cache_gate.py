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
         patch.object(dp, "_is_repo_cached", lambda repo: True):
        with dp.with_progress(model_id="Qwen/Qwen3-VL", category="caption", repo_id="Qwen/Qwen3-VL"):
            pass
    assert emitted == [], "cache hit must not emit download-progress events"


def test_with_progress_emits_when_repo_not_cached():
    emitted = []
    with patch.object(dp, "schedule_emit_from_thread", lambda p: emitted.append(p)), \
         patch.object(dp, "_is_repo_cached", lambda repo: False):
        with dp.with_progress(model_id="Qwen/Qwen3-VL", category="caption", repo_id="Qwen/Qwen3-VL"):
            pass
    statuses = [p.status for p in emitted]
    assert statuses == ["starting", "complete"]


def test_with_progress_emits_when_repo_id_omitted():
    # No repo_id → no cache check → always emit (single-file fetches etc.).
    emitted = []
    with patch.object(dp, "schedule_emit_from_thread", lambda p: emitted.append(p)):
        with dp.with_progress(model_id="facebook/sam3/merges.txt", category="mask"):
            pass
    statuses = [p.status for p in emitted]
    assert statuses == ["starting", "complete"]
