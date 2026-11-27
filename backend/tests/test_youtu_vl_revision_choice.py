"""Which reviewed Youtu-VL revision a load reads.

``load()`` reads one of the reviewed upstream commits: the newest whose
snapshot the hub cache holds whole, else the newest, fetched. A cache filled
before the newest commit existed holds only the older hash, and must still
caption offline with no download and no download indicator. ``refs/main`` is
never followed: it may name a commit nobody reviewed.

Every test runs on an empty hub cache and writes empty placeholder files: the
choice reads file presence only, never file content.
"""
import re
from unittest.mock import MagicMock, patch

import pytest

from app.api.events import download_progress as dp
from app.core.captioning.models.youtu_vl import (
    YOUTU_VL_LOAD_FILES,
    YOUTU_VL_REVISION,
    YOUTU_VL_REVISIONS,
    YoutuVLModel,
    resolve_youtu_vl_revision,
)
from tests.support.youtu_vl_doubles import persistent_rope_model, youtu_vl_hub_doubles

pytestmark = pytest.mark.usefixtures("empty_hub_cache")

# Written out, not imported: moving the reviewed list must touch this test too.
NEW = "68d50d82df573c7282ecc1f06f1c9473252c9e72"
OLD = "8d30a0e49662a1d628a472b12df264dbcd768753"
UNREVIEWED = "c" * 40


def _repo(cache):
    return cache / "models--tencent--Youtu-VL-4B-Instruct"


def _write_snapshot(cache, revision, names):
    snapshot = _repo(cache) / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in names:
        (snapshot / name).write_bytes(b"")


def _without(name):
    return [n for n in YOUTU_VL_LOAD_FILES if n != name]


def _point_refs_main(cache, revision):
    refs = _repo(cache) / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision, encoding="utf-8")


def _load():
    """Run `load()` on the hub doubles; return (doubles, emitted statuses)."""
    emitted = []
    with patch.object(dp, "schedule_emit_from_thread", lambda p: emitted.append(p)), \
         youtu_vl_hub_doubles(persistent_rope_model()) as hub:
        YoutuVLModel(service=MagicMock()).load()
    return hub, [p.status for p in emitted]


def _hub_calls(hub):
    return {
        "AutoConfig.from_pretrained": hub.config_cls.from_pretrained,
        "get_class_from_dynamic_module": hub.get_class,
        "AutoModelForCausalLM.from_pretrained": hub.model_cls.from_pretrained,
        "AutoProcessor.from_pretrained": hub.processor_cls.from_pretrained,
    }


def _assert_every_hub_call_reads(hub, revision):
    for label, mock in _hub_calls(hub).items():
        mock.assert_called_once()
        assert mock.call_args.kwargs.get("revision") == revision, label


def test_the_reviewed_revisions_are_full_hashes_newest_first():
    assert YOUTU_VL_REVISIONS == (NEW, OLD)
    for revision in YOUTU_VL_REVISIONS:
        assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert YOUTU_VL_REVISION == NEW
    assert len(YOUTU_VL_LOAD_FILES) == 17


def test_only_the_older_snapshot_cached_loads_it_on_every_hub_call_with_no_indicator(empty_hub_cache):
    _write_snapshot(empty_hub_cache, OLD, YOUTU_VL_LOAD_FILES)

    assert resolve_youtu_vl_revision() == (OLD, True)
    hub, statuses = _load()
    _assert_every_hub_call_reads(hub, OLD)
    assert statuses == []


def test_both_snapshots_cached_loads_the_newest(empty_hub_cache):
    _write_snapshot(empty_hub_cache, NEW, YOUTU_VL_LOAD_FILES)
    _write_snapshot(empty_hub_cache, OLD, YOUTU_VL_LOAD_FILES)

    assert resolve_youtu_vl_revision() == (NEW, True)
    hub, statuses = _load()
    _assert_every_hub_call_reads(hub, NEW)
    assert statuses == []


def test_nothing_cached_fetches_the_newest_and_shows_the_indicator():
    assert resolve_youtu_vl_revision() == (NEW, False)
    hub, statuses = _load()
    _assert_every_hub_call_reads(hub, NEW)
    assert statuses == ["starting", "complete"]


def test_a_partial_newest_snapshot_falls_back_to_the_complete_older_one(empty_hub_cache):
    _write_snapshot(empty_hub_cache, NEW, _without("model-00003-of-00003.safetensors"))
    _write_snapshot(empty_hub_cache, OLD, YOUTU_VL_LOAD_FILES)

    assert resolve_youtu_vl_revision() == (OLD, True)

    # The partial snapshot alone: it has a config.json, so a cache check on
    # the repo would call it cached and hide the download about to happen.
    old = _repo(empty_hub_cache) / "snapshots" / OLD
    for name in YOUTU_VL_LOAD_FILES:
        (old / name).unlink()
    old.rmdir()

    assert resolve_youtu_vl_revision() == (NEW, False)
    hub, statuses = _load()
    _assert_every_hub_call_reads(hub, NEW)
    assert statuses == ["starting", "complete"]


def test_an_unreviewed_hash_in_refs_main_is_ignored(empty_hub_cache):
    _write_snapshot(empty_hub_cache, UNREVIEWED, YOUTU_VL_LOAD_FILES)
    _point_refs_main(empty_hub_cache, UNREVIEWED)

    assert resolve_youtu_vl_revision() == (NEW, False)
    alone, _ = _load()

    _write_snapshot(empty_hub_cache, OLD, YOUTU_VL_LOAD_FILES)
    assert resolve_youtu_vl_revision() == (OLD, True)
    with_old, _ = _load()

    for hub in (alone, with_old):
        for label, mock in _hub_calls(hub).items():
            for call in mock.call_args_list:
                seen = (*call.args, *call.kwargs.values())
                assert UNREVIEWED not in seen, label
                assert "main" not in seen, label


@pytest.mark.parametrize("missing", YOUTU_VL_LOAD_FILES)
def test_each_missing_load_file_makes_a_snapshot_incomplete(empty_hub_cache, missing):
    _write_snapshot(empty_hub_cache, NEW, _without(missing))

    assert resolve_youtu_vl_revision() == (NEW, False)
