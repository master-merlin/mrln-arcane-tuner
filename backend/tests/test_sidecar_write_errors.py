"""Sidecar writes must fail as 4xx, never as an unhandled 500.

Both cases below were found by the traversal-containment work and are
pre-existing: neither is a regression from the client-side URL encoding, since
`{filename:path}` converters passed slashes through raw before that change too.

1. **Nested sidecar.** Captions live beside their image, so a dataset holding
   `sub/shot.png` writes `sub/shot.txt`. Nothing created `sub/` on the write
   path, so an ordinary dataset layout raised `FileNotFoundError` -> HTTP 500.
   This is the one that matters: images in sub-directories are normal, not
   exotic, so this was a live 500 on an ordinary workflow.

2. **Filename the filesystem refuses.** Windows forbids `< > : " | ? *` in a
   name. A legitimately-encoded `query?x=1.png` arrives intact -- which is
   correct, the client must not truncate it -- and then cannot be stored. That
   is a bad request, not a server fault.

Both were "a 500 where a 4xx belongs": the failure was neither silent nor
handled, and a 500 tells a user the server is broken when their input was
simply nested or invalid.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.core.dataset_manager import DatasetManager


@pytest.fixture()
def mock_settings():
    inst = MagicMock()
    inst.get_module_settings.return_value = {}
    inst.update_module_settings = MagicMock()
    with patch("app.core.dataset_manager.get_settings_manager", return_value=inst):
        yield inst


@pytest.fixture()
def manager(tmp_path, mock_settings):
    default_root = tmp_path / "datasets"
    default_root.mkdir(parents=True, exist_ok=True)

    with patch.object(DatasetManager, "__init__", lambda self, **kw: None):
        mgr = DatasetManager()
    mgr.root_dir = str(tmp_path)
    mgr.storage_file = str(tmp_path / "dataset_locations.json")
    mgr.default_root = str(default_root)
    mgr.settings_manager = mock_settings
    mgr.datasets = {}
    mgr._loop = None
    mgr._db = MagicMock()
    mgr._dataset_repo = MagicMock()
    mgr._media_repo = MagicMock()
    return mgr


def test_caption_in_subdirectory_is_written_not_500(manager, tmp_path):
    """The headline case: a caption for an image in a sub-folder must save."""
    ds = manager.create_dataset("nested")

    manager.save_caption(ds.name, "sub/shot.txt", "a caption")

    written = os.path.join(ds.path, "sub", "shot.txt")
    assert os.path.isfile(written), "the sidecar directory was not created"
    with open(written, encoding="utf-8") as f:
        assert f.read() == "a caption"
    assert manager.read_caption(ds.name, "sub/shot.txt") == "a caption"


def test_deeply_nested_caption_is_written(manager):
    """More than one missing level must also work -- `parents=True`, not one mkdir."""
    ds = manager.create_dataset("deep")
    manager.save_caption(ds.name, "a/b/c/shot.txt", "deep caption")
    assert manager.read_caption(ds.name, "a/b/c/shot.txt") == "deep caption"


def test_nested_lyrics_is_written(manager):
    """save_lyrics shares the write path, so it must share the fix.

    Pinned separately because the two were duplicated line-for-line before, and
    a fix applied to only one of them is exactly how they drift apart again.
    """
    ds = manager.create_dataset("nested_lyrics")
    manager.save_lyrics(ds.name, "sub/song.lyrics.txt", "verse one")
    assert manager.read_caption(ds.name, "sub/song.lyrics.txt") == "verse one"


def test_created_directory_stays_inside_the_dataset_root(manager):
    """Creating parents must not become a way to build directories elsewhere.

    Containment is established by `validate_path_within` BEFORE the write
    helper runs, so a traversal is refused before any directory is created.
    Pinned because the helper now has a side effect on the filesystem that it
    did not have before -- the guard order is load-bearing.
    """
    ds = manager.create_dataset("contained")
    outside = os.path.join(os.path.dirname(ds.path), "evil")

    with pytest.raises(HTTPException) as exc:
        manager.save_caption(ds.name, "../evil/x.txt", "pwned")

    assert exc.value.status_code == 403
    assert not os.path.exists(outside), "a traversal created a directory outside the root"


@pytest.mark.skipif(os.name != "nt", reason="'?' is only illegal on Windows")
def test_filesystem_illegal_filename_is_400_not_500(manager):
    """An unstorable name is a bad request, not a server fault."""
    ds = manager.create_dataset("illegal")

    with pytest.raises(HTTPException) as exc:
        manager.save_caption(ds.name, "query?x=1.txt", "content")

    assert exc.value.status_code == 400
    # The message must name the file, or the user cannot tell which of a batch
    # of captions failed.
    assert "query?x=1.txt" in str(exc.value.detail)


def test_ordinary_caption_still_round_trips(manager):
    """Prove the negative: the fix did not break the flat, common case."""
    ds = manager.create_dataset("flat")
    manager.save_caption(ds.name, "shot.txt", "plain caption")
    assert manager.read_caption(ds.name, "shot.txt") == "plain caption"
    assert os.path.isfile(os.path.join(ds.path, "shot.txt"))
