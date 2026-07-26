"""W1.T6: caption/lyrics IO must resolve through validate_path_within.

Pre-fix:
- ``read_caption`` had NO containment check at all — any ``filename``
  (including one with ``../`` segments) was joined onto ``dataset.path``
  and opened directly.
- ``save_caption`` / ``save_lyrics`` used a naive
  ``os.path.abspath(path).startswith(os.path.abspath(dataset.path))``
  check, which is vulnerable to prefix-collision: a dataset rooted at
  ``.../datasets/foo`` "contains" ``.../datasets/foobar`` under a plain
  string-prefix test (no separator boundary check), even though
  ``foobar`` is a sibling directory, not a subdirectory of ``foo``.

All three now resolve the target through the shared
``validate_path_within`` guard (``app/api/_path_guard.py``), which raises
``HTTPException(403)`` on escape and returns the resolved ``Path`` used
for the actual open/write.

Fixture mirrors ``test_dataset_manager.py`` / ``test_dataset_manager_audio.py``:
a ``DatasetManager`` rooted in ``tmp_path`` with the DB layer mocked out.
"""

from __future__ import annotations

import os

import pytest
from fastapi import HTTPException
from unittest.mock import MagicMock, patch

from app.core.dataset_manager import DatasetManager


# ── Fixtures (mirrors test_dataset_manager.py / test_dataset_manager_audio.py) ──


@pytest.fixture()
def mock_settings():
    """Patch SettingsManager so DatasetManager never touches real disk."""
    mock_instance = MagicMock()
    mock_instance.get_module_settings.return_value = {}
    mock_instance.update_module_settings = MagicMock()
    with patch(
        "app.core.dataset_manager.get_settings_manager", return_value=mock_instance
    ):
        yield mock_instance


@pytest.fixture()
def manager(tmp_path, mock_settings):
    """Create a DatasetManager rooted in tmp_path."""
    storage_file = str(tmp_path / "dataset_locations.json")
    default_root = str(tmp_path / "datasets")
    os.makedirs(default_root, exist_ok=True)

    with patch.object(DatasetManager, "__init__", lambda self, **kw: None):
        mgr = DatasetManager()

    mgr.root_dir = str(tmp_path)
    mgr.storage_file = storage_file
    mgr.default_root = default_root
    mgr.settings_manager = mock_settings
    mgr.datasets = {}
    mgr._loop = None
    mgr._db = MagicMock()
    mgr._dataset_repo = MagicMock()
    mgr._media_repo = MagicMock()
    return mgr


# ── read_caption: no guard at all pre-fix ───────────────────────────────


def test_read_caption_rejects_traversal(manager):
    """A "../" filename must not be able to read a file sitting next to
    (outside) the dataset directory."""
    ds = manager.create_dataset("foo")
    secret_path = os.path.join(os.path.dirname(ds.path), "secret.txt")
    with open(secret_path, "w", encoding="utf-8") as f:
        f.write("top secret")

    with pytest.raises(HTTPException):
        manager.read_caption(ds.name, "../secret.txt")


# ── save_caption / save_lyrics: prefix-collision pre-fix ───────────────


def test_save_caption_rejects_prefix_sibling(manager):
    """'.../datasets/foo' must not authorize writes into the sibling
    '.../datasets/foobar' — the classic startswith() prefix-collision bug.
    """
    ds = manager.create_dataset("foo")
    sibling_dir = os.path.join(os.path.dirname(ds.path), "foobar")
    os.makedirs(sibling_dir, exist_ok=True)

    with pytest.raises(HTTPException):
        manager.save_caption(ds.name, "../foobar/x.txt", "pwned")

    assert not os.path.exists(os.path.join(sibling_dir, "x.txt"))


def test_save_lyrics_rejects_prefix_sibling(manager):
    """Same prefix-collision guard, exercised on save_lyrics's sibling path."""
    ds = manager.create_dataset("foo")
    sibling_dir = os.path.join(os.path.dirname(ds.path), "foobar")
    os.makedirs(sibling_dir, exist_ok=True)

    with pytest.raises(HTTPException):
        manager.save_lyrics(ds.name, "../foobar/x.lyrics.txt", "pwned")

    assert not os.path.exists(os.path.join(sibling_dir, "x.lyrics.txt"))


# ── Sanity: legitimate round-trips are unaffected ──────────────────────


def test_save_and_read_caption_roundtrip_still_works(manager):
    ds = manager.create_dataset("roundtrip")
    manager.save_caption(ds.name, "img.txt", "a caption")
    assert manager.read_caption(ds.name, "img.txt") == "a caption"


def test_save_and_read_lyrics_roundtrip_still_works(manager):
    ds = manager.create_dataset("roundtrip_lyrics")
    manager.save_lyrics(ds.name, "song.lyrics.txt", "verse one")
    assert manager.read_caption(ds.name, "song.lyrics.txt") == "verse one"
