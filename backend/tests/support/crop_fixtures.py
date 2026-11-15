"""Shared pieces of the crop containment and error tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image


def bare_manager(tmp_path: Path):
    """A ``DatasetManager`` rooted in *tmp_path* with settings and the DB mocked."""
    from app.core.dataset_manager import DatasetManager

    settings = MagicMock()
    settings.get_module_settings.return_value = {}
    default_root = tmp_path / "datasets"
    default_root.mkdir(exist_ok=True)
    with patch("app.core.dataset_manager.get_settings_manager", return_value=settings), \
         patch.object(DatasetManager, "__init__", lambda self, **kw: None):
        mgr = DatasetManager()
    mgr.root_dir = str(tmp_path)
    mgr.storage_file = str(tmp_path / "dataset_locations.json")
    mgr.default_root = str(default_root)
    mgr.settings_manager = settings
    mgr.datasets = {}
    mgr._loop = None
    mgr._db = MagicMock()
    mgr._dataset_repo = MagicMock()
    mgr._media_repo = MagicMock()
    return mgr


def image(path: Path, size: tuple[int, int] = (64, 48), color: str = "blue") -> None:
    """Write a real image, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def assert_no_path_in(text: str, tmp_path: Path) -> None:
    """No form of the test tree's location leaks into a response body."""
    unescaped = text.replace("\\\\", "\\").lower()
    resolved = tmp_path.resolve()
    for form in {str(tmp_path), str(resolved), tmp_path.as_posix(),
                 resolved.as_posix(), tmp_path.name}:
        assert form.lower() not in unescaped, form
