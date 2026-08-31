"""Every "delete this image" dialog must name everything the delete removes.

RULE-20 class **C** (contract) guard, the second instance of the mechanism in
``test_harmonize_confirm_contract.py`` — proof that it generalises past the one
dialog that prompted it (LANE-54).

Three call sites open a confirm for the SAME endpoint
(``DELETE /api/datasets/{name}/pairs/{filename}`` ->
``DatasetManager.delete_media_pair``), and until 2026-08-31 they disagreed about
what it destroys. The Analyze modal said "caption and any masks"; the dataset
workspace said only "This image and its caption" — while the code has always
also deleted the masked copy and every control-slot image. That is the same
defect class as the harmonize over-claim pointing the other way, and it is the
worse direction: a user is told less will be destroyed than actually is.

Part A proves what ``delete_media_pair`` really removes. Part B asserts each of
the three dialogs names all of it. Both halves are positive facts (this file is
gone / this phrase is present), so there is no empty-offender list to control
for (CONVENTIONS "Tests" #11); the source-scanning half carries its own control
in ``test_extraction_finds_every_dialog``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.core.dataset_manager import DatasetManager

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "frontend" / "src" / "app"

# Every site that opens a confirm for delete_media_pair, with the anchor that
# identifies its message. Adding a fourth caller means adding it here.
DELETE_PAIR_DIALOGS = [
    (SRC / "modals" / "analyze" / "analyze.component.ts", "deleteDuplicate"),
    (SRC / "modals" / "analyze" / "analyze.component.ts", "deleteFile"),
    (SRC / "modals" / "similar-images" / "similar-images.component.ts", "similar-images delete"),
    (SRC / "workspace" / "dataset-workspace.component.ts", "workspace delete-pair"),
]

# What delete_media_pair removes, proven in Part A, in the words the dialogs use.
REQUIRED_PHRASES = ["caption", "mask", "masked copy", "control images"]


# ── Fixtures (mirroring test_dataset_manager.py) ─────────────────────────


@pytest.fixture()
def mock_settings():
    mock_instance = MagicMock()
    mock_instance.get_module_settings.return_value = {}
    mock_instance.update_module_settings = MagicMock()
    with patch("app.core.dataset_manager.get_settings_manager", return_value=mock_instance):
        yield mock_instance


@pytest.fixture()
def manager(tmp_path, mock_settings):
    default_root = str(tmp_path / "datasets")
    os.makedirs(default_root, exist_ok=True)

    with patch.object(DatasetManager, "__init__", lambda self, **kw: None):
        mgr = DatasetManager()

    mgr.root_dir = str(tmp_path)
    mgr.storage_file = str(tmp_path / "dataset_locations.json")
    mgr.default_root = default_root
    mgr.settings_manager = mock_settings
    mgr.datasets = {}
    mgr._loop = None
    mgr._db = MagicMock()
    mgr._dataset_repo = MagicMock()
    mgr._media_repo = MagicMock()
    return mgr


# ══ Part A — what delete_media_pair actually removes ═════════════════════


def test_delete_media_pair_removes_every_sidecar_the_dialogs_name(manager):
    ds = manager.create_dataset("delset")
    for sub in ("masks", "masked", "control", "control_2"):
        os.makedirs(os.path.join(ds.path, sub), exist_ok=True)

    Image.new("RGB", (32, 32), "red").save(os.path.join(ds.path, "a.jpg"))
    sidecars = {
        "caption": os.path.join(ds.path, "a.txt"),
        "mask": os.path.join(ds.path, "masks", "a.png"),
        "masked copy": os.path.join(ds.path, "masked", "a.jpg"),
        "masked caption": os.path.join(ds.path, "masked", "a.txt"),
        "control": os.path.join(ds.path, "control", "a.png"),
        "control_2": os.path.join(ds.path, "control_2", "a.jpg"),
    }
    for kind, path in sidecars.items():
        if path.endswith(".txt"):
            Path(path).write_text("x", encoding="utf-8")
        else:
            Image.new("RGB", (32, 32), "blue").save(path)

    manager.scan_dataset("delset")
    manager.delete_media_pair("delset", "a.jpg")

    assert not os.path.exists(os.path.join(ds.path, "a.jpg"))
    for kind, path in sidecars.items():
        assert not os.path.exists(path), (
            f"{kind} survived delete_media_pair — if that is now intended, the "
            f"three confirm dialogs in DELETE_PAIR_DIALOGS must stop naming it"
        )


# ══ Part B — what the three dialogs say ══════════════════════════════════


def _delete_messages() -> list[tuple[str, str]]:
    """Every `message:` in a confirm whose title asks about deleting an image."""
    found: list[tuple[str, str]] = []
    for path in {p for p, _ in DELETE_PAIR_DIALOGS}:
        if not path.is_file():
            continue
        src = path.read_text(encoding="utf-8")
        for m in re.finditer(
            r"title: '(?:Delete this image\?|Delete this entry\?)',\s*\n\s*message:(.*?)confirmLabel:",
            src,
            re.DOTALL,
        ):
            text = re.sub(r"\$\{[^}]*\}", "<path>", m.group(1))
            text = text.replace("`", "").replace("'", "").replace("+", " ")
            found.append((path.name, re.sub(r"\s+", " ", text).strip()))
    return found


@pytest.mark.skipif(not SRC.is_dir(), reason="frontend sources not present")
def test_extraction_finds_every_dialog():
    """Positive control: the scan below is not looking at an empty list."""
    msgs = _delete_messages()
    assert len(msgs) == len(DELETE_PAIR_DIALOGS), (
        f"expected {len(DELETE_PAIR_DIALOGS)} delete-pair confirms, extracted "
        f"{len(msgs)}: {msgs} — the extraction has drifted, or a call site was "
        f"added/removed without updating DELETE_PAIR_DIALOGS"
    )
    for name, msg in msgs:
        assert len(msg) > 40, f"{name}: implausibly short message {msg!r}"


@pytest.mark.skipif(not SRC.is_dir(), reason="frontend sources not present")
@pytest.mark.parametrize("phrase", REQUIRED_PHRASES)
def test_every_delete_dialog_names_every_sidecar(phrase):
    for name, msg in _delete_messages():
        assert phrase in msg, (
            f"{name} does not warn that the {phrase} is deleted too, but "
            f"delete_media_pair removes it — see "
            f"test_delete_media_pair_removes_every_sidecar_the_dialogs_name. "
            f"Message: {msg!r}"
        )
