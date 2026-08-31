"""The Harmonize confirm dialog must describe what ``harmonize_files`` does.

RULE-20 class **C** (contract) guard for the LANE-54 finding: from 2026-07-09
(``117918e2``) to 2026-08-31 the Analyze modal's Harmonize confirm promised it
"crops them to the majority aspect ratio". ``DatasetManager.harmonize_files``
has never contained a crop or a resize. The sweep commit that introduced the
sentence even announced it as a correction — "message corrected: converts to
JPG + renames + crops" — so no reviewer, and no test, could catch it: nothing
in the repository connected the words of a dialog to the code they describe.

That is the gap this file closes, and it deliberately closes it in ONE file
with BOTH halves, because two files drift:

* Part A asserts, behaviourally, what ``harmonize_files`` really does to a
  dataset on disk — including that pixel dimensions come out unchanged.
* Part B asserts that the confirm copy in ``analyze.component.ts`` says exactly
  those things and no more.

The lock works in both directions. Add cropping to ``harmonize_files`` and
``test_harmonize_preserves_pixel_dimensions`` goes red, forcing the author to
delete the "not resized or cropped" sentence — at which point Part B goes red
too and the copy has to be rewritten deliberately. Edit the copy without
touching the code and Part B alone goes red.

Every assertion here states a POSITIVE fact (this behaviour happens / this
phrase is present), so there is no empty-offender-list to control for
(CONVENTIONS "Tests" #11). The regex extraction in Part B is itself covered:
it must yield a non-empty message containing a known anchor token, so a
drifted pattern fails loudly instead of passing vacuously.

Audio is covered next door by ``test_dataset_manager_audio.py``
(``test_harmonize_skips_audio_pairs``); this file pins that the dialog SAYS so.
It lives in the Python suite for the reason given in
``test_frontend_integer_axis_guard.py``: it needs filesystem access, and the
frontend tsconfig has no ``@types/node``.
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
ANALYZE_TS = REPO_ROOT / "frontend" / "src" / "app" / "modals" / "analyze" / "analyze.component.ts"

# The one golden shared by both halves of this file. Part A renames a dataset
# with this display name and asserts the file that lands on disk; Part B asserts
# the dialog shows the same stem. `analyze.component.ts::harmonizedStem` is the
# TypeScript echo of `re.sub(r'[^a-zA-Z0-9]+', '_', name).strip('_').lower()`.
GOLDEN_DATASET_NAME = "Aston Martin Valkyrie"
GOLDEN_FIRST_FILE = "aston_martin_valkyrie_00001.jpg"


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


def _image(path: str, width: int, height: int) -> None:
    Image.new("RGB", (width, height), "red").save(path)


# ══ Part A — what harmonize_files actually does ══════════════════════════


class TestHarmonizeBehaviour:
    def test_harmonize_preserves_pixel_dimensions(self, manager):
        """No crop, no resize — the dialog is allowed to promise this.

        Two deliberately different aspect ratios: a majority-AR crop (what the
        dialog claimed for seven weeks) would have to change at least one of
        them. Both come out at their input size.
        """
        ds = manager.create_dataset(GOLDEN_DATASET_NAME)
        _image(os.path.join(ds.path, "wide.png"), 160, 90)
        _image(os.path.join(ds.path, "tall.png"), 90, 160)

        manager.scan_dataset(GOLDEN_DATASET_NAME)
        manager.harmonize_files(GOLDEN_DATASET_NAME)

        sizes = sorted(
            Image.open(os.path.join(ds.path, f)).size
            for f in os.listdir(ds.path)
            if f.endswith(".jpg")
        )
        assert sizes == [(90, 160), (160, 90)], (
            "harmonize_files changed image geometry — the Analyze confirm "
            "dialog says 'Pixels are not resized or cropped', so either the "
            "resize is a bug or the dialog copy must be rewritten (LANE-54)."
        )

    def test_harmonize_converts_to_jpg_and_renames_to_canonical_stem(self, manager):
        ds = manager.create_dataset(GOLDEN_DATASET_NAME)
        _image(os.path.join(ds.path, "zzz_original.png"), 64, 64)

        manager.scan_dataset(GOLDEN_DATASET_NAME)
        result = manager.harmonize_files(GOLDEN_DATASET_NAME)

        assert result["converted"] == 1
        assert result["renamed"] == 1
        assert not os.path.exists(os.path.join(ds.path, "zzz_original.png"))
        assert os.path.exists(os.path.join(ds.path, GOLDEN_FIRST_FILE)), (
            f"expected the canonical first name {GOLDEN_FIRST_FILE!r}; "
            f"found {sorted(f for f in os.listdir(ds.path) if f.endswith('.jpg'))}"
        )

    def test_harmonize_renames_caption_mask_masked_and_control_sidecars(self, manager):
        """The four sidecar kinds the dialog names, each proven to move."""
        ds = manager.create_dataset(GOLDEN_DATASET_NAME)
        for sub in ("masks", "masked", "control", "control_2"):
            os.makedirs(os.path.join(ds.path, sub), exist_ok=True)

        _image(os.path.join(ds.path, "aaa.png"), 64, 64)
        with open(os.path.join(ds.path, "aaa.txt"), "w", encoding="utf-8") as f:
            f.write("a caption")
        _image(os.path.join(ds.path, "masks", "aaa.png"), 64, 64)
        _image(os.path.join(ds.path, "masked", "aaa.jpg"), 64, 64)
        with open(os.path.join(ds.path, "masked", "aaa.txt"), "w", encoding="utf-8") as f:
            f.write("masked caption")
        _image(os.path.join(ds.path, "control", "aaa.png"), 64, 64)
        _image(os.path.join(ds.path, "control_2", "aaa.jpg"), 64, 64)

        manager.scan_dataset(GOLDEN_DATASET_NAME)
        manager.harmonize_files(GOLDEN_DATASET_NAME)

        stem = GOLDEN_FIRST_FILE[: -len(".jpg")]
        assert os.path.exists(os.path.join(ds.path, f"{stem}.txt")), "caption not renamed"
        assert os.path.exists(os.path.join(ds.path, "masks", f"{stem}.png")), "mask not renamed"
        assert os.path.exists(os.path.join(ds.path, "masked", f"{stem}.jpg")), (
            "masked copy not renamed"
        )
        assert os.path.exists(os.path.join(ds.path, "masked", f"{stem}.txt")), (
            "masked caption not renamed"
        )
        # Controls keep their own extension — renamed only, never converted.
        assert os.path.exists(os.path.join(ds.path, "control", f"{stem}.png")), (
            "control slot not renamed"
        )
        assert os.path.exists(os.path.join(ds.path, "control_2", f"{stem}.jpg")), (
            "control_2 slot not renamed"
        )
        assert not os.path.exists(os.path.join(ds.path, "control", "aaa.png"))


# ══ Part B — what the confirm dialog says ════════════════════════════════


def _harmonize_confirm_message() -> str:
    """Extract the Harmonize confirm ``message`` from the Analyze modal.

    Grabs everything between ``title: `Harmonize`` and the ``confirmLabel``
    that closes the same options object, then strips the TypeScript template
    plumbing so the assertions below read against prose.
    """
    src = ANALYZE_TS.read_text(encoding="utf-8")
    block = re.search(
        r"title: `Harmonize.*?confirmLabel:",
        src,
        re.DOTALL,
    )
    assert block, (
        "could not find the Harmonize confirm options object in "
        f"{ANALYZE_TS} — this guard's extraction has drifted, fix it here "
        "rather than deleting the test"
    )
    text = block.group(0)
    text = re.sub(r"\$\{[^}]*\}", "<stem>", text)  # interpolations
    text = text.replace("`", "").replace("' +", "").replace("+", " ")
    return re.sub(r"\s+", " ", text)


@pytest.mark.skipif(not ANALYZE_TS.is_file(), reason="frontend sources not present")
class TestHarmonizeConfirmCopy:
    def test_extraction_finds_the_dialog(self):
        """Positive control for the regex in ``_harmonize_confirm_message``.

        Without this, a drifted pattern would make every assertion below run
        against an empty string and the phrase checks would fail for the wrong
        reason (or, if they were written as absence checks, pass forever).
        """
        msg = _harmonize_confirm_message()
        assert len(msg) > 80, f"extracted message implausibly short: {msg!r}"
        assert "Harmonize" in msg
        assert "message:" in msg

    def test_copy_promises_no_geometry_change(self):
        """Pinned against ``test_harmonize_preserves_pixel_dimensions``.

        The dialog is allowed to say pixels are untouched only because that
        test proves it. If cropping is ever added, that test fails first and
        this sentence must go with it.
        """
        msg = _harmonize_confirm_message()
        assert "Pixels are not resized or cropped" in msg, (
            "the Harmonize dialog no longer tells the user their images keep "
            "their geometry. If harmonize_files now crops or resizes, "
            "test_harmonize_preserves_pixel_dimensions is the test to change "
            "first; if it does not, restore the sentence."
        )
        # The exact claim that was false for seven weeks, in the form it took.
        assert "majority aspect ratio" not in msg
        assert "crops them" not in msg

    @pytest.mark.parametrize(
        "phrase, proven_by",
        [
            ("converted to JPG", "test_harmonize_converts_to_jpg_and_renames_to_canonical_stem"),
            ("_00001.jpg", "test_harmonize_converts_to_jpg_and_renames_to_canonical_stem"),
            ("caption", "test_harmonize_renames_caption_mask_masked_and_control_sidecars"),
            ("mask", "test_harmonize_renames_caption_mask_masked_and_control_sidecars"),
            ("masked copy", "test_harmonize_renames_caption_mask_masked_and_control_sidecars"),
            ("control images", "test_harmonize_renames_caption_mask_masked_and_control_sidecars"),
            ("Audio pairs are left untouched", "test_harmonize_skips_audio_pairs (audio suite)"),
            ("cannot be undone", "harmonize_files rewrites files in place"),
        ],
    )
    def test_copy_names_each_operation_the_code_performs(self, phrase, proven_by):
        msg = _harmonize_confirm_message()
        assert phrase in msg, (
            f"the Harmonize dialog no longer mentions {phrase!r}, but the code "
            f"still does it — see {proven_by}"
        )
