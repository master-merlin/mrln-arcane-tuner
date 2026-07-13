"""
Tests for first-class AUDIO media support in the dataset layer (C0):
scanner classification + metadata extraction, ``get_dataset_pairs``
``media_type``, the lyrics sidecar round-trip, and the image-only-op
guards (crop/adjust/harmonize must skip/reject audio gracefully).

Mirrors the fixture conventions in ``test_dataset_manager.py`` — a
``DatasetManager`` rooted in ``tmp_path`` with the DB layer mocked out.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import soundfile as sf
from PIL import Image
from unittest.mock import patch, MagicMock

from app.core.dataset_manager import DatasetManager


# ── Helpers ──────────────────────────────────────────────────────────────


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


def _create_image(path: str, width: int = 100, height: int = 100):
    Image.new("RGB", (width, height), "red").save(path)


def _create_caption(path: str, text: str = "a test caption"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _create_audio(
    path: str,
    *,
    sample_rate: int = 8000,
    duration_s: float = 0.2,
    channels: int = 1,
    format: str | None = None,
    subtype: str | None = None,
) -> None:
    """Write a tiny generated sine-wave tone — no binary fixtures committed."""
    n = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    if channels > 1:
        tone = np.stack([tone] * channels, axis=-1)
    kwargs = {}
    if format:
        kwargs["format"] = format
    if subtype:
        kwargs["subtype"] = subtype
    sf.write(path, tone.astype(np.float32), sample_rate, **kwargs)


# ── Scan: classification + metadata extraction ───────────────────────────


class TestScanAudio:
    def test_scan_extracts_audio_metadata(self, manager):
        ds = manager.create_dataset("audioset")
        _create_audio(
            os.path.join(ds.path, "song.wav"), sample_rate=8000, duration_s=0.25, channels=1,
        )

        result = manager.scan_dataset("audioset")

        assert result.multimedia_count == 1
        assert result.file_count == 1
        meta = result.media_metadata["song.wav"]
        assert meta["is_audio"] is True
        assert meta["sample_rate"] == 8000
        assert meta["channels"] == 1
        assert meta["duration_s"] == pytest.approx(0.25, abs=0.01)

    @pytest.mark.parametrize(
        "ext,fmt,subtype",
        [
            (".wav", None, None),
            (".flac", "FLAC", None),
            (".ogg", "OGG", "VORBIS"),
            (".opus", "OGG", "OPUS"),
            (".mp3", None, None),
        ],
    )
    def test_scan_supports_every_contract_extension(self, manager, ext, fmt, subtype):
        """All five extensions fixed by the program plan must scan cleanly."""
        ds = manager.create_dataset(f"fmt{ext.strip('.')}")
        _create_audio(
            os.path.join(ds.path, f"clip{ext}"), sample_rate=8000, duration_s=0.1,
            format=fmt, subtype=subtype,
        )

        result = manager.scan_dataset(f"fmt{ext.strip('.')}")

        assert result.multimedia_count == 1
        meta = result.media_metadata[f"clip{ext}"]
        assert meta["sample_rate"] == 8000
        assert meta["duration_s"] > 0

    def test_scan_audio_counts_like_other_media(self, manager):
        """Mixed image+audio dataset: both count toward file/multimedia counts."""
        ds = manager.create_dataset("mixed")
        _create_image(os.path.join(ds.path, "pic.png"))
        _create_audio(os.path.join(ds.path, "song.wav"))

        with patch("app.core.dataset_manager.solide_hash_robust", return_value="deadbeef" * 4):
            result = manager.scan_dataset("mixed")

        assert result.multimedia_count == 2
        assert result.file_count == 2

    def test_scan_audio_caption_txt_unchanged(self, manager):
        """A plain .txt sidecar still works as a caption for an audio file."""
        ds = manager.create_dataset("audiocap")
        _create_audio(os.path.join(ds.path, "song.wav"))
        _create_caption(os.path.join(ds.path, "song.txt"), "a folk ballad")

        result = manager.scan_dataset("audiocap")

        assert result.caption_count == 1
        assert result.caption_coverage is True
        assert result.media_metadata["song.wav"]["has_caption"] is True

    def test_scan_audio_no_perceptual_hash_or_thumbnail_fields(self, manager):
        """Audio entries skip the image-only hash/score/dims pipeline."""
        ds = manager.create_dataset("nohash")
        _create_audio(os.path.join(ds.path, "song.wav"))

        result = manager.scan_dataset("nohash")

        meta = result.media_metadata["song.wav"]
        assert "solid_hash" not in meta
        assert "width" not in meta
        assert "quality_score" not in meta

    def test_scan_audio_never_becomes_preview_when_image_present(self, manager):
        """An audio file must not be elected preview_image (no thumbnail)."""
        ds = manager.create_dataset("previewmix")
        _create_audio(os.path.join(ds.path, "aaa_song.wav"))  # sorts first
        _create_image(os.path.join(ds.path, "zzz_pic.png"))

        with patch("app.core.dataset_manager.solide_hash_robust", return_value="deadbeef" * 4):
            result = manager.scan_dataset("previewmix")

        assert result.preview_image == "zzz_pic.png"

    def test_scan_lyrics_sidecar_not_counted_as_caption(self, manager):
        """`<stem>.lyrics.txt` must not inflate caption_count — it's a
        distinct sidecar tracked via has_lyrics, not has_caption."""
        ds = manager.create_dataset("lyricsscan")
        _create_audio(os.path.join(ds.path, "song.wav"))
        _create_caption(os.path.join(ds.path, "song.lyrics.txt"), "la la la")

        result = manager.scan_dataset("lyricsscan")

        assert result.caption_count == 0
        assert result.media_metadata["song.wav"]["has_caption"] is False
        assert result.media_metadata["song.wav"]["has_lyrics"] is True

    def test_scan_lyrics_sidecar_missing_defaults_false(self, manager):
        ds = manager.create_dataset("nolyrics")
        _create_audio(os.path.join(ds.path, "song.wav"))

        result = manager.scan_dataset("nolyrics")

        assert result.media_metadata["song.wav"]["has_lyrics"] is False

    def test_scan_bad_audio_file_does_not_abort_scan(self, manager):
        """A corrupt/undecodable audio file falls back to best-effort
        metadata (zeros) rather than crashing the whole scan."""
        ds = manager.create_dataset("badaudio")
        with open(os.path.join(ds.path, "broken.wav"), "wb") as f:
            f.write(b"not a real wav file")

        result = manager.scan_dataset("badaudio")

        assert result.multimedia_count == 1
        meta = result.media_metadata["broken.wav"]
        assert meta["duration_s"] == 0.0
        assert meta["sample_rate"] == 0


# ── get_dataset_pairs ──────────────────────────────────────────────────


class TestGetDatasetPairsAudio:
    def test_pairs_media_type_is_audio(self, manager):
        ds = manager.create_dataset("pairsaudio")
        _create_audio(os.path.join(ds.path, "song.wav"), duration_s=0.3)
        manager.scan_dataset("pairsaudio")

        pairs = manager.get_dataset_pairs("pairsaudio")

        assert len(pairs) == 1
        assert pairs[0]["media_type"] == "audio"
        assert pairs[0]["metadata"]["duration_s"] == pytest.approx(0.3, abs=0.01)

    def test_pairs_hydrate_lyrics_content(self, manager):
        ds = manager.create_dataset("pairslyrics")
        _create_audio(os.path.join(ds.path, "song.wav"))
        _create_caption(os.path.join(ds.path, "song.lyrics.txt"), "verse one")
        manager.scan_dataset("pairslyrics")

        pairs = manager.get_dataset_pairs("pairslyrics")

        assert pairs[0]["lyrics_file"] == "song.lyrics.txt"
        assert pairs[0]["lyrics_content"] == "verse one"

    def test_pairs_lyrics_empty_when_no_sidecar(self, manager):
        ds = manager.create_dataset("pairsnolyrics")
        _create_audio(os.path.join(ds.path, "song.wav"))
        manager.scan_dataset("pairsnolyrics")

        pairs = manager.get_dataset_pairs("pairsnolyrics")

        assert pairs[0]["lyrics_file"] is None
        assert pairs[0]["lyrics_content"] == ""


# ── Lyrics round-trip ──────────────────────────────────────────────────


class TestLyricsRoundTrip:
    def test_save_then_read_lyrics(self, manager):
        ds = manager.create_dataset("lyricsrt")
        _create_audio(os.path.join(ds.path, "song.wav"))
        manager.scan_dataset("lyricsrt")

        manager.save_lyrics("lyricsrt", "song.lyrics.txt", "hello world\nverse two")
        content = manager.read_caption("lyricsrt", "song.lyrics.txt")

        assert content == "hello world\nverse two"
        assert os.path.exists(os.path.join(ds.path, "song.lyrics.txt"))

    def test_save_lyrics_flips_has_lyrics_flag(self, manager):
        ds = manager.create_dataset("lyricsflag")
        _create_audio(os.path.join(ds.path, "song.wav"))
        manager.scan_dataset("lyricsflag")
        assert manager.datasets["lyricsflag"].media_metadata["song.wav"]["has_lyrics"] is False

        manager.save_lyrics("lyricsflag", "song.lyrics.txt", "some lyrics")

        assert manager.datasets["lyricsflag"].media_metadata["song.wav"]["has_lyrics"] is True

    def test_save_empty_lyrics_clears_flag(self, manager):
        ds = manager.create_dataset("lyricsclear")
        _create_audio(os.path.join(ds.path, "song.wav"))
        manager.scan_dataset("lyricsclear")
        manager.save_lyrics("lyricsclear", "song.lyrics.txt", "some lyrics")
        assert manager.datasets["lyricsclear"].media_metadata["song.wav"]["has_lyrics"] is True

        manager.save_lyrics("lyricsclear", "song.lyrics.txt", "   ")

        assert manager.datasets["lyricsclear"].media_metadata["song.wav"]["has_lyrics"] is False

    def test_lyrics_does_not_touch_caption_flag(self, manager):
        """Saving lyrics must not flip has_caption (distinct sidecars)."""
        ds = manager.create_dataset("lyricsvscap")
        _create_audio(os.path.join(ds.path, "song.wav"))
        manager.scan_dataset("lyricsvscap")

        manager.save_lyrics("lyricsvscap", "song.lyrics.txt", "some lyrics")

        assert manager.datasets["lyricsvscap"].media_metadata["song.wav"]["has_caption"] is False

    def test_save_lyrics_rejects_path_traversal(self, manager):
        manager.create_dataset("lyricstraversal")
        with pytest.raises(ValueError, match="traversal"):
            manager.save_lyrics("lyricstraversal", "../../evil.lyrics.txt", "x")


# ── Image-only-op guards ──────────────────────────────────────────────


class TestImageOpGuardsRejectAudio:
    def test_crop_media_rejects_audio(self, manager):
        ds = manager.create_dataset("cropaudio")
        _create_audio(os.path.join(ds.path, "song.wav"))

        with pytest.raises(ValueError, match="audio"):
            manager.crop_media("cropaudio", "song.wav", 10, 10)

    def test_apply_adjustments_rejects_audio(self, manager):
        ds = manager.create_dataset("adjaudio")
        _create_audio(os.path.join(ds.path, "song.wav"))

        with pytest.raises(ValueError, match="audio"):
            manager.apply_adjustments("adjaudio", "song.wav", {"contrast": 1.5})

    def test_harmonize_skips_audio_pairs(self, manager):
        """Harmonize converts image pairs to JPG but must leave audio files
        on disk, untouched — converting audio bytes through PIL would
        destroy them."""
        ds = manager.create_dataset("harmaudio")
        _create_image(os.path.join(ds.path, "pic.png"))
        _create_audio(os.path.join(ds.path, "song.wav"))

        with patch("app.core.dataset_manager.solide_hash_robust", return_value="deadbeef" * 4):
            manager.scan_dataset("harmaudio")
            result = manager.harmonize_files("harmaudio")

        # Only the image pair was processed/renamed/converted.
        assert result["processed"] == 1
        assert result["renamed"] == 1
        # Audio file survives untouched at its original name.
        assert os.path.exists(os.path.join(ds.path, "song.wav"))

    def test_harmonize_audio_only_dataset_is_a_noop(self, manager):
        ds = manager.create_dataset("audioonly")
        _create_audio(os.path.join(ds.path, "song.wav"))
        manager.scan_dataset("audioonly")

        result = manager.harmonize_files("audioonly")

        assert result == {"processed": 0, "converted": 0, "renamed": 0}
        assert os.path.exists(os.path.join(ds.path, "song.wav"))
