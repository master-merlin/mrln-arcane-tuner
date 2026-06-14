"""Tests for the snapshot byte-progress poller.

huggingface_hub only routes a caller's ``tqdm_class`` to the coarse
"Fetching N files" bar (never per-file byte transfers), so a full-repo
snapshot download surfaces real progress by polling on-disk cache growth
against the repo's total size. These tests cover that helper layer.
"""
from unittest.mock import patch

from app.api.events import download_progress as dp


# ── _on_disk_bytes ───────────────────────────────────────────────────────────


class TestOnDiskBytes:
    def test_sums_blob_file_sizes(self, tmp_path):
        blobs = tmp_path / "blobs"
        blobs.mkdir()
        (blobs / "a").write_bytes(b"x" * 100)
        (blobs / "b.incomplete").write_bytes(b"y" * 250)  # in-flight part counts
        assert dp._on_disk_bytes(str(tmp_path)) == 350

    def test_missing_dir_is_zero(self, tmp_path):
        assert dp._on_disk_bytes(str(tmp_path / "nope")) == 0

    def test_none_is_zero(self):
        assert dp._on_disk_bytes(None) == 0


# ── _repo_total_bytes ────────────────────────────────────────────────────────


class TestRepoTotalBytes:
    @patch("huggingface_hub.HfApi")
    def test_sums_sibling_sizes(self, MockApi):
        sib = lambda n: type("S", (), {"size": n})()  # noqa: E731
        MockApi.return_value.repo_info.return_value = type(
            "I", (), {"siblings": [sib(10), sib(20), sib(None)]},
        )()
        assert dp._repo_total_bytes("org/model") == 30

    @patch("huggingface_hub.HfApi")
    def test_metadata_failure_returns_none(self, MockApi):
        MockApi.return_value.repo_info.side_effect = RuntimeError("offline")
        assert dp._repo_total_bytes("org/model") is None


# ── snapshot_byte_progress context manager ───────────────────────────────────


class TestSnapshotByteProgress:
    def _cache_with_bytes(self, tmp_path, n: int) -> str:
        blobs = tmp_path / "blobs"
        blobs.mkdir()
        if n:
            (blobs / "blob").write_bytes(b"z" * n)
        return str(tmp_path)

    def test_starting_and_complete_emitted_with_real_bytes(self, tmp_path):
        cache_dir = self._cache_with_bytes(tmp_path, 400)
        emits = []
        with patch.object(dp, "_repo_total_bytes", return_value=1000), \
             patch.object(dp, "_repo_cache_dir", return_value=cache_dir), \
             patch.object(dp, "schedule_emit_from_thread", side_effect=emits.append):
            with dp.snapshot_byte_progress(
                repo_id="org/model", model_id="org/model", category="training",
            ):
                pass

        statuses = [p.status for p in emits]
        assert statuses[0] == "starting"
        assert statuses[-1] == "complete"
        # Resume baseline surfaces immediately: 400/1000 = 40%.
        start = emits[0]
        assert start.current_bytes == 400
        assert start.total_bytes == 1000
        assert start.percent == 40

    def test_error_emitted_and_reraised(self, tmp_path):
        cache_dir = self._cache_with_bytes(tmp_path, 0)
        emits = []
        with patch.object(dp, "_repo_total_bytes", return_value=None), \
             patch.object(dp, "_repo_cache_dir", return_value=cache_dir), \
             patch.object(dp, "schedule_emit_from_thread", side_effect=emits.append):
            try:
                with dp.snapshot_byte_progress(
                    repo_id="org/model", model_id="org/model", category="training",
                ):
                    raise RuntimeError("boom")
            except RuntimeError:
                pass

        assert emits[-1].status == "error"
        assert "boom" in (emits[-1].error or "")

    def test_unknown_total_is_indeterminate(self, tmp_path):
        cache_dir = self._cache_with_bytes(tmp_path, 50)
        emits = []
        with patch.object(dp, "_repo_total_bytes", return_value=None), \
             patch.object(dp, "_repo_cache_dir", return_value=cache_dir), \
             patch.object(dp, "schedule_emit_from_thread", side_effect=emits.append):
            with dp.snapshot_byte_progress(
                repo_id="org/model", model_id="org/model", category="training",
            ):
                pass

        # total unknown → percent None (frontend shows an indeterminate spinner)
        assert emits[0].total_bytes is None
        assert emits[0].percent is None
        assert emits[0].current_bytes == 50


class TestFileProgressModel:
    def test_download_progress_defaults_files_empty(self):
        from app.api.events.download_progress import DownloadProgress
        p = DownloadProgress(
            source="hf", model_id="org/m", category="training", status="starting",
        )
        assert p.files == []

    def test_make_payload_carries_files(self):
        from app.api.events.download_progress import _make_payload, FileProgress
        f = FileProgress(name="dit.safetensors", current_bytes=40, total_bytes=100)
        p = _make_payload(
            source="hf", model_id="org/m", category="training",
            status="downloading", current=40, total=100, files=[f],
        )
        assert p.files[0].name == "dit.safetensors"
        assert p.files[0].percent == 40


class TestSnapshotProgressRegistry:
    def test_update_and_snapshot(self):
        from app.api.events.download_progress import SnapshotProgressRegistry
        reg = SnapshotProgressRegistry(total=1000)
        reg.update("a.bin", current=40, total=100)
        reg.update("b.bin", current=10, total=50)
        snap = reg.snapshot()
        names = [f.name for f in snap]
        assert names == ["a.bin", "b.bin"]  # sorted by name, stable
        assert snap[0].percent == 40
        assert snap[1].percent == 20

    def test_update_overwrites_same_file(self):
        from app.api.events.download_progress import SnapshotProgressRegistry
        reg = SnapshotProgressRegistry(total=None)
        reg.update("a.bin", current=10, total=100)
        reg.update("a.bin", current=90, total=100)
        assert len(reg.snapshot()) == 1
        assert reg.snapshot()[0].current_bytes == 90

    def test_done_removes_and_returns_size(self):
        from app.api.events.download_progress import SnapshotProgressRegistry
        reg = SnapshotProgressRegistry(total=None)
        reg.update("a.bin", current=100, total=100)
        size = reg.done("a.bin")
        assert size == 100
        assert reg.snapshot() == []

    def test_done_unknown_file_is_safe(self):
        from app.api.events.download_progress import SnapshotProgressRegistry
        reg = SnapshotProgressRegistry(total=None)
        assert reg.done("missing") is None


class TestCapturePerFile:
    def test_byte_bar_records_into_registry(self):
        import sys
        import huggingface_hub.utils.tqdm  # noqa: F401  (registers the submodule)
        hf_tqdm_mod = sys.modules["huggingface_hub.utils.tqdm"]
        from app.api.events.download_progress import (
            SnapshotProgressRegistry, _capture_per_file,
        )
        reg = SnapshotProgressRegistry(total=None)
        with _capture_per_file(reg):
            bar = hf_tqdm_mod.tqdm(
                total=100, unit="B", desc="dit.safetensors",
            )
            bar.update(40)
            snap = reg.snapshot()
            assert snap[0].name == "dit.safetensors"
            assert snap[0].current_bytes == 40
            bar.close()
            assert reg.snapshot() == []  # close() marks done

    def test_file_count_bar_ignored(self):
        import sys
        import huggingface_hub.utils.tqdm  # noqa: F401  (registers the submodule)
        hf_tqdm_mod = sys.modules["huggingface_hub.utils.tqdm"]
        from app.api.events.download_progress import (
            SnapshotProgressRegistry, _capture_per_file,
        )
        reg = SnapshotProgressRegistry(total=None)
        with _capture_per_file(reg):
            # The outer "Fetching N files" bar has no unit="B" — must be ignored.
            bar = hf_tqdm_mod.tqdm(total=5, desc="Fetching 5 files")
            bar.update(1)
            assert reg.snapshot() == []
            bar.close()

    def test_original_tqdm_restored_even_on_error(self):
        import sys
        import huggingface_hub.utils.tqdm  # noqa: F401  (registers the submodule)
        hf_tqdm_mod = sys.modules["huggingface_hub.utils.tqdm"]
        from app.api.events.download_progress import (
            SnapshotProgressRegistry, _capture_per_file,
        )
        original = hf_tqdm_mod.tqdm
        reg = SnapshotProgressRegistry(total=None)
        try:
            with _capture_per_file(reg):
                assert hf_tqdm_mod.tqdm is not original  # patched in scope
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert hf_tqdm_mod.tqdm is original  # restored on exception
