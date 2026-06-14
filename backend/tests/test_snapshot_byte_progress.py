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
