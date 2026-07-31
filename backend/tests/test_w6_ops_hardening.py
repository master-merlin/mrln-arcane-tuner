"""Ops-robustness regressions from the full-backend review.

The unifying theme is work that could not be stopped or bounded: an ffmpeg run
with no timeout and no way to observe cancellation, an unknown-total task that
broadcast on every tick, an archive that could expand without limit, and a
retry chain with no total budget.
"""

from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

import pytest

from app.core.portable.envelope import MANIFEST_NAME, ManifestError


# ── ffmpeg runner: timeout + abort ────────────────────────────────────────


def _sleeper_argv(seconds: float) -> list[str]:
    """An argv that ignores ffmpeg entirely and just sleeps, so the runner's
    control flow is exercised without needing a real encode."""
    return [
        "-c",
        f"import time,sys; time.sleep({seconds}); sys.exit(0)",
    ]


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    """Point the runner's binary resolution at this interpreter."""
    from app.core.video import ffmpeg as ff

    monkeypatch.setattr(ff, "resolve_ffmpeg", lambda: sys.executable)
    return ff


class TestRunFfmpegControl:
    def test_timeout_kills_the_child(self, fake_ffmpeg):
        started = time.monotonic()
        with pytest.raises(fake_ffmpeg.FFmpegError, match="timed out"):
            fake_ffmpeg.run_ffmpeg(_sleeper_argv(30), timeout=1.0)
        assert time.monotonic() - started < 10, "did not kill promptly"

    def test_abort_stops_a_running_child(self, fake_ffmpeg):
        """Pre-fix the worker thread was parked inside subprocess.run, so a
        cancelled task could not take effect until the segment finished."""
        started = time.monotonic()
        with pytest.raises(fake_ffmpeg.FFmpegAborted):
            fake_ffmpeg.run_ffmpeg(
                _sleeper_argv(30), timeout=None, should_abort=lambda: True
            )
        assert time.monotonic() - started < 10

    def test_successful_run_returns_zero(self, fake_ffmpeg):
        assert fake_ffmpeg.run_ffmpeg(_sleeper_argv(0), timeout=30) == 0

    def test_nonzero_exit_raises_with_stderr_tail(self, fake_ffmpeg):
        argv = ["-c", "import sys; sys.stderr.write('boom-marker\\n'); sys.exit(3)"]
        with pytest.raises(fake_ffmpeg.FFmpegError, match="boom-marker"):
            fake_ffmpeg.run_ffmpeg(argv, timeout=30)

    def test_chatty_stderr_does_not_deadlock(self, fake_ffmpeg):
        """A child that floods stderr fills the OS pipe buffer; with no reader
        it blocks forever and so does the parent. ffmpeg does this routinely
        (non-monotonic DTS warnings on stream-copy splits)."""
        argv = [
            "-c",
            "import sys\n"
            "for i in range(20000): sys.stderr.write('warning line %d\\n' % i)\n"
            "sys.exit(0)",
        ]
        assert fake_ffmpeg.run_ffmpeg(argv, timeout=60) == 0


class TestSplitCancellation:
    def test_abort_marks_cancelled_and_removes_the_partial_clip(
        self, tmp_path, monkeypatch
    ):
        from app.core.tasks.task_manager import TaskStatus, task_manager
        from app.core.video import split_batch
        from app.core.video.ffmpeg import FFmpegAborted

        task_manager.set_loop(None)
        dataset = tmp_path / "ds"
        dataset.mkdir()
        (dataset / "long.mp4").write_bytes(b"\x00")

        monkeypatch.setattr(split_batch, "_resolve_source_dir", lambda n: dataset)
        monkeypatch.setattr(split_batch, "_scan", lambda n: None)
        monkeypatch.setattr(split_batch, "_nearest_keyframe", lambda p, t: 0.0)

        def _aborting(args, progress_cb=None, should_abort=None):
            # Emulate a killed encoder: partial output on disk, then abort.
            Path(args[-1]).write_bytes(b"partial")
            raise FFmpegAborted("cancelled")

        monkeypatch.setattr(split_batch, "_run_ffmpeg", _aborting)

        t = task_manager.create(
            type="video_split", title="x", total=2, dataset_name="ds"
        )
        split_batch.run_video_split_batch(
            t.id,
            dataset_name="ds",
            source_rel_path="long.mp4",
            segments=[{"start_s": 0.0, "end_s": 1.0}, {"start_s": 1.0, "end_s": 2.0}],
            mode="copy",
            archive_source=False,
        )

        assert task_manager.get(t.id).status == TaskStatus.CANCELLED
        assert task_manager.get(t.id).failed == 0, "abort counted as a failure"
        assert list(dataset.glob("*_000.mp4")) == [], "partial clip left on disk"


# ── task progress throttle ────────────────────────────────────────────────


class TestProgressThrottle:
    def test_unknown_total_is_time_throttled(self, monkeypatch):
        """total=0 has no percentage to step on, so the delta half of the rule
        cannot apply — previously it evaluated to "always emit"."""
        from app.core.tasks.task_manager import TaskManager

        tm = TaskManager()
        sent: list = []
        tm.set_loop(object())
        monkeypatch.setattr(
            tm, "_broadcast", TaskManager._broadcast.__get__(tm, TaskManager)
        )
        monkeypatch.setattr(
            "asyncio.run_coroutine_threadsafe",
            lambda coro, loop: (coro.close(), sent.append(1))[1],
        )

        task = tm.create(type="t", title="x", total=0)
        sent.clear()
        for i in range(50):
            tm.update(task.id, current=i)
        assert len(sent) <= 2, f"unknown-total task broadcast {len(sent)} times"


# ── archive hardening ─────────────────────────────────────────────────────


class TestArchiveHardening:
    def test_symlinked_file_is_not_baked_into_the_export(self, tmp_path):
        from app.core.portable.archive import write_zip

        root = tmp_path / "ds"
        root.mkdir()
        (root / "real.txt").write_text("inside")
        outside = tmp_path / "secret.txt"
        outside.write_text("SECRET")
        try:
            (root / "link.txt").symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this host")

        buf = write_zip(root, {"v": 1})
        with zipfile.ZipFile(buf) as zf:
            names = set(zf.namelist())
            assert "real.txt" in names
            assert "link.txt" not in names
            assert all(b"SECRET" not in zf.read(n) for n in names if n != MANIFEST_NAME)

    def test_oversized_archive_is_rejected_by_declared_size(self, tmp_path):
        from app.core.portable.archive import safe_extract

        zip_path = tmp_path / "bomb.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(MANIFEST_NAME, "{}")
            zf.writestr("big.bin", b"\x00" * (1024 * 1024))

        dest = tmp_path / "out"
        dest.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            with pytest.raises(ManifestError, match="import limit"):
                safe_extract(zf, dest, max_total_bytes=1024)

    def test_normal_archive_still_extracts(self, tmp_path):
        from app.core.portable.archive import safe_extract

        zip_path = tmp_path / "ok.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(MANIFEST_NAME, "{}")
            zf.writestr("a/b.txt", "hello")

        dest = tmp_path / "out"
        dest.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            safe_extract(zf, dest)
        assert (dest / "a" / "b.txt").read_text() == "hello"

    def test_traversal_member_still_rejected(self, tmp_path):
        from app.core.portable.archive import safe_extract

        zip_path = tmp_path / "eve.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(MANIFEST_NAME, "{}")
            zf.writestr("../escape.txt", "x")

        dest = tmp_path / "out"
        dest.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            with pytest.raises(ManifestError, match="Unsafe path"):
                safe_extract(zf, dest)


# ── provider URL validation ───────────────────────────────────────────────


class TestProviderUrl:
    @pytest.mark.parametrize(
        "bad", ["file:///etc/passwd", "ftp://host/x", "not-a-url", ""]
    )
    def test_non_http_scheme_is_rejected(self, bad):
        from app.core.llm.openai_compat import _validate_base_url

        with pytest.raises(ValueError):
            _validate_base_url(bad)

    @pytest.mark.parametrize(
        "good",
        [
            "https://api.openai.com/v1",
            "http://localhost:11434/v1",
            "https://api.openai.com/v1/",
        ],
    )
    def test_http_urls_pass_and_are_normalized(self, good):
        from app.core.llm.openai_compat import _validate_base_url

        assert _validate_base_url(good) == good.rstrip("/")


# ── bucket distribution ───────────────────────────────────────────────────


class TestBucketDistribution:
    def test_video_item_counts_once(self):
        """A video item was counted twice — once under the spatial key by
        get_bucket and again under the spatial x frames key — so the logged
        total was 2x the item count and every percentage was halved."""
        from app.engine.components.bucketing import BucketManager

        bm = BucketManager(base_resolutions=512, frame_buckets=[1, 5, 9])
        for _ in range(4):
            bm.get_bucket_for_video(640, 480, available_frames=9)

        total = sum(bm._distribution.values())
        assert total == 4, dict(bm._distribution)
        assert all("f" in key for key in bm._distribution)

    def test_image_item_still_counts_once(self):
        from app.engine.components.bucketing import BucketManager

        bm = BucketManager(base_resolutions=512)
        for _ in range(3):
            bm.get_bucket(640, 480)
        assert sum(bm._distribution.values()) == 3
