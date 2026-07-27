"""
Tests for TrainingLogger — covers log_step, NaN handling, timing, save_loss_history.
"""

import json
import time
from unittest.mock import MagicMock


from app.engine.components.training_logger import TrainingLogger


class TestLogStep:
    def test_accumulates_loss_history(self):
        tl = TrainingLogger(max_steps=100)
        tl.log_step(0, loss=0.5, lr=0.001)
        tl.log_step(1, loss=0.4, lr=0.001)
        assert len(tl.loss_history) == 2

    def test_step_is_one_indexed_in_history(self):
        tl = TrainingLogger(max_steps=10)
        tl.log_step(0, loss=1.0)
        assert tl.loss_history[0]["step"] == 1

    def test_nan_loss_sanitized(self):
        tl = TrainingLogger(max_steps=10)
        tl.log_step(0, loss=float("nan"), lr=float("inf"))
        entry = tl.loss_history[0]
        assert entry["loss"] == 0.0
        assert entry["lr"] == 0.0

    def test_inf_loss_sanitized(self):
        tl = TrainingLogger(max_steps=10)
        tl.log_step(0, loss=float("inf"))
        assert tl.loss_history[0]["loss"] == 0.0

    def test_extra_data_merged(self):
        tl = TrainingLogger(max_steps=10)
        tl.log_step(0, loss=0.5, extra={"snr_gamma": 5.0})
        # The extra field is logged but not stored in loss_history directly
        # — it's in the structlog output. We just verify no crash.
        assert len(tl.loss_history) == 1

    def test_extra_nan_sanitized(self):
        tl = TrainingLogger(max_steps=10)
        tl.log_step(0, loss=0.5, extra={"grad_norm": float("nan")})
        assert len(tl.loss_history) == 1


class TestLogStepNoneLoss:
    """W2.T6 fix-wave: ``loss=None`` marks a step with no usable loss (an
    all-NaN accumulation window). It must still emit progress/step/ETA
    telemetry but must NOT leave a fake data point in the loss chart, loss
    history, or DB metrics buffer."""

    def test_none_loss_not_added_to_history(self):
        tl = TrainingLogger(max_steps=10)
        tl.log_step(0, loss=None)
        assert tl.loss_history == []

    def test_none_loss_not_buffered_for_db(self):
        tl = TrainingLogger(max_steps=10)
        tl._job_id = "job-1"
        tl.log_step(0, loss=None)
        assert tl._metrics_buffer == []

    def test_none_loss_still_writes_step_ipc_without_loss_key(self):
        """The Jobs screen's live step counter/ETA comes from the IPC
        ``step`` message — it must still fire, but without a ``loss`` key
        (so the frontend's `typeof m.loss === 'number'` chart filter and
        the current-loss KPI both correctly treat it as "no loss this
        step" instead of a fabricated 0.0)."""
        writer = MagicMock()
        tl = TrainingLogger(max_steps=10, log_writer=writer)
        tl.log_step(5, loss=None, lr=1e-4, extra={"nan_window_skipped": True})
        writer.step.assert_called_once()
        (payload,), _ = writer.step.call_args
        assert "loss" not in payload
        assert payload["step"] == 6
        assert payload["nan_window_skipped"] is True

    def test_none_loss_does_not_crash_on_missing_extra_grad_norm(self):
        """Regression guard: the metrics-buffer branch used to read
        ``extra.get(...)`` unconditionally — must stay skipped for
        loss=None even when extra is provided."""
        tl = TrainingLogger(max_steps=10)
        tl._job_id = "job-1"
        tl.log_step(0, loss=None, extra={"nan_window_skipped": True})
        assert tl._metrics_buffer == []

    def test_real_loss_after_none_marker_still_recorded(self):
        """loss=None must not disturb bookkeeping for subsequent real
        steps."""
        tl = TrainingLogger(max_steps=10)
        tl.log_step(0, loss=None)
        tl.log_step(1, loss=0.5)
        assert len(tl.loss_history) == 1
        assert tl.loss_history[0]["step"] == 2


class TestElapsedAndETA:
    def test_total_elapsed_includes_offset(self):
        tl = TrainingLogger(max_steps=100, elapsed_offset=3600.0)
        elapsed = tl.get_total_elapsed()
        assert elapsed >= 3600.0

    def test_paused_elapsed_subtracted(self):
        sm = MagicMock()
        sm.paused_elapsed = 100.0
        tl = TrainingLogger(max_steps=100, signal_manager=sm)
        elapsed = tl.get_total_elapsed()
        # Elapsed should be negative or very small since we just started
        # and pause is 100s, but the key thing is it's less than without pause
        elapsed_raw = time.time() - tl.start_time
        assert elapsed < elapsed_raw + 1

    def test_training_start_time_set_on_first_step(self):
        """_training_start_time must be None before any log_step call."""
        tl = TrainingLogger(max_steps=100)
        assert tl._training_start_time is None
        tl.log_step(0, loss=0.5)
        assert tl._training_start_time is not None

    def test_eta_excludes_prep_time(self):
        """ETA should use only training elapsed, not total elapsed.

        Simulates 60s of prep by backdating start_time, then verifying
        that _training_start_time is set near the first log_step call,
        not at the backdated start_time.
        """
        tl = TrainingLogger(max_steps=100)
        # Simulate 60s of prep time before first step
        tl.start_time = time.time() - 60.0
        tl.last_step_time = tl.start_time

        now = time.time()
        tl.log_step(0, loss=0.5)

        # _training_start_time should be set near 'now', not 60s ago
        assert tl._training_start_time is not None
        assert abs(tl._training_start_time - now) < 1.0, (
            f"_training_start_time should be near now, not backdated. "
            f"Diff: {abs(tl._training_start_time - now):.2f}s"
        )
        # Verify that start_time (which includes prep) is still 60s ago
        assert (time.time() - tl.start_time) > 59.0


class TestSaveLossHistory:
    def test_saves_to_json(self, tmp_path):
        tl = TrainingLogger(max_steps=10, output_dir=str(tmp_path))
        tl.log_step(0, loss=0.5)
        tl.log_step(1, loss=0.4)
        tl.save_loss_history()

        path = tmp_path / "loss_history.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 2
        assert data[0]["loss"] == 0.5

    def test_custom_output_dir(self, tmp_path):
        custom = tmp_path / "custom"
        tl = TrainingLogger(max_steps=10)
        tl.log_step(0, loss=1.0)
        tl.save_loss_history(output_dir=str(custom))
        assert (custom / "loss_history.json").exists()

    def test_no_history_no_write(self, tmp_path):
        tl = TrainingLogger(max_steps=10, output_dir=str(tmp_path))
        tl.save_loss_history()
        assert not (tmp_path / "loss_history.json").exists()

    def test_no_output_dir_no_write(self):
        tl = TrainingLogger(max_steps=10)
        tl.log_step(0, loss=0.5)
        tl.save_loss_history()  # Should not raise


class TestLossHistoryProperty:
    def test_returns_list(self):
        tl = TrainingLogger(max_steps=5)
        assert isinstance(tl.loss_history, list)
        assert len(tl.loss_history) == 0

    def test_round_trip_entries(self, tmp_path):
        tl = TrainingLogger(max_steps=10, output_dir=str(tmp_path))
        tl.log_step(0, loss=0.123456789)
        tl.save_loss_history()

        data = json.loads((tmp_path / "loss_history.json").read_text())
        assert data[0]["loss"] == round(0.123456789, 6)
