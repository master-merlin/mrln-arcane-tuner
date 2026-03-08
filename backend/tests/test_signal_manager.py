"""
Tests for TrainingSignalManager — file-based IPC.

Covers: signal sending, reading, clearing, handle_signals logic
(pause, resume, soft_stop), and unknown action handling.
"""
import json

from app.engine.components.signal_manager import TrainingSignalManager, SIGNAL_FILENAME


# ── Send Signal ──────────────────────────────────────────────────────────


class TestSendSignal:
    """Tests for the static send_signal method."""

    def test_send_signal_creates_file(self, tmp_path):
        """send_signal should create signal.json in the output directory."""
        TrainingSignalManager.send_signal(str(tmp_path), "pause")

        signal_path = tmp_path / SIGNAL_FILENAME
        assert signal_path.exists()

        data = json.loads(signal_path.read_text())
        assert data["action"] == "pause"

    def test_send_signal_creates_directory(self, tmp_path):
        """send_signal should create the output directory if it doesn't exist."""
        new_dir = tmp_path / "nested" / "output"
        TrainingSignalManager.send_signal(str(new_dir), "soft_stop")

        assert (new_dir / SIGNAL_FILENAME).exists()

    def test_send_signal_overwrites_existing(self, tmp_path):
        """Sending a new signal should overwrite the previous one."""
        TrainingSignalManager.send_signal(str(tmp_path), "pause")
        TrainingSignalManager.send_signal(str(tmp_path), "resume")

        data = json.loads((tmp_path / SIGNAL_FILENAME).read_text())
        assert data["action"] == "resume"


# ── Check Signal ─────────────────────────────────────────────────────────


class TestCheckSignal:
    """Tests for check_signal reading."""

    def test_check_signal_reads_action(self, tmp_path):
        """check_signal should return the action string from signal.json."""
        TrainingSignalManager.send_signal(str(tmp_path), "soft_stop")
        mgr = TrainingSignalManager(str(tmp_path))

        assert mgr.check_signal() == "soft_stop"

    def test_check_signal_no_file(self, tmp_path):
        """check_signal with no signal file should return None."""
        mgr = TrainingSignalManager(str(tmp_path))
        assert mgr.check_signal() is None

    def test_check_signal_corrupt_json(self, tmp_path):
        """check_signal with corrupt JSON should return None gracefully."""
        signal_path = tmp_path / SIGNAL_FILENAME
        signal_path.write_text("not valid json{{{")

        mgr = TrainingSignalManager(str(tmp_path))
        assert mgr.check_signal() is None


# ── Clear Signal ─────────────────────────────────────────────────────────


class TestClearSignal:
    """Tests for clear_signal."""

    def test_clear_signal_removes_file(self, tmp_path):
        """clear_signal should delete the signal.json file."""
        TrainingSignalManager.send_signal(str(tmp_path), "pause")
        mgr = TrainingSignalManager(str(tmp_path))

        mgr.clear_signal()
        assert not (tmp_path / SIGNAL_FILENAME).exists()

    def test_clear_signal_noop_when_missing(self, tmp_path):
        """clear_signal with no file should not raise."""
        mgr = TrainingSignalManager(str(tmp_path))
        mgr.clear_signal()  # Should not raise


# ── Handle Signals ───────────────────────────────────────────────────────


class TestHandleSignals:
    """Tests for handle_signals dispatch logic."""

    def test_no_signal_returns_none(self, tmp_path):
        """handle_signals with no signal should return None."""
        mgr = TrainingSignalManager(str(tmp_path))
        assert mgr.handle_signals() is None

    def test_soft_stop_returns_soft_stop(self, tmp_path):
        """handle_signals with 'soft_stop' signal should return 'soft_stop'."""
        TrainingSignalManager.send_signal(str(tmp_path), "soft_stop")
        mgr = TrainingSignalManager(str(tmp_path))

        result = mgr.handle_signals()
        assert result == "soft_stop"
        # Signal file should be cleared
        assert not (tmp_path / SIGNAL_FILENAME).exists()

    def test_resume_without_pause_returns_none(self, tmp_path):
        """'resume' signal without being paused should clear and return None."""
        TrainingSignalManager.send_signal(str(tmp_path), "resume")
        mgr = TrainingSignalManager(str(tmp_path))

        result = mgr.handle_signals()
        assert result is None
        assert not (tmp_path / SIGNAL_FILENAME).exists()

    def test_unknown_action_clears_and_returns_none(self, tmp_path):
        """Unknown action should be cleared without side effects."""
        signal_path = tmp_path / SIGNAL_FILENAME
        signal_path.write_text(json.dumps({"action": "explode"}))

        mgr = TrainingSignalManager(str(tmp_path))
        result = mgr.handle_signals()
        assert result is None
        assert not (tmp_path / SIGNAL_FILENAME).exists()


# ── Paused Elapsed ───────────────────────────────────────────────────────


class TestPausedElapsed:
    """Tests for paused time tracking."""

    def test_initial_paused_elapsed_is_zero(self, tmp_path):
        """New manager should have 0 paused elapsed time."""
        mgr = TrainingSignalManager(str(tmp_path))
        assert mgr.paused_elapsed == 0.0
