"""
Training Signal Manager — File-based IPC for pause/resume/soft-stop.

The server writes a signal file to the job's output directory.
The training loop checks this file each step and acts accordingly.

Protocol:
    signal.json → { "action": "pause" | "soft_stop" | "resume" }
"""
import json
import os
import time
import structlog

logger = structlog.get_logger(__name__)

SIGNAL_FILENAME = "signal.json"


class TrainingSignalManager:
    """Used inside the trainer subprocess to check for signals from the server.

    Args:
        output_dir: The job's output directory where signal.json will be placed.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.signal_path = os.path.join(output_dir, SIGNAL_FILENAME)
        self._paused_elapsed: float = 0.0  # Accumulated time spent paused

    @property
    def paused_elapsed(self) -> float:
        """Total time spent in paused state (for ETA correction)."""
        return self._paused_elapsed

    def check_signal(self) -> str | None:
        """Read and return the current signal action, or None if no signal."""
        if not os.path.exists(self.signal_path):
            return None
        try:
            with open(self.signal_path, "r") as f:
                data = json.load(f)
            return data.get("action")
        except (json.JSONDecodeError, OSError):
            return None

    def clear_signal(self):
        """Remove the signal file after acting on it."""
        try:
            if os.path.exists(self.signal_path):
                os.remove(self.signal_path)
        except OSError:
            pass

    def handle_signals(self) -> str | None:
        """Check for signals and handle pause blocking.

        Returns:
            "soft_stop" if a soft stop was requested, None otherwise.
            Blocks in-place if paused until resumed.
        """
        action = self.check_signal()
        if action is None:
            return None

        if action == "pause":
            logger.info("training_paused_by_signal")
            self.clear_signal()
            pause_start = time.time()
            # Block until resumed
            while True:
                time.sleep(1.0)
                resume_action = self.check_signal()
                if resume_action == "resume":
                    self.clear_signal()
                    paused_duration = time.time() - pause_start
                    self._paused_elapsed += paused_duration
                    logger.info(
                        "training_resumed_by_signal",
                        paused_seconds=round(paused_duration, 1),
                    )
                    return None
                if resume_action == "soft_stop":
                    self.clear_signal()
                    paused_duration = time.time() - pause_start
                    self._paused_elapsed += paused_duration
                    logger.info("soft_stop_while_paused")
                    return "soft_stop"

        if action == "soft_stop":
            logger.info("training_soft_stop_by_signal")
            self.clear_signal()
            return "soft_stop"

        if action == "resume":
            # Resume without being paused — just clear
            self.clear_signal()
            return None

        # Unknown action
        self.clear_signal()
        return None

    @staticmethod
    def send_signal(output_dir: str, action: str):
        """Write a signal file to the job's output directory.

        Used by the server process (JobManager) to send commands
        to the trainer subprocess.

        Args:
            output_dir: The job's output directory.
            action: One of "pause", "resume", "soft_stop".
        """
        signal_path = os.path.join(output_dir, SIGNAL_FILENAME)
        os.makedirs(output_dir, exist_ok=True)
        with open(signal_path, "w") as f:
            json.dump({"action": action}, f)
        logger.info("signal_sent", path=signal_path, action=action)
