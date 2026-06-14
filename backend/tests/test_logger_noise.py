"""The noisy third-party loggers used during HF downloads must be pinned to a
WARNING threshold (drops their INFO/DEBUG chatter, keeps real warnings/errors)."""
import logging

from app.core.logger import _quiet_noisy_loggers


def test_quiet_noisy_loggers_pins_threshold():
    # Pre-dirty: make one of them DEBUG so we can prove it gets raised.
    logging.getLogger("filelock").setLevel(logging.DEBUG)

    _quiet_noisy_loggers()

    for name in ("filelock", "urllib3", "hf_xet", "websockets"):
        assert logging.getLogger(name).level >= logging.WARNING, name


def test_info_record_dropped_but_warning_kept():
    _quiet_noisy_loggers()
    flog = logging.getLogger("filelock")
    # isEnabledFor reflects the effective threshold the handler chain will see.
    assert flog.isEnabledFor(logging.INFO) is False
    assert flog.isEnabledFor(logging.WARNING) is True
