import time
from app.api.events.download_progress import RateLimiter


def test_starting_and_complete_always_pass():
    rl = RateLimiter(min_interval_s=10.0, min_delta_pct=99.0)
    assert rl.allow("starting", percent=0)
    # downloading immediately after starting is suppressed
    assert not rl.allow("downloading", percent=0)
    # complete still passes despite tight throttle
    assert rl.allow("complete", percent=100)


def test_error_always_passes():
    rl = RateLimiter()
    rl.allow("starting", percent=0)
    assert rl.allow("error", percent=0)


def test_downloading_throttled_by_time():
    rl = RateLimiter(min_interval_s=0.2, min_delta_pct=99.0)
    assert rl.allow("starting", percent=0)
    # too soon
    assert not rl.allow("downloading", percent=1)
    time.sleep(0.21)
    assert rl.allow("downloading", percent=2)


def test_downloading_passes_on_percent_delta():
    rl = RateLimiter(min_interval_s=999.0, min_delta_pct=5.0)
    assert rl.allow("starting", percent=0)
    assert not rl.allow("downloading", percent=2)
    assert rl.allow("downloading", percent=10)


def test_downloading_allows_null_percent_when_time_elapsed():
    # Margin is generous (sleep 3x threshold) — Windows timer jitter under
    # load can otherwise push a 60ms sleep below the 50ms threshold.
    rl = RateLimiter(min_interval_s=0.05, min_delta_pct=5.0)
    assert rl.allow("starting", percent=None)
    time.sleep(0.15)
    # Unknown total → percent None; should still pass after time threshold
    assert rl.allow("downloading", percent=None)
