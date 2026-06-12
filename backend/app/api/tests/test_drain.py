import pytest

from app.core.drain import DrainActive, is_draining, set_draining


@pytest.fixture(autouse=True)
def _reset_drain():
    set_draining(False)
    yield
    set_draining(False)


def test_starts_not_draining():
    assert is_draining() is False


def test_set_draining_toggles():
    set_draining(True)
    assert is_draining() is True
    set_draining(False)
    assert is_draining() is False


def test_drain_active_is_runtime_error():
    assert issubclass(DrainActive, RuntimeError)
