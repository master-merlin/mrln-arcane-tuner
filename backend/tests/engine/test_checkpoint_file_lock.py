"""LANE-170: a transient Windows file lock must not crash a checkpoint save.

The incident: ``os.replace(staging, final)`` raised ``PermissionError`` with
``winerror=5`` 0.27 s after a 1.24 GB write, and the training run died. Windows
refuses a rename while any process holds a handle on the target.

The tests patch the rename and the sleep and set ``winerror`` by hand, so they
run on any OS and never really wait.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.engine.utils import safe_save as safe_save_mod
from app.engine.utils.safe_save import retry_file_lock, safe_save_file

_REAL_REPLACE = os.replace
_BACKOFF = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]


def _locked(winerror: int | None = 5) -> PermissionError:
    err = PermissionError(13, "locked")
    if winerror is not None:
        err.winerror = winerror
    return err


class _Rename:
    """Raises ``errors`` in turn, then does the real rename (``real``) or nothing."""

    def __init__(self, *errors: BaseException, real=None) -> None:
        self.errors = list(errors)
        self.real = real
        self.calls = 0

    def __call__(self, *args) -> None:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        if self.real is not None:
            self.real(*args)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    # Replaces this module's `time`, not `time.sleep`: other threads keep theirs.
    seen: list[float] = []
    monkeypatch.setattr(safe_save_mod, "time", SimpleNamespace(sleep=seen.append))
    return seen


# ── retry_file_lock ──────────────────────────────────────────────────────


@pytest.mark.parametrize("winerror", [5, 32])
def test_a_locked_rename_is_retried_until_it_succeeds(sleeps, winerror):
    rename = _Rename(_locked(winerror), _locked(winerror))

    retry_file_lock(rename, "a", "b")

    assert rename.calls == 3
    assert sleeps == _BACKOFF[:2]


def test_the_retries_are_bounded_and_the_lock_error_is_raised(sleeps):
    rename = _Rename(*[_locked() for _ in range(99)])

    with pytest.raises(PermissionError):
        retry_file_lock(rename, "a", "b")

    assert rename.calls == len(_BACKOFF) + 1
    assert sleeps == _BACKOFF


@pytest.mark.parametrize(
    "error",
    [_locked(None), _locked(1224), FileNotFoundError(2, "gone"), OSError(28, "disk full")],
    ids=["no-winerror", "other-winerror", "not-found", "os-error"],
)
def test_any_other_error_raises_at_once_without_sleeping(sleeps, error):
    rename = _Rename(error)

    with pytest.raises(type(error)):
        retry_file_lock(rename, "a", "b")

    assert rename.calls == 1
    assert sleeps == []


# ── safe_save_file ───────────────────────────────────────────────────────


class _SaveFile:
    """Counts serializations and writes a small file; raises ``errors`` first."""

    def __init__(self, *errors: BaseException) -> None:
        self.errors = list(errors)
        self.calls = 0

    def __call__(self, _tensors, path, metadata=None) -> None:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        Path(path).write_bytes(b"data")


def test_safe_save_retries_only_the_rename_not_the_serialization(tmp_path, sleeps, monkeypatch):
    save = _SaveFile()
    rename = _Rename(_locked(), _locked(32), real=_REAL_REPLACE)
    monkeypatch.setattr(safe_save_mod, "save_file", save)
    monkeypatch.setattr(os, "replace", rename)
    target = tmp_path / "lora.safetensors"

    safe_save_file({}, target)

    assert target.read_bytes() == b"data"
    assert save.calls == 1
    assert rename.calls == 3


def test_safe_save_removes_its_tmp_when_the_lock_outlasts_the_bound(tmp_path, sleeps, monkeypatch):
    save = _SaveFile()
    rename = _Rename(*[_locked() for _ in range(99)])
    monkeypatch.setattr(safe_save_mod, "save_file", save)
    monkeypatch.setattr(os, "replace", rename)
    target = tmp_path / "lora.safetensors"

    with pytest.raises(PermissionError):
        safe_save_file({}, target)

    assert save.calls == 1
    assert rename.calls == len(_BACKOFF) + 1
    assert not target.exists()
    assert not Path(f"{target}.tmp").exists()


def test_safe_save_still_rewrites_the_whole_file_on_the_mmap_error(tmp_path, sleeps, monkeypatch):
    save = _SaveFile(OSError("os error 1224"))
    monkeypatch.setattr(safe_save_mod, "save_file", save)
    target = tmp_path / "lora.safetensors"

    safe_save_file({}, target)

    assert target.read_bytes() == b"data"
    assert save.calls == 2
    assert sleeps == [safe_save_mod._RETRY_DELAY]
