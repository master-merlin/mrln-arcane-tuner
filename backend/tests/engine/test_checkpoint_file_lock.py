"""LANE-170: a transient Windows file lock must not crash a checkpoint save.

The incident: ``os.replace(staging, final)`` raised ``PermissionError`` with
``winerror=5`` 0.27 s after a 1.24 GB write, and the training run died. Windows
refuses a rename while any process holds a handle on the target.

The tests patch the rename and the sleep and set ``winerror`` by hand, so they
run on any OS and never really wait.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import app.engine.components.checkpoints as ckpt_mod
from app.engine.components.checkpoints import (
    CheckpointFolderLocked,
    CheckpointManager,
    _publish_folder,
    checkpoint_resumable,
)
from app.engine.utils import safe_save as safe_save_mod
from app.engine.utils.safe_save import retry_file_lock, safe_save_file

_REAL_REPLACE = os.replace
_REAL_RENAME = os.rename
_BACKOFF = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]


def _locked(winerror: int | None = 5) -> PermissionError:
    err = PermissionError(13, "locked")
    if winerror is not None:
        err.winerror = winerror
    return err


class _Rename:
    """Takes ``errors`` in turn, raising each (``None`` lets that call through).

    A call let through, and every call after the list, does the real rename
    (``real``) or nothing.
    """

    def __init__(self, *errors: BaseException | None, real=None) -> None:
        self.errors = list(errors)
        self.real = real
        self.calls = 0

    def __call__(self, *args) -> None:
        self.calls += 1
        err = self.errors.pop(0) if self.errors else None
        if err is not None:
            raise err
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


# ── _publish_folder ──────────────────────────────────────────────────────


def _locks(n: int = 99) -> list[PermissionError]:
    return [_locked() for _ in range(n)]


def _folders(tmp_path: Path, *, with_final: bool = True) -> tuple[Path, Path]:
    final = tmp_path / "checkpoint-000010"
    staging = tmp_path / "checkpoint-000010.staging-0123456789ab"
    staging.mkdir()
    (staging / "state.txt").write_text("new")
    if with_final:
        final.mkdir()
        (final / "state.txt").write_text("old")
    return staging, final


def _replaced(final: Path) -> list[Path]:
    return list(final.parent.glob(f"{final.name}{ckpt_mod.REPLACED_MARK}*"))


def test_publish_retries_a_locked_move_onto_final(tmp_path, sleeps, monkeypatch):
    staging, final = _folders(tmp_path, with_final=False)
    monkeypatch.setattr(os, "replace", _Rename(_locked(), real=_REAL_REPLACE))

    _publish_folder(str(staging), str(final))

    assert (final / "state.txt").read_text() == "new"
    assert not staging.exists()
    assert sleeps == _BACKOFF[:1]


def test_publish_retries_a_locked_rename_aside(tmp_path, sleeps, monkeypatch):
    staging, final = _folders(tmp_path)
    monkeypatch.setattr(os, "rename", _Rename(_locked(), real=_REAL_RENAME))

    _publish_folder(str(staging), str(final))

    assert (final / "state.txt").read_text() == "new"
    assert _replaced(final) == []


def test_rename_aside_exhausted_names_the_folder_and_moves_nothing(tmp_path, sleeps, monkeypatch):
    staging, final = _folders(tmp_path)
    rename = _Rename(*_locks())
    monkeypatch.setattr(os, "rename", rename)

    with pytest.raises(CheckpointFolderLocked, match=final.name):
        _publish_folder(str(staging), str(final))

    assert rename.calls == len(_BACKOFF) + 1
    assert (final / "state.txt").read_text() == "old"
    assert (staging / "state.txt").read_text() == "new"


def test_move_exhausted_rolls_back_to_the_earlier_folder(tmp_path, sleeps, monkeypatch):
    staging, final = _folders(tmp_path)
    replace = _Rename(*_locks())
    monkeypatch.setattr(os, "replace", replace)

    with pytest.raises(CheckpointFolderLocked, match=final.name):
        _publish_folder(str(staging), str(final))

    assert replace.calls == len(_BACKOFF) + 1
    assert (final / "state.txt").read_text() == "old"
    assert _replaced(final) == []


def test_the_rollback_gets_its_own_retries(tmp_path, sleeps, monkeypatch):
    staging, final = _folders(tmp_path)
    monkeypatch.setattr(os, "replace", _Rename(*_locks()))
    monkeypatch.setattr(os, "rename", _Rename(None, _locked(), real=_REAL_RENAME))

    with pytest.raises(CheckpointFolderLocked):
        _publish_folder(str(staging), str(final))

    assert (final / "state.txt").read_text() == "old"
    assert sleeps == [*_BACKOFF, _BACKOFF[0]]


def test_move_and_rollback_exhausted_name_the_surviving_folder(tmp_path, sleeps, monkeypatch):
    staging, final = _folders(tmp_path)
    monkeypatch.setattr(os, "replace", _Rename(*_locks()))
    monkeypatch.setattr(os, "rename", _Rename(None, *_locks(), real=_REAL_RENAME))

    with pytest.raises(CheckpointFolderLocked) as caught:
        _publish_folder(str(staging), str(final))

    [survivor] = _replaced(final)
    assert survivor.name in str(caught.value)
    assert (survivor / "state.txt").read_text() == "old"


def test_a_publish_error_that_is_not_a_lock_keeps_its_type(tmp_path, sleeps, monkeypatch):
    staging, final = _folders(tmp_path)
    monkeypatch.setattr(os, "replace", _Rename(OSError(28, "disk full")))

    with pytest.raises(OSError, match="disk full") as caught:
        _publish_folder(str(staging), str(final))

    assert not isinstance(caught.value, CheckpointFolderLocked)
    assert sleeps == []
    assert (final / "state.txt").read_text() == "old"


# ── _atomic_write_bytes_file ─────────────────────────────────────────────


def test_atomic_write_retries_a_locked_replace(tmp_path, sleeps, monkeypatch):
    monkeypatch.setattr(os, "replace", _Rename(_locked(32), real=_REAL_REPLACE))
    target = tmp_path / "state.json"

    ckpt_mod._atomic_write_json(str(target), {"step": 1})

    assert json.loads(target.read_text()) == {"step": 1}
    assert sleeps == _BACKOFF[:1]


def test_atomic_write_removes_its_tmp_when_the_lock_outlasts_the_bound(tmp_path, sleeps, monkeypatch):
    monkeypatch.setattr(os, "replace", _Rename(*_locks()))
    target = tmp_path / "state.json"

    with pytest.raises(PermissionError):
        ckpt_mod._atomic_write_json(str(target), {"step": 1})

    assert not target.exists()
    assert not Path(f"{target}.tmp").exists()


# ── save_checkpoint ──────────────────────────────────────────────────────


def _lock_folder_moves(errors: list[PermissionError]):
    """``os.replace`` that locks only the move of a staging folder."""

    def replace(src, dst):
        if ckpt_mod.STAGING_MARK in os.path.basename(str(src)) and errors:
            raise errors.pop(0)
        _REAL_REPLACE(src, dst)

    return replace


def _save(run: Path, step: int) -> Path:
    model = nn.Linear(4, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    CheckpointManager(str(run)).save_checkpoint(
        step, {"model": model}, optimizer=opt, config={"lora_name": "t"}
    )
    return run / f"checkpoint-{step:06d}"


def test_a_real_save_survives_a_transient_lock_on_its_publish(tmp_path, sleeps, monkeypatch):
    monkeypatch.setattr(os, "replace", _lock_folder_moves(_locks(2)))

    folder = _save(tmp_path, 10)

    assert checkpoint_resumable(folder) == (True, "")


def test_a_lock_that_outlasts_the_bound_fails_the_save_by_name(tmp_path, sleeps, monkeypatch):
    earlier = _save(tmp_path, 5)
    monkeypatch.setattr(os, "replace", _lock_folder_moves(_locks()))

    with pytest.raises(CheckpointFolderLocked, match="checkpoint-000010"):
        _save(tmp_path, 10)

    assert checkpoint_resumable(earlier) == (True, "")
    assert not list(tmp_path.glob(f"*{ckpt_mod.STAGING_MARK}*"))
