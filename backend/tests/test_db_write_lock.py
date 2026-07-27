"""Regression: the write-lock in ``DatabaseEngine._WriteContext`` must be
released on EVERY path — including commit()/connect() failures.

Before the fix, ``__exit__`` ran commit()/rollback()/close() with no
try/finally around the lock release, and ``__enter__`` acquired the lock
before ``_make_connection()`` with no cleanup on a connect failure. Either
failure leaked the lock, permanently wedging every subsequent ``db.write()``
in the process (job status persistence, scans, everything) until restart.
This is a realistic path: the trainer subprocess writes the same SQLite file,
so a cross-process SQLITE_BUSY at commit (after the 10s busy timeout) can hit
the API process's write.
"""

import sqlite3

import pytest

from app.core.db.engine import DatabaseEngine


@pytest.fixture()
def db_engine(tmp_path):
    """Isolated DatabaseEngine backed by a temp SQLite file (never initialized —
    these tests replace ``_make_connection`` entirely, so no real file I/O
    happens)."""
    db_path = str(tmp_path / "test.db")
    return DatabaseEngine(db_path)


class _FakeConn:
    """Minimal stand-in for ``sqlite3.Connection`` used by ``_WriteContext``."""

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


def _raise_oserror():
    raise OSError("connect failed")


def test_lock_released_when_commit_raises(db_engine, monkeypatch):
    class _BoomConn(_FakeConn):
        def commit(self):
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db_engine, "_make_connection", lambda: _BoomConn())
    with pytest.raises(sqlite3.OperationalError):
        with db_engine.write():
            pass
    # The lock MUST be free again:
    assert db_engine._write_lock.acquire(timeout=1)
    db_engine._write_lock.release()


def test_lock_released_when_connect_raises(db_engine, monkeypatch):
    monkeypatch.setattr(db_engine, "_make_connection", _raise_oserror)
    with pytest.raises(OSError):
        db_engine.write().__enter__()
    assert db_engine._write_lock.acquire(timeout=1)
    db_engine._write_lock.release()


def test_lock_released_when_close_raises(db_engine, monkeypatch):
    """close() raising (e.g. I/O error flushing WAL) must not leak the lock
    either — it sits between commit/rollback and the lock release."""

    class _CloseBoomConn(_FakeConn):
        def close(self):
            raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(db_engine, "_make_connection", lambda: _CloseBoomConn())
    with pytest.raises(sqlite3.OperationalError):
        with db_engine.write():
            pass
    assert db_engine._write_lock.acquire(timeout=1)
    db_engine._write_lock.release()


def test_lock_released_after_normal_write_success(db_engine, monkeypatch):
    """Sanity: the happy path still commits, closes, and releases exactly once."""
    conn = _FakeConn()
    calls: list[str] = []
    conn.commit = lambda: calls.append("commit")
    conn.close = lambda: calls.append("close")
    monkeypatch.setattr(db_engine, "_make_connection", lambda: conn)

    with db_engine.write() as c:
        assert c is conn

    assert calls == ["commit", "close"]
    assert db_engine._write_lock.acquire(timeout=1)
    db_engine._write_lock.release()
