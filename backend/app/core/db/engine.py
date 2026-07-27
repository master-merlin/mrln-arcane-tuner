"""SQLite database engine — singleton connection manager.

Provides thread-safe access to the application database with:
- WAL journal mode for concurrent read/write
- Foreign key enforcement
- Thread-local connections (sqlite3 is not thread-safe)
- Write serialization via threading.Lock
"""

import os
import sqlite3
import threading

import structlog

logger = structlog.get_logger(__name__)


class DatabaseEngine:
    """Singleton SQLite connection manager.

    Usage::

        db = DatabaseEngine.get_instance()
        conn = db.connection()        # thread-local read connection
        with db.write() as conn:      # exclusive write connection
            conn.execute(...)
    """

    _instance: "DatabaseEngine | None" = None
    _lock = threading.Lock()

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            env_path = os.environ.get("MRLN_DB_PATH")
            if env_path:
                db_path = env_path
            else:
                root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                db_path = os.path.join(root, "arcane_tuner.db")

        self.db_path = db_path
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "DatabaseEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = DatabaseEngine()
        return cls._instance

    # ── Lifecycle ────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create DB file (if needed), set pragmas, run migrations."""
        if self._initialized:
            return

        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        conn = self._make_connection()

        # journal_mode=WAL is persisted in the database file itself (not
        # per-connection like the pragmas _make_connection already sets on
        # every connection), so it only needs setting once here.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()

        # Run schema migrations
        from .migrations import run_migrations
        run_migrations(self)

        self._initialized = True
        logger.info("database_initialized", path=self.db_path)

    def close(self) -> None:
        """Close thread-local connection if open."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ── Connections ──────────────────────────────────────────────────

    def _make_connection(self) -> sqlite3.Connection:
        """Create a new connection with standard settings.

        Pragmas are per-connection in SQLite — ``initialize()`` used to set
        ``synchronous``/``busy_timeout`` only on its own throwaway init
        connection (closed immediately after), so every REAL connection the
        app actually uses (every thread-local read via ``connection()``,
        every write via ``write()``) ran with SQLite's defaults instead.
        Setting them here applies them to every connection this factory
        produces.
        """
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=10.0,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def connection(self) -> sqlite3.Connection:
        """Get thread-local read connection (lazy-created)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._make_connection()
            self._local.conn = conn
        return conn

    class _WriteContext:
        """Context manager that serializes writes via a lock."""

        def __init__(self, engine: "DatabaseEngine"):
            self._engine = engine
            self._conn: sqlite3.Connection | None = None

        def __enter__(self) -> sqlite3.Connection:
            self._engine._write_lock.acquire()
            try:
                self._conn = self._engine._make_connection()
            except BaseException:
                # Connect failed — the lock must not leak, or every subsequent
                # db.write() in the process blocks forever.
                self._engine._write_lock.release()
                raise
            return self._conn

        def __exit__(self, exc_type, exc_val, exc_tb):
            try:
                try:
                    if exc_type is None:
                        self._conn.commit()
                    else:
                        self._conn.rollback()
                finally:
                    self._conn.close()
            finally:
                # Always release, even if commit()/rollback()/close() raises
                # (e.g. a cross-process SQLITE_BUSY at commit) — an unreleased
                # lock wedges every future write until the backend restarts.
                self._engine._write_lock.release()
            return False  # don't suppress exceptions

    def write(self) -> "_WriteContext":
        """Get an exclusive write connection as a context manager.

        Commits on success, rolls back on exception::

            with db.write() as conn:
                conn.execute("INSERT INTO ...")
        """
        return self._WriteContext(self)


def get_db() -> DatabaseEngine:
    """Convenience accessor for FastAPI dependency injection."""
    return DatabaseEngine.get_instance()
