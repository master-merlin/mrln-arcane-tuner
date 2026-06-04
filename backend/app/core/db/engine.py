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

        # Pragmas for performance + safety
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
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
        """Create a new connection with standard settings."""
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=10.0,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
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
            self._conn = self._engine._make_connection()
            return self._conn

        def __exit__(self, exc_type, exc_val, exc_tb):
            conn = self._conn
            if conn is not None:
                if exc_type is None:
                    conn.commit()
                else:
                    conn.rollback()
                conn.close()
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
