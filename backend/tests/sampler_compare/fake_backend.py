"""A stand-in for the comparison's backend in its tests. Standard library only.

Answers ``GET /api/datasets/<name>`` on ``127.0.0.1:$PORT`` from the database ``MRLN_DB_PATH``
names; ``FAKE_BACKEND=never_answers`` binds nothing.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PREFIX = "/api/datasets/"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server API
        if not self.path.startswith(PREFIX):
            self.send_error(404)
            return
        name = urllib.parse.unquote(self.path[len(PREFIX):])
        db = sqlite3.connect(f"{Path(os.environ['MRLN_DB_PATH']).resolve().as_uri()}?mode=ro", uri=True)
        try:
            row = db.execute("select path from datasets where name=?", (name,)).fetchone()
        finally:
            db.close()
        if row is None:
            self.send_error(404)
            return
        body = json.dumps({"name": name, "path": row[0]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs):
        pass


def _started(port: int) -> None:
    sys.stdout.write(json.dumps({"event": "fake_backend_started", "pid": os.getpid(), "port": str(port),
                                 "db": os.environ.get("MRLN_DB_PATH")}) + "\n")
    sys.stdout.flush()


#: The exit code of the real server whose bind is refused: uvicorn's ``config.STARTUP_FAILURE``,
#: with which its single-process start exits at once (``uvicorn/server.py``, no retry).
STARTUP_FAILURE = 3

#: Whether the real server's socket has ``SO_REUSEADDR``: asyncio's ``create_server`` sets it
#: exactly when ``os.name == "posix" and sys.platform != "cygwin"`` (``asyncio/base_events.py``).
#: With it, a closed connection still waiting on the port blocks no new server; without it (Windows),
#: a reuse bind could share a port another server holds.
REUSE_ADDRESS = os.name == "posix" and sys.platform != "cygwin"


class _Server(ThreadingHTTPServer):
    allow_reuse_address = REUSE_ADDRESS


def main() -> int:
    port = int(os.environ["PORT"])
    if os.environ.get("FAKE_BACKEND") == "never_answers":
        _started(port)
        while True:
            time.sleep(1)
    try:
        server = _Server(("127.0.0.1", port), Handler)  # bound before it says it started
    except OSError as exc:
        sys.stderr.write(f"fake_backend: cannot bind 127.0.0.1:{port}: {exc}\n")
        return STARTUP_FAILURE
    _started(port)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
