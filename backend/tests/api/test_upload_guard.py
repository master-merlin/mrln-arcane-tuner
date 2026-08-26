"""Uploads are bounded, and never leave a partial temp file behind.

Streaming an archive to a temp file instead of into RAM was already right — a
dataset export can embed multi-GB video. What was missing is that "read until
the client stops sending" is not a bound (ARCHITECTURE D10 invariant 6). A
single request could fill the volume *before* any extractor limit applied,
because those limits live in the extractor and the extractor never ran.

The cleanup half matters as much as the cap: a client that disconnects
mid-upload leaves a partial file, and enough of those exhaust a volume just as
effectively as one oversized one.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api._upload_guard import CHUNK_BYTES, spooled_upload


class FakeUpload:
    """Minimal UploadFile stand-in.

    Not a mock of the guard's own logic — it only supplies bytes, which is the
    seam a real client is on the other side of. `read` honours the requested
    size so the guard's chunking is exercised rather than bypassed.
    """

    def __init__(self, payload: bytes, *, fail_after: int | None = None):
        self._buf = payload
        self._pos = 0
        self._fail_after = fail_after
        self.reads = 0

    async def read(self, size: int = -1) -> bytes:
        self.reads += 1
        if self._fail_after is not None and self.reads > self._fail_after:
            raise ConnectionResetError("client went away mid-upload")
        if self._pos >= len(self._buf):
            return b""
        end = len(self._buf) if size < 0 else min(self._pos + size, len(self._buf))
        chunk = self._buf[self._pos : end]
        self._pos = end
        return chunk


def _temp_files() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob("*.ziptest"))


@pytest.mark.asyncio
async def test_upload_under_the_limit_is_written_whole():
    payload = b"x" * (3 * CHUNK_BYTES + 17)
    async with spooled_upload(
        FakeUpload(payload), suffix=".ziptest", max_bytes=len(payload) + 1
    ) as path:
        assert path.read_bytes() == payload


@pytest.mark.asyncio
async def test_upload_over_the_limit_is_413():
    payload = b"x" * (2 * CHUNK_BYTES)
    with pytest.raises(HTTPException) as exc:
        async with spooled_upload(
            FakeUpload(payload), suffix=".ziptest", max_bytes=CHUNK_BYTES
        ):
            raise AssertionError("body must not run for an oversized upload")
    assert exc.value.status_code == 413
    # The message must state the limit, or a user cannot tell how much to trim.
    assert str(CHUNK_BYTES) in str(exc.value.detail)


@pytest.mark.asyncio
async def test_oversized_upload_leaves_no_temp_file():
    """The cap is worthless if the rejected bytes stay on disk."""
    before = _temp_files()
    payload = b"x" * (4 * CHUNK_BYTES)
    with pytest.raises(HTTPException):
        async with spooled_upload(
            FakeUpload(payload), suffix=".ziptest", max_bytes=CHUNK_BYTES
        ):
            pass
    assert _temp_files() == before


@pytest.mark.asyncio
async def test_client_disconnect_mid_upload_leaves_no_temp_file():
    """The case the plan called out: a partial file is slow disk exhaustion."""
    before = _temp_files()
    payload = b"x" * (8 * CHUNK_BYTES)
    with pytest.raises(ConnectionResetError):
        async with spooled_upload(
            FakeUpload(payload, fail_after=2), suffix=".ziptest"
        ):
            raise AssertionError("body must not run when the client vanished")
    assert _temp_files() == before


@pytest.mark.asyncio
async def test_failure_inside_the_caller_body_still_cleans_up():
    """Cleanup is unconditional, not just on the guard's own error paths."""
    before = _temp_files()
    with pytest.raises(ZeroDivisionError):
        async with spooled_upload(FakeUpload(b"small"), suffix=".ziptest"):
            1 / 0  # noqa: B018
    assert _temp_files() == before


@pytest.mark.asyncio
async def test_success_path_also_removes_the_temp_file():
    """Prove the negative: the happy path is not the leak either."""
    before = _temp_files()
    async with spooled_upload(FakeUpload(b"payload"), suffix=".ziptest") as path:
        assert path.exists()
        kept = path
    assert not kept.exists()
    assert _temp_files() == before


@pytest.mark.asyncio
async def test_limit_is_on_bytes_read_not_a_declared_length():
    """A Content-Length header can lie, and a chunked upload may omit it.

    Pinned because trusting the header is the usual shortcut, and it leaves the
    real stream unbounded — exactly the defect being fixed.
    """
    payload = b"x" * (3 * CHUNK_BYTES)
    upload = FakeUpload(payload)
    upload.headers = {"content-length": "1"}  # the lie
    with pytest.raises(HTTPException) as exc:
        async with spooled_upload(upload, suffix=".ziptest", max_bytes=CHUNK_BYTES):
            pass
    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_no_more_than_one_chunk_is_written_past_the_limit():
    """Bound the overshoot, not just the total.

    Checking only after the whole stream would make the 'limit' advisory.
    """
    seen: list[int] = []
    payload = b"x" * (10 * CHUNK_BYTES)

    class Watcher(FakeUpload):
        async def read(self, size: int = -1) -> bytes:
            chunk = await super().read(size)
            seen.append(len(chunk))
            return chunk

    with pytest.raises(HTTPException):
        async with spooled_upload(
            Watcher(payload), suffix=".ziptest", max_bytes=2 * CHUNK_BYTES
        ):
            pass

    # 2 chunks fit under the limit, the 3rd trips it; nothing beyond is read.
    assert sum(seen) <= 3 * CHUNK_BYTES
