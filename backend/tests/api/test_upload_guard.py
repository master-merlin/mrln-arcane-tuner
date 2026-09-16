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
from uuid import uuid4

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


def _unique_suffix() -> str:
    """A suffix no other test — and no other xdist worker — can produce.

    ``tempfile.gettempdir()`` is one directory per MACHINE, so a glob for
    ``*.ziptest`` also sees the IN-FLIGHT temp file of a sibling worker running
    these same tests under ``-n 4`` and reports it as this test's leak
    (observed 2026-09-16: ``test_client_disconnect_mid_upload_leaves_no_temp_file``
    failed on ``D:/docker/tmp/tmpw0v9oog2.ziptest``, a file it never created).
    The invariant is "the files THIS upload left behind", so the identity of
    this test's uploads — its suffix — is what the glob must be keyed on.
    """
    return f".{uuid4().hex}.ziptest"


def _temp_files(suffix: str) -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob(f"*{suffix}"))


@pytest.mark.asyncio
async def test_upload_under_the_limit_is_written_whole():
    payload = b"x" * (3 * CHUNK_BYTES + 17)
    async with spooled_upload(
        FakeUpload(payload), suffix=_unique_suffix(), max_bytes=len(payload) + 1
    ) as path:
        assert path.read_bytes() == payload


@pytest.mark.asyncio
async def test_upload_over_the_limit_is_413():
    payload = b"x" * (2 * CHUNK_BYTES)
    with pytest.raises(HTTPException) as exc:
        async with spooled_upload(
            FakeUpload(payload), suffix=_unique_suffix(), max_bytes=CHUNK_BYTES
        ):
            raise AssertionError("body must not run for an oversized upload")
    assert exc.value.status_code == 413
    # The message must state the limit, or a user cannot tell how much to trim.
    assert str(CHUNK_BYTES) in str(exc.value.detail)


@pytest.mark.asyncio
async def test_oversized_upload_leaves_no_temp_file():
    """The cap is worthless if the rejected bytes stay on disk."""
    suffix = _unique_suffix()
    before = _temp_files(suffix)
    payload = b"x" * (4 * CHUNK_BYTES)
    with pytest.raises(HTTPException):
        async with spooled_upload(
            FakeUpload(payload), suffix=suffix, max_bytes=CHUNK_BYTES
        ):
            pass
    assert _temp_files(suffix) == before


@pytest.mark.asyncio
async def test_client_disconnect_mid_upload_leaves_no_temp_file():
    """The case the plan called out: a partial file is slow disk exhaustion."""
    suffix = _unique_suffix()
    before = _temp_files(suffix)
    payload = b"x" * (8 * CHUNK_BYTES)
    with pytest.raises(ConnectionResetError):
        async with spooled_upload(
            FakeUpload(payload, fail_after=2), suffix=suffix
        ):
            raise AssertionError("body must not run when the client vanished")
    assert _temp_files(suffix) == before


@pytest.mark.asyncio
async def test_a_sibling_workers_temp_file_is_not_this_tests_leak():
    """The parallel-gate defect these tests carried, pinned (LANE-63).

    Under ``-n 4`` a sibling worker holds its own ``*.ziptest`` open in the
    SAME machine temp dir for the length of its upload. A leak check that
    globs the whole suffix therefore fails on a file it never created — a
    red gate that says nothing about the code. Here that file is present for
    real; the check must still be about this test's own uploads, and must
    not touch the stranger's file.
    """
    foreign = Path(tempfile.gettempdir()) / f"tmp{uuid4().hex}.ziptest"

    class SiblingWorkerUpload(FakeUpload):
        """Its first chunk is when the OTHER worker opens its own temp file —
        i.e. after this test took its 'before' snapshot, which is what makes
        the stranger indistinguishable from a leak to a whole-suffix glob."""

        async def read(self, size: int = -1) -> bytes:
            if self.reads == 0:
                foreign.write_bytes(b"a sibling xdist worker's upload, in flight")
            return await super().read(size)

    try:
        suffix = _unique_suffix()
        before = _temp_files(suffix)
        with pytest.raises(ConnectionResetError):
            async with spooled_upload(
                SiblingWorkerUpload(b"x" * (8 * CHUNK_BYTES), fail_after=2),
                suffix=suffix,
            ):
                raise AssertionError("body must not run when the client vanished")
        assert _temp_files(suffix) == before
        assert foreign.exists(), "the check reached into another worker's files"
    finally:
        foreign.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_failure_inside_the_caller_body_still_cleans_up():
    """Cleanup is unconditional, not just on the guard's own error paths."""
    suffix = _unique_suffix()
    before = _temp_files(suffix)
    with pytest.raises(ZeroDivisionError):
        async with spooled_upload(FakeUpload(b"small"), suffix=suffix):
            1 / 0  # noqa: B018
    assert _temp_files(suffix) == before


@pytest.mark.asyncio
async def test_success_path_also_removes_the_temp_file():
    """Prove the negative: the happy path is not the leak either."""
    suffix = _unique_suffix()
    before = _temp_files(suffix)
    async with spooled_upload(FakeUpload(b"payload"), suffix=suffix) as path:
        assert path.exists()
        kept = path
    assert not kept.exists()
    assert _temp_files(suffix) == before


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
        async with spooled_upload(upload, suffix=_unique_suffix(),
                                  max_bytes=CHUNK_BYTES):
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
            Watcher(payload), suffix=_unique_suffix(), max_bytes=2 * CHUNK_BYTES
        ):
            pass

    # 2 chunks fit under the limit, the 3rd trips it; nothing beyond is read.
    assert sum(seen) <= 3 * CHUNK_BYTES
