"""Bounded streaming of uploaded files to a temporary path.

Every archive-ingest route read the upload to EOF. Streaming to a temp file
rather than into RAM was already right -- a dataset export can embed multi-GB
video -- but "stream until the client stops sending" is not a bound
(ARCHITECTURE D10 invariant 6: every queue, buffer and wait is bounded). A
single request could fill the volume before any archive limit was consulted,
because those limits live in the extractor and the extractor never ran.

This lives in layer L3, not beside the layer L0 guards, on purpose: it takes a
FastAPI ``UploadFile``, so it is HTTP-surface machinery, and pushing it into
``core`` would drag the web framework down into layer L0 for no gain. The
outbound URL guard went the other way because both the engine and the API need
it; only the API can use this one.

Cleanup is unconditional. A client that disconnects mid-upload leaves a partial
temp file, and enough of those fill a volume just as effectively as one large
one -- so callers get a context manager rather than a path plus a promise to
remember a ``finally``.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.core.logger import get_logger

logger = get_logger(__name__)

#: Ceiling on a single uploaded archive. High enough that a real video dataset
#: export passes -- the point is to bound the stream, not to police dataset
#: size. The extractor applies the limits that actually matter (volume-relative
#: expansion, member count, compression ratio) once the bytes are on disk.
MAX_UPLOAD_BYTES = 64 * 1024**3  # 64 GiB

CHUNK_BYTES = 1024 * 1024


@asynccontextmanager
async def spooled_upload(
    file: UploadFile,
    *,
    suffix: str = ".zip",
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> AsyncIterator[Path]:
    """Stream *file* to a temp path, bounded, and always clean up.

    Yields the path to the completed temp file. Raises ``HTTPException(413)``
    once the upload passes *max_bytes*, having written no more than one chunk
    beyond it.

    The cap is enforced on bytes ACTUALLY READ, never on ``Content-Length``: a
    header can lie, and a chunked upload need not send one at all, so trusting
    it would leave the real stream unbounded.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tmp.name)
    written = 0
    try:
        while chunk := await file.read(CHUNK_BYTES):
            written += len(chunk)
            if written > max_bytes:
                logger.warning(
                    "upload_rejected_too_large", limit=max_bytes, read=written
                )
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Upload exceeds the {max_bytes}-byte limit. Split the "
                        "export, or import from a path on the server instead."
                    ),
                )
            tmp.write(chunk)
        tmp.close()
        yield tmp_path
    finally:
        # Unconditional: covers the 413 above, a client disconnecting
        # mid-stream, and any failure in the caller's body. A partial temp file
        # left behind is the same disk-exhaustion problem in slow motion.
        try:
            tmp.close()
        except OSError:
            pass
        tmp_path.unlink(missing_ok=True)
