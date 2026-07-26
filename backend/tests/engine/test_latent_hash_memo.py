"""Task W2.T1 — hash_source_file must memoize on (path, size, mtime).

``LatentManager.hash_source_file`` is the biggest speed win identified in the
backend hardening review: it reads the ENTIRE source media file in 64KB
chunks and SHA-256s it purely to rebuild a cache filename, and it is called
every training step (per item, per control slot) via ``_fname_for`` from
``load_cached_latents``/``load_cached_latent_windows`` — even though the
digest is invariant for the life of a run unless the file itself changes.

The fix memoizes the digest keyed on ``(abspath, st_size, st_mtime_ns)`` so
an unchanged file is never re-read, while a modified file (new size and/or
mtime) still gets a fresh digest.

House contract: the latent cache is content-addressed and its filenames must
stay byte-identical for unchanged inputs, so an existing warm cache on a
user's disk keeps hitting after this change. The memoization must not alter
the digest VALUE the pre-memo code produced — only skip redundant re-reads.
"""

import hashlib
import time

import pytest

from app.engine.components.latents import LatentManager


def test_second_hash_serves_from_memo(tmp_path, monkeypatch):
    """A second call for the SAME (path, size, mtime) must not re-read the file."""
    f = tmp_path / "clip.bin"
    f.write_bytes(b"x" * 200_000)

    opens = {"n": 0}
    real_open = open

    def counting_open(*a, **k):
        opens["n"] += 1
        return real_open(*a, **k)

    monkeypatch.setattr("builtins.open", counting_open)

    h1 = LatentManager.hash_source_file(str(f))
    first = opens["n"]
    assert first >= 1  # sanity: the first call actually read the file

    h2 = LatentManager.hash_source_file(str(f))
    assert h1 == h2
    assert opens["n"] == first  # no second read — served from the memo


def test_modified_file_rehashes(tmp_path):
    """Same path, new content + mtime -> a NEW digest (memo must not go stale)."""
    f = tmp_path / "clip.bin"
    f.write_bytes(b"a" * 1000)
    h1 = LatentManager.hash_source_file(str(f))

    time.sleep(0.01)  # force a distinguishable st_mtime_ns
    f.write_bytes(b"b" * 1000)  # same size, different content/mtime
    h2 = LatentManager.hash_source_file(str(f))

    assert h1 != h2


def test_digest_value_matches_independent_sha256(tmp_path):
    """House contract: the memoized digest is byte-identical to a plain
    sha256 of the file contents — an existing on-disk cache filename (built
    from the pre-memo digest) must keep hitting after this change."""
    f = tmp_path / "photo.bin"
    content = b"\xff\xd8\xff\xe0" + (b"\x42" * 50_000)
    f.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    assert LatentManager.hash_source_file(str(f)) == expected

    # ...and the memo-served second call still matches (not corrupted by the cache).
    assert LatentManager.hash_source_file(str(f)) == expected


def test_missing_file_raises_without_being_swallowed(tmp_path):
    """A vanished file must still raise loudly — the os.stat() added for the
    memo key replaces the read as the first I/O call, but must not turn a
    hard failure into a silent success."""
    missing = tmp_path / "gone.bin"
    with pytest.raises(OSError):
        LatentManager.hash_source_file(str(missing))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
