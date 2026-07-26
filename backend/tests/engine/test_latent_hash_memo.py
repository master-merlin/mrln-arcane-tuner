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


# ── Gap fix: latent_filename's extra_key branch must share the same memo ───
#
# video_trim_extra_key() (pipeline_data.py) returns a non-empty extra_key for
# EVERY video item, so every Bernini-R / WAN / LTX video run routes through
# latent_filename's `if extra_key:` branch — which, pre-fix, ran its OWN
# inline 64KB-chunked read + SHA-256 of the source file on every single call,
# completely bypassing hash_source_file's memo. These tests pin that the
# extra_key branch now reads the file at most once per (path, size, mtime),
# while still producing sha256(file_bytes + extra_key) — bit-identical to
# the pre-fix digest, so a user's existing warm video-latent cache keeps
# hitting.


def _extra_key_expected_filename(img_id: str, content: bytes, extra_key: str) -> str:
    """Reference implementation: the exact pre-fix algorithm, independently
    computed (NOT by calling latent_filename)."""
    h = hashlib.sha256(content)
    h.update(extra_key.encode("utf-8"))
    return f"{img_id}_{h.hexdigest()[:16]}.safetensors"


def test_extra_key_second_call_reads_file_once(tmp_path, monkeypatch):
    """A video-shaped call (non-empty extra_key) for the SAME unchanged file
    must not re-read it on the second lookup — this is the actual per-step
    hot path for video/Bernini-R runs."""
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"v" * 300_000)

    opens = {"n": 0}
    real_open = open

    def counting_open(*a, **k):
        opens["n"] += 1
        return real_open(*a, **k)

    monkeypatch.setattr("builtins.open", counting_open)

    name1 = LatentManager.latent_filename("clip.mp4", str(f), extra_key="t0.0-1.625")
    first = opens["n"]
    assert first >= 1  # sanity: the first call actually read the file

    name2 = LatentManager.latent_filename("clip.mp4", str(f), extra_key="t0.0-1.625")
    assert name1 == name2
    assert opens["n"] == first  # no second read — served from the memo


def test_extra_key_digest_matches_pre_fix_algorithm(tmp_path):
    """Critical pin: the returned filename must equal what the OLD inline
    algorithm (full read, then h.update(extra_key)) would have produced —
    computed independently here, not by calling the function twice."""
    f = tmp_path / "clip.mp4"
    content = b"\x00\x00\x00\x18ftypmp42" + (b"\x37" * 40_000)
    f.write_bytes(content)

    extra_key = "t0.0-1.625"
    expected = _extra_key_expected_filename("clip.mp4", content, extra_key)

    # Cold call.
    assert LatentManager.latent_filename("clip.mp4", str(f), extra_key) == expected
    # Memo-served call must match too (not corrupted by caching).
    assert LatentManager.latent_filename("clip.mp4", str(f), extra_key) == expected


def test_extra_key_different_values_differ_and_match_independent_expectations(
    tmp_path,
):
    """Two different extra_keys for the SAME file source must produce
    different filenames (the tiled/trimmed-window collision guarantee), AND
    each must independently match sha256(file_bytes + that extra_key) — this
    is the .copy() hazard: if the memoized hasher were mutated in place
    instead of copied, the SECOND call's digest would silently include the
    FIRST call's extra_key bytes too."""
    f = tmp_path / "clip.mp4"
    content = b"\x00\x00\x00\x18ftypmp42" + (b"\x99" * 40_000)
    f.write_bytes(content)

    key_a = "t0.0-1.625"
    key_b = "t1.625-3.25"

    name_a = LatentManager.latent_filename("tiled_src", str(f), key_a)
    name_b = LatentManager.latent_filename("tiled_src", str(f), key_b)

    assert name_a != name_b
    assert name_a == _extra_key_expected_filename("tiled_src", content, key_a)
    assert name_b == _extra_key_expected_filename("tiled_src", content, key_b)


def test_extra_key_modified_file_rehashes(tmp_path):
    """Same path, new content + mtime -> a NEW digest on the extra_key path
    too (the memo must not go stale for video sources any more than it does
    for the plain hash_source_file path)."""
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"a" * 1000)
    extra_key = "t0.0-1.625"
    name1 = LatentManager.latent_filename("clip.mp4", str(f), extra_key)

    time.sleep(0.01)  # force a distinguishable st_mtime_ns
    f.write_bytes(b"b" * 1000)  # same size, different content/mtime
    name2 = LatentManager.latent_filename("clip.mp4", str(f), extra_key)

    assert name1 != name2
    content_after = f.read_bytes()
    assert name2 == _extra_key_expected_filename("clip.mp4", content_after, extra_key)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
