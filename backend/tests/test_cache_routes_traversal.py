"""Purge segments must never escape the dataset .cache root."""

from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api.cache_routes import _purge_cache, _validate_cache_segment

try:
    import _winapi
except ImportError:  # pragma: no cover - this project targets Windows
    _winapi = None


@pytest.mark.parametrize("bad", ["..", ".", "a/b", "a\\b", "..\\..", ""])
def test_bad_segments_rejected(bad):
    with pytest.raises(HTTPException) as exc:
        _validate_cache_segment(bad)
    assert exc.value.status_code == 400


@pytest.mark.parametrize("ok", ["qwen_image", "v3", "latents", "original_512x512"])
def test_plain_names_pass(ok):
    assert _validate_cache_segment(ok) == ok


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


# ── Server-discovered names: skip-and-continue, never abort the purge ──────
#
# `_purge_cache` enumerates `models`/`types` from disk via `iterdir()` when
# the request doesn't filter them. Those names never passed through client
# validation — a legacy directory, a manual copy, or a future naming
# convention the segment regex doesn't admit must not abort an in-progress
# destructive purge after earlier siblings have already been deleted.


def test_purge_skips_invalid_discovered_model_and_continues(tmp_path):
    """A non-conforming on-disk model dir (discovered via iterdir(), not
    requested by the client) is skipped — the purge still completes and
    still removes its well-formed sibling."""
    root = tmp_path / ".cache"
    _write(root / "sdxl" / "1.0.0" / "latents" / "orig" / "a.npy", b"L" * 10)

    # A directory name the segment regex rejects (leading dot), sitting
    # right next to a legitimate model — both are discovered the same way
    # since models=None.
    bad_dir = root / ".legacy_cache_dir"
    _write(bad_dir / "stray.npy", b"x")

    result = _purge_cache(root, models=None, types=None, variants=None)

    assert not (root / "sdxl").exists()  # legitimate sibling still purged
    assert bad_dir.exists()  # non-conforming dir left alone, not fatal
    assert result["deleted"] > 0


def test_purge_client_supplied_invalid_model_still_aborts(tmp_path):
    """The fail-fast security property is unchanged: a CLIENT-supplied
    segment that fails validation still 400s before any path is built —
    only server-discovered names get skip-and-continue treatment."""
    root = tmp_path / ".cache"
    _write(root / "sdxl" / "1.0.0" / "latents" / "orig" / "a.npy", b"L" * 10)

    with pytest.raises(HTTPException) as exc:
        _purge_cache(root, models=["../../escape"], types=None, variants=None)
    assert exc.value.status_code == 400

    # Fail-fast means nothing was deleted before the guard fired.
    assert (root / "sdxl").exists()


# ── Containment layer: `validate_path_within` must actually be exercised ───
#
# A purely name-based check (`_validate_cache_segment`) cannot catch an
# on-disk directory whose NAME is perfectly plain but which is itself a
# symlink/junction resolving outside the cache root. These tests drive a
# real NTFS junction through `_purge_cache` end to end; they fail if the
# `validate_path_within(...)` calls were ever removed from `_purge_cache`
# (the junction's target would then get deleted / silently traversed).


def _seed_escaping_junction(tmp_path: Path) -> tuple[Path, Path]:
    """Build a `.cache` root with a legitimate "sdxl" model plus a junction
    named "escaped_model" that resolves to a `victim/` tree OUTSIDE the
    cache root — itself shaped like a real cache subtree (version/type/file)
    so that, if containment were skipped, `_purge_cache`'s normal delete
    logic would actually remove real files out there. Returns
    ``(cache_root, victim_file)``. Skips if this environment cannot create
    an NTFS junction (e.g. no privilege).
    """
    cache_root = tmp_path / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    _write(cache_root / "sdxl" / "1.0.0" / "latents" / "orig" / "a.npy", b"L" * 10)

    victim_root = tmp_path / "victim"
    victim_file = victim_root / "1.0.0" / "latents" / "orig" / "victim.npy"
    _write(victim_file, b"do-not-delete")

    if _winapi is None:
        pytest.skip("_winapi unavailable — this guard targets Windows/NTFS junctions")

    junction_path = cache_root / "escaped_model"
    try:
        _winapi.CreateJunction(str(victim_root), str(junction_path))
    except OSError as exc:
        pytest.skip(f"cannot create an NTFS junction in this environment: {exc}")

    return cache_root, victim_file


def test_purge_containment_blocks_discovered_junction_escape(tmp_path):
    """A server-discovered "model" dir that is actually a junction escaping
    the cache root is skipped (not deleted, not fatal) — the containment
    check, not the regex, is what catches this, since "escaped_model" is a
    perfectly plain name."""
    cache_root, victim_file = _seed_escaping_junction(tmp_path)

    result = _purge_cache(cache_root, models=None, types=None, variants=None)

    assert victim_file.exists()  # the escape was blocked, nothing there was touched
    assert not (cache_root / "sdxl").exists()  # skip-and-continue: sibling still purged
    assert result["deleted"] > 0  # from the legitimate sibling only


def test_purge_containment_blocks_client_supplied_junction_escape(tmp_path):
    """The same junction escape, but requested explicitly by name via the
    client-supplied `models` filter, still 403s (fail-fast) rather than
    being silently skipped."""
    cache_root, victim_file = _seed_escaping_junction(tmp_path)

    with pytest.raises(HTTPException) as exc:
        _purge_cache(cache_root, models=["escaped_model"], types=None, variants=None)
    assert exc.value.status_code == 403

    assert victim_file.exists()
