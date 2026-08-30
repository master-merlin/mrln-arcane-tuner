"""Tests for the dataset thumbnails module."""

from __future__ import annotations

from pathlib import Path
from PIL import Image

from app.core.dataset import thumbnails


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_image(path: Path, size: tuple[int, int] = (800, 600), fmt: str = "JPEG") -> None:
    """Write a tiny PIL image to *path*."""
    img = Image.new("RGB", size, (123, 45, 67))
    img.save(path, fmt)


# ── Path helpers ────────────────────────────────────────────────────────


def test_thumbnail_dir_returns_dot_thumbnails_subfolder(tmp_path):
    assert thumbnails.thumbnail_dir(str(tmp_path)) == tmp_path / ".thumbnails"


def test_thumbnail_path_for_uses_stem_and_webp(tmp_path):
    # The size is a directory, not a name suffix — see the collision guards
    # at the bottom of this file for why.
    expected = tmp_path / ".thumbnails" / "256" / "img_001.webp"
    assert thumbnails.thumbnail_path_for(str(tmp_path), "img_001.jpg") == expected


def test_thumbnail_path_for_puts_the_size_in_a_directory(tmp_path):
    assert thumbnails.thumbnail_path_for(str(tmp_path), "img.jpg", 1024) == (
        tmp_path / ".thumbnails" / "1024" / "img.webp"
    )


def test_thumbnail_path_for_refuses_an_unlisted_edge(tmp_path):
    """The edge is a path component; the allowlist is enforced here too."""
    import pytest

    with pytest.raises(ValueError, match="max_edge"):
        thumbnails.thumbnail_path_for(str(tmp_path), "img.jpg", 999)
    with pytest.raises(ValueError, match="max_edge"):
        thumbnails.thumbnail_path_for(str(tmp_path), "img.jpg", "../../etc")


def test_thumbnail_path_for_normalizes_backslash_keys(tmp_path):
    # rel_path may arrive with Windows-style backslashes from OS APIs;
    # the helper normalizes them before building the name. Subdir sources
    # are namespaced (control/img1.jpg must not collide with img1.png),
    # so both separator styles resolve to the same prefixed thumbnail.
    expected = tmp_path / ".thumbnails" / "256" / "sub__img_001.webp"
    assert thumbnails.thumbnail_path_for(str(tmp_path), "sub\\img_001.png") == expected
    assert thumbnails.thumbnail_path_for(str(tmp_path), "sub/img_001.png") == expected


# ── generate_thumbnail (images) ─────────────────────────────────────────


def test_generate_thumbnail_creates_webp_with_max_edge_256(tmp_path):
    src = tmp_path / "src.jpg"
    _make_image(src, size=(800, 600))
    dst = tmp_path / ".thumbnails" / "src.webp"

    ok = thumbnails.generate_thumbnail(str(src), dst)

    assert ok is True
    assert dst.exists()
    with Image.open(dst) as img:
        assert img.format == "WEBP"
        # Long edge clamped to 256, aspect ratio preserved
        assert max(img.size) == 256
        assert img.size == (256, 192)


def test_generate_thumbnail_creates_parent_dir_lazily(tmp_path):
    src = tmp_path / "src.jpg"
    _make_image(src)
    dst = tmp_path / ".thumbnails" / "nested" / "src.webp"

    ok = thumbnails.generate_thumbnail(str(src), dst)

    assert ok is True
    assert dst.exists()


def test_generate_thumbnail_returns_false_on_corrupt_source(tmp_path):
    src = tmp_path / "broken.jpg"
    src.write_bytes(b"not a valid image")
    dst = tmp_path / ".thumbnails" / "broken.webp"

    ok = thumbnails.generate_thumbnail(str(src), dst)

    assert ok is False
    assert not dst.exists()


def test_generate_thumbnail_atomic_write_no_partial_files(tmp_path):
    """After success, no .tmp files should remain."""
    src = tmp_path / "src.png"
    _make_image(src, fmt="PNG")
    dst = tmp_path / ".thumbnails" / "src.webp"

    thumbnails.generate_thumbnail(str(src), dst)

    tmp_files = list(dst.parent.glob("*.tmp"))
    assert tmp_files == []


# ── generate_thumbnail (GIF) ─────────────────────────────────────────────


def test_generate_thumbnail_handles_gif(tmp_path):
    """A GIF's first frame is extracted (now via PyAV — .gif joined the
    canonical VIDEO_EXTENSIONS set used for first-frame thumbnails)."""
    src = tmp_path / "src.gif"
    frames = [
        Image.new("RGB", (400, 300), (255, 0, 0)),
        Image.new("RGB", (400, 300), (0, 255, 0)),
    ]
    frames[0].save(src, save_all=True, append_images=frames[1:], format="GIF")

    dst = tmp_path / ".thumbnails" / "src.webp"
    ok = thumbnails.generate_thumbnail(str(src), dst)

    assert ok is True
    assert dst.exists()
    with Image.open(dst) as img:
        assert max(img.size) == 256


# ── generate_thumbnail (MP4) ─────────────────────────────────────────────


def test_generate_thumbnail_handles_mp4_first_frame(tmp_path):
    """Encode a 2-frame MP4 with PyAV and verify thumbnail extraction."""
    import av
    import numpy as np

    src = tmp_path / "clip.mp4"
    with av.open(str(src), mode="w") as container:
        stream = container.add_stream("h264", rate=24)
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = "yuv420p"

        for color in ((200, 50, 50), (50, 200, 50)):
            frame_arr = np.full((240, 320, 3), color, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    dst = tmp_path / ".thumbnails" / "clip.webp"
    ok = thumbnails.generate_thumbnail(str(src), dst)

    assert ok is True
    assert dst.exists()
    with Image.open(dst) as img:
        assert max(img.size) == 256


# ── ensure_thumbnail ────────────────────────────────────────────────


def test_ensure_thumbnail_generates_when_missing(tmp_path):
    _make_image(tmp_path / "a.jpg")

    result = thumbnails.ensure_thumbnail(str(tmp_path), "a.jpg")

    assert result is not None
    assert result.exists()
    assert result == tmp_path / ".thumbnails" / "256" / "a.webp"


def test_ensure_thumbnail_returns_existing_without_regeneration(tmp_path):
    """Second call must reuse the existing thumbnail (mtime unchanged)."""
    _make_image(tmp_path / "a.jpg")

    first = thumbnails.ensure_thumbnail(str(tmp_path), "a.jpg")
    assert first is not None
    first_mtime = first.stat().st_mtime_ns

    second = thumbnails.ensure_thumbnail(str(tmp_path), "a.jpg")
    assert second == first
    assert second.stat().st_mtime_ns == first_mtime


def test_ensure_thumbnail_returns_none_when_source_missing(tmp_path):
    result = thumbnails.ensure_thumbnail(str(tmp_path), "ghost.jpg")
    assert result is None


def test_ensure_thumbnail_returns_none_when_source_corrupt(tmp_path):
    (tmp_path / "broken.jpg").write_bytes(b"junk")

    result = thumbnails.ensure_thumbnail(str(tmp_path), "broken.jpg")
    assert result is None


# ── invalidate_thumbnail ────────────────────────────────────────────────


def test_invalidate_thumbnail_removes_existing_file(tmp_path):
    _make_image(tmp_path / "a.jpg")
    thumbnails.ensure_thumbnail(str(tmp_path), "a.jpg")
    assert (tmp_path / ".thumbnails" / "256" / "a.webp").exists()

    thumbnails.invalidate_thumbnail(str(tmp_path), "a.jpg")

    assert not (tmp_path / ".thumbnails" / "256" / "a.webp").exists()


def test_invalidate_thumbnail_silent_when_missing(tmp_path):
    """Must not raise when there's nothing to delete."""
    thumbnails.invalidate_thumbnail(str(tmp_path), "ghost.jpg")  # no exception


# ── Derived-name collisions (RULE-20 guard) ─────────────────────────────
#
# A derived filename is a NAMESPACE. Encoding the rendition size into the
# stem as `<stem>@<edge>` used a separator the source alphabet can contain:
# `foo.png`@512 and `foo@512.png`@256 both resolved to `foo@512.webp`, so one
# file served two sources and invalidating either deleted the other's cache.
# These tests pin the property, not the scheme: distinct (source, edge) keys
# get distinct paths, and invalidation touches only its own source.

# `*` and `?` are illegal in NTFS filenames, so they are exercised at the
# path-computation level only; `[` and `@` are legal everywhere and get
# real files on disk.
_HOSTILE_STEMS = (
    "foo", "foo@512", "foo@256", "plain", "a[1]", "a1", "a*", "a?", "a@b*c?[d]",
)


def test_distinct_source_and_edge_keys_never_share_a_thumbnail_path(tmp_path):
    """Every (source, max_edge) pair must own its own file.

    The reproducer: thumbnail_path_for('foo.png', 512) and
    thumbnail_path_for('foo@512.png', 256) both returned `foo@512.webp`.
    """
    seen: dict[Path, tuple[str, int]] = {}
    for stem in _HOSTILE_STEMS:
        rel = f"{stem}.png"
        for edge in thumbnails.ALLOWED_MAX_EDGES:
            path = thumbnails.thumbnail_path_for(str(tmp_path), rel, edge)
            assert path not in seen, (
                f"collision: ({rel}, {edge}) and {seen[path]} both map to "
                f"{path} - one thumbnail would serve two sources"
            )
            seen[path] = (rel, edge)


def test_invalidate_does_not_delete_a_siblings_rendition(tmp_path):
    """`foo@512.png` keeps its thumbnail when `foo.png` is invalidated."""
    _make_image(tmp_path / "foo.png", fmt="PNG")
    _make_image(tmp_path / "foo@512.png", fmt="PNG")

    victim = thumbnails.ensure_thumbnail(str(tmp_path), "foo@512.png", 256)
    assert victim is not None and victim.exists()
    assert thumbnails.ensure_thumbnail(str(tmp_path), "foo.png", 512) is not None

    thumbnails.invalidate_thumbnail(str(tmp_path), "foo.png")

    assert victim.exists(), (
        "invalidating foo.png deleted the thumbnail of the unrelated file "
        "foo@512.png - the grid now regenerates it from the wrong source"
    )


def test_renditions_of_two_sources_hold_their_own_pixels(tmp_path):
    """Observable output, not just paths: each thumbnail shows its source."""
    Image.new("RGB", (900, 900), (255, 0, 0)).save(tmp_path / "foo.png")
    Image.new("RGB", (900, 900), (0, 0, 255)).save(tmp_path / "foo@512.png")

    red = thumbnails.ensure_thumbnail(str(tmp_path), "foo.png", 512)
    blue = thumbnails.ensure_thumbnail(str(tmp_path), "foo@512.png", 256)
    assert red is not None and blue is not None

    with Image.open(red) as img:
        assert img.convert("RGB").getpixel((5, 5))[0] > 200, "foo.png is red"
    with Image.open(blue) as img:
        assert img.convert("RGB").getpixel((5, 5))[2] > 200, "foo@512.png is blue"


def test_invalidation_never_pattern_matches_a_neighbour(tmp_path):
    """`a[1].png` is a literal name, not a character class.

    Matching siblings with a glob would make invalidating `a[1].png` delete
    `a1.png`'s renditions. Whatever resolves the sibling set must treat the
    source stem as literal text.
    """
    _make_image(tmp_path / "a[1].png", fmt="PNG")
    _make_image(tmp_path / "a1.png", fmt="PNG")
    neighbour = thumbnails.ensure_thumbnail(str(tmp_path), "a1.png", 512)
    assert neighbour is not None and neighbour.exists()
    assert thumbnails.ensure_thumbnail(str(tmp_path), "a[1].png", 512) is not None

    thumbnails.invalidate_thumbnail(str(tmp_path), "a[1].png")

    assert neighbour.exists(), "glob semantics ate a literal neighbour"


def test_invalidate_removes_every_rendition_of_its_own_source(tmp_path):
    _make_image(tmp_path / "a[1].png", fmt="PNG")
    made = [
        thumbnails.ensure_thumbnail(str(tmp_path), "a[1].png", edge)
        for edge in thumbnails.ALLOWED_MAX_EDGES
    ]
    assert all(p is not None and p.exists() for p in made)

    thumbnails.invalidate_thumbnail(str(tmp_path), "a[1].png")

    assert [p for p in made if p is not None and p.exists()] == []


# ── Legacy flat layout (migration) ──────────────────────────────────────


def test_invalidate_also_drops_this_source_legacy_flat_renditions(tmp_path):
    """An item that churns cleans up its own pre-`<edge>/` orphans."""
    thumb_root = tmp_path / ".thumbnails"
    thumb_root.mkdir()
    legacy_default = thumb_root / "a.webp"
    legacy_sized = thumb_root / "a@512.webp"
    for p in (legacy_default, legacy_sized):
        p.write_bytes(b"old")
    stranger = thumb_root / "b@512.webp"
    stranger.write_bytes(b"someone else")

    thumbnails.invalidate_thumbnail(str(tmp_path), "a.jpg")

    assert not legacy_default.exists()
    assert not legacy_sized.exists()
    assert stranger.exists(), "invalidation reached a different source's file"


def test_purge_legacy_layout_clears_orphans_and_spares_live_renditions(tmp_path):
    """Orphans are removed on the first scan, not left to accumulate."""
    _make_image(tmp_path / "a.jpg")
    live = thumbnails.ensure_thumbnail(str(tmp_path), "a.jpg", 512)
    assert live is not None and live.exists()

    thumb_root = tmp_path / ".thumbnails"
    orphans = [
        thumb_root / "a.webp",
        thumb_root / "a@512.webp",
        thumb_root / "a[1]@1024.webp",
        thumb_root / "gone.webp.tmp",
    ]
    for p in orphans:
        p.write_bytes(b"orphan")

    removed = thumbnails.purge_legacy_layout(str(tmp_path))

    assert removed == len(orphans)
    assert [p for p in orphans if p.exists()] == []
    assert live.exists(), "the purge reached into a per-size directory"
    # Idempotent: nothing left to do on the next scan.
    assert thumbnails.purge_legacy_layout(str(tmp_path)) == 0


def test_purge_legacy_layout_is_a_noop_without_a_cache(tmp_path):
    assert thumbnails.purge_legacy_layout(str(tmp_path)) == 0
