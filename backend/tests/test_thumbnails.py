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
    expected = tmp_path / ".thumbnails" / "img_001.webp"
    assert thumbnails.thumbnail_path_for(str(tmp_path), "img_001.jpg") == expected


def test_thumbnail_path_for_normalizes_backslash_keys(tmp_path):
    # rel_path may arrive with Windows-style backslashes from OS APIs;
    # the helper normalizes them before building the name. Subdir sources
    # are namespaced (control/img1.jpg must not collide with img1.png),
    # so both separator styles resolve to the same prefixed thumbnail.
    expected = tmp_path / ".thumbnails" / "sub__img_001.webp"
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
    """PIL opens the first frame of a GIF by default."""
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
    assert result == tmp_path / ".thumbnails" / "a.webp"


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
    assert (tmp_path / ".thumbnails" / "a.webp").exists()

    thumbnails.invalidate_thumbnail(str(tmp_path), "a.jpg")

    assert not (tmp_path / ".thumbnails" / "a.webp").exists()


def test_invalidate_thumbnail_silent_when_missing(tmp_path):
    """Must not raise when there's nothing to delete."""
    thumbnails.invalidate_thumbnail(str(tmp_path), "ghost.jpg")  # no exception
