"""Tests for the dataset thumbnails module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
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
    # the helper normalizes them before extracting the stem.
    expected = tmp_path / ".thumbnails" / "img_001.webp"
    assert thumbnails.thumbnail_path_for(str(tmp_path), "sub\\img_001.png") == expected


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
