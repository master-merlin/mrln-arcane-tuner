"""Guards for sized thumbnail renditions.

Dataset covers are painted into cards 260-630px wide but `preview_image` is a
full-size training source (measured on the library grid: median 2.36 MP, one at
58 MP, 598 MP of decoded bitmap for 3.9 MP of screen), which is what made fast
scrolling impossible. Covers now request a bounded rendition via `max_edge`.

Three things must hold, and each is a way this could go wrong quietly:
  * the size is part of the cache key, or a card gets whatever size happened to
    be generated first;
  * invalidation drops EVERY rendition, or an edited image keeps painting its
    pre-edit pixels at some other size;
  * the size is allowlisted, because it lands in a filename.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.core.dataset import thumbnails


def _make_image(path: Path, size: tuple[int, int] = (2000, 1200)) -> None:
    Image.new("RGB", size, (123, 45, 67)).save(path, "JPEG")


# ── The size is part of the key ──────────────────────────────────────────


def test_default_edge_keeps_the_bare_stem(tmp_path):
    """Existing thumbnails on disk stay valid — the default name is unchanged."""
    assert thumbnails.thumbnail_path_for(str(tmp_path), "a.jpg") == (
        tmp_path / ".thumbnails" / "a.webp"
    )
    assert thumbnails.thumbnail_path_for(str(tmp_path), "a.jpg", 256) == (
        tmp_path / ".thumbnails" / "a.webp"
    )


def test_non_default_edge_is_a_distinct_file(tmp_path):
    assert thumbnails.thumbnail_path_for(str(tmp_path), "a.jpg", 512) == (
        tmp_path / ".thumbnails" / "a@512.webp"
    )


def test_sized_key_composes_with_subdirectory_prefixing(tmp_path):
    assert thumbnails.thumbnail_path_for(str(tmp_path), "control/img1.jpg", 512) == (
        tmp_path / ".thumbnails" / "control__img1@512.webp"
    )


@pytest.mark.parametrize("max_edge", thumbnails.ALLOWED_MAX_EDGES)
def test_ensure_generates_at_the_requested_edge(tmp_path, max_edge):
    _make_image(tmp_path / "a.jpg")

    result = thumbnails.ensure_thumbnail(str(tmp_path), "a.jpg", max_edge)

    assert result is not None
    with Image.open(result) as img:
        assert max(img.size) == max_edge, "requested edge is not what was written"


def test_two_edges_coexist_without_clobbering_each_other(tmp_path):
    """The bug this prevents: one size overwriting the other's file."""
    _make_image(tmp_path / "a.jpg")

    small = thumbnails.ensure_thumbnail(str(tmp_path), "a.jpg", 256)
    large = thumbnails.ensure_thumbnail(str(tmp_path), "a.jpg", 512)

    assert small != large
    with Image.open(small) as s, Image.open(large) as lg:
        assert max(s.size) == 256
        assert max(lg.size) == 512


# ── Invalidation drops every rendition ───────────────────────────────────


def test_invalidate_removes_all_sizes_not_just_the_default(tmp_path):
    _make_image(tmp_path / "a.jpg")
    small = thumbnails.ensure_thumbnail(str(tmp_path), "a.jpg", 256)
    large = thumbnails.ensure_thumbnail(str(tmp_path), "a.jpg", 512)
    assert small.exists() and large.exists()

    thumbnails.invalidate_thumbnail(str(tmp_path), "a.jpg")

    assert not small.exists()
    assert not large.exists(), "a stale rendition keeps painting pre-edit pixels"


def test_invalidate_leaves_a_different_source_alone(tmp_path):
    """`a` must not take `ab` with it — prefix matching, not substring."""
    _make_image(tmp_path / "a.jpg")
    _make_image(tmp_path / "ab.jpg")
    keep = thumbnails.ensure_thumbnail(str(tmp_path), "ab.jpg", 512)

    thumbnails.invalidate_thumbnail(str(tmp_path), "a.jpg")

    assert keep.exists()


def test_glob_metacharacters_in_a_filename_do_not_break_invalidation(tmp_path):
    """A source named `shot[1].jpg` is legal; a glob pattern would mis-read it."""
    _make_image(tmp_path / "shot[1].jpg")
    small = thumbnails.ensure_thumbnail(str(tmp_path), "shot[1].jpg", 256)
    large = thumbnails.ensure_thumbnail(str(tmp_path), "shot[1].jpg", 512)

    thumbnails.invalidate_thumbnail(str(tmp_path), "shot[1].jpg")

    assert not small.exists()
    assert not large.exists()


def test_paths_for_is_silent_when_nothing_was_ever_generated(tmp_path):
    found = thumbnails.thumbnail_paths_for(str(tmp_path), "ghost.jpg")
    assert found == [tmp_path / ".thumbnails" / "ghost.webp"]
    thumbnails.invalidate_thumbnail(str(tmp_path), "ghost.jpg")  # no exception


def test_a_suffix_that_is_not_a_size_is_not_treated_as_a_rendition(tmp_path):
    """`a@notasize.webp` is somebody else's file, not our rendition."""
    thumb_dir = tmp_path / ".thumbnails"
    thumb_dir.mkdir()
    stranger = thumb_dir / "a@notasize.webp"
    stranger.write_bytes(b"x")

    assert stranger not in thumbnails.thumbnail_paths_for(str(tmp_path), "a.jpg")
