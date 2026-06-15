"""Media extension-set contracts.

Pins the invariant that broke ``.mkv``/``.avi`` datasets: the dataset
"pair" accept-list (``MULTIMEDIA_EXTENSIONS``) must admit every trainable
video container, otherwise those clips are dropped at ingestion and the
dataset scans to an empty grid (0 videos, no captions, no preview) while the
trainer's ``/pairs`` inventory comes back empty.
"""

from __future__ import annotations

from app.core.dataset.media_types import (
    BROWSER_VIDEO_EXTENSIONS,
    MULTIMEDIA_EXTENSIONS,
    MULTIMEDIA_IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    is_probeable_video,
)


def test_multimedia_admits_every_trainable_video():
    """Every VIDEO_EXTENSIONS container must be an ingestable dataset item.

    This is the anti-drift contract: if a video ext is trainable (probe,
    thumbnail, collation all handle it) it MUST also pass the accept-list,
    or the clip never becomes a media item at all.
    """
    assert VIDEO_EXTENSIONS <= MULTIMEDIA_EXTENSIONS


def test_multimedia_includes_mkv_and_avi():
    """Regression: .mkv/.avi were silently dropped (empty grid + empty
    trainer inventory) because the legacy accept-list re-listed only
    .mp4/.webm/.gif."""
    assert ".mkv" in MULTIMEDIA_EXTENSIONS
    assert ".avi" in MULTIMEDIA_EXTENSIONS


def test_multimedia_image_side_stays_curated():
    """The image side remains a curated subset — .bmp/.tiff are intentionally
    NOT dataset-item sources here (display/train targets only)."""
    assert ".bmp" not in MULTIMEDIA_EXTENSIONS
    assert ".tiff" not in MULTIMEDIA_EXTENSIONS
    assert MULTIMEDIA_IMAGE_EXTENSIONS == MULTIMEDIA_EXTENSIONS - VIDEO_EXTENSIONS


def test_browser_playable_is_a_video_subset():
    """Inline-playable containers are a subset of trainable ones; .mkv/.avi
    are trainable-but-not-web-native (grid shows the poster thumbnail)."""
    assert BROWSER_VIDEO_EXTENSIONS <= VIDEO_EXTENSIONS
    assert ".mkv" not in BROWSER_VIDEO_EXTENSIONS
    assert ".avi" not in BROWSER_VIDEO_EXTENSIONS


def test_gif_is_video_but_not_probeable():
    """.gif renders as an animated tile (in VIDEO_EXTENSIONS) but isn't a
    trainable clip — unchanged by this fix."""
    assert ".gif" in VIDEO_EXTENSIONS
    assert ".gif" in MULTIMEDIA_EXTENSIONS
    assert not is_probeable_video(".gif")
    assert is_probeable_video(".mkv")
    assert is_probeable_video(".avi")
