"""Canonical media file-extension sets — single source of truth.

Several scan / masking / thumbnail paths historically hardcoded their own
copies of these extension sets, which drift apart silently (the audit found 5
duplicated literals). Import the constant for the concept you need instead of
re-declaring a set literal.

The three sets are intentionally distinct (NOT ``IMAGE | VIDEO``):

* ``IMAGE_EXTENSIONS``      — still-image formats the masking/image pipelines accept.
* ``VIDEO_EXTENSIONS``      — video/animation formats the scanner probes for dimensions
                              and skips for image-only hashing.
* ``MULTIMEDIA_EXTENSIONS`` — the curated accept-list of files that count as a dataset
                              "pair" source. It accepts only ``.mp4``/``.gif`` among
                              video formats and omits ``.bmp``/``.tiff`` among images,
                              matching the importer's long-standing behavior.
"""

from __future__ import annotations

# Ordered by source-resolution preference (``.jpg`` first) — the masking
# pipeline iterates this to pick a stem's source image when several formats
# could exist. ``IMAGE_EXTENSIONS`` is the membership set derived from it.
IMAGE_EXTENSION_PREFERENCE: tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".tiff",
)

IMAGE_EXTENSIONS: frozenset[str] = frozenset(IMAGE_EXTENSION_PREFERENCE)

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".gif", ".webm", ".mkv", ".avi"}
)

MULTIMEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".avif", ".mp4", ".gif"}
)
