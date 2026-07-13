"""Canonical media file-extension sets — single source of truth.

Several scan / masking / thumbnail paths historically hardcoded their own
copies of these extension sets, which drift apart silently (the audit found 5
duplicated literals). Import the constant for the concept you need instead of
re-declaring a set literal.

The four sets are intentionally distinct (NOT ``IMAGE | VIDEO | AUDIO``):

* ``IMAGE_EXTENSIONS``      — still-image formats the masking/image pipelines accept.
* ``VIDEO_EXTENSIONS``      — video/animation formats the scanner probes for dimensions
                              and skips for image-only hashing.
* ``AUDIO_EXTENSIONS``      — audio formats the scanner probes for duration/sample
                              rate/channels via ``soundfile`` (the primary decoder —
                              preferred over torchaudio because the docker image's
                              torchaudio 2.11 trio is installed ``--no-deps`` and may
                              carry different backends than the local 2.12.1 install).
                              All five contract extensions (``.wav .mp3 .flac .ogg
                              .opus``) were verified to round-trip write+read via
                              ``soundfile`` + libsndfile 1.2.2 on this stack — no
                              extension was excluded.
* ``MULTIMEDIA_EXTENSIONS`` — the accept-list of files that count as a dataset
                              "pair" source (drives the grid AND the trainer's
                              ``/pairs`` inventory). Its image side is a curated
                              subset (omits ``.bmp``/``.tiff`` — not display/train
                              targets here); its video side is the FULL
                              ``VIDEO_EXTENSIONS`` set, so every trainable clip is
                              ingestable; its audio side is the FULL
                              ``AUDIO_EXTENSIONS`` set. Historically it re-listed only
                              ``.mp4``/``.webm``/``.gif`` and silently dropped
                              ``.mkv``/``.avi`` clips at the front door — those
                              datasets then scanned to an empty grid (0 videos, no
                              captions, no preview) AND fed the trainer an empty
                              inventory, even though every downstream stage (probe,
                              thumbnail, ``media_type``, collation) already handled
                              them. Deriving the video (and now audio) side from the
                              canonical extension sets fixes that and prevents the
                              sets from drifting apart.
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

# Video containers a browser can play inline (drives the grid <video> tag).
# A subset of VIDEO_EXTENSIONS — .mkv/.avi are trainable but not web-native.
BROWSER_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".webm"})

# Audio formats the scanner accepts (music/voice LoRA datasets, e.g. ACE-Step).
# Fixed by the program plan — see module docstring for the decoder recon that
# verified all five round-trip via soundfile + libsndfile 1.2.2.
AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".opus"}
)

# Curated still-image subset accepted as dataset items (no .bmp/.tiff).
MULTIMEDIA_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".avif"}
)

# Image subset ∪ the FULL trainable-video set ∪ the FULL audio set. The union
# (rather than a second hand-maintained literal) is the single source of
# truth: any container added to VIDEO_EXTENSIONS/AUDIO_EXTENSIONS becomes an
# ingestable dataset item automatically. Adds .mkv/.avi over the legacy
# mp4/webm/gif list — the clips that previously scanned to an empty grid and
# an empty trainer inventory.
MULTIMEDIA_EXTENSIONS: frozenset[str] = (
    MULTIMEDIA_IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
)


def is_probeable_video(ext: str) -> bool:
    """Whether *ext* is a trainable video clip worth probing for metadata.

    ``.gif`` lives in ``VIDEO_EXTENSIONS`` (it renders as an animated tile in
    the grid) but is an animated *image*, not a trainable video — it has no
    framerate/duration/codec the trim + clip-health layer cares about, so it
    is excluded from probing. ``ext`` is matched case-insensitively.
    """
    ext = ext.lower()
    return ext in VIDEO_EXTENSIONS and ext != ".gif"


def is_probeable_audio(ext: str) -> bool:
    """Whether *ext* is an audio file worth probing for duration/sample
    rate/channels via ``soundfile``. ``ext`` is matched case-insensitively.
    """
    return ext.lower() in AUDIO_EXTENSIONS
