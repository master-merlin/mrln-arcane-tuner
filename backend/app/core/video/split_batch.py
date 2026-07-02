"""Background worker — split a long source video into training clips.

Mirrors the crop-batch adopter shape (per-segment cancel checks, ok/failed
counts, isolated per-segment failures) but runs on the ``cpu`` lane so it never
blocks the GPU lane. For each :class:`Segment` it cuts a ``.mp4`` into the
dataset's source directory, choosing per-segment between two strategies:

  * **stream-copy** (``-c copy``) — lossless and near-instant, but only frame-
    accurate when the segment start lands on a keyframe.
  * **re-encode** (``libx264 -crf 16``) — frame-accurate at any start, slower.

``mode``:
  * ``"auto"``     — copy when a keyframe sits within ~0.1s of the segment
                     start (so the cut is clean), else re-encode.
  * ``"copy"``     — force stream-copy for every segment.
  * ``"reencode"`` — force re-encode for every segment.

On success, the source long video is optionally archived into
``<dataset>/.video_sources/`` (a dot-dir the scanner ignores), then
``scan_dataset`` is called so the new clips are probed before the task's
automatic ``dataset.invalidated`` broadcast fires.

Module-level seams (monkeypatchable in tests):
  ``_run_ffmpeg``        → app.core.video.ffmpeg.run_ffmpeg
  ``_nearest_keyframe``  → app.core.video.ffmpeg.nearest_keyframe_before
  ``_scan``              → dataset_manager.scan_dataset
  ``_resolve_source_dir``→ (dataset_name) -> Path to the dataset source dir
"""

from __future__ import annotations

import shutil
from pathlib import Path

from app.core.logger import get_logger
from app.core.tasks.task_manager import task_manager

logger = get_logger(__name__)

# A segment start within this many seconds of a keyframe is "clean enough" to
# stream-copy in auto mode.
_KEYFRAME_SNAP_S = 0.1


# ── Seams ───────────────────────────────────────────────────────────────────


def _run_ffmpeg(args: list[str], progress_cb=None) -> int:
    from app.core.video.ffmpeg import run_ffmpeg

    return run_ffmpeg(args, progress_cb=progress_cb)


def _nearest_keyframe(path: str, t: float) -> float:
    from app.core.video.ffmpeg import nearest_keyframe_before

    return nearest_keyframe_before(path, t)


def _scan(dataset_name: str) -> None:
    from app.core.dataset_manager import dataset_manager as dm

    dm.scan_dataset(dataset_name, force_full=False)


def _resolve_source_dir(dataset_name: str) -> Path:
    """Return the dataset's on-disk source directory (where clips land)."""
    from app.core.dataset_manager import dataset_manager as dm

    ds = dm.get_dataset(dataset_name)
    if ds is None:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    return Path(ds.path)


# ── Per-segment cut ─────────────────────────────────────────────────────────


def _decide_mode(source_path: str, start_s: float, mode: str) -> str:
    """Resolve ``auto`` to ``copy`` / ``reencode`` for one segment."""
    if mode in ("copy", "reencode"):
        return mode
    # auto: copy iff a keyframe is within the snap window of the start.
    kf = _nearest_keyframe(source_path, start_s)
    if abs(start_s - kf) <= _KEYFRAME_SNAP_S:
        return "copy"
    return "reencode"


def _build_args(
    source_path: str, out_path: str, start_s: float, end_s: float, effective_mode: str
) -> list[str]:
    """Assemble the ffmpeg argv (sans executable) for one segment cut."""
    if effective_mode == "copy":
        # Input-seek BEFORE -i for a fast keyframe-aligned copy.
        return [
            "-y",
            "-ss",
            f"{start_s:.6f}",
            "-to",
            f"{end_s:.6f}",
            "-i",
            source_path,
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            out_path,
        ]
    # Re-encode: output-seek (-ss after -i) for sample-accurate trimming.
    return [
        "-y",
        "-i",
        source_path,
        "-ss",
        f"{start_s:.6f}",
        "-to",
        f"{end_s:.6f}",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        out_path,
    ]


def _cut_segment(
    source_path: str, out_path: str, start_s: float, end_s: float, mode: str
) -> str:
    """Cut a single segment. Returns the effective mode used ('copy'/'reencode')."""
    effective_mode = _decide_mode(source_path, start_s, mode)
    args = _build_args(source_path, out_path, start_s, end_s, effective_mode)
    _run_ffmpeg(args)
    return effective_mode


# ── Worker ──────────────────────────────────────────────────────────────────


def run_video_split_batch(
    task_id: str,
    *,
    dataset_name: str,
    source_rel_path: str,
    segments: list,
    mode: str = "auto",
    output_prefix: str | None = None,
    archive_source: bool = True,
) -> None:
    """Synchronous worker — runs on the cpu lane thread.

    Cuts each segment to ``{prefix or source_stem}_{i:03d}.mp4`` in the dataset
    source dir. Per-segment failures are isolated (logged, counted, the rest
    continue); cancellation is checked before each segment. On any successful
    cut, the source is archived (if requested) and a rescan runs so the new
    clips are probed before the auto ``dataset.invalidated``.

    ``segments`` items may be Pydantic ``Segment`` objects or plain dicts with
    ``start_s`` / ``end_s``.
    """
    ok = 0
    failed = 0
    cancelled = False
    produced: list[str] = []

    try:
        source_dir = _resolve_source_dir(dataset_name)
        source_path = source_dir / source_rel_path
        if not source_path.exists():
            task_manager.fail(task_id, f"source not found: {source_rel_path}")
            return

        prefix = output_prefix or Path(source_rel_path).stem

        for i, seg in enumerate(segments):
            if task_manager.is_cancelled(task_id):
                cancelled = True
                break

            start_s = seg["start_s"] if isinstance(seg, dict) else seg.start_s
            end_s = seg["end_s"] if isinstance(seg, dict) else seg.end_s

            out_name = f"{prefix}_{i:03d}.mp4"
            out_path = source_dir / out_name

            try:
                used = _cut_segment(
                    str(source_path),
                    str(out_path),
                    float(start_s),
                    float(end_s),
                    mode,
                )
                ok += 1
                produced.append(out_name)
                logger.info(
                    "video_segment_cut",
                    task_id=task_id,
                    out=out_name,
                    mode=used,
                    start=start_s,
                    end=end_s,
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.warning(
                    "video_segment_failed",
                    task_id=task_id,
                    segment=i,
                    error=str(exc),
                )

            task_manager.update(
                task_id, current=i + 1, item=out_name, ok=ok, failed=failed
            )

    except Exception as exc:  # unrecoverable setup error
        task_manager.fail(task_id, str(exc))
        return

    # Archive the source + rescan only if at least one clip was produced.
    if produced:
        if archive_source:
            try:
                archive_dir = source_dir / ".video_sources"
                archive_dir.mkdir(exist_ok=True)
                dest = archive_dir / source_path.name
                shutil.move(str(source_path), str(dest))
                logger.info("video_source_archived", task_id=task_id, dest=str(dest))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "video_source_archive_failed", task_id=task_id, error=str(exc)
                )
        try:
            _scan(dataset_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("video_split_rescan_failed", task_id=task_id, error=str(exc))

    if cancelled:
        task_manager.finish_cancelled(task_id)
    else:
        task_manager.complete(task_id)
