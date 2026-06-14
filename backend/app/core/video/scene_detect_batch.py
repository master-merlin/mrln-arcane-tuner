"""Background worker — auto scene-detection over a long source video.

Runs PySceneDetect's ``ContentDetector`` on a source clip and writes the
detected cut points as a list of ``{start_s, end_s}`` proposals to
``<dataset>/.video/proposals/<stem>.json`` (a dot-dir the scanner ignores).
This task ONLY proposes — the user reviews the proposals in the UI and a
SEPARATE ``/video/split`` call performs the actual cut. Runs on the ``cpu``
lane (opencv decode is CPU-bound and must not block the GPU lane).

Module-level seams (monkeypatchable in tests):
  ``_detect_scenes(path, threshold, min_scene_len_s, progress_cb)``
        → returns ``[(start_s, end_s), ...]`` via PySceneDetect.
  ``_resolve_dataset_path(dataset_name)`` → Path to the dataset root.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.logger import get_logger
from app.core.tasks.task_manager import task_manager

logger = get_logger(__name__)


# ── Seams ───────────────────────────────────────────────────────────────────


def _resolve_dataset_path(dataset_name: str) -> Path:
    from app.core.dataset_manager import dataset_manager as dm

    ds = dm.get_dataset(dataset_name)
    if ds is None:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    return Path(ds.path)


def _detect_scenes(
    path: str, threshold: float, min_scene_len_s: float, progress_cb=None
) -> list[tuple[float, float]]:
    """Run ContentDetector and return ``[(start_s, end_s), ...]``.

    ``min_scene_len_s`` is converted to a frame count via the clip fps so very
    short flickers don't fragment the proposal list.
    """
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(path)
    fps = float(video.frame_rate) or 0.0
    min_scene_len = int(round(min_scene_len_s * fps)) if fps > 0 else 0

    detector = ContentDetector(threshold=threshold, min_scene_len=max(0, min_scene_len))
    scene_manager = SceneManager()
    scene_manager.add_detector(detector)

    def _cb(frame_image, frame_num):  # PySceneDetect callback signature
        if progress_cb is not None:
            try:
                progress_cb(int(frame_num))
            except Exception:  # noqa: BLE001
                pass

    scene_manager.detect_scenes(video, callback=_cb if progress_cb else None)
    scenes = scene_manager.get_scene_list()

    out: list[tuple[float, float]] = []
    for start, end in scenes:
        out.append((float(start.seconds), float(end.seconds)))
    return out


# ── Worker ──────────────────────────────────────────────────────────────────


def _proposals_path(dataset_root: Path, source_rel_path: str) -> Path:
    """Where proposals for a given source clip are written."""
    stem = Path(source_rel_path).stem
    return dataset_root / ".video" / "proposals" / f"{stem}.json"


def run_scene_detect_batch(
    task_id: str,
    *,
    dataset_name: str,
    source_rel_path: str,
    threshold: float = 27.0,
    min_scene_len_s: float = 1.0,
) -> None:
    """Synchronous worker — runs on the cpu lane thread.

    Detects scenes in the source clip and writes ``{start_s,end_s}`` proposals to
    ``<dataset>/.video/proposals/<stem>.json``. Progress is reported in frames
    processed. Failures finalize the task as failed (the proposals file is not
    written).
    """
    try:
        dataset_root = _resolve_dataset_path(dataset_name)
        source_path = dataset_root / source_rel_path
        if not source_path.exists():
            task_manager.fail(task_id, f"source not found: {source_rel_path}")
            return

        def progress_cb(frame_num: int) -> None:
            task_manager.update(task_id, current=frame_num, item=source_rel_path)

        scenes = _detect_scenes(
            str(source_path), threshold, min_scene_len_s, progress_cb=progress_cb
        )

        proposals = [{"start_s": s, "end_s": e} for s, e in scenes]
        out_path = _proposals_path(dataset_root, source_rel_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"source_rel_path": source_rel_path, "segments": proposals}),
            encoding="utf-8",
        )

        logger.info(
            "scene_detect_done",
            task_id=task_id,
            source=source_rel_path,
            count=len(proposals),
        )
        task_manager.update(
            task_id,
            current=task_manager.get(task_id).total or len(proposals),
            ok=len(proposals),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("scene_detect_failed", task_id=task_id, error=str(exc))
        task_manager.fail(task_id, str(exc))
        return

    task_manager.complete(task_id)
