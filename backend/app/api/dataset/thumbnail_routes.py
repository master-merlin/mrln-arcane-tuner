"""Migration off the pre-``<edge>/`` flat thumbnail layout (LANE-40).

`a5003618` moved every rendition to ``.thumbnails/<edge>/<stem>.webp`` and
called :func:`purge_legacy_layout` from ``_prepare_scan``. That purge is
correct and does nothing until a scan runs, so a library of datasets nobody
rescans keeps its flat renditions forever.

**What is and is not wrong with those datasets.** Their covers are *correct*:
``thumbnail_path_for`` cannot name a flat file, so `ensure_thumbnail` falls
back to the source. What is wrong is what that fallback COSTS — a full decode
of a 25 MB source per tile, MEASURED at 38.69 s for one 25-image dataset — and
LANE-40 shipped this sweep believing the answer was to delete the very bytes
that make it unnecessary. It no longer does: :func:`purge_legacy_layout`
adopts every rendition whose provenance it can prove into ``<edge>/`` and
deletes only what it can name a reason for (LANE-53). **The sweep is now a
repair, and the reclaim is the part that is left over.**

It is still **not a rescan**, and that distinction is the rest of the design:
a rescan rehashes and rescores every file, which would put minutes of GPU/CPU
work behind a button that moves files within one directory. The routes here
never scan, never touch the database, and never broadcast
``dataset.invalidated`` — no counter, no ``media_metadata`` key and no
servable path changes (an adopted rendition lands on the path
``thumbnail_path_for`` already names), so fanning out N dataset refetches
would cost the client a full reload for a zero observable delta. The client
re-runs the survey when the task finishes instead.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.dataset import thumbnails
from app.core.dataset_manager import Dataset, dataset_manager
from app.core.logger import get_logger
from app.core.tasks.task_manager import task_manager

router = APIRouter()
logger = get_logger(__name__)

#: ``Task.type`` for the sweep. Reserved in ECOSYSTEM §6 (LANE-40).
TASK_TYPE = "thumbnail_migration"
#: Terminal statuses — anything else means a sweep is still in flight, which
#: is what single-flight keys on.
_LIVE_STATUSES = ("pending", "running")


# ── Wire models (ECOSYSTEM §6, LANE-40) ──────────────────────────────────


class ThumbnailLegacyEntry(BaseModel):
    """One dataset that still holds flat-layout renditions."""

    name: str                               # dataset name (registry key)
    files: int                              # flat renditions found
    # Size the flat layout occupies, display-only. NOT a reclaim forecast:
    # most of it is adopted rather than freed (LANE-53), and the split is only
    # knowable per file, after the gates have run.
    bytes: int


class ThumbnailLegacySurveyResponse(BaseModel):
    """Which datasets have not been migrated, and what it costs to leave them.

    The totals are computed here rather than reduced client-side (D10: compute
    at write time). ``total_files == 0`` is the single fact the banner's
    positive-only ``@if`` gates on, so a client that cannot parse a
    later-added per-entry field still hides the affordance correctly.
    """

    datasets: list[ThumbnailLegacyEntry]    # unmigrated only; migrated omitted
    dataset_count: int                      # == len(datasets)
    total_files: int                        # sum of entry.files
    total_bytes: int                        # sum of entry.bytes


class ThumbnailMigrationStartedResponse(BaseModel):
    """Acknowledgement of an enqueued sweep."""

    task_id: str                            # follow on the Task Center / /ws
    dataset_count: int                      # datasets the sweep will visit
    files: int                              # flat renditions it will deal with
    bytes: int                              # bytes those renditions occupy


# ── Detection ────────────────────────────────────────────────────────────


def _survey() -> list[tuple[Dataset, int, int]]:
    """Return ``(dataset, files, bytes)`` for every UNMIGRATED dataset.

    One ``scandir`` of one directory per dataset (see
    :func:`thumbnails._iter_legacy_entries`) — a library of 30 datasets costs
    30 directory reads, not 30 tree walks. Datasets whose path is gone are
    skipped silently: ``missing`` is the dataset screen's story to tell, and a
    disk-reclaim survey has nothing to add to it.
    """
    found: list[tuple[Dataset, int, int]] = []
    for dataset in list(dataset_manager.datasets.values()):
        files, size = thumbnails.legacy_layout_survey(dataset.path)
        if files:
            found.append((dataset, files, size))
    found.sort(key=lambda row: row[0].name)
    return found


@router.get("/datasets/thumbnails/legacy", response_model=ThumbnailLegacySurveyResponse)
async def get_legacy_thumbnail_survey() -> ThumbnailLegacySurveyResponse:
    """Report which datasets still hold pre-``<edge>/`` flat renditions.

    Off the event loop because it touches the filesystem once per dataset —
    cheap, but not zero, and the loop must not do blocking IO.
    """
    rows = await asyncio.to_thread(_survey)
    return ThumbnailLegacySurveyResponse(
        datasets=[
            ThumbnailLegacyEntry(name=d.name, files=f, bytes=b) for d, f, b in rows
        ],
        dataset_count=len(rows),
        total_files=sum(f for _, f, _ in rows),
        total_bytes=sum(b for _, _, b in rows),
    )


# ── Migration ────────────────────────────────────────────────────────────


def _live_migration_id() -> str | None:
    """Return the id of a sweep that has not finished, if one exists."""
    for task in task_manager.list():
        if task.type == TASK_TYPE and task.status.value in _LIVE_STATUSES:
            return task.id
    return None


def run_thumbnail_migration(task_id: str, datasets: list[Dataset]) -> None:
    """Migrate the flat layout of each of *datasets*, one dataset per step.

    Runs on the shared non-GPU ``background`` lane, so it is bounded twice
    over: the work is a fixed list of datasets decided before the task was
    created (it cannot grow while running), and the cancel flag is checked
    before every dataset, so the user can hand the lane back at any step
    boundary. The per-dataset sweep itself is the unit of interruption — it is
    a directory of renames and unlinks and splitting it finer would buy
    nothing.

    ``ok`` counts every rendition the sweep DEALT WITH, adopted or removed,
    because the Task Center's progress must not fall when a dataset turns out
    to be salvageable. What actually happened to those files is the difference
    between adopted and removed, which is what the log line and the follow-up
    survey report — LANE-53: most of what this used to delete is a rename.

    Finalizes itself (complete / cancelled); a raised exception is left to the
    lane runner's failure path.
    """
    ok = 0
    adopted = removed = kept = 0
    for index, dataset in enumerate(datasets):
        if task_manager.is_cancelled(task_id):
            logger.info(
                "thumbnail_migration_cancelled",
                task_id=task_id, done=index, total=len(datasets), files=ok,
                adopted=adopted, removed=removed, kept=kept,
            )
            task_manager.finish_cancelled(task_id)
            return
        outcome = thumbnails.purge_legacy_layout(dataset.path)
        adopted += outcome.adopted
        removed += outcome.removed
        kept += outcome.kept
        ok = adopted + removed
        task_manager.update(
            task_id, current=index + 1, item=dataset.name, ok=ok,
        )
    logger.info(
        "thumbnail_migration_complete",
        task_id=task_id, datasets=len(datasets), files=ok,
        adopted=adopted, removed=removed, kept=kept,
    )
    task_manager.complete(task_id)


@router.post(
    "/datasets/thumbnails/migrate",
    response_model=ThumbnailMigrationStartedResponse,
)
async def start_thumbnail_migration() -> ThumbnailMigrationStartedResponse:
    """Enqueue one bounded sweep that reclaims the flat-layout renditions.

    Single-flight: a sweep already pending or running answers 409 rather than
    queueing a second one, so a double click cannot put two redundant passes
    on a lane other work shares. Nothing to do also answers 409 — an empty
    task would show up in the Task Center saying nothing.
    """
    if (live := _live_migration_id()) is not None:
        raise HTTPException(
            status_code=409,
            detail=f"A thumbnail migration is already running (task {live}).",
        )

    rows = await asyncio.to_thread(_survey)
    if not rows:
        raise HTTPException(
            status_code=409, detail="There is nothing to migrate.",
        )

    datasets = [d for d, _, _ in rows]
    files = sum(f for _, f, _ in rows)
    size = sum(b for _, _, b in rows)
    task = task_manager.create(
        type=TASK_TYPE,
        title=f"Thumbnail cache repair · {len(datasets)} datasets",
        total=len(datasets),
    )
    task_manager.enqueue(
        task.id,
        lambda tid: run_thumbnail_migration(tid, datasets),
        lane="background",
    )
    logger.info(
        "thumbnail_migration_enqueued",
        task_id=task.id, datasets=len(datasets), files=files, bytes=size,
    )
    return ThumbnailMigrationStartedResponse(
        task_id=task.id, dataset_count=len(datasets), files=files, bytes=size,
    )
