"""Dataset cache administration — list and purge latent/embedding caches."""

from __future__ import annotations

import asyncio
import re
import threading as _threading
import time as _time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api._deps import dataset_or_404
from app.api._path_guard import safe_rmtree, validate_path_within
from app.core.dataset_manager import Dataset, dataset_manager
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\- ]*$")


def _validate_cache_segment(name: str) -> str:
    """A CLIENT-SUPPLIED purge segment must be a plain directory name — no
    separators, no dot-navigation. Anything else 400s before any path is
    built. This is the security boundary for values from the request body
    (``PurgeCacheRequest.models`` / ``.types`` / ``.variants``) — invalid
    input aborts the whole purge, since the caller can simply retry with a
    corrected request."""
    if not name or name in {".", ".."} or not _SEGMENT_RE.fullmatch(name):
        raise HTTPException(status_code=400, detail=f"Invalid cache segment: {name!r}")
    return name


def _validate_discovered_segment(name: str) -> str | None:
    """A SERVER-DISCOVERED purge segment (an on-disk name from ``iterdir()``)
    that doesn't look like a plain segment is skipped, not fatal. Unlike a
    client-supplied name, there is no "retry with corrected input" for a
    directory that is already sitting on disk (legacy naming, a manual copy,
    a future family-naming convention) — a destructive purge must not abort
    after earlier siblings have already been deleted just because one
    unrelated on-disk entry doesn't parse. Returns ``None`` (caller should
    skip this entry) instead of raising."""
    if not name or name in {".", ".."} or not _SEGMENT_RE.fullmatch(name):
        logger.warning(
            "cache_segment_skipped", name=name, reason="invalid_segment_name"
        )
        return None
    return name


def _require_segment_dir(parent: Path, name: str, cache_root: Path) -> Path:
    """Join + validate a CLIENT-SUPPLIED segment onto *parent*. Fail-fast:
    an invalid name (400) or a resolved path escaping *cache_root* (403)
    aborts the whole purge — the security property for request-body input."""
    segment_dir = parent / _validate_cache_segment(name)
    return validate_path_within(segment_dir, cache_root)


def _discover_segment_dir(parent: Path, name: str, cache_root: Path) -> Path | None:
    """Join + validate a SERVER-DISCOVERED segment onto *parent*. Skip and
    continue: an invalid name, or a resolved path escaping *cache_root*
    (e.g. an on-disk symlink/junction), is logged and this entry is skipped
    — never fatal. Returns ``None`` when the entry was skipped."""
    segment = _validate_discovered_segment(name)
    if segment is None:
        return None
    try:
        return validate_path_within(parent / segment, cache_root)
    except HTTPException:
        logger.warning(
            "cache_segment_skipped", name=name, reason="path_escapes_cache_root"
        )
        return None


def get_dataset_or_404(name: str) -> Dataset:
    """Path-operation dependency: resolve a dataset by name or 404."""
    return dataset_or_404(dataset_manager.get_dataset(name))


# ── Request Models ───────────────────────────────────────────────────────

class PurgeCacheRequest(BaseModel):
    """Request body for cache purge."""
    models: list[str] | None = None
    versions: list[str] | None = None    # ["1.0.0", ...] — None = all versions
    types: list[str] | None = None       # ["latents", "embeddings"]
    variants: list[str] | None = None    # ["original", "masked"]


# ── Response Models ──────────────────────────────────────────────────────

class CacheStatsResponse(BaseModel):
    """Aggregated cache size statistics across all datasets."""
    total_bytes: int
    latent_bytes: int
    embedding_bytes: int
    cached_datasets: int
    dataset_root_bytes: int


class PurgeCacheResponse(BaseModel):
    """Summary of a cache purge operation."""
    dataset: str
    deleted: int
    freed_bytes: int


class CacheListResponse(BaseModel):
    """The cache tree for one dataset (``GET /datasets/{name}/cache/list``).

    ``cache`` stays ``dict[str, Any]`` rather than a fully-typed nested
    model: its keys are discovered at scan time (model name -> version ->
    cache-type name), and a "variant" value is EITHER a list of resolution
    strings or a bare ``True`` depending on whether sub-directories exist
    (see ``_scan_cache_tree``'s docstring) — a strict schema would either
    reject one of those two shapes or have to model a union at every leaf,
    for no real safety gain over documenting the outer envelope.
    """

    dataset: str
    cache: dict[str, Any]


# ── Helpers ──────────────────────────────────────────────────────────────

def _dir_size(path: Path) -> int:
    """Calculate total size in bytes of a directory tree."""
    total = 0
    for fp in path.rglob("*"):
        if fp.is_file():
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def _scan_cache_tree(cache_root: Path) -> dict[str, Any]:
    """Scan .cache/ and build a structured tree.

    Dynamically discovers all cache type directories (latents, te1, te2, etc.)
    under each version.

    Returns::

        {
            "model_name": {
                "version": {
                    "types": {
                        "latents": { "variant_name": ["1024x1024", ...] },
                        "te1": { "variant_name": true },
                        ...
                    },
                    "size_bytes": 123456
                }
            }
        }
    """
    tree: dict[str, Any] = {}

    if not cache_root.is_dir():
        return tree

    for model_entry in cache_root.iterdir():
        if not model_entry.is_dir():
            continue
        model_name = model_entry.name
        tree[model_name] = {}

        for version_entry in model_entry.iterdir():
            if not version_entry.is_dir():
                continue
            version = version_entry.name
            version_data: dict[str, Any] = {
                "types": {},
                "size_bytes": _dir_size(version_entry),
            }

            # Scan all type directories dynamically
            for type_entry in version_entry.iterdir():
                if not type_entry.is_dir():
                    continue
                type_name = type_entry.name
                variants: dict[str, Any] = {}
                has_files = False

                for child in type_entry.iterdir():
                    if child.is_dir():
                        # Check if variants have sub-dirs (resolutions) or just files
                        sub_items = sorted(
                            r.name for r in child.iterdir() if r.is_dir()
                        )
                        if sub_items:
                            variants[child.name] = sub_items
                        else:
                            variants[child.name] = True
                    elif child.is_file():
                        has_files = True

                # If dir has flat files but no variant subdirs, mark as flat
                if not variants and has_files:
                    variants["_flat"] = True

                version_data["types"][type_name] = variants

            tree[model_name][version] = version_data

    return tree


def _purge_cache(
    cache_root: Path,
    models: list[str] | None,
    types: list[str] | None,
    variants: list[str] | None,
    versions: list[str] | None = None,
) -> dict[str, Any]:
    """Delete selected cache subtrees.

    Filters are ANDed: ``models`` → ``versions`` → ``types`` → ``variants``.
    A ``None`` filter means "all at that level". Returns a summary of what was
    deleted.
    """
    if not cache_root.is_dir():
        return {"deleted": 0, "freed_bytes": 0}

    deleted = 0
    freed = 0

    # Determine which models to target
    target_models = models if models else [
        e.name for e in cache_root.iterdir() if e.is_dir()
    ]

    for model_name in target_models:
        if models:
            model_dir = _require_segment_dir(cache_root, model_name, cache_root)
        else:
            model_dir = _discover_segment_dir(cache_root, model_name, cache_root)
            if model_dir is None:
                continue
        if not model_dir.is_dir():
            continue

        for version_entry in model_dir.iterdir():
            if not version_entry.is_dir():
                continue
            # Filter by requested versions (None = all)
            if versions and version_entry.name not in versions:
                continue

            # Discover types dynamically or filter by request
            if types:
                type_names = types
            else:
                type_names = [
                    e.name for e in version_entry.iterdir() if e.is_dir()
                ]

            for cache_type in type_names:
                if types:
                    type_dir = _require_segment_dir(
                        version_entry, cache_type, cache_root
                    )
                else:
                    type_dir = _discover_segment_dir(
                        version_entry, cache_type, cache_root
                    )
                    if type_dir is None:
                        continue
                if not type_dir.is_dir():
                    continue

                if variants:
                    # Check if _flat is requested (flat-file types with no variant subdirs)
                    if "_flat" in variants:
                        # Delete entire type directory for flat-file types
                        size = _dir_size(type_dir)
                        safe_rmtree(type_dir)
                        deleted += 1
                        freed += size
                        logger.info(
                            "cache_type_purged",
                            model=model_name,
                            version=version_entry.name,
                            type=cache_type,
                            freed_bytes=size,
                        )
                        continue
                    # Delete specific variants only — `variants` is only ever
                    # populated from the client request (there is no on-disk
                    # discovery branch for this filter), so it always stays
                    # fail-fast via `_require_segment_dir`.
                    for variant in variants:
                        variant_dir = _require_segment_dir(
                            type_dir, variant, cache_root
                        )
                        if variant_dir.is_dir():
                            size = _dir_size(variant_dir)
                            safe_rmtree(variant_dir)
                            deleted += 1
                            freed += size
                            logger.info(
                                "cache_variant_purged",
                                model=model_name,
                                version=version_entry.name,
                                type=cache_type,
                                variant=variant,
                                freed_bytes=size,
                            )
                else:
                    # Delete entire type directory
                    size = _dir_size(type_dir)
                    safe_rmtree(type_dir)
                    deleted += 1
                    freed += size
                    logger.info(
                        "cache_type_purged",
                        model=model_name,
                        version=version_entry.name,
                        type=cache_type,
                        freed_bytes=size,
                    )

        # Clean up empty directories bottom-up
        if model_dir.is_dir():
            for version_entry2 in list(model_dir.iterdir()):
                if not version_entry2.is_dir():
                    continue
                # Clean empty type dirs (latents/, embeddings/)
                for type_entry in list(version_entry2.iterdir()):
                    if type_entry.is_dir() and not any(type_entry.iterdir()):
                        type_entry.rmdir()
                # Clean empty version dirs
                if not any(version_entry2.iterdir()):
                    version_entry2.rmdir()
            # Clean empty model dirs
            if not any(model_dir.iterdir()):
                model_dir.rmdir()

    return {"deleted": deleted, "freed_bytes": freed}


# ── Routes ───────────────────────────────────────────────────────────────


def _aggregate_cache_stats() -> dict[str, Any]:
    """Scan all datasets and aggregate cache sizes by type.

    Returns the existing per-`.cache/` totals (`total_bytes`, `latent_bytes`,
    `embedding_bytes`, `cached_datasets`) PLUS `dataset_root_bytes` — the full
    on-disk size of every `<dataset>/` folder (images + captions + masks +
    `.cache/` + anything else). The dataset-root walk overlaps with the
    per-cache iteration (two passes), but the redundancy is intentional: the
    DATASETS KPI corner needs the "what does this actually take on my disk"
    answer, not just the cache subset.
    """
    all_datasets = dataset_manager.list_datasets()
    total_bytes = 0
    latent_bytes = 0
    embedding_bytes = 0
    cached_datasets = 0
    dataset_root_bytes = 0   # full folder size including images, masks, etc.

    for ds in all_datasets:
        ds_path = Path(ds.path)
        if ds_path.is_dir():
            dataset_root_bytes += _dir_size(ds_path)
        cache_root = ds_path / ".cache"
        if not cache_root.is_dir():
            continue
        cached_datasets += 1
        for model_dir in cache_root.iterdir():
            if not model_dir.is_dir():
                continue
            for version_dir in model_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                for type_dir in version_dir.iterdir():
                    if not type_dir.is_dir():
                        continue
                    size = _dir_size(type_dir)
                    total_bytes += size
                    if type_dir.name == "latents":
                        latent_bytes += size
                    else:
                        embedding_bytes += size

    return {
        "total_bytes": total_bytes,
        "latent_bytes": latent_bytes,
        "embedding_bytes": embedding_bytes,
        "cached_datasets": cached_datasets,
        "dataset_root_bytes": dataset_root_bytes,
    }


# ── TTL cache for the expensive cross-dataset aggregation ────────────────

# Process-wide cache for the (expensive) cross-dataset aggregation. Served
# instantly within the TTL; recomputed on a cold/stale GET and warmed at startup
# by a silent background task.
_CACHE_STATS_TTL_S = 120.0
_cache_stats_value: dict | None = None
_cache_stats_at: float = 0.0
# Single-flight guard for the background recompute. Set from the event loop
# (request handler), cleared from a worker thread, so it takes a lock.
_cache_stats_lock = _threading.Lock()
_cache_stats_refreshing = False


def _get_fresh_cache_stats() -> dict | None:
    if _cache_stats_value is not None and (_time.time() - _cache_stats_at) < _CACHE_STATS_TTL_S:
        return _cache_stats_value
    return None


def _store_cache_stats(stats: dict) -> None:
    global _cache_stats_value, _cache_stats_at
    _cache_stats_value = stats
    _cache_stats_at = _time.time()


def _begin_refresh() -> bool:
    """Claim the right to start a background refresh. False if one is running."""
    global _cache_stats_refreshing
    with _cache_stats_lock:
        if _cache_stats_refreshing:
            return False
        _cache_stats_refreshing = True
        return True


def _end_refresh() -> None:
    global _cache_stats_refreshing
    with _cache_stats_lock:
        _cache_stats_refreshing = False


def _schedule_cache_stats_refresh() -> None:
    """Queue a silent recompute on the background lane; no-op if one is live."""
    if not _begin_refresh():
        return
    from app.core.tasks.task_manager import task_manager
    try:
        task = task_manager.create(
            type="cache_stats_warmup", title="Cache stats", user_visible=False,
        )
        task_manager.enqueue(task.id, run_cache_stats_refresh, lane="background")
    except Exception:  # noqa: BLE001 - never let a refresh failure break the GET
        _end_refresh()
        raise


def run_cache_stats_refresh(task_id: str) -> None:
    """Silent background worker — recompute the cross-dataset cache aggregation
    and warm the in-memory cache. Runs on the non-GPU 'background' lane."""
    from app.core.tasks.task_manager import task_manager
    try:
        stats = _aggregate_cache_stats()
        _store_cache_stats(stats)
    except Exception as exc:  # noqa: BLE001
        task_manager.fail(task_id, str(exc))
        return
    finally:
        _end_refresh()
    task_manager.complete(task_id)


@router.get("/datasets/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    """Aggregate cache size statistics across all datasets.

    Stale-while-revalidate. `_aggregate_cache_stats` walks every dataset root
    twice (cache subtrees + full on-disk size), so its cost scales with the
    whole library and is dominated by disk latency — seconds, not milliseconds,
    whenever the OS page cache is cold. It used to run INLINE whenever the value
    was older than the TTL, which meant the first visit to the Datasets screen
    after any idle gap longer than the TTL paid the full sweep. The startup
    warm-up hid that: it made the cost look like a once-per-boot event.

    Now an existing value is ALWAYS returned immediately and a recompute is
    queued on the background lane when it is stale. This is a "size on disk"
    KPI, so serving it one refresh-interval old is the right trade. The only
    remaining inline wait is the cold case where no value exists at all, which
    the startup warm-up normally covers before the first navigation.
    """
    if _cache_stats_value is not None:
        if _get_fresh_cache_stats() is None:
            _schedule_cache_stats_refresh()
        return _cache_stats_value

    stats = await asyncio.to_thread(_aggregate_cache_stats)
    _store_cache_stats(stats)
    return stats


@router.get("/datasets/{name}/cache/list", response_model=CacheListResponse)
async def list_cache(name: str, dataset: Dataset = Depends(get_dataset_or_404)):
    """List the cache tree for a dataset."""
    cache_root = Path(dataset.path) / ".cache"
    tree = await asyncio.to_thread(_scan_cache_tree, cache_root)
    return {"dataset": name, "cache": tree}


@router.post("/datasets/{name}/cache/purge", response_model=PurgeCacheResponse)
async def purge_cache(
    name: str, request: PurgeCacheRequest, dataset: Dataset = Depends(get_dataset_or_404),
):
    """Purge selected cache subtrees for a dataset."""
    cache_root = Path(dataset.path) / ".cache"
    logger.info(
        "cache_purge_requested",
        dataset=name,
        models=request.models,
        versions=request.versions,
        types=request.types,
        variants=request.variants,
    )

    result = await asyncio.to_thread(
        _purge_cache,
        cache_root,
        request.models,
        request.types,
        request.variants,
        request.versions,
    )
    return {"dataset": name, **result}
