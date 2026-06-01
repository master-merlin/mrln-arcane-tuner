"""Dataset cache administration — list and purge latent/embedding caches."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api._path_guard import safe_rmtree
from app.core.dataset_manager import dataset_manager
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


# ── Request Models ───────────────────────────────────────────────────────

class PurgeCacheRequest(BaseModel):
    """Request body for cache purge."""
    models: list[str] | None = None
    types: list[str] | None = None       # ["latents", "embeddings"]
    variants: list[str] | None = None    # ["original", "masked"]


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
) -> dict[str, Any]:
    """Delete selected cache subtrees.

    Returns summary of what was deleted.
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
        model_dir = cache_root / model_name
        if not model_dir.is_dir():
            continue

        for version_entry in model_dir.iterdir():
            if not version_entry.is_dir():
                continue

            # Discover types dynamically or filter by request
            if types:
                type_names = types
            else:
                type_names = [
                    e.name for e in version_entry.iterdir() if e.is_dir()
                ]

            for cache_type in type_names:
                type_dir = version_entry / cache_type
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
                    # Delete specific variants only
                    for variant in variants:
                        variant_dir = type_dir / variant
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



@router.get("/datasets/cache/stats")
async def cache_stats():
    """Aggregate cache size statistics across all datasets."""
    return await asyncio.to_thread(_aggregate_cache_stats)


@router.get("/datasets/{name}/cache/list")
async def list_cache(name: str):
    """List the cache tree for a dataset."""
    dataset = dataset_manager.get_dataset(name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    cache_root = Path(dataset.path) / ".cache"
    tree = await asyncio.to_thread(_scan_cache_tree, cache_root)
    return {"dataset": name, "cache": tree}


@router.post("/datasets/{name}/cache/purge")
async def purge_cache(name: str, request: PurgeCacheRequest):
    """Purge selected cache subtrees for a dataset."""
    dataset = dataset_manager.get_dataset(name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    cache_root = Path(dataset.path) / ".cache"
    logger.info(
        "cache_purge_requested",
        dataset=name,
        models=request.models,
        types=request.types,
        variants=request.variants,
    )

    result = await asyncio.to_thread(
        _purge_cache, cache_root, request.models, request.types, request.variants,
    )
    return {"dataset": name, **result}
