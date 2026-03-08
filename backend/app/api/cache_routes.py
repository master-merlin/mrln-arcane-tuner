"""Dataset cache administration — list and purge latent/embedding caches."""

from __future__ import annotations

import asyncio
import os
import shutil
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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

def _dir_size(path: str) -> int:
    """Calculate total size in bytes of a directory tree."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _scan_cache_tree(cache_root: str) -> dict[str, Any]:
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

    if not os.path.isdir(cache_root):
        return tree

    for model_entry in os.scandir(cache_root):
        if not model_entry.is_dir():
            continue
        model_name = model_entry.name
        tree[model_name] = {}

        for version_entry in os.scandir(model_entry.path):
            if not version_entry.is_dir():
                continue
            version = version_entry.name
            version_data: dict[str, Any] = {
                "types": {},
                "size_bytes": _dir_size(version_entry.path),
            }

            # Scan all type directories dynamically
            for type_entry in os.scandir(version_entry.path):
                if not type_entry.is_dir():
                    continue
                type_name = type_entry.name
                variants: dict[str, Any] = {}
                has_files = False

                for child in os.scandir(type_entry.path):
                    if child.is_dir():
                        # Check if variants have sub-dirs (resolutions) or just files
                        sub_items = [
                            r.name for r in os.scandir(child.path)
                            if r.is_dir()
                        ]
                        if sub_items:
                            variants[child.name] = sorted(sub_items)
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
    cache_root: str,
    models: list[str] | None,
    types: list[str] | None,
    variants: list[str] | None,
) -> dict[str, Any]:
    """Delete selected cache subtrees.

    Returns summary of what was deleted.
    """
    if not os.path.isdir(cache_root):
        return {"deleted": 0, "freed_bytes": 0}

    deleted = 0
    freed = 0

    # Determine which models to target
    target_models = models if models else [
        e.name for e in os.scandir(cache_root) if e.is_dir()
    ]

    # Determine which types to target (discover dynamically if not specified)
    # types filter is applied per-version below

    for model_name in target_models:
        model_dir = os.path.join(cache_root, model_name)
        if not os.path.isdir(model_dir):
            continue

        for version_entry in os.scandir(model_dir):
            if not version_entry.is_dir():
                continue

            # Discover types dynamically or filter by request
            if types:
                type_names = types
            else:
                type_names = [
                    e.name for e in os.scandir(version_entry.path)
                    if e.is_dir()
                ]

            for cache_type in type_names:
                type_dir = os.path.join(version_entry.path, cache_type)
                if not os.path.isdir(type_dir):
                    continue

                if variants:
                    # Check if _flat is requested (flat-file types with no variant subdirs)
                    if "_flat" in variants:
                        # Delete entire type directory for flat-file types
                        size = _dir_size(type_dir)
                        shutil.rmtree(type_dir)
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
                        variant_dir = os.path.join(type_dir, variant)
                        if os.path.isdir(variant_dir):
                            size = _dir_size(variant_dir)
                            shutil.rmtree(variant_dir)
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
                    shutil.rmtree(type_dir)
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
        if os.path.isdir(model_dir):
            for version_entry2 in list(os.scandir(model_dir)):
                if not version_entry2.is_dir():
                    continue
                # Clean empty type dirs (latents/, embeddings/)
                for type_entry in list(os.scandir(version_entry2.path)):
                    if type_entry.is_dir() and not any(os.scandir(type_entry.path)):
                        os.rmdir(type_entry.path)
                # Clean empty version dirs
                if not any(os.scandir(version_entry2.path)):
                    os.rmdir(version_entry2.path)
            # Clean empty model dirs
            if not any(os.scandir(model_dir)):
                os.rmdir(model_dir)

    return {"deleted": deleted, "freed_bytes": freed}


# ── Routes ───────────────────────────────────────────────────────────────

@router.get("/datasets/{name}/cache/list")
async def list_cache(name: str):
    """List the cache tree for a dataset."""
    dataset = dataset_manager.get_dataset(name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    cache_root = os.path.join(dataset.path, ".cache")
    tree = await asyncio.to_thread(_scan_cache_tree, cache_root)
    return {"dataset": name, "cache": tree}


@router.post("/datasets/{name}/cache/purge")
async def purge_cache(name: str, request: PurgeCacheRequest):
    """Purge selected cache subtrees for a dataset."""
    dataset = dataset_manager.get_dataset(name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    cache_root = os.path.join(dataset.path, ".cache")
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
