"""Model path resolution utilities.

Handles local paths, relative paths, and ``huggingface:`` URI scheme
resolution.  Sets HF Hub symlink env-vars to work around WinError 1314
on Windows.
"""

from __future__ import annotations

import os

import structlog
from huggingface_hub import hf_hub_download, snapshot_download

# Prevent WinError 1314: symlink permission errors on Windows.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

logger = structlog.get_logger(__name__)

class ModelPathResolver:
    """Resolve model component paths from local or ``huggingface:`` URIs."""

    @staticmethod
    def resolve(
        path_str: str,
        base_dir: str | None = None,
        *,
        local_files_only: bool = False,
    ) -> str | None:
        """Resolve a path string to an absolute local path.

        Supports:
        - ``huggingface:<repo_id>`` — downloads full snapshot
        - ``huggingface:<repo_id>:<filename>`` — downloads single file
        - Absolute local paths — returned as-is
        - Relative local paths — joined with *base_dir* or cwd

        Args:
            path_str: Path or HuggingFace URI.
            base_dir: Optional base directory for relative paths.
            local_files_only: When ``True``, skip HF download and only
                use files already in the local cache.  Raises
                ``FileNotFoundError`` if not cached.

        Returns:
            Absolute local path, or ``None`` if *path_str* is empty.
        """
        if not path_str:
            return None
            
        # 1. HuggingFace Handling
        if path_str.startswith("huggingface:"):
            return ModelPathResolver._resolve_hf(
                path_str, local_files_only=local_files_only,
            )
            
        # 2. Local Path Handling
        # If absolute, return as is
        if os.path.isabs(path_str):
            # We return it even if it doesn't exist, to let caller fail with clear message
            return path_str
            
        # If relative, join with base_dir or cwd
        if base_dir:
            full_path = os.path.join(base_dir, path_str)
        else:
            full_path = os.path.abspath(path_str)
            
        return full_path

    @staticmethod
    def _resolve_hf(
        path_str: str,
        *,
        local_files_only: bool = False,
    ) -> str:
        """Download from HuggingFace Hub and return the local cache path."""
        from app.api.events.download_progress import make_progress_tqdm, with_progress

        clean = path_str.replace("huggingface:", "")
        parts = clean.split(":")
        repo_id = parts[0]
        filename = parts[1] if len(parts) > 1 else None
        # model_id for the WS payload — disambiguate single-file vs snapshot
        progress_id = f"{repo_id}/{filename}" if filename else repo_id

        # Offline / skip-update mode: cache only, never hit the network.
        # We do NOT use this as a fast-path when online: a previously
        # interrupted download leaves a *partial* snapshot in the cache, and
        # ``snapshot_download(local_files_only=True)`` returns it as if it were
        # complete (it does not verify the file manifest offline). The loader
        # then fails on a missing subfolder (e.g. ``tokenizer/``) with
        # "Unrecognized model ... should have a ``model_type`` key". So when
        # online we always run the resumable download below, which re-checks
        # every file's etag and fetches only what's missing — self-healing a
        # partial cache.
        if local_files_only:
            try:
                if filename:
                    return hf_hub_download(
                        repo_id=repo_id, filename=filename, local_files_only=True,
                    )
                return snapshot_download(repo_id=repo_id, local_files_only=True)
            except Exception:
                raise FileNotFoundError(
                    f"Model '{repo_id}' not found in local HF cache. "
                    "Disable offline / skip-update mode or download "
                    "the model first.",
                )

        # Real download — wrap with progress emits. Bind the WS metadata with a
        # tqdm SUBCLASS (not functools.partial): snapshot_download fetches files
        # concurrently and calls the classmethod tqdm_class.get_lock(), which a
        # partial cannot provide ("'functools.partial' object has no attribute
        # 'get_lock'").
        bound_tqdm = make_progress_tqdm(
            source="hf", model_id=progress_id, category="training",
        )
        try:
            with with_progress(model_id=progress_id, category="training"):
                if filename:
                    logger.info("downloading_file_from_hub", repo=repo_id, file=filename)
                    # hf_hub_download() does NOT accept tqdm_class (huggingface_hub
                    # >= 0.36 — only snapshot_download does); passing it raises
                    # "unexpected keyword argument 'tqdm_class'" and aborts the
                    # download. with_progress still emits coarse start/complete.
                    return hf_hub_download(repo_id=repo_id, filename=filename)
                else:
                    logger.info("downloading_snapshot_from_hub", repo=repo_id)
                    return snapshot_download(repo_id=repo_id, tqdm_class=bound_tqdm)
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("hf_download_failed", repo=repo_id, file=filename, error=str(e))
            raise

    @staticmethod
    def find_component(
        definition: any,
        component_key: str,
        root_path: str | None = None,
        candidates: list[str] | None = None,
    ) -> str | None:
        """Smart discovery of a component path.

        1. Checks explicit entry in ``definition.components[key]``.
        2. Falls back to scanning *root_path* for *candidates* files/dirs.

        Args:
            definition: Model definition with a ``components`` dict.
            component_key: Component to look up (e.g. ``"vae"``).
            root_path: Optional model root directory for fallback scanning.
            candidates: File/dir names to probe inside *root_path*.

        Returns:
            Resolved path, or ``None`` if not found.
        """
        # 1. Explicit Definition
        comp = definition.components.get(component_key)
        if comp:
            return ModelPathResolver.resolve(comp.path)
            
        # 2. Discovery in Root
        if root_path and candidates:
            for candidate in candidates:
                candidate_path = os.path.join(root_path, candidate)
                if os.path.exists(candidate_path):
                    logger.info("component_discovered_in_root", key=component_key, path=candidate_path)
                    return candidate_path
        
        return None
