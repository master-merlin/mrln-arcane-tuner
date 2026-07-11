"""Model path resolution utilities.

Handles local paths, relative paths, and ``huggingface:`` URI scheme
resolution.  Sets HF Hub symlink env-vars to work around WinError 1314
on Windows.
"""

from __future__ import annotations

import os

import structlog
from huggingface_hub import hf_hub_download, snapshot_download

from app.engine.utils.hf_download_guard import download_with_stall_guard

# Prevent WinError 1314: symlink permission errors on Windows.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

logger = structlog.get_logger(__name__)


def _snapshot_fully_cached(repo_id: str, revision: str | None = None) -> bool:
    """True iff EVERY file in *repo_id* is already in the local HF cache.

    A bare ``snapshot_download(repo_id, local_files_only=True)`` is NOT a
    reliable "is it cached" signal: it returns whatever snapshot folder exists
    even when the previous download was interrupted and files are still missing
    (it does not verify the manifest offline — see ``_resolve_hf``). Using it to
    gate the progress bar means a *partial* cache is treated as a cache hit, so
    the resume runs SILENTLY — the top-bar download indicator stays dark and the
    job looks idle while a large model quietly transfers in the background.

    Cross-checking the Hub's file list against ``try_to_load_from_cache`` makes a
    partial cache report ``False``, so the caller attaches the emitting tqdm and
    the remainder downloads WITH progress. Only a genuinely complete snapshot
    returns ``True`` (and skips the bar, avoiding a spurious flash).

    Any error (offline, repo removed, auth) → ``False``: prefer showing progress
    over a misleading silent download. The caller's online ``snapshot_download``
    then surfaces any real error as before.
    """
    try:
        from huggingface_hub import HfApi, try_to_load_from_cache

        rev_kwargs = {"revision": revision} if revision else {}
        files = HfApi().list_repo_files(repo_id, **rev_kwargs)
        if not files:
            return False
        return all(
            isinstance(try_to_load_from_cache(repo_id, f, **rev_kwargs), str)
            for f in files
        )
    except Exception as e:
        logger.debug("snapshot_cache_probe_failed", repo=repo_id, error=str(e))
        return False


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
        - ``huggingface:<repo_id>@<revision>[:<filename>]`` — pins a branch /
          tag / commit (e.g. the DreamLite checkpoints' ``diffusers`` branch)
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
        from app.api.events.download_progress import (
            snapshot_byte_progress,
            with_progress,
        )

        clean = path_str.replace("huggingface:", "")
        parts = clean.split(":")
        repo_id = parts[0]
        filename = parts[1] if len(parts) > 1 else None
        # Optional "@revision" suffix on the repo id (branch / tag / commit).
        revision = None
        if "@" in repo_id:
            repo_id, revision = repo_id.split("@", 1)
        # Passed conditionally so revision-less calls keep their legacy
        # kwargs shape (no ``revision=None`` noise).
        rev_kwargs = {"revision": revision} if revision else {}
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
                        repo_id=repo_id, filename=filename,
                        local_files_only=True, **rev_kwargs,
                    )
                return snapshot_download(
                    repo_id=repo_id, local_files_only=True, **rev_kwargs,
                )
            except Exception:
                raise FileNotFoundError(
                    f"Model '{repo_id}' not found in local HF cache. "
                    "Disable offline / skip-update mode or download "
                    "the model first.",
                )

        # Every ONLINE call below routes through download_with_stall_guard,
        # which runs the actual snapshot_download/hf_hub_download in a
        # killable child process — see hf_download_guard's module docstring
        # for why: an in-process HF call cannot be aborted on a stall (Python
        # threads/socket reads are un-abortable), and this resolve runs both
        # in the API process AND inside the detached trainer subprocess,
        # where a wedged download is invisible and survives backend
        # restarts. local_files_only (above) is unaffected — offline
        # resolves never hit the network, so there's nothing to stall on.
        try:
            if filename:
                logger.info("downloading_file_from_hub", repo=repo_id, file=filename)
                # with_progress emits a coarse start/complete pair for the
                # indicator; the guard's child process can't be attached to
                # an in-process tqdm (see hf_download_guard's docstring for
                # the accepted per-file-breakdown regression).
                with with_progress(model_id=progress_id, category="training"):
                    return download_with_stall_guard(
                        repo_id=repo_id, filename=filename, revision=revision,
                    )

            # Snapshot (full repo). Only surface the download indicator on a
            # REAL transfer: a snapshot fully on disk is loaded, not downloaded.
            # We must NOT gate that on a bare local_files_only probe — it returns
            # a *partial* snapshot as if complete, so an interrupted download
            # would resume SILENTLY (dark indicator, job stuck looking idle).
            # _snapshot_fully_cached cross-checks the Hub's file list against the
            # cache, so only a genuinely complete snapshot skips the bar.
            if _snapshot_fully_cached(repo_id, revision):
                logger.info("snapshot_cache_hit", repo=repo_id)
                return download_with_stall_guard(repo_id=repo_id, revision=revision)

            # A real (or partial-resume) transfer. We can't attach an emitting
            # tqdm for BYTE progress — HF only routes tqdm_class to the coarse
            # "Fetching N files" bar (its docstring: "tqdm_class is not passed to
            # each individual download"), which sits frozen at 0/N while a single
            # multi-GB shard downloads. snapshot_byte_progress instead polls the
            # on-disk cache growth against the repo's total size for true,
            # resume-aware byte progress — the SAME on-disk signal the guard's
            # own stall watchdog polls, so it keeps working when the bytes are
            # written by a child process instead of this one. The online
            # download re-checks etags and fetches only what's missing, so a
            # partial cache (whether from an old interruption or a guard-killed
            # attempt) self-heals.
            logger.info("downloading_snapshot_from_hub", repo=repo_id)
            with snapshot_byte_progress(
                repo_id=repo_id, model_id=progress_id, category="training",
            ):
                return download_with_stall_guard(repo_id=repo_id, revision=revision)
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("hf_download_failed", repo=repo_id, file=filename, error=str(e))
            raise

    @staticmethod
    def ensure_definition_cached(definition: any) -> None:
        """Pre-fetch a definition's Hugging Face components in-process.

        Training runs in a detached subprocess (``run_trainer.py``) where the
        download-progress WS bridge is a no-op — ``_APP_LOOP`` is only captured
        in the API process by ``main.lifespan``, so ``schedule_emit_from_thread``
        silently drops every event there. Calling this from the API process
        *before* launching the trainer downloads the base model through the
        progress-emitting ``_resolve_hf`` path, so the top-bar download
        indicator updates; the subprocess then loads from the warm HF cache.

        Respects per-model source overrides (mirrors ``GenericComponentLoader.
        _resolve_root``):
        - ``LOCAL_DIFFUSERS`` / ``LOCAL_SAFETENSORS`` → weights are already
          local, nothing to fetch.
        - ``HF_HUB`` + skip-update / global-offline → resolve cache-only
          (``local_files_only=True``); a missing repo raises here, surfacing the
          problem as a clean pre-flight error rather than a trainer crash.

        Best-effort: only ``huggingface:`` component paths are pre-fetched; any
        non-HF (absolute / relative) paths are left to the loader.
        """
        from app.core.schemas.model_overrides import ModelSourceType
        from app.engine.utils.model_override_manager import ModelOverrideManager

        source_type, _local_path, local_files_only = (
            ModelOverrideManager.resolve_effective_source(definition.id)
        )
        if source_type in (
            ModelSourceType.LOCAL_DIFFUSERS,
            ModelSourceType.LOCAL_SAFETENSORS,
        ):
            return

        for comp in definition.components.values():
            path = getattr(comp, "path", None)
            if isinstance(path, str) and path.startswith("huggingface:"):
                ModelPathResolver.resolve(path, local_files_only=local_files_only)

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
