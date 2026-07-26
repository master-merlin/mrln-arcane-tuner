"""Disk-persisted text embedding cache.

Mirrors the ``LatentManager`` pattern for text embeddings:
- Saves each caption's embedding(s) as a ``.safetensors`` file.
- File key: SHA-256 of the caption string for uniqueness.
- Supports multi-TE families (e.g. Flux1 with T5 + CLIP pooled)
  via separate sub-directories (``te1/``, ``te2/``).
- Includes TE quantization scheme in the path so FP8 and full-precision
  embeddings are cached separately and never collide.

Cache directory layout::

    {dataset_path}/.cache/{model_name}/{dataset_version}/embeddings/{te_quant}/te1/{source}_{hash}.safetensors
    {dataset_path}/.cache/{model_name}/{dataset_version}/embeddings/{te_quant}/te2/{source}_{hash}.safetensors
"""

import hashlib
import os
import re

import structlog
import torch
from safetensors.torch import load_file

from app.engine.utils.safe_save import safe_save_file

logger = structlog.get_logger(__name__)

# Maximum characters for the human-readable source segment of the filename.
_SOURCE_MAX_LEN = 40
_SAFE_CHARS_RE = re.compile(r"[^a-zA-Z0-9_\-]")


def _sanitize_source(raw: str) -> str:
    """Create a filesystem-safe, human-readable prefix from a source hint.

    Examples:
        ``"my_image (1).png"`` → ``"my_image_1_png"``
        ``""`` → ``"caption"``
    """
    if not raw:
        return "caption"
    s = _SAFE_CHARS_RE.sub("_", raw)
    # Collapse consecutive underscores
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:_SOURCE_MAX_LEN] or "caption"


class TextEmbeddingCache:
    """Disk-backed text embedding cache manager.

    Each caption is hashed (SHA-256) and stored as a safetensors file.
    The filename includes a human-readable source hint for inspectability.
    """

    # ── Path helpers ──────────────────────────────────────────────────────

    @staticmethod
    def resolve_te_cache_dir(
        dataset_path: str,
        model_name: str,
        dataset_version: str,
        te_slot: str = "te1",
        te_quant: str = "none",
    ) -> str:
        """Build the standard TE cache directory path.

        Includes the TE quantization scheme as a path segment so that
        FP8-quantized and full-precision embeddings are cached separately.

        Path layout::

            {ds}/.cache/{model}/{version}/embeddings/{te_quant}/te1/

        Args:
            dataset_path: Root of the dataset on disk.
            model_name: Model identifier (e.g. ``"flux2-dev"``).
            dataset_version: Dataset version string (e.g. ``"1.0.0"``).
            te_slot: ``"te1"`` for primary TE, ``"te2"`` for secondary.
            te_quant: TE quantization scheme (e.g. ``"none"``, ``"fp8"``,
                ``"nf4"``).  Different schemes produce numerically different
                embeddings and must not share a cache directory.

        Returns:
            Absolute directory path.
        """
        return os.path.join(
            dataset_path, ".cache", model_name, dataset_version,
            "embeddings", te_quant, te_slot,
        )

    @staticmethod
    def caption_to_filename(caption: str, source_hint: str = "") -> str:
        """Derive a unique filename from a caption string.

        Format: ``{sanitized_source}_{sha256_hex}.safetensors``

        Args:
            caption: The full caption text to hash.
            source_hint: Human-readable origin (image filename, "dropout",
                "sample_0", etc.).

        Returns:
            Filename string (no directory component).
        """
        sha = hashlib.sha256(caption.encode("utf-8")).hexdigest()
        source = _sanitize_source(source_hint)
        return f"{source}_{sha}.safetensors"

    # ── Save / Load ───────────────────────────────────────────────────────

    @staticmethod
    def save(
        caption: str,
        tensor: torch.Tensor,
        cache_dir: str,
        source_hint: str = "",
    ) -> str:
        """Save a single embedding tensor to disk.

        Args:
            caption: Caption string (used for filename hashing).
            tensor: Embedding tensor (stored on CPU).
            cache_dir: Directory to write into.
            source_hint: Human-readable origin for the filename.

        Returns:
            Absolute path of the saved file.
        """
        os.makedirs(cache_dir, exist_ok=True)
        fname = TextEmbeddingCache.caption_to_filename(caption, source_hint)
        path = os.path.join(cache_dir, fname)
        safe_save_file({"emb": tensor.detach().cpu()}, path)
        return path

    @staticmethod
    def load(
        caption: str,
        cache_dir: str,
        source_hint: str = "",
    ) -> torch.Tensor | None:
        """Load a cached embedding from disk.

        Args:
            caption: Caption string (used for filename hashing).
            cache_dir: Directory to search in.
            source_hint: Same hint used when saving.

        Returns:
            The embedding tensor (CPU), or ``None`` if not found / corrupt.
        """
        fname = TextEmbeddingCache.caption_to_filename(caption, source_hint)
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            return None
        try:
            data = load_file(path)
            return data["emb"]
        except Exception as e:
            logger.warning("te_cache_load_failed", path=path, error=str(e))
            return None

    @staticmethod
    def check_coverage(
        captions: list[str],
        cache_dir: str,
        source_hints: list[str] | None = None,
    ) -> tuple[int, int, list[str]]:
        """Count cached vs missing embeddings on disk.

        Args:
            captions: All caption strings to check.
            cache_dir: Directory to scan.
            source_hints: Optional per-caption source hints.  If ``None``,
                defaults are used (works because hash alone is unique).

        Returns:
            ``(cached_count, missing_count, sample_missing_captions)``
            where *sample_missing_captions* contains up to 10 samples.
        """
        cached = 0
        missing = 0
        missing_samples: list[str] = []
        hints = source_hints or [""] * len(captions)

        for caption, hint in zip(captions, hints):
            fname = TextEmbeddingCache.caption_to_filename(caption, hint)
            path = os.path.join(cache_dir, fname)
            if os.path.exists(path):
                cached += 1
            else:
                missing += 1
                if len(missing_samples) < 10:
                    missing_samples.append(caption[:80])

        return cached, missing, missing_samples
