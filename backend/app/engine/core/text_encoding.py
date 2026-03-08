"""Text encoding output and caching infrastructure.

Provides:
- ``TextEncoderOutput`` — unified return type for all family encoders
- ``TextEncodingCache`` — lazy in-memory cache with TE offload guard

Usage::

    # Driver implements the raw encoding
    class Flux2Driver(IModelDriver):
        def encode_text(self, captions, dtype) -> TextEncoderOutput:
            return self.encode_layer_concat(captions, dtype)

    # Pipeline uses the cache wrapper
    cache = TextEncodingCache(driver, device)
    output = cache.encode(captions, dtype)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
import torch
from torch import Tensor

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# TextEncoderOutput
# ---------------------------------------------------------------------------

@dataclass
class TextEncoderOutput:
    """Unified text encoding result — eliminates side-effect storage.

    All family encoders return this struct. Consumers access fields
    explicitly rather than relying on hidden state (``_pooled_embeds``,
    ``_clip_pooled``).

    Attributes:
        embeddings: Primary text embeddings.  Shape varies by family:
            - Flux2: ``[B, L, D*N]`` (layer concatenation)
            - Flux1: ``[B, L, 4096]`` (T5 sequence)
            - SDXL:  ``[B, L, D1+D2]`` (dual CLIP hidden concat)
            - QwenImage: ``[B, L, D]``
            - ZImage: ``list[Tensor]`` — variable-length per sample
        attention_mask: Encoder attention mask (QwenImage only).
        pooled: Pooled text embeddings (SDXL: dual CLIP pooled,
            Flux1: CLIP pooled).
    """

    embeddings: Tensor | list[Tensor]
    attention_mask: Tensor | None = None
    pooled: Tensor | None = None

    def require_pooled(self) -> Tensor:
        """Guard: raise if pooled embeddings are missing.

        Call this in ``forward_pass()`` for families that need pooled
        embeddings (SDXL, Flux1).

        Raises:
            ValueError: If ``pooled`` is ``None``.
        """
        if self.pooled is None:
            raise ValueError(
                "TextEncoderOutput.pooled is None — this family's "
                "forward_pass() requires pooled embeddings but the "
                "driver's encode_text() did not produce them. "
                "Check the driver implementation."
            )
        return self.pooled

    def require_attention_mask(self) -> Tensor:
        """Guard: raise if attention mask is missing.

        Call this in ``forward_pass()`` for families that need an
        encoder attention mask (QwenImage).

        Raises:
            ValueError: If ``attention_mask`` is ``None``.
        """
        if self.attention_mask is None:
            raise ValueError(
                "TextEncoderOutput.attention_mask is None — this family's "
                "forward_pass() requires an attention mask but the "
                "driver's encode_text() did not produce one. "
                "Check the driver implementation."
            )
        return self.attention_mask


# ---------------------------------------------------------------------------
# TextEncodingCache
# ---------------------------------------------------------------------------

class TextEncodingCache:
    """Lazy in-memory cache for text embeddings.

    Wraps a driver's ``encode_text()`` method with per-caption caching.
    Handles TE GPU ↔ CPU offload transparently.

    The cache stores ``TextEncoderOutput`` fields on CPU to free GPU
    memory.  On retrieval, tensors are moved to the target device.

    Args:
        driver: Any object implementing ``encode_text(captions, dtype)``
            and providing ``get_text_encoders()`` and ``device``.
        device: Target device for returned tensors.
    """

    def __init__(self, driver: Any, device: torch.device):
        self.driver = driver
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)

        # Cache: caption_str → _CacheEntry
        self._cache: dict[str, _CacheEntry] = {}

    def encode(
        self,
        captions: list[str],
        dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Encode captions, using cache when possible.

        Args:
            captions: Batch of processed caption strings.
            dtype: Target dtype for the returned tensors.

        Returns:
            Batched ``TextEncoderOutput`` on ``self.device``.
        """
        uncached: list[tuple[int, str]] = []

        for i, cap in enumerate(captions):
            if cap not in self._cache:
                uncached.append((i, cap))

        if uncached:
            self._encode_uncached(uncached, dtype)

        return self._assemble_batch(captions, dtype)

    def clear(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Number of cached captions."""
        return len(self._cache)

    # --- Internal ---

    def _encode_uncached(
        self,
        uncached: list[tuple[int, str]],
        dtype: torch.dtype,
    ) -> None:
        """Encode uncached captions and store results."""
        text_encoders = self.driver.get_text_encoders()
        if not text_encoders:
            raise RuntimeError(
                "Text encoder was unloaded but encountered uncached "
                f"caption(s): {', '.join(cap[:50] for _, cap in uncached)}"
            )

        # TE offload guard — move to GPU if on CPU
        moved: list[tuple[str, torch.nn.Module]] = []
        for name, te in text_encoders.items():
            te_device = next(te.parameters()).device
            if te_device != self.device:
                self.logger.warning(
                    "te_cache_miss_after_offload",
                    encoder=name,
                    count=len(uncached),
                    hint="pre-caching should have covered all captions",
                )
                te.to(self.device)
                moved.append((name, te))

        try:
            # Encode one-by-one for per-caption cache granularity
            for _, cap in uncached:
                output = self.driver.encode_text([cap], dtype)
                self._cache[cap] = _CacheEntry.from_output(output)
        finally:
            # Restore offloaded TEs to CPU
            for name, te in moved:
                te.to("cpu")
            if moved:
                torch.cuda.empty_cache()

        self.logger.debug(
            "text_embeddings_cached",
            new=len(uncached),
            total=len(self._cache),
        )

    def _assemble_batch(
        self,
        captions: list[str],
        dtype: torch.dtype,
    ) -> TextEncoderOutput:
        """Reconstruct a batched TextEncoderOutput from cached entries."""
        entries = [self._cache[cap] for cap in captions]

        # Check if embeddings are variable-length (list[Tensor])
        first = entries[0]
        if first.is_variable_length:
            embeddings: Tensor | list[Tensor] = [
                e.embeddings.to(self.device, dtype=dtype)
                for e in entries
            ]
        else:
            embeddings = torch.stack(
                [e.embeddings.to(self.device, dtype=dtype) for e in entries],
                dim=0,
            )

        # Pooled (optional)
        pooled: Tensor | None = None
        if first.pooled is not None:
            pooled = torch.stack(
                [e.pooled.to(self.device, dtype=dtype) for e in entries],
                dim=0,
            )

        # Attention mask (optional)
        attention_mask: Tensor | None = None
        if first.attention_mask is not None:
            attention_mask = torch.stack(
                [e.attention_mask.to(self.device) for e in entries],
                dim=0,
            )

        return TextEncoderOutput(
            embeddings=embeddings,
            pooled=pooled,
            attention_mask=attention_mask,
        )


# ---------------------------------------------------------------------------
# Internal cache entry
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    """Single-caption cache entry stored on CPU."""

    embeddings: Tensor
    pooled: Tensor | None = None
    attention_mask: Tensor | None = None
    is_variable_length: bool = False

    @classmethod
    def from_output(cls, output: TextEncoderOutput) -> _CacheEntry:
        """Create a CPU cache entry from a TextEncoderOutput."""
        if isinstance(output.embeddings, list):
            # Variable-length (ZImage): store first element only
            # (encode is called with [single_caption])
            return cls(
                embeddings=output.embeddings[0].cpu(),
                pooled=output.pooled.cpu() if output.pooled is not None else None,
                attention_mask=(
                    output.attention_mask.cpu()
                    if output.attention_mask is not None
                    else None
                ),
                is_variable_length=True,
            )

        # Fixed-length: squeeze batch dim for single-caption
        emb = output.embeddings.squeeze(0).cpu()
        return cls(
            embeddings=emb,
            pooled=(
                output.pooled.squeeze(0).cpu()
                if output.pooled is not None
                else None
            ),
            attention_mask=(
                output.attention_mask.squeeze(0).cpu()
                if output.attention_mask is not None
                else None
            ),
            is_variable_length=False,
        )
