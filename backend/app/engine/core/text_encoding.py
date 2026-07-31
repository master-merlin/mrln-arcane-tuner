"""Text encoding output type — the unified return of every family encoder.

``TextEncoderOutput`` replaces the side-effect storage the drivers used to rely
on (``_pooled_embeds``, ``_clip_pooled``): consumers read named fields, and the
``require_*`` guards turn "this family needs pooled embeddings but the driver
never produced them" into a named error at the call site instead of a
``NoneType`` crash deeper in ``forward_pass``.

Usage::

    class Flux2Driver(IModelDriver):
        def encode_text(self, captions, dtype) -> TextEncoderOutput:
            return self.encode_layer_concat(captions, dtype)

This module also carried a ``TextEncodingCache`` (plus its ``_CacheEntry``) —
a per-caption in-memory cache with a TE-offload guard. It was never
instantiated anywhere in the application; the live caching path is the
pipeline's own ``text_cache`` dict plus the on-disk TE cache. Removed rather
than left as a second, untaken implementation of a subtle piece of behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


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
