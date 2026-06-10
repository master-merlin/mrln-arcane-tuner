# backend/app/core/captioning/tokenizer_service.py
"""Cached caption tokenizers + token counting with an overflow cutoff index.

Tokenizers are loaded lazily on first use (CLIP/T5 fast tokenizers, vocab-only,
small) and cached for the process lifetime. Families with ``tokenizer_kind ==
"heuristic"`` use a ~4-chars-per-token estimate and need no download.
"""

from __future__ import annotations

import math
from typing import Any

from app.core.logger import get_logger
from app.engine.core.caption_target import CaptionTarget

logger = get_logger(__name__)

# Average characters per token for the heuristic fallback.
_HEURISTIC_CHARS_PER_TOKEN = 4


class TokenizerService:
    """Singleton-style service; also instantiable directly in tests."""

    _instance: "TokenizerService | None" = None

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> "TokenizerService":
        if cls._instance is None:
            cls._instance = TokenizerService()
        return cls._instance

    def _get_tokenizer(self, tokenizer_id: str) -> Any:
        tok = self._cache.get(tokenizer_id)
        if tok is None:
            from transformers import AutoTokenizer

            logger.info("loading_tokenizer", tokenizer_id=tokenizer_id)
            tok = AutoTokenizer.from_pretrained(tokenizer_id)
            self._cache[tokenizer_id] = tok
        return tok

    def count_with_cutoff(self, text: str, target: CaptionTarget) -> tuple[int, int | None]:
        """Return ``(token_count, cutoff_char_index)``.

        ``cutoff_char_index`` is the character offset where ``usable_limit`` is
        exceeded (text from there on would be truncated), or ``None`` when the
        text fits.
        """
        if not text:
            return 0, None

        if target.tokenizer_kind == "heuristic":
            tokens = math.ceil(len(text) / _HEURISTIC_CHARS_PER_TOKEN)
            if tokens > target.usable_limit:
                return tokens, target.usable_limit * _HEURISTIC_CHARS_PER_TOKEN
            return tokens, None

        tok = self._get_tokenizer(target.tokenizer_id)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        tokens = len(ids)
        if tokens > target.usable_limit:
            # End char of the last in-budget token.
            cutoff = offsets[target.usable_limit - 1][1]
            return tokens, int(cutoff)
        return tokens, None
