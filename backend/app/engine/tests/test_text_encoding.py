"""Tests for TextEncoderOutput and TextEncodingCache."""

from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from app.engine.core.text_encoding import (
    TextEncoderOutput,
    TextEncodingCache,
    _CacheEntry,
)


# ---------------------------------------------------------------------------
# TextEncoderOutput
# ---------------------------------------------------------------------------

class TestTextEncoderOutput:
    """Validate TextEncoderOutput fields and guards."""

    def test_basic_embeddings_only(self):
        emb = torch.randn(2, 10, 768)
        out = TextEncoderOutput(embeddings=emb)
        assert out.embeddings is emb
        assert out.pooled is None
        assert out.attention_mask is None

    def test_with_pooled(self):
        emb = torch.randn(2, 10, 768)
        pooled = torch.randn(2, 768)
        out = TextEncoderOutput(embeddings=emb, pooled=pooled)
        assert out.pooled is pooled

    def test_with_attention_mask(self):
        emb = torch.randn(2, 10, 768)
        mask = torch.ones(2, 10)
        out = TextEncoderOutput(embeddings=emb, attention_mask=mask)
        assert out.attention_mask is mask

    def test_variable_length_embeddings(self):
        emb_list = [torch.randn(5, 768), torch.randn(7, 768)]
        out = TextEncoderOutput(embeddings=emb_list)
        assert isinstance(out.embeddings, list)
        assert len(out.embeddings) == 2

    def test_require_pooled_success(self):
        pooled = torch.randn(2, 768)
        out = TextEncoderOutput(embeddings=torch.randn(2, 10, 768), pooled=pooled)
        result = out.require_pooled()
        assert result is pooled

    def test_require_pooled_raises(self):
        out = TextEncoderOutput(embeddings=torch.randn(2, 10, 768))
        with pytest.raises(ValueError, match="pooled is None"):
            out.require_pooled()

    def test_require_attention_mask_success(self):
        mask = torch.ones(2, 10)
        out = TextEncoderOutput(
            embeddings=torch.randn(2, 10, 768), attention_mask=mask,
        )
        result = out.require_attention_mask()
        assert result is mask

    def test_require_attention_mask_raises(self):
        out = TextEncoderOutput(embeddings=torch.randn(2, 10, 768))
        with pytest.raises(ValueError, match="attention_mask is None"):
            out.require_attention_mask()


# ---------------------------------------------------------------------------
# _CacheEntry
# ---------------------------------------------------------------------------

class TestCacheEntry:
    """Validate internal cache entry serialization."""

    def test_from_fixed_length_output(self):
        out = TextEncoderOutput(
            embeddings=torch.randn(1, 10, 768),
            pooled=torch.randn(1, 768),
        )
        entry = _CacheEntry.from_output(out)
        assert entry.embeddings.shape == (10, 768)  # squeezed
        assert entry.pooled.shape == (768,)  # squeezed
        assert entry.is_variable_length is False

    def test_from_variable_length_output(self):
        out = TextEncoderOutput(
            embeddings=[torch.randn(5, 4096)],
        )
        entry = _CacheEntry.from_output(out)
        assert entry.embeddings.shape == (5, 4096)
        assert entry.is_variable_length is True

    def test_from_output_with_mask(self):
        out = TextEncoderOutput(
            embeddings=torch.randn(1, 10, 768),
            attention_mask=torch.ones(1, 10),
        )
        entry = _CacheEntry.from_output(out)
        assert entry.attention_mask is not None
        assert entry.attention_mask.shape == (10,)  # squeezed

    def test_cpu_storage(self):
        """Cache entries are stored on CPU."""
        out = TextEncoderOutput(
            embeddings=torch.randn(1, 10, 768),
            pooled=torch.randn(1, 768),
        )
        entry = _CacheEntry.from_output(out)
        assert entry.embeddings.device == torch.device("cpu")
        assert entry.pooled.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# TextEncodingCache
# ---------------------------------------------------------------------------

class _MockDriver:
    """Minimal driver mock for cache testing."""

    def __init__(self, device: torch.device, dim: int = 768):
        self.device = device
        self.dim = dim
        self.encode_call_count = 0

        # Fake text encoder — parameters() must return a fresh iterator each call
        self._te = MagicMock()
        self._te_param = torch.nn.Parameter(torch.zeros(1))
        self._te_param.data = self._te_param.data.to(device)
        self._te.parameters = lambda: iter([self._te_param])

    def get_text_encoders(self) -> dict[str, torch.nn.Module]:
        return {"text_encoder": self._te}

    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        self.encode_call_count += 1
        B = len(captions)
        return TextEncoderOutput(
            embeddings=torch.randn(B, 10, self.dim, dtype=dtype),
        )


class _MockPooledDriver(_MockDriver):
    """Driver that produces pooled embeddings."""

    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        self.encode_call_count += 1
        B = len(captions)
        return TextEncoderOutput(
            embeddings=torch.randn(B, 10, self.dim, dtype=dtype),
            pooled=torch.randn(B, self.dim, dtype=dtype),
        )


class _MockVariableLengthDriver(_MockDriver):
    """Driver that produces variable-length embeddings."""

    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
    ) -> TextEncoderOutput:
        self.encode_call_count += 1
        emb_list = [torch.randn(5 + i, self.dim, dtype=dtype) for i in range(len(captions))]
        return TextEncoderOutput(embeddings=emb_list)


class TestTextEncodingCache:
    """Test TextEncodingCache."""

    def test_cache_hit(self):
        """Second call for same caption returns cached result."""
        driver = _MockDriver(torch.device("cpu"))
        cache = TextEncodingCache(driver, torch.device("cpu"))

        # First call — cache miss
        cache.encode(["hello"], torch.float32)
        assert driver.encode_call_count == 1

        # Second call — cache hit
        cache.encode(["hello"], torch.float32)
        assert driver.encode_call_count == 1  # no new encode

    def test_cache_miss(self):
        """New caption triggers encoding."""
        driver = _MockDriver(torch.device("cpu"))
        cache = TextEncodingCache(driver, torch.device("cpu"))

        cache.encode(["hello"], torch.float32)
        cache.encode(["world"], torch.float32)
        assert driver.encode_call_count == 2

    def test_cache_size(self):
        driver = _MockDriver(torch.device("cpu"))
        cache = TextEncodingCache(driver, torch.device("cpu"))

        cache.encode(["a", "b", "c"], torch.float32)
        assert cache.size == 3

    def test_clear(self):
        driver = _MockDriver(torch.device("cpu"))
        cache = TextEncodingCache(driver, torch.device("cpu"))

        cache.encode(["a"], torch.float32)
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0

    def test_batch_assembly_fixed_length(self):
        """Batch assembly returns correct shape for fixed-length embeddings."""
        driver = _MockDriver(torch.device("cpu"), dim=768)
        cache = TextEncodingCache(driver, torch.device("cpu"))

        result = cache.encode(["a", "b"], torch.float32)
        assert isinstance(result.embeddings, Tensor)
        assert result.embeddings.shape[0] == 2  # batch dim

    def test_batch_assembly_with_pooled(self):
        """Pooled embeddings are assembled across batch."""
        driver = _MockPooledDriver(torch.device("cpu"), dim=768)
        cache = TextEncodingCache(driver, torch.device("cpu"))

        result = cache.encode(["a", "b"], torch.float32)
        assert result.pooled is not None
        assert result.pooled.shape[0] == 2

    def test_batch_assembly_variable_length(self):
        """Variable-length embeddings return as list."""
        driver = _MockVariableLengthDriver(torch.device("cpu"), dim=768)
        cache = TextEncodingCache(driver, torch.device("cpu"))

        result = cache.encode(["a", "b"], torch.float32)
        assert isinstance(result.embeddings, list)
        assert len(result.embeddings) == 2

    def test_error_on_unloaded_te(self):
        """RuntimeError when TE is gone and caption is uncached."""
        driver = _MockDriver(torch.device("cpu"))
        cache = TextEncodingCache(driver, torch.device("cpu"))

        # Make TE unavailable
        driver.get_text_encoders = lambda: {}

        with pytest.raises(RuntimeError, match="unloaded"):
            cache.encode(["never_cached"], torch.float32)
