"""Tests for TextEncoderOutput and TextEncodingCache."""


import pytest
import torch

from app.engine.core.text_encoding import (
    TextEncoderOutput,
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
