"""
Tests for EmbeddingManager: text encoder embedding caching and offloading.
Phase 6: Text Encoder Offloading & Embedding Caching.
"""

import pytest
import os
import torch
from unittest.mock import MagicMock
from types import SimpleNamespace

from app.engine.components.embedding_manager import EmbeddingManager


# ── Mock Text Encoder ────────────────────────────────────────────────────


def make_mock_encoder(hidden_dim: int = 768, num_layers: int = 4, has_pooled: bool = False):
    """Create a mock text encoder that returns hidden_states."""
    def forward(input_ids, output_hidden_states=False):
        B, L = input_ids.shape
        hidden_states = tuple(torch.randn(B, L, hidden_dim) for _ in range(num_layers))
        result = SimpleNamespace(hidden_states=hidden_states)
        if has_pooled:
            result.text_embeds = torch.randn(B, hidden_dim)
        return result
    
    encoder = MagicMock()
    encoder.side_effect = forward
    encoder.to = MagicMock(return_value=encoder)
    return encoder


def make_mock_tokenizer(max_length: int = 77):
    """Create a mock tokenizer that returns dummy input_ids."""
    def tokenize(text, padding=None, max_length=77, truncation=None, return_tensors=None):
        B = len(text) if isinstance(text, list) else 1
        return SimpleNamespace(input_ids=torch.randint(0, 1000, (B, max_length)))
    
    tokenizer = MagicMock()
    tokenizer.side_effect = tokenize
    return tokenizer


# ── SDXL Encoding Tests ─────────────────────────────────────────────────


class TestSDXLEncoding:
    """Tests for SDXL dual-CLIP text encoding."""

    def test_encode_sdxl_returns_expected_keys(self):
        """SDXL encoding should return prompt_embeds and pooled_embeds."""
        em = EmbeddingManager(model_family="sdxl", device="cpu")
        tok1 = make_mock_tokenizer()
        tok2 = make_mock_tokenizer()
        enc1 = make_mock_encoder(hidden_dim=768, num_layers=4)
        enc2 = make_mock_encoder(hidden_dim=1280, num_layers=4, has_pooled=True)

        result = em.encode_sdxl(
            ["a cat", "a dog"],
            tokenizer_1=tok1, tokenizer_2=tok2,
            text_encoder_1=enc1, text_encoder_2=enc2,
            device="cpu",
        )

        assert "prompt_embeds" in result
        assert "pooled_embeds" in result

    def test_encode_sdxl_prompt_embeds_shape(self):
        """prompt_embeds should be [B, L, 768+1280=2048]."""
        em = EmbeddingManager(model_family="sdxl", device="cpu")
        tok1 = make_mock_tokenizer()
        tok2 = make_mock_tokenizer()
        enc1 = make_mock_encoder(hidden_dim=768, num_layers=4)
        enc2 = make_mock_encoder(hidden_dim=1280, num_layers=4, has_pooled=True)

        result = em.encode_sdxl(
            ["hello"],
            tokenizer_1=tok1, tokenizer_2=tok2,
            text_encoder_1=enc1, text_encoder_2=enc2,
            device="cpu",
        )

        assert result["prompt_embeds"].shape[0] == 1          # batch
        assert result["prompt_embeds"].shape[2] == 768 + 1280  # concat dim


# ── Flux2 Encoding Tests ────────────────────────────────────────────────


class TestFlux2Encoding:
    """Tests for Flux2 single-TE text encoding."""

    def test_encode_flux2_returns_ctx(self):
        """Flux2 encoding should return ctx key."""
        em = EmbeddingManager(model_family="flux2", device="cpu")
        tok = make_mock_tokenizer(max_length=512)
        enc = make_mock_encoder(hidden_dim=4096, num_layers=6)

        result = em.encode_flux2(
            ["a landscape"], tokenizer=tok, text_encoder=enc,
            device="cpu", concat_layers=3,
        )

        assert "ctx" in result

    def test_encode_flux2_concat_shape(self):
        """ctx should be [B, L, D*concat_layers]."""
        em = EmbeddingManager(model_family="flux2", device="cpu")
        tok = make_mock_tokenizer(max_length=512)
        enc = make_mock_encoder(hidden_dim=4096, num_layers=6)

        result = em.encode_flux2(
            ["test"], tokenizer=tok, text_encoder=enc,
            device="cpu", concat_layers=3,
        )

        assert result["ctx"].shape[2] == 4096 * 3

    def test_encode_flux2_single_layer(self):
        """With concat_layers=1, ctx dim should match single hidden dim."""
        em = EmbeddingManager(model_family="flux2", device="cpu")
        tok = make_mock_tokenizer(max_length=512)
        enc = make_mock_encoder(hidden_dim=4096, num_layers=4)

        result = em.encode_flux2(
            ["test"], tokenizer=tok, text_encoder=enc,
            device="cpu", concat_layers=1,
        )

        assert result["ctx"].shape[2] == 4096


# ── Cache Tests ──────────────────────────────────────────────────────────


class TestEmbeddingCaching:
    """Tests for embedding caching (save and load)."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Saved embeddings should load back identically."""
        em = EmbeddingManager(model_family="sdxl", cache_dir=str(tmp_path), device="cpu")

        embeddings = {
            "prompt_embeds": torch.randn(2, 77, 2048),
            "pooled_embeds": torch.randn(2, 1280),
        }
        ids = ["img_001", "img_002"]

        em.save_embeddings(embeddings, ids)

        loaded = em.load_cached_embeddings(ids)
        assert loaded is not None
        assert torch.allclose(loaded["prompt_embeds"][0], embeddings["prompt_embeds"][0])
        assert torch.allclose(loaded["pooled_embeds"][1], embeddings["pooled_embeds"][1])

    def test_load_returns_none_when_missing(self, tmp_path):
        """load_cached_embeddings should return None when cache is incomplete."""
        em = EmbeddingManager(model_family="sdxl", cache_dir=str(tmp_path), device="cpu")
        result = em.load_cached_embeddings(["nonexistent_id"])
        assert result is None

    def test_per_item_cache_dirs(self, tmp_path):
        """Each item should be saved to its specific cache directory."""
        em = EmbeddingManager(model_family="flux2", device="cpu")

        dir1 = str(tmp_path / "ds1")
        dir2 = str(tmp_path / "ds2")

        embeddings = {"ctx": torch.randn(2, 512, 12288)}
        ids = ["a", "b"]
        cache_dirs = [dir1, dir2]

        em.save_embeddings(embeddings, ids, cache_dirs)

        assert os.path.exists(os.path.join(dir1, "a.safetensors"))
        assert os.path.exists(os.path.join(dir2, "b.safetensors"))

    def test_partial_cache_returns_none(self, tmp_path):
        """If any item is missing from cache, return None (consistent batch)."""
        em = EmbeddingManager(model_family="sdxl", cache_dir=str(tmp_path), device="cpu")

        embeddings = {"prompt_embeds": torch.randn(1, 77, 2048)}
        em.save_embeddings(embeddings, ["exists"])

        result = em.load_cached_embeddings(["exists", "missing"])
        assert result is None

    def test_cache_stats_tracking(self, tmp_path):
        """Cache hits and misses should be tracked."""
        em = EmbeddingManager(model_family="sdxl", cache_dir=str(tmp_path), device="cpu")

        # Miss
        em.load_cached_embeddings(["missing"])

        # Hit
        embeddings = {"prompt_embeds": torch.randn(1, 77, 2048)}
        em.save_embeddings(embeddings, ["hit_item"])
        em.load_cached_embeddings(["hit_item"])

        assert em._cache_hits == 1
        assert em._cache_misses == 1


# ── Offloading Tests ─────────────────────────────────────────────────────


class TestTextEncoderOffloading:
    """Tests for text encoder offloading."""

    def test_offload_calls_to_cpu(self):
        """Offloading should move encoders to CPU."""
        enc1 = MagicMock()
        enc2 = MagicMock()

        EmbeddingManager.offload_text_encoders(enc1, enc2)

        enc1.to.assert_called_once_with("cpu")
        enc2.to.assert_called_once_with("cpu")

    def test_offload_handles_none(self):
        """Offloading should gracefully skip None encoders."""
        enc1 = MagicMock()
        EmbeddingManager.offload_text_encoders(enc1, None)
        enc1.to.assert_called_once_with("cpu")


# ── Caption Prefix Tests ─────────────────────────────────────────────────


class TestCaptionPrefix:
    """Tests for caption prefix prepending."""

    def test_prefix_prepended(self):
        """Prefix should be prepended with comma-space delimiter."""
        em = EmbeddingManager(model_family="sdxl")
        result = em.apply_caption_prefix(["a cat", "a dog"], "masterpiece, best quality")
        assert result[0] == "masterpiece, best quality, a cat"
        assert result[1] == "masterpiece, best quality, a dog"

    def test_empty_prefix_noop(self):
        """Empty prefix should return captions unchanged."""
        em = EmbeddingManager(model_family="sdxl")
        captions = ["hello", "world"]
        result = em.apply_caption_prefix(captions, "")
        assert result == captions

    def test_empty_caption_gets_prefix_only(self):
        """Empty caption should become just the prefix."""
        em = EmbeddingManager(model_family="sdxl")
        result = em.apply_caption_prefix([""], "quality")
        assert result[0] == "quality"

    def test_none_prefix_noop(self):
        """None prefix should return captions unchanged."""
        em = EmbeddingManager(model_family="sdxl")
        result = em.apply_caption_prefix(["test"], None)
        assert result == ["test"]


# ── Encode-and-Cache Integration Tests ───────────────────────────────────


class TestEncodeAndCache:
    """Integration tests for encode_and_cache dispatch."""

    def test_sdxl_encode_and_cache(self, tmp_path):
        """encode_and_cache with sdxl family should encode and save."""
        em = EmbeddingManager(model_family="sdxl", device="cpu")
        tok1 = make_mock_tokenizer()
        tok2 = make_mock_tokenizer()
        enc1 = make_mock_encoder(hidden_dim=768, num_layers=4)
        enc2 = make_mock_encoder(hidden_dim=1280, num_layers=4, has_pooled=True)

        c_dir = str(tmp_path / "cache")
        result = em.encode_and_cache(
            captions=["a test"],
            ids=["test_001"],
            cache_dirs=[c_dir],
            tokenizer_1=tok1, tokenizer_2=tok2,
            text_encoder_1=enc1, text_encoder_2=enc2,
            device="cpu",
        )

        assert "prompt_embeds" in result
        assert os.path.exists(os.path.join(c_dir, "test_001.safetensors"))

    def test_flux2_encode_and_cache(self, tmp_path):
        """encode_and_cache with flux2 family should encode and save."""
        em = EmbeddingManager(model_family="flux2", device="cpu")
        tok = make_mock_tokenizer(max_length=512)
        enc = make_mock_encoder(hidden_dim=4096, num_layers=6)

        c_dir = str(tmp_path / "cache")
        result = em.encode_and_cache(
            captions=["a test"],
            ids=["test_002"],
            cache_dirs=[c_dir],
            tokenizer=tok, text_encoder=enc,
            device="cpu",
        )

        assert "ctx" in result
        assert os.path.exists(os.path.join(c_dir, "test_002.safetensors"))

    def test_unknown_family_raises(self):
        """encode_and_cache with unknown family should raise ValueError."""
        em = EmbeddingManager(model_family="unknown", device="cpu")
        with pytest.raises(ValueError, match="Unknown model family"):
            em.encode_and_cache(["x"], ["id1"])


# ── Cache Dir Resolution Tests ───────────────────────────────────────────


class TestCacheDirResolution:
    """Tests for standardized cache directory resolution."""

    def test_resolve_cache_dir_structure(self):
        """Cache dir should follow dataset/.cache/model/version/embeddings/variant pattern."""
        result = EmbeddingManager.resolve_cache_dir("/data/cats", "sdxl-base", "v1")
        expected = os.path.join("/data/cats", ".cache", "sdxl-base", "v1", "embeddings", "original")
        assert result == expected

    def test_resolve_cache_dir_variant(self):
        """Explicit variant should appear in the path."""
        result = EmbeddingManager.resolve_cache_dir("/data/cats", "sdxl-base", "v1", "masked")
        expected = os.path.join("/data/cats", ".cache", "sdxl-base", "v1", "embeddings", "masked")
        assert result == expected
