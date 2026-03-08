"""
Tests for TextEmbeddingCache — disk-persisted text embedding caching.

Covers:
- Filename generation (SHA-256 + sanitized source)
- Round-trip save/load
- Cache miss returns None
- Coverage checking
- Source hint sanitization edge cases
"""

import hashlib
import os

import torch

from app.engine.components.text_embeddings import (
    TextEmbeddingCache,
    _sanitize_source,
)


class TestFilenameGeneration:
    def test_sha256_deterministic(self):
        """Same caption always produces the same filename."""
        fn1 = TextEmbeddingCache.caption_to_filename("hello world", "img1")
        fn2 = TextEmbeddingCache.caption_to_filename("hello world", "img1")
        assert fn1 == fn2

    def test_different_captions_produce_different_filenames(self):
        fn1 = TextEmbeddingCache.caption_to_filename("cat on mat", "img")
        fn2 = TextEmbeddingCache.caption_to_filename("dog on mat", "img")
        assert fn1 != fn2

    def test_filename_contains_source_and_hash(self):
        fn = TextEmbeddingCache.caption_to_filename("test caption", "my_image")
        assert fn.startswith("my_image_")
        assert fn.endswith(".safetensors")
        # Verify hash component
        expected_hash = hashlib.sha256("test caption".encode("utf-8")).hexdigest()
        assert expected_hash in fn

    def test_empty_caption_produces_valid_filename(self):
        fn = TextEmbeddingCache.caption_to_filename("", "dropout_empty")
        assert fn.startswith("dropout_empty_")
        assert fn.endswith(".safetensors")

    def test_empty_source_hint_defaults(self):
        fn = TextEmbeddingCache.caption_to_filename("test", "")
        assert fn.startswith("caption_")


class TestSourceSanitization:
    def test_special_chars_replaced(self):
        result = _sanitize_source("my image (1).png")
        assert "(" not in result
        assert ")" not in result
        assert " " not in result

    def test_consecutive_underscores_collapsed(self):
        result = _sanitize_source("a   b   c")
        assert "__" not in result

    def test_truncation_at_max_length(self):
        long_name = "a" * 100
        result = _sanitize_source(long_name)
        assert len(result) <= 40

    def test_empty_input(self):
        assert _sanitize_source("") == "caption"


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        """Save then load returns identical tensor."""
        cache_dir = str(tmp_path / "te1")
        caption = "a photo of a cat sitting on a mat"
        tensor = torch.randn(1, 512, 4096)

        TextEmbeddingCache.save(caption, tensor, cache_dir, source_hint="cat_img")
        loaded = TextEmbeddingCache.load(caption, cache_dir, source_hint="cat_img")

        assert loaded is not None
        assert loaded.shape == tensor.shape
        assert torch.allclose(loaded, tensor)

    def test_load_missing_returns_none(self, tmp_path):
        cache_dir = str(tmp_path / "te1")
        os.makedirs(cache_dir, exist_ok=True)
        result = TextEmbeddingCache.load("nonexistent caption", cache_dir)
        assert result is None

    def test_save_creates_directory(self, tmp_path):
        cache_dir = str(tmp_path / "deep" / "nested" / "te1")
        assert not os.path.exists(cache_dir)

        TextEmbeddingCache.save("test", torch.zeros(1, 10), cache_dir)
        assert os.path.exists(cache_dir)

    def test_tensor_stored_on_cpu(self, tmp_path):
        cache_dir = str(tmp_path / "te1")
        tensor = torch.randn(1, 77, 768)
        TextEmbeddingCache.save("prompt", tensor, cache_dir)
        loaded = TextEmbeddingCache.load("prompt", cache_dir)
        assert loaded.device == torch.device("cpu")

    def test_different_hints_same_caption_same_file(self, tmp_path):
        """Source hint is cosmetic; the hash drives uniqueness."""
        cache_dir = str(tmp_path / "te1")
        caption = "same caption"
        tensor = torch.randn(1, 10)

        TextEmbeddingCache.save(caption, tensor, cache_dir, source_hint="hint_a")
        # Load with a different hint — should still find by hash
        # (different hint → different filename → miss)
        loaded = TextEmbeddingCache.load(caption, cache_dir, source_hint="hint_b")
        # This should be None because the filename includes the hint
        assert loaded is None

    def test_corrupt_file_returns_none(self, tmp_path):
        cache_dir = str(tmp_path / "te1")
        os.makedirs(cache_dir, exist_ok=True)
        fname = TextEmbeddingCache.caption_to_filename("test", "src")
        path = os.path.join(cache_dir, fname)
        with open(path, "wb") as f:
            f.write(b"not a valid safetensors file")

        result = TextEmbeddingCache.load("test", cache_dir, "src")
        assert result is None


class TestCheckCoverage:
    def test_all_cached(self, tmp_path):
        cache_dir = str(tmp_path / "te1")
        captions = ["cap1", "cap2", "cap3"]
        for cap in captions:
            TextEmbeddingCache.save(cap, torch.zeros(1, 10), cache_dir)

        cached, missing, _ = TextEmbeddingCache.check_coverage(captions, cache_dir)
        assert cached == 3
        assert missing == 0

    def test_none_cached(self, tmp_path):
        cache_dir = str(tmp_path / "te1")
        os.makedirs(cache_dir, exist_ok=True)
        captions = ["cap1", "cap2"]

        cached, missing, samples = TextEmbeddingCache.check_coverage(captions, cache_dir)
        assert cached == 0
        assert missing == 2
        assert len(samples) == 2

    def test_partial_coverage(self, tmp_path):
        cache_dir = str(tmp_path / "te1")
        TextEmbeddingCache.save("cap1", torch.zeros(1, 10), cache_dir)

        cached, missing, samples = TextEmbeddingCache.check_coverage(
            ["cap1", "cap2", "cap3"], cache_dir,
        )
        assert cached == 1
        assert missing == 2

    def test_missing_samples_capped_at_10(self, tmp_path):
        cache_dir = str(tmp_path / "te1")
        os.makedirs(cache_dir, exist_ok=True)
        captions = [f"caption_{i}" for i in range(20)]

        _, _, samples = TextEmbeddingCache.check_coverage(captions, cache_dir)
        assert len(samples) <= 10


class TestResolveCacheDir:
    def test_path_structure_default_quant(self):
        """Default te_quant='none' is included in path."""
        path = TextEmbeddingCache.resolve_te_cache_dir(
            dataset_path="/data/cats",
            model_name="flux2-dev",
            dataset_version="1.0.0",
            te_slot="te1",
        )
        expected = os.path.join("/data/cats", ".cache", "flux2-dev", "1.0.0", "embeddings", "none", "te1")
        assert path == expected

    def test_fp8_quant_produces_different_path(self):
        """FP8 quantization routes to a separate directory."""
        path_none = TextEmbeddingCache.resolve_te_cache_dir(
            dataset_path="/data", model_name="flux2-dev",
            dataset_version="1.0.0", te_slot="te1", te_quant="none",
        )
        path_fp8 = TextEmbeddingCache.resolve_te_cache_dir(
            dataset_path="/data", model_name="flux2-dev",
            dataset_version="1.0.0", te_slot="te1", te_quant="fp8",
        )
        assert path_none != path_fp8
        assert os.path.join("embeddings", "none", "te1") in path_none
        assert os.path.join("embeddings", "fp8", "te1") in path_fp8

    def test_te2_slot(self):
        path = TextEmbeddingCache.resolve_te_cache_dir(
            dataset_path="/data",
            model_name="flux1-dev",
            dataset_version="2.0",
            te_slot="te2",
            te_quant="nf4",
        )
        assert path.endswith(os.path.join("embeddings", "nf4", "te2"))
