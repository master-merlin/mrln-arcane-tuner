"""
EmbeddingManager: Text encoder embedding caching and offloading.

Mirrors the LatentManager pattern — encodes text captions into embeddings,
caches them to disk as safetensors, and enables text encoder offloading
when ``train_text_encoder=False``.

Supports both SDXL (dual CLIP, prompt_embeds + pooled_embeds) and
Flux2 (single TE, last-N-layer concatenation).
"""

import os
import torch

from safetensors.torch import save_file, load_file
import structlog

logger = structlog.get_logger(__name__)


class EmbeddingManager:
    """
    Manages text encoder embedding caching and offloading.

    Args:
        model_family: Either ``"sdxl"`` or ``"flux2"``.
        cache_dir: Default cache directory (can be overridden per-item).
        device: Target device for encoded/loaded embeddings.
    """

    def __init__(self, model_family: str, cache_dir: str | None = None, device: str = "cuda"):
        self.model_family = model_family
        self.cache_dir = cache_dir
        self.device = device
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("embedding_manager_init", model_family=model_family, cache_dir=cache_dir)

    @staticmethod
    def resolve_cache_dir(dataset_path: str, model_name: str, dataset_version: str, variant: str = "original") -> str:
        """
        Standardized embedding cache directory.

        Path: ``{dataset_path}/.cache/{model_name}/{dataset_version}/embeddings/{variant}``
        """
        return os.path.join(dataset_path, ".cache", model_name, dataset_version, "embeddings", variant)

    def encode_sdxl(
        self,
        captions: list[str],
        tokenizer_1,
        tokenizer_2,
        text_encoder_1,
        text_encoder_2,
        device: str | None = None,
        max_length: int = 77,
    ) -> dict[str, torch.Tensor]:
        """
        Encode captions using dual CLIP text encoders (SDXL).

        Returns:
            Dict with ``prompt_embeds`` [B, L, 2048] and ``pooled_embeds`` [B, 1280].
        """
        dev = device or self.device

        with torch.no_grad():
            t1 = tokenizer_1(
                captions, padding="max_length", max_length=max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(dev)
            t2 = tokenizer_2(
                captions, padding="max_length", max_length=max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(dev)

            e1 = text_encoder_1(t1, output_hidden_states=True)
            e2 = text_encoder_2(t2, output_hidden_states=True)

            # Penultimate hidden states
            h1 = e1.hidden_states[-2]
            h2 = e2.hidden_states[-2]
            prompt_embeds = torch.cat([h1, h2], dim=-1)
            pooled_embeds = e2.text_embeds

        logger.debug(
            "sdxl_embeddings_encoded",
            batch_size=prompt_embeds.shape[0],
            prompt_shape=list(prompt_embeds.shape),
            pooled_shape=list(pooled_embeds.shape),
        )
        return {"prompt_embeds": prompt_embeds, "pooled_embeds": pooled_embeds}

    def encode_flux2(
        self,
        captions: list[str],
        tokenizer,
        text_encoder,
        device: str | None = None,
        max_length: int = 512,
        concat_layers: int = 3,
    ) -> dict[str, torch.Tensor]:
        """
        Encode captions using single text encoder (Flux2 Klein).

        Concatenates last ``concat_layers`` hidden states.

        Returns:
            Dict with ``ctx`` [B, L, D*concat_layers].
        """
        dev = device or self.device

        with torch.no_grad():
            text_inputs = tokenizer(
                captions, padding="max_length", max_length=max_length,
                truncation=True, return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(dev)

            outputs = text_encoder(input_ids, output_hidden_states=True)
            ctx = torch.cat(outputs.hidden_states[-concat_layers:], dim=-1)

        logger.debug(
            "flux2_embeddings_encoded",
            batch_size=ctx.shape[0],
            ctx_shape=list(ctx.shape),
            concat_layers=concat_layers,
        )
        return {"ctx": ctx}

    def save_embeddings(
        self,
        embeddings: dict[str, torch.Tensor],
        ids: list[str],
        cache_dirs: list[str] | None = None,
    ):
        """
        Save encoded embeddings to disk as safetensors.

        Each item ``i`` is saved to ``{cache_dirs[i]}/{ids[i]}.safetensors`` or
        ``{self.cache_dir}/{ids[i]}.safetensors``.
        """
        batch_size = next(iter(embeddings.values())).shape[0]
        for i in range(batch_size):
            c_dir = cache_dirs[i] if cache_dirs else self.cache_dir
            if not c_dir:
                continue

            os.makedirs(c_dir, exist_ok=True)
            path = os.path.join(c_dir, f"{ids[i]}.safetensors")
            item = {k: v[i].detach().cpu() for k, v in embeddings.items()}
            save_file(item, path)

        logger.debug("embeddings_saved", count=batch_size)

    def load_cached_embeddings(
        self,
        ids: list[str],
        cache_dirs: list[str] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """
        Try to load cached embeddings from disk.

        Returns:
            Dict with stacked tensors, or ``None`` if any item is missing.
        """
        if not cache_dirs and not self.cache_dir:
            return None

        loaded: list[dict[str, torch.Tensor]] = []
        for i, img_id in enumerate(ids):
            c_dir = cache_dirs[i] if cache_dirs else self.cache_dir
            if not c_dir:
                self._cache_misses += len(ids)
                return None

            path = os.path.join(c_dir, f"{img_id}.safetensors")
            if not os.path.exists(path):
                self._cache_misses += len(ids) - len(loaded)
                return None

            try:
                data = load_file(path)
                loaded.append(data)
            except (OSError, KeyError) as e:
                logger.warning("embedding_cache_load_failed", path=path, error=str(e))
                self._cache_misses += len(ids) - len(loaded)
                return None

        self._cache_hits += len(ids)

        # Stack into batched tensors
        result = {}
        for key in loaded[0]:
            result[key] = torch.stack([item[key] for item in loaded]).to(self.device)

        return result

    def encode_and_cache(
        self,
        captions: list[str],
        ids: list[str],
        cache_dirs: list[str] | None = None,
        **encode_kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Encode captions and cache the results. Dispatches to the correct
        encoder based on ``model_family``.

        Args:
            captions: List of text captions.
            ids: List of unique IDs for cache filenames.
            cache_dirs: Per-item cache directories.
            **encode_kwargs: Passed to the family-specific encode method.

        Returns:
            Dict of embedding tensors.
        """
        if self.model_family == "sdxl":
            embeddings = self.encode_sdxl(captions, **encode_kwargs)
        elif self.model_family == "flux2":
            embeddings = self.encode_flux2(captions, **encode_kwargs)
        else:
            raise ValueError(f"Unknown model family: {self.model_family}")

        self.save_embeddings(embeddings, ids, cache_dirs)
        return embeddings

    @staticmethod
    def offload_text_encoders(*encoders, logger_ref=None):
        """
        Move text encoders to CPU and clear CUDA cache.

        Call this after all embeddings are cached and ``train_text_encoder=False``.
        """
        for enc in encoders:
            if enc is not None and hasattr(enc, "to"):
                enc.to("cpu")
        torch.cuda.empty_cache()

        freed_msg = "text_encoders_offloaded"
        if logger_ref:
            logger_ref.info(freed_msg, count=len(encoders))
        else:
            logger.info(freed_msg, count=len(encoders))

    def log_cache_stats(self):
        """Log cache hit/miss statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        logger.info(
            "embedding_cache_stats",
            hits=self._cache_hits,
            misses=self._cache_misses,
            hit_rate=round(hit_rate, 1),
        )

    def apply_caption_prefix(self, captions: list[str], prefix: str) -> list[str]:
        """
        Prepend a caption prefix to each caption.

        Args:
            captions: Raw captions from dataset.
            prefix: Prefix string (e.g. ``"masterpiece, best quality"``).

        Returns:
            List of prefixed captions.
        """
        if not prefix:
            return captions
        return [f"{prefix}, {c}" if c else prefix for c in captions]
