"""Z-Image Trainer -- family-specific hooks for the generic training pipeline.

All shared logic (optimizer, EMA, gradient accumulation, noise offset,
checkpointing, signals, logging) lives in ``GenericTrainingPipeline``.
This module implements Z-Image-specific behaviour:
- S3-DiT (single-stream) transformer architecture
- Single text encoder encoding
- Flow matching with configurable timestep sampling
- CFG support with negative prompts
"""

import os

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from .driver import ZImageDriver
from .loader import ZImageLoader
from .saver import ZImageSaver

logger = structlog.get_logger(__name__)


class ZImageTrainer(GenericTrainingPipeline):
    """Z-Image Base LoRA trainer.

    ~6B S3-DiT (single-stream DiT) by Alibaba Tongyi-MAI.
    Supports CFG with negative prompts (non-distilled).
    Uses 3D Unified RoPE.
    """

    # -- Setup --

    def _setup_family(self) -> None:
        """Initialize Z-Image-specific loader, saver, driver, and caches."""
        self.driver = ZImageDriver(self.definition, self.device)
        self.loader = ZImageLoader(self.device)
        self.saver = ZImageSaver()

    def _create_sampler(self):
        """Create a ZImageSampler if sampling is configured."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import ZImageSampler
            return ZImageSampler(self)
        return None

    # -- Component Assignment --

    def _assign_components(self) -> None:
        """Wire components via driver + set Z-Image-specific aliases."""
        super()._assign_components()
        self.model = self.components["unet"]

        # Architecture params
        arch = getattr(self.definition, "architecture_params", {}) or {}
        self.max_length = int(arch.get("te.max_length", 512))

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep self.model in sync after PEFT/quantization wrapping."""
        self.model = new_model
        self.components["unet"] = new_model
        # Also update driver's reference
        self.driver.model = new_model

    # -- Disk-backed TE Pre-caching --

    def _pre_cache_text_embeddings(self) -> None:
        """Warm text embedding cache from disk + encode missing.

        1. Build the full set of captions (training, dropout, sampling).
        2. Try to load from disk cache (te1/).
        3. Encode only truly uncached captions on GPU.
        4. Save newly encoded embeddings to disk for future runs.
        """
        if not self.config.get("cache_text_embeddings", True):
            return
        if self.text_encoder is None:
            return

        from app.engine.components.text_embeddings import TextEmbeddingCache

        te_cache_dirs = self._resolve_te_cache_dirs()
        # Include TE quantization scheme so FP8 / bf16 embeddings don't collide
        te_quant = self.config.get("te_quantization", "none")
        te1_dir = os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te1") if te_cache_dirs else ""

        # -- Build full caption set (shared base class logic) --
        caption_hints = self._build_caption_hints()

        # -- Phase 1: Load from disk --
        disk_loaded = 0
        need_encode: list[tuple[str, str]] = []

        for caption, hint in caption_hints.items():
            if caption in self.text_cache:
                continue
            if te1_dir:
                tensor = TextEmbeddingCache.load(caption, te1_dir, hint)
                if tensor is not None:
                    self.text_cache[caption] = tensor
                    disk_loaded += 1
                    continue
            need_encode.append((caption, hint))

        total = len(caption_hints)
        self.logger.info(
            "te_disk_cache_status",
            total=total,
            from_memory=total - disk_loaded - len(need_encode),
            from_disk=disk_loaded,
            need_encode=len(need_encode),
        )

        if not need_encode:
            if getattr(self, "_log_writer", None):
                self._log_writer.status("TE Cache Loaded from Disk")
            self.logger.info(
                "text_embedding_cache_complete",
                cached=len(self.text_cache), source="disk",
            )
            return

        # -- Phase 2: Encode missing (batched) --
        if getattr(self, "_log_writer", None):
            self._log_writer.status("Caching Text Embeddings (0%)")
        encode_total = len(need_encode)
        batch_size = 4
        dtype = self._resolve_loading_dtype()

        with torch.no_grad():
            for i in range(0, encode_total, batch_size):
                batch_items = need_encode[i : i + batch_size]
                batch_caps = [cap for cap, _ in batch_items]

                # Single batched forward pass through the TE
                emb_list = self._encode_text_direct(batch_caps, dtype)

                for j, (cap, hint) in enumerate(batch_items):
                    emb = emb_list[j].cpu()
                    self.text_cache[cap] = emb
                    if te1_dir:
                        TextEmbeddingCache.save(cap, emb, te1_dir, hint)

                pct = int(min(i + batch_size, encode_total) / encode_total * 100)
                if pct % 10 == 0 or (i + batch_size) >= encode_total:
                    if getattr(self, "_log_writer", None):
                        self._log_writer.status(f"Caching Text Embeddings ({pct}%)")

        self.logger.info(
            "text_embedding_cache_complete",
            cached=len(self.text_cache), newly_encoded=encode_total,
        )

    # -- TE Offloading --

    # _offload_text_encoders — inherited from base class.
    # Base uses _get_text_encoders() to discover and offload/unload TEs
    # and properly cleans self.components to prevent stale references.

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None,
    ) -> list[torch.Tensor]:
        """Encode captions through the Z-Image text encoder.

        Matches ``ZImagePipeline._encode_prompt``: applies Qwen3 chat template
        with ``enable_thinking=True``, uses ``hidden_states[-2]``, and returns
        **variable-length** per-sample tensors (non-padding only).

        Args:
            captions: Processed captions.
            dtype: Target dtype.

        Returns:
            List of text embedding tensors ``[Li, D]`` (one per caption).
        """
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(captions, dtype)

        return self._encode_text_direct(captions, dtype)

    def _encode_text_direct(
        self, captions: list[str], dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Encode captions directly via the driver (no cache).

        Delegates to ``driver.encode_text`` (single source of truth for the
        Qwen3 variable-length encoding) and unwraps the ``TextEncoderOutput``
        to the ``list[Tensor]`` ``[Li, D]`` contract the training loop expects.
        """
        return self.driver.encode_text(captions, dtype).embeddings

    def _get_cached_text_embeddings(
        self, captions: list[str], dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Encode on first encounter, reuse thereafter."""
        results: list[torch.Tensor] = []
        uncached: list[tuple[int, str]] = []

        for i, cap in enumerate(captions):
            if cap not in self.text_cache:
                uncached.append((i, cap))

        if uncached and self.text_encoder is not None:
            # Guard: if TE was offloaded to CPU, temporarily move back
            te_device = next(self.text_encoder.parameters()).device
            te_was_offloaded = te_device != self.device
            if te_was_offloaded:
                self.logger.warning(
                    "te_cache_miss_after_offload",
                    count=len(uncached),
                    hint="pre-caching should have covered all captions",
                )
                self.text_encoder.to(self.device)

            for _, cap in uncached:
                single_list = self._encode_text_direct([cap], dtype)
                # Cache the single variable-length tensor on CPU
                self.text_cache[cap] = single_list[0].cpu()

            if te_was_offloaded:
                self.text_encoder.to("cpu")
                torch.cuda.empty_cache()

            self.logger.debug(
                "text_embeddings_cached", new=len(uncached), total=len(self.text_cache),
            )
        elif uncached:
            raise RuntimeError(
                "Text encoder was unloaded but encountered uncached caption(s): "
                + ", ".join(cap[:50] for _, cap in uncached)
            )

        for cap in captions:
            results.append(self.text_cache[cap].to(self.device, dtype=dtype))

        return results

    # -- Target --

    def compute_target(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Z-Image velocity target: ``latents - noise`` (data → noise inverted).

        Z-Image uses inverted timestep convention (t=1 is clean, t=0 is noise).
        The model predicts ``data - noise`` (velocity toward clean data).
        The reference pipeline confirms this by negating model output
        before feeding to the scheduler (``noise_pred = -noise_pred``).

        Default generic target ``noise - latents`` has the wrong sign for
        this model, which causes training to destroy model weights.
        """
        return latents - noise

    # -- Forward Pass --
    # Delegated to ``ZImageDriver.forward_pass`` via the base
    # ``PipelineBaseMixin.forward_pass``.  The driver forward is self-contained
    # (inverted timestep + per-sample list API); no trainer-level copy needed.
