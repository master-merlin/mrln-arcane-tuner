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
from typing import Any

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
            print("[STATUS:TE Cache Loaded from Disk]", flush=True)
            self.logger.info(
                "text_embedding_cache_complete",
                cached=len(self.text_cache), source="disk",
            )
            return

        # -- Phase 2: Encode missing (batched) --
        print("[STATUS:Caching Text Embeddings (0%)]", flush=True)
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
                    print(f"[STATUS:Caching Text Embeddings ({pct}%)]", flush=True)

        self.logger.info(
            "text_embedding_cache_complete",
            cached=len(self.text_cache), newly_encoded=encode_total,
        )

    # -- TE Offloading --

    # _offload_text_encoders — inherited from base class.
    # Base uses _get_text_encoders() to discover and offload/unload TEs
    # and properly cleans self.components to prevent stale references.

    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
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
        """Text encoding matching ``ZImagePipeline._encode_prompt``.

        1. Wrap each caption via Qwen3 chat template (enable_thinking=True)
        2. Tokenize with max_length padding
        3. Forward through TE, take hidden_states[-2]
        4. Extract only non-padding tokens per sample

        Returns:
            List of tensors ``[Li, D]`` (variable length, no padding).
        """
        # 1. Apply Qwen3 chat template
        templated: list[str] = []
        for cap in captions:
            messages = [{"role": "user", "content": cap}]
            txt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            templated.append(txt)

        # 2. Tokenize
        text_inputs = self.tokenizer(
            templated, padding="max_length", max_length=self.max_length,
            truncation=True, return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attn_mask = text_inputs.attention_mask.to(self.device).bool()

        # 3. Forward — use hidden_states[-2] (second-to-last)
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
        hidden = outputs.hidden_states[-2]

        # 4. Extract non-padding tokens per sample
        embeddings_list: list[torch.Tensor] = []
        for i in range(len(hidden)):
            embeddings_list.append(hidden[i][attn_mask[i]].to(dtype=dtype))

        return embeddings_list

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

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: list[torch.Tensor],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """ZImageTransformer2DModel forward pass.

        The Z-Image S3-DiT uses a non-standard forward() signature:
            forward(x: List[Tensor], t, cap_feats: List[Tensor], ...)
        - ``x``: list of per-sample latent tensors [C, 1, H, W]
        - ``t``: timestep tensor [B]
        - ``cap_feats``: list of per-sample text embeddings [Li, D]
            (variable length, non-padding only)

        Args:
            noisy_input: Noisy latents [B, C, H, W].
            timesteps: Scaled timesteps [0, 1000].
            text_embeddings: List of per-sample embeddings [Li, D]
                from ``encode_text()``.
            batch: Full batch dict.

        Returns:
            Model prediction [B, C, H, W].
        """
        # Z-Image convention: t=1 → clean, t=0 → noise (inverted).
        # Our training loop uses [0, 1000] with 0=clean, 1000=noise,
        # so invert: (1000-t)/1000 maps 0→1.0 (clean), 1000→0.0 (noise).
        model_timesteps = (1000.0 - timesteps) / 1000.0

        # Z-Image forward() expects lists of per-sample tensors
        # patchify_and_embed expects [C, F, H, W] (4D with frame dim)
        # Per-sample slicing gives [C, H, W], so add frame dim: [C, 1, H, W]
        x_list = [noisy_input[i].unsqueeze(1) for i in range(noisy_input.shape[0])]

        # text_embeddings is already a list of per-sample [Li, D] tensors
        cap_list = text_embeddings

        output = self.model(
            x=x_list,
            t=model_timesteps,
            cap_feats=cap_list,
            return_dict=False,
        )

        # output is (list_of_tensors,) where each tensor is [C, F, H, W]
        # Need to squeeze frame dim (F=1) and re-batch to [B, C, H, W]
        sample_list = output[0] if isinstance(output, tuple) else output
        # Each sample: [C, 1, H, W] → [C, H, W], then stack to [B, C, H, W]
        return torch.stack([s.squeeze(1) for s in sample_list], dim=0)
