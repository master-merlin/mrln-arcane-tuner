"""Qwen-Image Trainer -- family-specific hooks for the generic training pipeline.

All shared logic (optimizer, EMA, gradient accumulation, noise offset,
checkpointing, signals, logging) lives in ``GenericTrainingPipeline``.
This module implements Qwen-Image-specific behaviour:
- Single Qwen2.5-VL text encoder (used in text-only mode)
- Flow matching with configurable timestep sampling
- QwenImageTransformer2DModel forward pass (hidden_states + encoder_hidden_states)
- Patchified latent preparation (patch_size=2)
"""

import os
from typing import Any

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from .driver import QwenImageDriver
from .loader import QwenImageLoader
from .saver import QwenImageSaver

logger = structlog.get_logger(__name__)

# ── Prompt template (from QwenImagePipeline.__init__) ────────────────────
PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_DROP_IDX = 34  # system preamble tokens to drop
TOKENIZER_MAX_LENGTH = 1024


class QwenImageTrainer(GenericTrainingPipeline):
    """Qwen-Image (2512) LoRA trainer.

    20B MMDiT with single Qwen2.5-VL text encoder, 60 transformer layers,
    and flow matching noise schedule.
    """

    # -- Setup --

    def _setup_family(self) -> None:
        """Initialize Qwen-Image-specific loader, saver, driver, and caches."""
        self.driver = QwenImageDriver(self.definition, self.device)
        self.loader = QwenImageLoader(self.device)
        self.saver = QwenImageSaver()

    def _create_sampler(self):
        """Create a QwenImageSampler if sampling is configured."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import QwenImageSampler
            return QwenImageSampler(self)
        return None

    # -- Component Assignment --

    def _assign_components(self) -> None:
        """Wire components via driver + set Qwen-Image-specific aliases."""
        super()._assign_components()
        self.model = self.components["unet"]
        # Architecture params
        self.max_length = TOKENIZER_MAX_LENGTH

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep self.model in sync after PEFT/quantization wrapping."""
        self.model = new_model
        self.components["unet"] = new_model
        # Also update driver's reference
        self.driver.model = new_model

    @property
    def transformer(self) -> torch.nn.Module:
        """Alias for sampler compatibility (sampler accesses .transformer)."""
        return self.model

    # -- Disk-backed TE Pre-caching --

    def _pre_cache_text_embeddings(self) -> None:
        """Warm text embedding cache from disk + encode missing.

        Qwen-Image caches (embedding, mask) tuples:
        - te1/ stores embedding tensors
        - te2/ stores attention mask tensors
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
        te2_dir = os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te2") if te_cache_dirs else ""

        # -- Build full caption set (shared base class logic) --
        caption_hints = self._build_caption_hints()

        # -- Phase 1: Load from disk --
        disk_loaded = 0
        need_encode: list[tuple[str, str]] = []

        for caption, hint in caption_hints.items():
            if caption in self.text_cache:
                continue
            if te1_dir and te2_dir:
                emb_tensor = TextEmbeddingCache.load(caption, te1_dir, hint)
                mask_tensor = TextEmbeddingCache.load(caption, te2_dir, hint)
                if emb_tensor is not None and mask_tensor is not None:
                    self.text_cache[caption] = (emb_tensor, mask_tensor)
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
                emb_batch, mask_batch = self._encode_text_direct(batch_caps, dtype)

                for j, (cap, hint) in enumerate(batch_items):
                    emb_cpu = emb_batch[j].cpu()
                    mask_cpu = mask_batch[j].cpu()
                    self.text_cache[cap] = (emb_cpu, mask_cpu)
                    if te1_dir:
                        TextEmbeddingCache.save(cap, emb_cpu, te1_dir, hint)
                    if te2_dir:
                        TextEmbeddingCache.save(cap, mask_cpu, te2_dir, hint)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions through Qwen2.5-VL in text-only mode.

        Returns a ``(embeddings, attention_mask)`` tuple.  The base
        pipeline passes this opaquely to ``forward_pass()`` which unpacks it.

        Args:
            captions: Processed captions.
            dtype: Target dtype.

        Returns:
            (text_embeddings [B, L, D], attention_mask [B, L]).
        """
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(captions, dtype)

        return self._encode_text_direct(captions, dtype)

    @staticmethod
    def _extract_masked_hidden(
        hidden_states: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Extract only non-padding tokens per batch element."""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    def _encode_text_direct(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Text encoding matching ``QwenImagePipeline._get_qwen_prompt_embeds``.

        1. Wrap each caption in the system/user/assistant template
        2. Tokenize with ``max_length = 1024 + 34``
        3. Extract non-padding tokens via ``_extract_masked_hidden``
        4. Drop first 34 tokens (system preamble)
        5. Re-pad to max actual length in batch

        Returns:
            (hidden_states [B, L, D], attention_mask [B, L]).
        """
        # 1. Wrap in template
        txt = [PROMPT_TEMPLATE.format(cap) for cap in captions]

        # 2. Tokenize
        max_len = self.max_length + PROMPT_TEMPLATE_DROP_IDX
        text_inputs = self.tokenizer(
            txt, max_length=max_len, padding=True,
            truncation=True, return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attn_mask = text_inputs.attention_mask.to(self.device)

        # 3. Forward through text encoder
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states[-1]

        # 4. Extract non-padding tokens
        split_hs = self._extract_masked_hidden(hidden_states, attn_mask)

        # 5. Drop first 34 tokens (system preamble), re-pad
        split_hs = [e[PROMPT_TEMPLATE_DROP_IDX:] for e in split_hs]
        attn_mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=self.device)
            for e in split_hs
        ]
        max_seq_len = max(e.size(0) for e in split_hs)
        prompt_embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
            for u in split_hs
        ])
        encoder_attn_mask = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
            for u in attn_mask_list
        ])

        return prompt_embeds.to(dtype=dtype), encoder_attn_mask

    def _get_cached_text_embeddings(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode on first encounter, reuse thereafter.

        Caches both embeddings and attention masks.

        Returns:
            (text_embeddings [B, L, D], attention_mask [B, L]).
        """
        emb_results: list[torch.Tensor] = []
        mask_results: list[torch.Tensor] = []
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
                single_ctx, single_mask = self._encode_text_direct([cap], dtype)
                self.text_cache[cap] = (
                    single_ctx.squeeze(0).cpu(),
                    single_mask.squeeze(0).cpu(),
                )

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
            cached_emb, cached_mask = self.text_cache[cap]
            emb_results.append(cached_emb.to(self.device, dtype=dtype))
            mask_results.append(cached_mask.to(self.device))

        return torch.stack(emb_results, dim=0), torch.stack(mask_results, dim=0)


    # -- Forward Pass --

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """QwenImageTransformer2DModel forward pass.

        The transformer expects patchified sequence input:
            hidden_states: [B, (H/p)*(W/p), C*p*p]  (e.g. p=2, C=16 → 64)
        Plus spatial metadata for RoPE positional embeddings.

        Args:
            noisy_input: Noisy latents [B, C, H, W].
            timesteps: Scaled timesteps [0, 1000].
            text_embeddings: ``(embeddings [B, L, D], attention_mask [B, L])``
                tuple from ``encode_text()``.  Falls back gracefully if
                only a plain tensor is passed (legacy path).
            batch: Full batch dict.

        Returns:
            Model prediction [B, C, H, W].
        """
        # Unpack (embeddings, mask) tuple from encode_text()
        if isinstance(text_embeddings, tuple):
            enc_hs, enc_mask = text_embeddings
        else:
            enc_hs = text_embeddings
            enc_mask = None

        B, C, H, W = noisy_input.shape
        patch_size = getattr(self.model.config, "patch_size", 2)

        # Patchify: [B, C, H, W] → [B, (H/p)*(W/p), C*p*p]
        pH = H // patch_size
        pW = W // patch_size
        # Reshape: [B, C, pH, p, pW, p] → [B, pH*pW, C*p*p]
        x = noisy_input.reshape(B, C, pH, patch_size, pW, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)       # [B, pH, pW, C, p, p]
        x = x.reshape(B, pH * pW, C * patch_size * patch_size)

        # Qwen-Image model expects timesteps in [0, 1]
        model_timesteps = timesteps / 1000.0

        # img_shapes: [(1, pH, pW)] per sample — single-frame image
        img_shapes = [(1, pH, pW)] * B

        # txt_seq_lens: actual valid token count from attention mask
        # (matches reference QwenImagePipeline)
        if enc_mask is not None:
            txt_seq_lens = enc_mask.sum(dim=1).tolist()
        else:
            txt_seq_lens = [enc_hs.shape[1]] * B

        output = self.model(
            hidden_states=x,
            encoder_hidden_states=enc_hs,
            encoder_hidden_states_mask=enc_mask,
            timestep=model_timesteps,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )

        # Handle tuple output — first element is the noise prediction
        pred = output[0] if isinstance(output, tuple) else output

        # Unpatchify: [B, pH*pW, out_channels*p*p] → [B, out_channels, H, W]
        out_channels = getattr(self.model.config, "out_channels", C)
        pred = pred.reshape(B, pH, pW, out_channels, patch_size, patch_size)
        pred = pred.permute(0, 3, 1, 4, 2, 5)  # [B, out_C, pH, p, pW, p]
        pred = pred.reshape(B, out_channels, H, W)

        return pred

