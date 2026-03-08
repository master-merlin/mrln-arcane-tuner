import os
from typing import Any

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline

from .driver import Flux1Driver
from .loader import Flux1Loader
from .saver import Flux1Saver
from .utils import pack_latents

logger = structlog.get_logger(__name__)


class Flux1Trainer(GenericTrainingPipeline):
    """FLUX.1 (Dev / Schnell) LoRA trainer."""

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_family(self) -> None:
        """Initialize Flux1-specific loader, saver, driver, and caches."""
        self.driver = Flux1Driver(self.definition, self.device)
        self.loader = Flux1Loader(self.device)
        self.saver = Flux1Saver()
        self._clip_pooled_cache: dict[str, torch.Tensor] = {}

        # Flux1 patchifies latents (2× down per spatial dim) before the
        # transformer.  Tell the flux_shift sampler so it computes
        # seq_len = (H/2)*(W/2) instead of H*W.
        self.config.setdefault("flux_shift_patchify_factor", 2)

    def _create_sampler(self):
        """Create a Flux1Sampler if sampling is configured."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import Flux1Sampler
            return Flux1Sampler(self)
        return None

    def _assign_components(self) -> None:
        """Wire components via driver + set Flux1-specific aliases."""
        super()._assign_components()
        self.clip_encoder = self.components.get("text_encoder")
        self.clip_tokenizer = self.components.get("tokenizer")
        self.t5_encoder = self.components.get("text_encoder_2")
        self.t5_tokenizer = self.components.get("tokenizer_2")

        # Cache architecture params
        arch = getattr(self.definition, "architecture_params", {}) or {}
        self.use_guidance_embed = arch.get("transformer.guidance_embeds", True)
        self.te_clip_max_length = arch.get("te.clip_max_length", 77)
        self.te_t5_max_length = arch.get("te.t5_max_length", 512)

        self.logger.info(
            "flux1_config",
            guidance_embed=self.use_guidance_embed,
            clip_max_len=self.te_clip_max_length,
            t5_max_len=self.te_t5_max_length,
        )

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep self.transformer in sync after PEFT/quantization wrapping."""
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model


    # ── Staged VRAM Management ───────────────────────────────────────────


    def _pre_cache_text_embeddings(self) -> None:
        """Warm T5+CLIP text embedding cache from disk + encode missing.

        1. Build the full set of captions (training, dropout variants, sampling).
        2. For each caption, try to load from disk cache first (te1/ for T5,
           te2/ for CLIP pooled).
        3. Encode only truly uncached captions on GPU.
        4. Save newly encoded embeddings to disk for future runs.
        """
        if not self.config.get("cache_text_embeddings", True):
            return
        if self.t5_encoder is None:
            return

        from app.engine.components.text_embeddings import TextEmbeddingCache

        # Resolve disk cache directories (te1=T5, te2=CLIP pooled)
        te_cache_dirs = self._resolve_te_cache_dirs()
        # Include TE quantization scheme so FP8 / bf16 embeddings don't collide
        te_quant = self.config.get("te_quantization", "none")
        te1_dir = os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te1") if te_cache_dirs else ""
        te2_dir = os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te2") if te_cache_dirs else ""

        # ── Build full caption set (shared base class logic) ────────────────
        caption_hints = self._build_caption_hints()

        # ── Phase 1: Load from disk cache ─────────────────────────────────
        disk_loaded = 0
        need_encode: list[tuple[str, str]] = []

        for caption, hint in caption_hints.items():
            if caption in self.text_cache:
                continue
            if te1_dir and te2_dir:
                t5_tensor = TextEmbeddingCache.load(caption, te1_dir, hint)
                clip_tensor = TextEmbeddingCache.load(caption, te2_dir, hint)
                if t5_tensor is not None and clip_tensor is not None:
                    self.text_cache[caption] = t5_tensor
                    self._clip_pooled_cache[caption] = clip_tensor
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
                cached_t5=len(self.text_cache),
                cached_clip=len(self._clip_pooled_cache),
                source="disk",
            )
            return

        # ── Phase 2: Encode missing captions on GPU ───────────────────────
        print("[STATUS:Caching Text Embeddings (0%)]", flush=True)
        encode_total = len(need_encode)
        batch_size = 4

        with torch.no_grad():
            for i in range(0, encode_total, batch_size):
                batch_items = need_encode[i : i + batch_size]
                batch_caps = [cap for cap, _ in batch_items]
                self._get_cached_text_embeddings(batch_caps, self._resolve_loading_dtype())

                # Save newly encoded to disk (T5 → te1/, CLIP → te2/)
                for cap, hint in batch_items:
                    if cap in self.text_cache and te1_dir:
                        TextEmbeddingCache.save(
                            cap, self.text_cache[cap], te1_dir, hint,
                        )
                    if cap in self._clip_pooled_cache and te2_dir:
                        TextEmbeddingCache.save(
                            cap, self._clip_pooled_cache[cap], te2_dir, hint,
                        )

                pct = round(min(i + batch_size, encode_total) / encode_total * 100)
                print(f"[STATUS:Caching Text Embeddings ({pct}%)]", flush=True)
                if (i + batch_size) % 20 == 0 or (i + batch_size) >= encode_total:
                    self.logger.info(
                        "te_cache_progress",
                        cached=min(i + batch_size, encode_total),
                        total=encode_total,
                    )

        self.logger.info(
            "text_embedding_cache_complete",
            cached_t5=len(self.text_cache),
            cached_clip=len(self._clip_pooled_cache),
            newly_encoded=encode_total,
        )

    def _offload_text_encoders(self) -> None:
        """Offload or unload CLIP + T5 text encoders + tokenizers.

        Delegates TE handling (including ``self.components`` cleanup)
        to the base class via ``_get_text_encoders()``, then cleans up
        Flux1-specific instance attributes and tokenizers.
        """
        unloading = self.config.get("unload_text_encoder", False)
        super()._offload_text_encoders()
        # Base cleans self.components + sets text_encoder/text_encoder_2
        # to None.  Flux1 uses different attr names, so also None-ify:
        if unloading and not hasattr(self, "_tok_cleaned"):
            self.clip_encoder = None
            self.t5_encoder = None
            self.clip_tokenizer = None
            self.t5_tokenizer = None
            self.components.pop("tokenizer", None)
            self.components.pop("tokenizer_2", None)
            self._tok_cleaned = True


    # ── Text Encoding ────────────────────────────────────────────────────

    def encode_text(
        self, captions: list[str], dtype: torch.dtype
    ) -> torch.Tensor:
        """Dual CLIP+T5 text encoding with lazy caching.

        CLIP provides pooled embeddings stored on ``self._clip_pooled``.
        T5 provides the sequence context returned as the main embedding.

        Args:
            captions: Batch of caption strings.
            dtype: Target dtype.

        Returns:
            T5 text embeddings ``[B, L_txt, 4096]``.
        """
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(captions, dtype)
        return self._encode_fresh(captions, dtype)

    def _encode_fresh(
        self, captions: list[str], dtype: torch.dtype
    ) -> torch.Tensor:
        """Encode captions without caching."""
        # T5 sequence embeddings
        t5_inputs = self.t5_tokenizer(
            captions,
            padding="max_length",
            max_length=self.te_t5_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            t5_out = self.t5_encoder(
                t5_inputs.input_ids.to(self.device),
            )
        t5_emb = t5_out.last_hidden_state.to(dtype=dtype)

        # CLIP pooled embeddings
        clip_inputs = self.clip_tokenizer(
            captions,
            padding="max_length",
            max_length=self.te_clip_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            clip_out = self.clip_encoder(
                clip_inputs.input_ids.to(self.device),
                output_hidden_states=False,
            )
        self._clip_pooled = clip_out.pooler_output.to(dtype=dtype)

        return t5_emb

    def _get_cached_text_embeddings(
        self, captions: list[str], dtype: torch.dtype
    ) -> torch.Tensor:
        """Encode on first encounter; reuse thereafter."""
        t5_results: list[torch.Tensor] = []
        pooled_results: list[torch.Tensor] = []
        uncached: list[tuple[int, str]] = []

        for i, cap in enumerate(captions):
            if cap in self.text_cache:
                t5_results.append(self.text_cache[cap])
                pooled_results.append(self._clip_pooled_cache[cap])
            else:
                uncached.append((i, cap))
                t5_results.append(None)  # placeholder
                pooled_results.append(None)

        if uncached and self.t5_encoder is not None:
            # Guard: if TE was offloaded to CPU, temporarily move back for encoding
            te_device = next(self.t5_encoder.parameters()).device
            te_was_offloaded = te_device != self.device
            if te_was_offloaded:
                self.logger.warning(
                    "te_cache_miss_after_offload",
                    count=len(uncached),
                    hint="pre-caching should have covered all captions",
                )
                self.t5_encoder.to(self.device)
                self.clip_encoder.to(self.device)

            batch_caps = [c for _, c in uncached]
            t5_emb = self._encode_fresh(batch_caps, dtype)
            clip_pooled = self._clip_pooled

            for j, (orig_idx, cap) in enumerate(uncached):
                t5_single = t5_emb[j : j + 1]
                pooled_single = clip_pooled[j : j + 1]
                self.text_cache[cap] = t5_single.cpu()
                self._clip_pooled_cache[cap] = pooled_single.cpu()
                t5_results[orig_idx] = t5_single.cpu()
                pooled_results[orig_idx] = pooled_single.cpu()

            # Move back to CPU if we temporarily brought it up
            if te_was_offloaded:
                self.t5_encoder.to("cpu")
                self.clip_encoder.to("cpu")
                torch.cuda.empty_cache()

        t5_batch = torch.cat(
            [t.to(self.device, dtype=dtype) for t in t5_results], dim=0
        )
        pooled_batch = torch.cat(
            [p.to(self.device, dtype=dtype) for p in pooled_results], dim=0
        )
        self._clip_pooled = pooled_batch
        return t5_batch

    # ── Latent Packing ───────────────────────────────────────────────────

    def prepare_latents_for_training(self, latents: torch.Tensor) -> torch.Tensor:
        """Pack image latents ``[B, C, H, W]`` → ``[B, L, C]``.

        Stores ``_current_img_ids`` as 2-D ``[L, 3]`` for the forward pass.
        Also stores ``_latent_h`` / ``_latent_w`` for unpack in sampling.
        """
        self._latent_h = latents.shape[2]
        self._latent_w = latents.shape[3]

        packed, img_ids = pack_latents(latents)
        self._current_img_ids = img_ids.to(self.device)
        return packed.to(self.device, dtype=self.autocast_dtype)

    # ── Forward Pass ─────────────────────────────────────────────────────

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """FluxTransformer2DModel forward: predict velocity.

        Args:
            noisy_input: Packed noisy latents ``[B, L, 64]``.
            timesteps: Scaled timesteps ``[0, 1000]``.
            text_embeddings: T5 context ``[B, L_txt, 4096]``.
            batch: Full batch dict.

        Returns:
            Velocity prediction ``[B, L, 64]``.
        """
        # Diffusers model multiplies timestep by 1000 internally
        model_timesteps = timesteps / 1000.0

        # txt_ids: zeros [L_txt, 3]
        txt_seq_len = text_embeddings.shape[1]
        txt_ids = torch.zeros(
            txt_seq_len, 3,
            device=self.device, dtype=text_embeddings.dtype,
        )

        # CLIP pooled embedding
        pooled = getattr(self, "_clip_pooled", None)
        if pooled is None:
            pooled_dim = self.transformer.config.pooled_projection_dim
            pooled = torch.zeros(
                noisy_input.shape[0], pooled_dim,
                device=self.device, dtype=self.autocast_dtype,
            )

        # Guidance (Dev uses guidance_embed; Schnell does not)
        guidance = None
        if self.use_guidance_embed:
            guidance_scale = float(self.config.get("guidance_scale", 3.5))
            guidance = torch.full(
                (noisy_input.shape[0],), guidance_scale,
                device=self.device, dtype=self.autocast_dtype,
            )

        output = self.transformer(
            hidden_states=noisy_input,
            encoder_hidden_states=text_embeddings,
            pooled_projections=pooled,
            timestep=model_timesteps,
            img_ids=self._current_img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )

        # return_dict=False → tuple; first element is the sample
        return output[0] if isinstance(output, tuple) else output
