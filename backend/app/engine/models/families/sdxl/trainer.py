"""
SDXL Trainer — family-specific hooks for the generic training pipeline.

All shared logic (optimizer, EMA, gradient accumulation, noise offset,
checkpointing, signals, logging) lives in ``GenericTrainingPipeline``.
This module only implements SDXL-specific behaviour:
- DDPMScheduler with discrete timesteps
- Dual CLIP text encoding (penultimate hidden states + pooled)
- UNet forward with SDXL ``added_cond_kwargs`` (text_embeds, time_ids)
- Epsilon prediction target
- Min-SNR gamma loss weighting
- LoRA targets from definition YAML or comprehensive defaults
"""

import os
from typing import Any

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from .driver import SDXLDriver
from .loader import SDXLLoader
from .saver import SDXLSaver

logger = structlog.get_logger(__name__)


class SDXLTrainer(GenericTrainingPipeline):
    """SDXL (1.0 / Turbo) LoRA trainer."""

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_family(self) -> None:
        """Initialize SDXL-specific loader, saver, driver, and caches."""
        self.driver = SDXLDriver(self.definition, self.device)
        self.loader = SDXLLoader(self.device)
        self.saver = SDXLSaver()
        self._pooled_cache: dict[str, torch.Tensor] = {}

    def _create_sampler(self):
        """Create an SDXLSampler if sampling is configured."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import SDXLSampler
            return SDXLSampler(self)
        return None

    # ── Component Assignment ─────────────────────────────────────────────

    def _assign_components(self) -> None:
        """Wire components via driver + set SDXL-specific aliases."""
        super()._assign_components()
        self.unet = self.components["unet"]
        # TEs may have been offloaded (removed from components dict)
        self.text_encoder_1 = self.components.get("text_encoder_1", getattr(self, "text_encoder_1", None))
        self.text_encoder_2 = self.components.get("text_encoder_2", getattr(self, "text_encoder_2", None))

        arch = getattr(self.definition, "architecture_params", {}) or {}
        self.te_max_length = arch.get("te.max_length", 77)
        self.scheduler_beta_start = float(arch.get("scheduler.beta_start", 0.00085))
        self.scheduler_beta_end = float(arch.get("scheduler.beta_end", 0.012))
        self.scheduler_prediction_type = arch.get("scheduler.prediction_type", "epsilon")

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        self.unet = new_model
        self.components["unet"] = new_model
        self.driver.unet = new_model

    def _resolve_loading_dtype(self) -> torch.dtype:
        """SDXL loads in fp32 — AMP GradScaler requires fp32 params/grads."""
        return torch.float32

    # ── Scheduler ────────────────────────────────────────────────────────

    def init_scheduler(self) -> Any:
        """Create DDPMScheduler for epsilon prediction."""
        from diffusers import DDPMScheduler

        return DDPMScheduler(
            beta_start=self.scheduler_beta_start,
            beta_end=self.scheduler_beta_end,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=self.scheduler_prediction_type,
        )

    # ── Disk-backed TE Pre-caching ────────────────────────────────────────

    def _pre_cache_text_embeddings(self) -> None:
        """Warm dual-CLIP text embedding cache from disk + encode missing.

        1. Build the full set of captions (training, dropout, sampling).
        2. Try to load from disk cache (te1/ for prompt, te2/ for pooled).
        3. Encode only truly uncached captions on GPU.
        4. Save newly encoded embeddings to disk for future runs.
        """
        if not self.config.get("cache_text_embeddings", True):
            return
        if self.text_encoder_1 is None:
            return

        from app.engine.components.text_embeddings import TextEmbeddingCache

        # Resolve disk cache directories
        te_cache_dirs = self._resolve_te_cache_dirs()
        # Include TE quantization scheme so FP8 / bf16 embeddings don't collide
        te_quant = self.config.get("te_quantization", "none")
        te1_dir = os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te1") if te_cache_dirs else ""
        te2_dir = os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te2") if te_cache_dirs else ""

        # ── Build full caption set (shared base class logic) ────────────────
        caption_hints = self._build_caption_hints()

        # ── Phase 1: Load from disk cache ─────────────────────────────
        disk_loaded = 0
        need_encode: list[tuple[str, str]] = []

        for caption, hint in caption_hints.items():
            if caption in self.text_cache:
                continue
            if te1_dir and te2_dir:
                prompt_tensor = TextEmbeddingCache.load(caption, te1_dir, hint)
                pooled_tensor = TextEmbeddingCache.load(caption, te2_dir, hint)
                if prompt_tensor is not None and pooled_tensor is not None:
                    self.text_cache[caption] = prompt_tensor
                    self._pooled_cache[caption] = pooled_tensor
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
                cached_prompt=len(self.text_cache),
                cached_pooled=len(self._pooled_cache),
                source="disk",
            )
            return

        # ── Phase 2: Encode missing captions on GPU ───────────────────
        if getattr(self, "_log_writer", None):
            self._log_writer.status("Caching Text Embeddings (0%)")
        encode_total = len(need_encode)
        batch_size = 4

        with torch.no_grad():
            for i in range(0, encode_total, batch_size):
                batch_items = need_encode[i : i + batch_size]
                batch_caps = [cap for cap, _ in batch_items]

                # Encode fresh
                prompt_embeds, pooled_embeds = self._encode_text_direct(batch_caps, self._resolve_loading_dtype())

                # Cache and save each
                for j, (cap, hint) in enumerate(batch_items):
                    p_emb = prompt_embeds[j].cpu()
                    pool_emb = pooled_embeds[j].cpu()
                    self.text_cache[cap] = p_emb
                    self._pooled_cache[cap] = pool_emb
                    if te1_dir:
                        TextEmbeddingCache.save(cap, p_emb, te1_dir, hint)
                    if te2_dir:
                        TextEmbeddingCache.save(cap, pool_emb, te2_dir, hint)

                pct = int((i + len(batch_items)) / encode_total * 100)
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Caching Text Embeddings ({pct}%)")

        self.logger.info(
            "text_embedding_cache_complete",
            cached_prompt=len(self.text_cache),
            cached_pooled=len(self._pooled_cache),
            newly_encoded=encode_total,
        )

    # ── TE Offloading ─────────────────────────────────────────────────────

    # _offload_text_encoders — inherited from base class.
    # SDXL uses text_encoder_1/text_encoder_2 attr names matching
    # the component dict keys, so the base implementation works.
    # Base properly cleans self.components to prevent stale references.

    # ── Text Encoding ────────────────────────────────────────────────────

    def _encode_text_direct(
        self, captions: list[str], dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dual CLIP encoding without caching.

        Delegates to ``driver.encode_dual_clip`` (single source of truth) and
        unwraps the ``TextEncoderOutput`` to the ``(prompt_embeds, pooled)``
        tuple contract the caching layer + ``forward_pass`` expect.

        Returns:
            (prompt_embeds [B, L, D1+D2], pooled_embeds [B, D2]).
        """
        out = self.driver.encode_text(captions, dtype)
        return out.embeddings, out.pooled

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None
    ) -> torch.Tensor:
        """Dual CLIP encoding with lazy caching.

        Stores ``self._pooled_embeds`` for use in ``forward_pass()``.

        Returns:
            Prompt embeddings [B, L, D1+D2].
        """
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(captions, dtype)

        with torch.no_grad():
            prompt_embeds, pooled_embeds = self._encode_text_direct(captions, dtype)
            self._pooled_embeds = pooled_embeds
            # forward_pass is delegated to driver.forward_pass, which reads the
            # pooled off the driver — keep it in sync (sampler still reads
            # trainer._pooled_embeds).
            self.driver._pooled_embeds = self._pooled_embeds
        return prompt_embeds

    def _get_cached_text_embeddings(
        self, captions: list[str], dtype: torch.dtype
    ) -> torch.Tensor:
        """Encode on first encounter; reuse thereafter."""
        prompt_results: list[torch.Tensor] = []
        pooled_results: list[torch.Tensor] = []
        uncached: list[tuple[int, str]] = []

        for i, cap in enumerate(captions):
            # Check BOTH caches: a caption restored from an old (pre-fix)
            # checkpoint can be present in ``text_cache`` but missing from
            # ``_pooled_cache`` (old checkpoints never saved pooled entries)
            # — treat that as a cache miss too so it re-warms instead of
            # KeyError-ing on ``self._pooled_cache[cap]`` below.
            if cap not in self.text_cache or cap not in self._pooled_cache:
                uncached.append((i, cap))

        if uncached and self.text_encoder_1 is not None:
            # Guard: if TEs were offloaded to CPU, temporarily move back
            te_device = next(self.text_encoder_1.parameters()).device
            te_was_offloaded = te_device != self.device
            if te_was_offloaded:
                self.logger.warning(
                    "te_cache_miss_after_offload",
                    count=len(uncached),
                    hint="pre-caching should have covered all captions",
                )
                self.text_encoder_1.to(self.device)
                self.text_encoder_2.to(self.device)

            for _, cap in uncached:
                with torch.no_grad():
                    p_emb, pool_emb = self._encode_text_direct([cap], dtype)
                self.text_cache[cap] = p_emb.squeeze(0).cpu()
                self._pooled_cache[cap] = pool_emb.squeeze(0).cpu()

            if te_was_offloaded:
                self.text_encoder_1.to("cpu")
                self.text_encoder_2.to("cpu")
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
            prompt_results.append(self.text_cache[cap].to(self.device, dtype=dtype))
            pooled_results.append(self._pooled_cache[cap].to(self.device, dtype=dtype))

        self._pooled_embeds = torch.stack(pooled_results, dim=0)
        # Mirror onto the driver for the delegated forward_pass (see encode_text).
        self.driver._pooled_embeds = self._pooled_embeds
        return torch.stack(prompt_results, dim=0)

    # ── Timestep Sampling ────────────────────────────────────────────────

    def sample_timesteps(
        self, batch_size: int, latents: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Discrete uniform timesteps [0, num_train_timesteps)."""
        return torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device,
        ).long()

    # ── Noise Addition ───────────────────────────────────────────────────

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """DDPM forward diffusion: add noise at discrete timesteps."""
        return self.scheduler.add_noise(latents, noise, timesteps)

    # ── Forward Pass ─────────────────────────────────────────────────────
    # Delegated to ``SDXLDriver.forward_pass`` via the base
    # ``PipelineBaseMixin.forward_pass`` (UNet + added_cond_kwargs).  The pooled
    # embedding is synced onto the driver in the encode path above; ``time_ids``
    # arrive on the batch from ``build_batch_extra``.

    # ── Loss Target ──────────────────────────────────────────────────────

    def compute_target(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Epsilon prediction: target = noise."""
        return noise

    # ── Min-SNR Gamma Weighting ──────────────────────────────────────────

    def compute_loss_weight(self, timesteps: torch.Tensor) -> torch.Tensor | None:
        """Min-SNR gamma weighting for epsilon prediction.

        Reduces the contribution of high-SNR (easy/clean) timesteps during
        training to stabilize convergence.  Based on:
        https://arxiv.org/abs/2303.09556

        Args:
            timesteps: Discrete timesteps [0, 1000).

        Returns:
            Per-sample weight tensor [B], or ``None`` if gamma ≤ 0.
        """
        gamma = float(self.config.get("min_snr_gamma", 5.0))
        if gamma <= 0:
            return None

        alphas_cumprod = self.scheduler.alphas_cumprod.to(timesteps.device)
        at = alphas_cumprod[timesteps]
        snr = at / (1 - at)

        weight = torch.stack(
            [snr, torch.ones_like(snr) * gamma], dim=1
        ).min(dim=1)[0] / snr

        return weight

    # ── Checkpoint Resume: pooled TE cache ───────────────────────────────
    #
    # The base ``PipelineBaseMixin.get_te_cache``/``set_te_cache`` only
    # persist ``self.text_cache`` (the penultimate-hidden-state cache).
    # SDXL also caches a SECOND dict, ``self._pooled_cache`` (pooled CLIP-G
    # embeds used in ``added_cond_kwargs["text_embeds"]``), keyed by the same
    # captions. Without this override, ``_pooled_cache`` silently vanished on
    # every checkpoint save — a resumed run with offloaded/unloaded TEs would
    # then hit a missing pooled entry for any caption not re-warmed this run.

    def get_te_cache(self) -> dict[str, dict[str, torch.Tensor]] | None:
        """Return prompt + pooled caches for checkpoint persistence."""
        if not self.text_cache:
            return None
        return {
            "prompt": dict(self.text_cache),
            "pooled": dict(self._pooled_cache),
        }

    def set_te_cache(self, caches: dict[str, dict[str, torch.Tensor]]) -> None:
        """Restore prompt + pooled caches from a checkpoint.

        Backward compatible with pre-fix checkpoints that only ever saved
        ``{"te": ...}`` (no ``"pooled"`` subcache at all): the prompt cache
        unions ``"te"`` (legacy key) with ``"prompt"`` (current key); a
        missing ``"pooled"`` subcache leaves ``_pooled_cache`` untouched
        (empty on first resume from an old checkpoint) rather than raising —
        ``_get_cached_text_embeddings`` re-warms any caption whose pooled
        entry is missing (or raises the existing, non-silent "not
        pre-cached" error if the TE has since been unloaded).
        """
        prompt_data = {**caches.get("te", {}), **caches.get("prompt", {})}
        if prompt_data:
            self.text_cache = prompt_data
        pooled_data = caches.get("pooled")
        if pooled_data:
            self._pooled_cache = pooled_data
        self.logger.info(
            "te_cache_restored",
            prompt_entries=len(self.text_cache),
            pooled_entries=len(self._pooled_cache),
        )

    # ── SDXL Batch Extra (time_ids) ──────────────────────────────────────

    def build_batch_extra(self, items: list[dict]) -> dict[str, Any]:
        """Build SDXL time_ids (6-component conditioning vector).

        Components: (original_h, original_w, crop_top, crop_left, target_h, target_w)
        """

        time_ids_list = []
        for item in items:
            tw, th = item["target_w"], item["target_h"]

            # Replicate the resize/crop math to get accurate crop offsets
            w, h = item.get("orig_w", tw), item.get("orig_h", th)
            scale = max(tw / w, th / h)
            nw, nh = int(w * scale), int(h * scale)
            left = (nw - tw) // 2
            top = (nh - th) // 2

            t_id = [nh, nw, top, left, th, tw]
            time_ids_list.append(torch.tensor(t_id, dtype=torch.float32))

        return {"time_ids": torch.stack(time_ids_list).to(self.device)}
