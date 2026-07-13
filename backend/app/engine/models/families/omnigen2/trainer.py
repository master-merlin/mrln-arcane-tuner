"""OmniGen2Trainer — family-specific trainer for OmniGen2.

Wires :class:`OmniGen2Driver` into the generic training pipeline. The
binding contracts (all load-bearing — each has a bug-class precedent):

1. **Convention delegation.** ``PipelineBaseMixin.add_noise`` /
   ``.compute_target`` / ``.sample_timesteps`` hardcode the STANDARD
   (non-inverted) flow-match convention. OmniGen2 is INVERTED (``t=0``
   noise, ``t=1`` clean; ``target = x0 - noise``; raw ``[0, 1)`` timesteps
   — see ``driver.py`` module docstring §3-4). All three are overridden
   below to delegate to the driver's versions. Skipping any one would
   silently train a pure-noise LoRA (the boogu_image/sdxl precedent).
2. **``encode_text`` tuple contract (krea2 C1/C2 pattern).** The driver's
   ``forward_pass`` expects a plain ``(embeddings, attention_mask)`` tuple;
   ``encode_text`` / ``_encode_text_direct`` unwrap the driver's
   ``TextEncoderOutput``.
3. **``_update_primary_model`` driver sync (krea2 C3 pattern).** The driver
   stores its primary model on ``self.driver.transformer`` (lumina2/chroma
   lineage) — synced after PEFT wrap so the optimizer sees the wrapped
   modules. Pinned by the parametrized seam-contract test.
4. **``init_scheduler`` delegation (boogu_image clobber lesson).**
   ``load_model()`` calls the TRAINER's ``init_scheduler()``; the base hook
   returns ``None`` and would clobber the loader-provided VENDORED
   scheduler in ``self.components["scheduler"]`` — delegate to the driver's
   version, which returns the loader instance.
5. **Ragged TE-cache entries (boogu_image Finding-1 pattern).** OmniGen2
   tokenizes ``padding="longest"`` (pipeline ~L305-311), so per-caption
   cache entries are VARIABLE-LENGTH — entries are stored trimmed to their
   true (mask) length and reassembly pads embeddings+masks to the batch
   max (mask=0 == ignored position; the model derives per-sample caption
   lengths from ``attention_mask.sum(dim=1)``).
6. **Disk-cache key template identity (lumina2/boogu pattern).** The chat
   template (system prompt) is baked into every encode; ``_disk_cache_key``
   bakes ``_TE_TEMPLATE_ID`` (version + prompt-text fingerprint from the
   driver's ``te_template_fingerprint()``) into the hashed string so a
   template edit can never silently reuse a stale on-disk embedding. The
   in-memory ``self.text_cache`` stays keyed by the raw caption.

NO uncond/negative-prompt asymmetry (contrast lumina2): the pipeline
chat-templates the CFG negative prompt IDENTICALLY to positives
(pipeline L392 + L413-418), so the sampler's negative encodes through this
same ``encode_text`` path and shares the same cache.

NO composite (caption, control) TE keys (contrast boogu_image): OmniGen2's
text embeddings are control-independent — the mllm never sees image pixels
(driver.py recon §1). ``trainer_edit.py`` documents this explicitly.
"""

from __future__ import annotations

import os

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline

from .driver import te_template_fingerprint

logger = structlog.get_logger(__name__)

# Disk-cache key template identity — see module docstring §6.
_TE_TEMPLATE_VERSION = "v1"
_TE_TEMPLATE_FINGERPRINT = te_template_fingerprint()
_TE_TEMPLATE_ID = (
    f"omnigen2/chatml_system_prompt/{_TE_TEMPLATE_VERSION}/"
    f"{_TE_TEMPLATE_FINGERPRINT}"
)


def _disk_cache_key(caption: str) -> str:
    """Compose the string hashed by ``TextEmbeddingCache.caption_to_filename``.

    Baking ``_TE_TEMPLATE_ID`` into the hashed string (instead of passing
    the raw caption) means a future template change produces a DIFFERENT
    on-disk filename for the same caption text, instead of silently reusing
    a stale embedding encoded under the old template.
    """
    return f"{_TE_TEMPLATE_ID}::{caption}"


class OmniGen2Trainer(GenericTrainingPipeline):
    """OmniGen2 LoRA trainer.

    ~4B DiT (32 joint + 2 noise-refiner + 2 ref-image-refiner + 2
    context-refiner Lumina2-style blocks, vendored) with a frozen
    Qwen2.5-VL-3B text encoder, the FLUX.1-dev VAE, and OmniGen2's own
    INVERTED flow-match convention.
    """

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_family(self) -> None:
        """Initialize OmniGen2-specific loader and driver."""
        from .loader import OmniGen2Loader  # noqa: PLC0415
        from .driver import OmniGen2Driver  # noqa: PLC0415

        self.loader = OmniGen2Loader(self.device)
        self.driver = OmniGen2Driver(self.definition, self.device)

    def _create_sampler(self):
        """Create an OmniGen2Sampler if sampling is configured.

        The Edit subclass overrides this to return the control-fed
        ``OmniGen2EditSampler``.
        """
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import OmniGen2Sampler  # noqa: PLC0415

            return OmniGen2Sampler(self)
        return None

    def init_scheduler(self) -> object:
        """Delegate to the driver (module docstring §4 — clobber lesson)."""
        return self.driver.init_scheduler()

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep self.transformer + driver.transformer in sync after PEFT wrap."""
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

    # ── Convention delegation (load-bearing — module docstring §1) ───────

    def sample_timesteps(
        self, batch_size: int, latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Delegate to the driver's NATIVE ``[0, 1)`` reversed-clock sampler.

        Mirrors ``PipelineBaseMixin.sample_timesteps``'s progress
        calculation but returns UNSCALED native timesteps (never ``*1000``)
        — without this override the base mixin's ``sample_scaled``
        (``*1000``, wrong scale AND wrong direction) would run instead.
        """
        max_steps = getattr(self, "max_train_steps", 1)
        progress = getattr(self, "global_step", 0) / max(max_steps, 1)
        return self.driver.sample_timesteps(
            batch_size, self.device, self.config, latents=latents, progress=progress,
        )

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate to the driver's INVERTED lerp ``x_t = (1-t)*noise + t*x0``."""
        return self.driver.add_noise(latents, noise, timesteps)

    def compute_target(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate to the driver's ``x0 - noise`` velocity target."""
        return self.driver.compute_target(latents, noise, timesteps)

    # ── Text Encoding (krea2 C1/C2 pattern — module docstring §2) ────────

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions through Qwen2.5-VL (text-only, chat template).

        ``batch`` is accepted for hook compatibility and ignored — OmniGen2
        text embeddings are control-independent BY DESIGN (driver.py recon
        §1), so even the Edit subclass never conditions the encode on the
        control image.

        Returns:
            (text_embeddings ``[B, L, 2048]``, attention_mask ``[B, L]``).
        """
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(captions, dtype)

        return self._encode_text_direct(captions, dtype)

    def _encode_text_direct(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions directly via the driver (no cache)."""
        out = self.driver.encode_text(captions, dtype)
        return out.embeddings, out.attention_mask

    @staticmethod
    def _trim_entry(
        emb: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trim a per-caption cache entry to its TRUE (mask) length.

        ``padding="longest"`` means an entry sliced out of a padded batch
        carries that batch's padding — length-normalize so reassembly is
        independent of the batch an entry was first encoded in (boogu
        Finding-1 / kandinsky5 precedent). Valid positions are a prefix
        (right padding).
        """
        true_len = int(mask.sum().item())
        return emb[:true_len], mask[:true_len]

    def _get_cached_text_embeddings(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode on first encounter, reuse thereafter (ragged-aware).

        Entries are stored TRIMMED to true length; reassembly pads
        embeddings AND masks to the batch max with zeros (module docstring
        §5). In-memory cache keyed by the RAW caption string.
        """
        uncached: list[tuple[int, str]] = []

        for i, cap in enumerate(captions):
            if cap not in self.text_cache:
                uncached.append((i, cap))

        if uncached and self.text_encoder is not None:
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
                single_emb, single_mask = self._encode_text_direct([cap], dtype)
                self.text_cache[cap] = self._trim_entry(
                    single_emb.squeeze(0).cpu(),
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

        # Mask-aware padded reassembly (entries are ragged).
        entries = [self.text_cache[cap] for cap in captions]
        max_len = max(e.shape[0] for e, _ in entries)

        emb_results: list[torch.Tensor] = []
        mask_results: list[torch.Tensor] = []
        for cached_emb, cached_mask in entries:
            emb = cached_emb.to(self.device, dtype=dtype)
            mask = cached_mask.to(self.device)
            pad_rows = max_len - emb.shape[0]
            if pad_rows > 0:
                emb = torch.cat(
                    [emb, emb.new_zeros(pad_rows, *emb.shape[1:])], dim=0,
                )
                mask = torch.cat([mask, mask.new_zeros(pad_rows)], dim=0)
            emb_results.append(emb)
            mask_results.append(mask)

        return torch.stack(emb_results, dim=0), torch.stack(mask_results, dim=0)

    # ── Disk-backed TE Pre-caching ────────────────────────────────────────

    def _sample_prompt_texts(self) -> list[str]:
        """Expanded sample-prompt strings to pre-cache (mirrors the
        sampler's own wildcard expansion so cache keys match exactly)."""
        from app.engine.core.sampling import expand_prompt_wildcards  # noqa: PLC0415

        texts: list[str] = []
        for sp in self.config.get("sample_prompts", []) or []:
            raw = (
                sp.get("prompt", "")
                if isinstance(sp, dict)
                else getattr(sp, "prompt", "")
            )
            if raw:
                expanded = expand_prompt_wildcards(raw, self.config)
                if expanded not in texts:
                    texts.append(expanded)
        return texts

    def _pre_cache_text_embeddings(self) -> None:
        """Warm the TE cache from disk + encode missing (boogu layout).

        - ``embeddings/{te_quant}/te1`` stores embeddings ``[L, 2048]``
        - ``embeddings/{te_quant}/te2`` stores attention masks ``[L]``
        Disk save/load uses :func:`_disk_cache_key` (template-baked). The
        CFG negative prompt is warmed through the SAME path (no lumina2
        asymmetry — module docstring).
        """
        if not self.config.get("cache_text_embeddings", True):
            return
        if self.text_encoder is None:
            return

        from app.engine.components.text_embeddings import TextEmbeddingCache  # noqa: PLC0415

        te_cache_dirs = self._resolve_te_cache_dirs()
        te_quant = self.config.get("te_quantization", "none")
        te1_dir = os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te1") if te_cache_dirs else ""
        te2_dir = os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te2") if te_cache_dirs else ""

        caption_hints = self._build_caption_hints()

        disk_loaded = 0
        need_encode: list[tuple[str, str]] = []

        for caption, hint in caption_hints.items():
            if caption in self.text_cache:
                continue
            if te1_dir and te2_dir:
                emb_tensor = TextEmbeddingCache.load(_disk_cache_key(caption), te1_dir, hint)
                mask_tensor = TextEmbeddingCache.load(_disk_cache_key(caption), te2_dir, hint)
                if emb_tensor is not None and mask_tensor is not None:
                    self.text_cache[caption] = (emb_tensor, mask_tensor)
                    disk_loaded += 1
                    continue
            need_encode.append((caption, hint))

        # Warm sample + negative prompts so the TE stays offloaded during
        # sampling (krea2/ovis VRAM-spike lesson).
        sample_texts = self._sample_prompt_texts()
        queued = {cap for cap, _ in need_encode}
        for sp in sample_texts:
            if sp not in self.text_cache and sp not in queued:
                need_encode.append((sp, ""))
                queued.add(sp)
        if sample_texts:
            neg = str(self.config.get("sample_negative_prompt", "") or "")
            if neg not in self.text_cache and neg not in queued:
                need_encode.append((neg, ""))

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

        if getattr(self, "_log_writer", None):
            self._log_writer.status("Caching Text Embeddings (0%)")
        encode_total = len(need_encode)
        batch_size = 4
        dtype = self._resolve_loading_dtype()

        with torch.no_grad():
            for i in range(0, encode_total, batch_size):
                batch_items = need_encode[i : i + batch_size]
                batch_caps = [cap for cap, _ in batch_items]

                emb_batch, mask_batch = self._encode_text_direct(batch_caps, dtype)

                for j, (cap, hint) in enumerate(batch_items):
                    # Trim out of the sub-batch's own padding (each sub-batch
                    # pads to ITS OWN max — boogu Finding 1).
                    emb_cpu, mask_cpu = self._trim_entry(
                        emb_batch[j].cpu(), mask_batch[j].cpu(),
                    )
                    self.text_cache[cap] = (emb_cpu, mask_cpu)
                    if te1_dir:
                        TextEmbeddingCache.save(_disk_cache_key(cap), emb_cpu, te1_dir, hint)
                    if te2_dir:
                        TextEmbeddingCache.save(_disk_cache_key(cap), mask_cpu, te2_dir, hint)

                pct = int(min(i + batch_size, encode_total) / encode_total * 100)
                if pct % 10 == 0 or (i + batch_size) >= encode_total:
                    if getattr(self, "_log_writer", None):
                        self._log_writer.status(f"Caching Text Embeddings ({pct}%)")

        self.logger.info(
            "text_embedding_cache_complete",
            cached=len(self.text_cache), newly_encoded=encode_total,
        )
