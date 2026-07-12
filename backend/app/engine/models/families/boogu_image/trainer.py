"""BooguImageTrainer — family-specific trainer for Boogu-Image (Task 5).

Wires :class:`BooguImageDriver` into the generic training pipeline and
completes the binding handoffs documented in ``driver.py``'s module
docstring ("Interface handoff to Task 5 (Trainer)"):

1. **Convention delegation (load-bearing).** ``PipelineBaseMixin.add_noise``
   / ``.compute_target`` / ``.sample_timesteps`` hardcode the STANDARD
   (non-inverted) flow-match convention. Boogu-Image is INVERTED
   (``target = x0 - noise``, raw ``[0, 1)`` timesteps — see
   ``driver.py``'s module docstring for the full derivation). All three are
   overridden below to delegate to the driver's own versions. Skipping any
   one of these would silently train a pure-noise LoRA.
2. **``encode_text`` tuple contract (the krea2 C1/C2 pattern).** The
   driver's ``forward_pass`` expects a plain ``(embeddings,
   attention_mask)`` tuple, not the raw ``TextEncoderOutput`` dataclass the
   base ``PipelineBaseMixin.encode_text`` passes through unchanged.
   ``encode_text`` / ``_encode_text_direct`` below unwrap it.
3. **``progress`` pass-through.** ``sample_timesteps`` computes
   ``global_step / max_train_steps`` (mirroring
   ``PipelineBaseMixin.sample_timesteps``'s own calculation) and forwards it
   to the driver so the ``radc`` curriculum timestep mode advances with
   training instead of silently pinning at ``progress=0``.
4. **The rest of the house override trio** (``_update_primary_model`` /
   ``transformer`` property / ``_assign_components``) — mirrors
   ``Krea2Trainer`` exactly: the driver stores its primary model on
   ``self.driver.model`` (not ``.transformer``), so the base
   ``_assign_components`` loop skips the ``transformer`` alias (it's a
   property on this subclass) and ``_update_primary_model`` must explicitly
   sync ``self.driver.model`` (the base method does not reach into the
   driver). Pinned by the parametrized seam-contract test
   (``backend/tests/engine/families/test_trainer_seam_contract.py``).

## encode_text — the Boogu VLM path + TE cache

The actual Qwen3-VL forward (chat template, system prompt, last-layer tap)
lives on ``BooguImageDriver.encode_text`` (see its docstring + module
docstring for the full evidence trail, including two findings that
contradict a literal reading of the upstream pipeline). This trainer's job
is the house TE-cache plumbing around it (mirrors
``Krea2Trainer``'s ``_pre_cache_text_embeddings`` / ``_get_cached_text_embeddings``
/ ``_encode_text_direct`` almost verbatim), plus one Boogu-specific addition:

- **Disk-cache key template identity.** ``TextEmbeddingCache.caption_to_filename``
  hashes ONLY the string it is given. If the raw caption were passed
  directly, a future differently-templated encoding of the SAME caption
  text (e.g. a hypothetical TI2I/edit template) would silently collide with
  and load a STALE embedding encoded under today's T2I template. ``_disk_cache_key``
  bakes ``_TE_TEMPLATE_ID`` into the string passed to
  ``TextEmbeddingCache.save``/``.load`` so a template version bump always
  produces a fresh on-disk filename. The IN-MEMORY ``self.text_cache`` dict
  stays keyed by the raw caption (matching the krea2/ernie/ideogram4
  convention the cross-family seam contract test expects).
- **Cached-embed dtype cast.** Already handled centrally by
  ``_get_cached_text_embeddings``'s ``.to(self.device, dtype=dtype)`` on
  read (the same house mechanism krea2 relies on) — verified, not
  reimplemented here.
- **Ragged-length entries (review Finding 1).** Boogu is the first
  VARIABLE-LENGTH family on this cache pattern (``padding="longest"``, no
  fixed crop), so per-caption cache entries have different lengths and a
  plain ``torch.stack`` reassembly would crash on any real mixed-length
  batch. Entries are stored TRIMMED to their true (mask) length —
  :meth:`BooguImageTrainer._trim_entry`, the kandinsky5 precedent — in both
  the lazy path and ``_pre_cache_text_embeddings`` (whose per-4 sub-batches
  each pad to their own max), and reassembly pads embeddings AND masks to
  the batch max with zeros (mask=0 == ignored position, equivalent to the
  direct encode path's processor-side batch padding).
"""

from __future__ import annotations

import os

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline

from .driver import te_template_fingerprint

logger = structlog.get_logger(__name__)

# Disk-cache key template identity — see module docstring.
#
# Version history:
#   v1 — T2I system prompt for ALL captions (including dropout ""). WRONG
#        for empty captions (review Finding 2): the base checkpoint's
#        learned unconditional anchor lives under the DROP prompt, so v1
#        dropout embeddings on disk must never be reused — hence the bump.
#   v2 — per-caption adaptive prompt (T2I for real captions, DROP for
#        empty/whitespace dropout), matching upstream's
#        ``_apply_chat_template`` adaptive branch.
#
# The id also embeds a fingerprint HASHED FROM the actual prompt strings
# (via the driver's public ``te_template_fingerprint()`` — never a direct
# import of the driver's private ``_SYSTEM_PROMPT_*`` constants), so any
# future edit to either prompt text changes every disk-cache key
# automatically — a prompt tweak can never silently forget the version bump.
_TE_TEMPLATE_VERSION = "v2"
_TE_TEMPLATE_FINGERPRINT = te_template_fingerprint()
_TE_TEMPLATE_ID = (
    f"boogu_image/chatml_system_prompt/{_TE_TEMPLATE_VERSION}/"
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


class BooguImageTrainer(GenericTrainingPipeline):
    """Boogu-Image (Base / Turbo) LoRA trainer.

    40-layer mixed single/double-stream DiT (Lumina2-style AdaLN blocks)
    with a Qwen3-VL mllm text encoder (full-VLM, chat-template + last-layer
    tap) and Boogu's own INVERTED flow-match convention.
    """

    # ── Setup (override trio 1/3) ──────────────────────────────────────────

    def _setup_family(self) -> None:
        """Initialize Boogu-Image-specific loader and driver.

        Lazy-imported so registry discovery (which just imports this
        module) never trips on a missing saver/sampler module from a later
        task.
        """
        from .loader import BooguImageLoader  # noqa: PLC0415
        from .driver import BooguImageDriver  # noqa: PLC0415

        self.loader = BooguImageLoader(self.device)
        self.driver = BooguImageDriver(self.definition, self.device)

    def _create_sampler(self):
        """Create a BooguImageSampler if sampling is configured and lands."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import BooguImageSampler  # noqa: PLC0415

            return BooguImageSampler(self)
        return None

    def init_scheduler(self) -> object:
        """Delegate to the driver's ``init_scheduler()`` (real-path fix).

        ``PipelineLoadingMixin.load_model()`` calls ``self.init_scheduler()``
        on the TRAINER (this class), not the driver — the trainer-level hook
        (``PipelineBaseMixin.init_scheduler``) defaults to ``None`` and, left
        un-overridden, ``load_model()`` writes that ``None`` back into
        ``self.components["scheduler"]``, clobbering the real vendored
        scheduler the loader placed there. The clobbered ``None`` then
        propagates to ``driver.scheduler`` the next time
        ``self._assign_components()`` re-syncs from ``self.components`` —
        which happens unconditionally in ``_quantize_text_encoders()`` on
        EVERY real job (``prepare_for_training()`` ->
        ``_quantize_components()`` -> ``_quantize_text_encoders()``),
        regardless of ``te_quantization``. The result: ``driver.scheduler``
        is ``None`` by the time sampling runs, tripping the sampler's
        deliberate fail-loud guard (``sampler.py``'s ``_denoise_base``).

        ``BooguImageDriver.init_scheduler()`` already correctly reads
        ``self._components["scheduler"]`` (the loader-provided vendored
        instance, never a fresh/stock one) — this override just makes sure
        the real path actually calls it, mirroring the sdxl precedent of a
        trainer-level ``init_scheduler()`` override but reusing the
        driver's existing implementation instead of duplicating it.
        """
        return self.driver.init_scheduler()

    # ── Component Assignment (override trio 2/3 continuation) ─────────────

    def _assign_components(self) -> None:
        """Wire components via driver + set the Boogu-Image ``model`` alias.

        The base loop skips ``transformer`` (a read-only property on this
        subclass) and does not know about a ``model`` alias at all — set it
        explicitly so it exists before the first possible
        ``_update_primary_model`` call (mirrors ``Krea2Trainer``).
        """
        super()._assign_components()
        self.model = self.driver.model

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep ``self.model`` and ``driver.model`` in sync after PEFT/quant wrap.

        The base method updates ``self.components["unet"]`` and
        ``self.model`` (if present) but does NOT reach into
        ``self.driver.model`` — without this override the PEFT-wrapped
        model is absent from the driver's forward graph and the optimizer's
        trainable-param list is empty (the historical krea2 C3 bug class;
        pinned by the cross-family seam-contract test).
        """
        super()._update_primary_model(new_model)
        self.model = new_model
        self.driver.model = new_model

    @property
    def transformer(self) -> torch.nn.Module | None:
        """Read-only alias delegating to ``driver.model``.

        ``BooguImageDriver`` stores its primary model on ``self.driver.model``
        (not ``.transformer``), so the base ``_assign_components`` loop sets
        ``self.transformer = None`` unless short-circuited by this property.
        Sampler code that reads ``trainer.transformer`` (or
        ``next(self.pipeline.transformer.parameters())``) must never hit
        ``None`` — and, post-PEFT-wrap, must resolve to the wrapped model
        (the historical krea2 C4 bug class; pinned by the seam-contract
        test's ``property_alias`` check).
        """
        return self.driver.model if self.driver is not None else None

    # ── Convention delegation (load-bearing — see module docstring #1) ────

    def sample_timesteps(
        self, batch_size: int, latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Delegate to the driver's raw ``[0, 1)`` sampler, forwarding progress.

        Mirrors ``PipelineBaseMixin.sample_timesteps``'s own progress
        calculation (``global_step / max_train_steps``) but returns
        UNSCALED ``[0, 1)`` timesteps (never ``*1000``) — Boogu's own
        scheduler is already ``[0, 1)``-native (contract 2). Without this
        override the base mixin's ``sample_scaled`` (``*1000``, wrong
        convention entirely) would run instead.
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
        """Delegate to the driver's INVERTED lerp — ``x_t = (1-t)*noise + t*x0``.

        The base ``PipelineBaseMixin.add_noise`` hardcodes the STANDARD
        (non-inverted) convention via ``NoiseInterpolation("linear")``
        (``(1-t)*x0 + t*noise``, and expects ``t`` pre-scaled to
        ``[0, 1000]``) — using it un-overridden with Boogu's raw ``[0, 1)``
        ``t`` would be wrong in BOTH sign and scale (see the load-bearing
        test in ``test_boogu_image_trainer.py``).
        """
        return self.driver.add_noise(latents, noise, timesteps)

    def compute_target(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate to the driver's ``x0 - noise`` velocity target.

        The base ``PipelineBaseMixin.compute_target`` hardcodes
        ``noise - latents`` (the STANDARD convention's opposite sign).
        Mirrors the SDXL precedent (``families/sdxl/trainer.py`` overrides
        all three directly for its own, different, convention).
        """
        return self.driver.compute_target(latents, noise, timesteps)

    # ── Text Encoding (the krea2 C1/C2 pattern — module docstring #2) ─────

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions through Qwen3-VL (full-VLM, chat template).

        ``batch`` is accepted for hook compatibility and ignored on THIS
        (text-only) trainer — the Base/Turbo definitions are
        ``control_inputs: 0``. The Edit variant (``BooguImageEditTrainer``,
        trainer_edit.py — task A4) OVERRIDES this method to condition the
        VLM on the batch's control image(s) under a composite
        ``(caption, control)`` cache key.

        Returns a ``(embeddings, attention_mask)`` tuple — the base pipeline
        passes this opaquely to ``forward_pass()`` which passes it to
        ``driver.forward_pass()`` that unpacks the tuple. Passing the raw
        ``TextEncoderOutput`` through unchanged (the base mixin's default)
        would break at that unpack (the krea2 C1/C2 bug class).

        Args:
            captions: Processed captions.
            dtype: Target dtype.
            batch: Ignored here (consumed by the Edit subclass's override).

        Returns:
            (text_embeddings ``[B, L<=256, 4096]``, attention_mask ``[B, L]``).
        """
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(captions, dtype)

        return self._encode_text_direct(captions, dtype)

    def _encode_text_direct(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions directly via the driver (no cache).

        Delegates to ``driver.encode_text``, which returns a
        ``TextEncoderOutput``. Unwraps to the ``(embeddings, attention_mask)``
        tuple contract ``driver.forward_pass`` expects.

        Returns:
            (embeddings ``[B, L, 4096]``, attention_mask ``[B, L]``).
        """
        out = self.driver.encode_text(captions, dtype)
        return out.embeddings, out.attention_mask

    @staticmethod
    def _trim_entry(
        emb: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trim a single per-caption cache entry to its TRUE (mask) length.

        Boogu tokenizes with ``padding="longest"`` and no fixed crop
        (driver.encode_text), so a per-caption entry sliced out of a padded
        batch carries whatever padding THAT batch happened to have. Cache
        entries must be length-normalized (the kandinsky5 precedent:
        "per-caption te1 entries are trimmed to the caption's TRUE length")
        so reassembly padding is well-defined and independent of the batch
        an entry was first encoded in.

        Args:
            emb: Per-caption embeddings ``[L_padded, D]``.
            mask: Per-caption attention mask ``[L_padded]``
                (``padding_side="right"``, so valid positions are a prefix).

        Returns:
            ``(emb [L_true, D], mask [L_true])``.
        """
        true_len = int(mask.sum().item())
        return emb[:true_len], mask[:true_len]

    def _get_cached_text_embeddings(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode on first encounter, reuse thereafter.

        Mirrors ``Krea2Trainer._get_cached_text_embeddings`` with ONE
        Boogu-specific addition (review Finding 1): cache entries are
        VARIABLE-LENGTH (``padding="longest"``, no fixed crop — boogu is
        the first such family on this pattern; krea2 crops to 34 tokens,
        longcat pads to a fixed 512), so a plain ``torch.stack`` over
        ragged ``[L_i, D]`` entries would raise RuntimeError on any real
        batch whose captions tokenize to different lengths. Entries are
        stored TRIMMED to their true length and reassembly PADS to the
        batch max — embeddings zero-padded, masks zero-padded (mask=0 ==
        ignored position; the model derives per-sample caption lengths
        from ``mask.sum(dim=1)``, so this is equivalent to the direct
        encode path's processor-side ``padding="longest"`` batch).

        The in-memory ``self.text_cache`` dict is keyed by the RAW caption
        string (not the disk-cache's template-baked key — see module
        docstring).

        Returns:
            (text_embeddings ``[B, L_max, 4096]``, attention_mask ``[B, L_max]``).
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

        # Mask-aware padded reassembly (see docstring — entries are ragged).
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

    # ── Disk-backed TE Pre-caching ──────────────────────────────────────────

    def _sample_prompt_texts(self) -> list[str]:
        """Expanded sample-prompt strings to pre-cache.

        Mirrors the sampler's wildcard expansion (GenericSamplingPipeline
        calls ``_expand_wildcards`` -> ``expand_prompt_wildcards`` before
        ``encode_prompt``) so the cache key matches the exact string the
        sampler requests via :meth:`encode_text`.
        """
        from app.engine.core.sampling import expand_prompt_wildcards

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
        """Warm text embedding cache from disk + encode missing.

        Mirrors ``Krea2Trainer._pre_cache_text_embeddings``:
        - te1/ stores embedding tensors ``[L, 4096]``
        - te2/ stores attention mask tensors ``[L]``

        Disk save/load calls use :func:`_disk_cache_key` (template-baked)
        rather than the raw caption — see module docstring.
        """
        if not self.config.get("cache_text_embeddings", True):
            return
        if self.text_encoder is None:
            return

        from app.engine.components.text_embeddings import TextEmbeddingCache

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
        # sampling. The sampler expands wildcards identically via
        # _expand_wildcards before calling encode_text, so the key produced
        # by _sample_prompt_texts matches.
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
                    # Trim each entry out of the sub-batch's own padding
                    # (each sub-batch of 4 pads to ITS OWN max — un-trimmed
                    # entries would carry inconsistent cross-sub-batch
                    # padding; review Finding 1). Both memory and disk store
                    # the trimmed entry; reassembly pads to the batch max.
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
