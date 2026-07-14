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

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from .driver import QwenImageDriver, te_template_fingerprint
from .loader import QwenImageLoader
from .saver import QwenImageSaver

logger = structlog.get_logger(__name__)

# The prompt template + preamble-drop encoding now lives solely in
# ``QwenImageDriver`` (single source of truth; the trainer delegates via
# ``driver.encode_text``).  ``PROMPT_TEMPLATE_DROP_IDX`` is retained here for the
# edit subclass import; ``TOKENIZER_MAX_LENGTH`` is the production context window
# the trainer syncs onto the driver in ``_assign_components``.
PROMPT_TEMPLATE_DROP_IDX = 34  # system preamble tokens to drop
TOKENIZER_MAX_LENGTH = 1024

# Disk-cache key template identity (the boogu_image precedent).
# ``TextEmbeddingCache.caption_to_filename`` hashes ONLY the string it is given;
# passing the raw caption meant a future edit to the driver's ``PROMPT_TEMPLATE``
# / preamble-drop would silently reuse embeddings encoded under the OLD template.
# Baking a fingerprint of the transformation into the hashed string makes a
# template bump always produce a fresh on-disk filename. ``te_template_fingerprint``
# hashes the actual template + drop-idx, so a prompt tweak can never forget it.
# The IN-MEMORY ``self.text_cache`` stays keyed by the raw caption (the
# krea2/ernie/ideogram4 convention the cross-family seam contract expects).
_TE_TEMPLATE_VERSION = "v1"
_TE_TEMPLATE_ID = f"qwen_image/chatml_system_prompt/{_TE_TEMPLATE_VERSION}/{te_template_fingerprint()}"


def _disk_cache_key(caption: str) -> str:
    """Compose the string hashed by ``TextEmbeddingCache.caption_to_filename``.

    Baking ``_TE_TEMPLATE_ID`` into the hashed string (instead of the raw
    caption) means a future template change produces a DIFFERENT on-disk
    filename for the same caption text, instead of silently reusing a stale
    embedding encoded under the old template.
    """
    return f"{_TE_TEMPLATE_ID}::{caption}"


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
        # The trainer's text path delegates to ``driver.encode_text``; keep the
        # driver's context window in lock-step with the trainer's production
        # value (the driver default drifted to 512 historically).
        self.driver.max_length = self.max_length

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
                emb_tensor = TextEmbeddingCache.load(_disk_cache_key(caption), te1_dir, hint)
                mask_tensor = TextEmbeddingCache.load(_disk_cache_key(caption), te2_dir, hint)
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
                emb_batch, mask_batch = self._encode_text_direct(batch_caps, dtype)

                for j, (cap, hint) in enumerate(batch_items):
                    # Trim each entry out of the sub-batch's own padding (each
                    # sub-batch of 4 pads to ITS OWN max — un-trimmed entries
                    # would carry inconsistent cross-sub-batch padding and crash
                    # the ragged reassembly stack; W3-4). Store the trimmed
                    # entry in memory AND on disk; reassembly re-pads to the
                    # batch max.
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

    # -- TE Offloading --

    # _offload_text_encoders — inherited from base class.
    # Base uses _get_text_encoders() to discover and offload/unload TEs
    # and properly cleans self.components to prevent stale references.

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions through Qwen2.5-VL in text-only mode.

        ``batch`` is accepted for hook compatibility and ignored here — the
        paired-edit subclass (:class:`QwenImageEditTrainer`) uses it to feed
        the control image to the VL encoder and key the cache compositely.

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

    def _encode_text_direct(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions directly via the driver (no cache).

        Delegates to ``driver.encode_text`` (single source of truth for the
        Qwen2.5-VL template + preamble-drop encoding) and unwraps the
        ``TextEncoderOutput`` to the ``(embeddings, attention_mask)`` tuple
        contract the base pipeline hands opaquely to ``forward_pass``.

        Returns:
            (hidden_states [B, L, D], attention_mask [B, L]).
        """
        out = self.driver.encode_text(captions, dtype)
        return out.embeddings, out.attention_mask

    @staticmethod
    def _trim_entry(
        emb: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trim a single per-caption cache entry to its TRUE (mask) length.

        Qwen2.5-VL tokenizes with ``padding=True`` (``pad="longest"``) and no
        fixed crop, so an entry sliced out of a padded batch carries whatever
        padding THAT batch happened to have. Cache entries must be
        length-normalized (the kandinsky5 / boogu_image precedent) so
        reassembly padding is well-defined and independent of the batch an
        entry was first encoded in — otherwise mixed-length batches raise a
        ragged ``torch.stack`` RuntimeError (W3-4).

        Args:
            emb: Per-caption embeddings ``[L_padded, D]``.
            mask: Per-caption attention mask ``[L_padded]`` (right-padded, so
                valid positions are a prefix).

        Returns:
            ``(emb [L_true, D], mask [L_true])``.
        """
        true_len = int(mask.sum().item())
        return emb[:true_len], mask[:true_len]

    def _get_cached_text_embeddings(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode on first encounter, reuse thereafter.

        Caches both embeddings and attention masks. Qwen2.5-VL is a
        VARIABLE-LENGTH encoder (``pad="longest"``, no fixed crop), so
        per-caption entries are stored TRIMMED to their true length and
        reassembly PADS to the batch max — embeddings zero-padded, masks
        zero-padded (mask=0 == ignored position). This is byte-equivalent to
        the direct encode path, which itself re-pads to the batch max and
        passes the mask as ``encoder_hidden_states_mask``
        (``QwenImageDriver.forward_pass``). A plain ``torch.stack`` over ragged
        entries would raise on any real mixed-length batch (W3-4).

        Returns:
            (text_embeddings [B, L_max, D], attention_mask [B, L_max]).
        """
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
                self.text_cache[cap] = self._trim_entry(
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

        # Mask-aware padded reassembly (entries are ragged — see docstring).
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

    # -- Forward Pass --
    # Delegated to ``QwenImageDriver.forward_pass`` via the base
    # ``PipelineBaseMixin.forward_pass`` (patchify → transformer → unpatchify).
    # The edit subclass keeps its own override for the control-concat path and
    # calls ``super().forward_pass`` (→ driver) for standard batches.

