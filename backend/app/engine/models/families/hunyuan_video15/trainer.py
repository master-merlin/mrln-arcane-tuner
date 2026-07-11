"""HunyuanVideo 1.5 trainer — wires the family loader/driver/saver/sampler.

All shared training mechanics live in ``GenericTrainingPipeline``. This
trainer:

- selects the hv15 loader/driver/saver in ``_setup_family``,
- delegates dual-TE text encoding to the driver, caching the FULL
  ``(emb, mask, emb2, mask2)`` 4-tuple per caption in memory and as an
  LTX-2-style te1/te2/te3 disk triple (te1 = Qwen embedding ``[1, 1000, 3584]``,
  te2 = ByT5 glyph embedding ``[1, 256, 1472]``, te3 = the two int64 attention
  masks concatenated ``[1, 1000 + 256]``) with te1 written LAST as the
  commit marker,
- pre-caches Siglip first-frame image embeddings for I2V runs (aux cache,
  mirroring LTX-2's audio-latent aux pre-cache) and attaches them to each
  batch via ``build_batch_extra``,
- keeps the transformer references in sync after PEFT/quant wrapping
  (``_update_primary_model`` — the seam-contract trio).
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline

from .driver import Hv15Driver
from .loader import Hv15Loader
from .saver import Hv15Saver

logger = structlog.get_logger(__name__)

# Cached entry: (emb [1,L1,D1], mask [1,L1], emb2 [1,L2,D2], mask2 [1,L2]) on CPU.
_Entry = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class Hv15Trainer(GenericTrainingPipeline):
    """HunyuanVideo 1.5 (480p T2V / I2V) LoRA trainer.

    ``is_video_family`` is derived from the model's ``is_video`` capability by
    the base — no per-trainer flag.
    """

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_family(self) -> None:
        self.driver = Hv15Driver(self.definition, self.device)
        self.loader = Hv15Loader(self.device)
        self.saver = Hv15Saver(mode=self.driver.mode)

    def _create_sampler(self):
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import Hv15Sampler

            return Hv15Sampler(self)
        return None

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep transformer references in sync after PEFT/quant wrapping."""
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

    # ── Text Encoding (dual TE via driver, 4-tuple cache) ────────────────

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None
    ) -> _Entry:
        """Batched ``(emb, mask, emb2, mask2)`` from the warm cache.

        Caching off → encode directly via the driver. A miss while the TEs are
        resident is encoded on the fly (and cached); a miss AFTER offload is a
        hard error (the pre-cache must cover every caption).
        """
        if not self.config.get("cache_text_embeddings", True):
            return self.driver.encode_text(captions, dtype)

        embs: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        embs2: list[torch.Tensor] = []
        masks2: list[torch.Tensor] = []
        for cap in captions:
            entry = self.text_cache.get(cap)
            if entry is None:
                if self.driver.text_encoder is None:
                    raise RuntimeError(
                        "Text encoder offloaded and caption not pre-cached: "
                        f"{cap[:60]!r}"
                    )
                out = self.driver.encode_text([cap], dtype)
                entry = self._slice_te_output(out, 0)
                self.text_cache[cap] = entry
            emb_c, mask_c, emb2_c, mask2_c = entry
            embs.append(emb_c)
            masks.append(mask_c)
            embs2.append(emb2_c)
            masks2.append(mask2_c)

        return (
            torch.cat([e.to(self.device, dtype=dtype) for e in embs], dim=0),
            torch.cat([m.to(self.device) for m in masks], dim=0),
            torch.cat([e.to(self.device, dtype=dtype) for e in embs2], dim=0),
            torch.cat([m.to(self.device) for m in masks2], dim=0),
        )

    @staticmethod
    def _slice_te_output(out: _Entry, j: int) -> _Entry:
        """Extract item ``j`` of a driver TE batch as a CPU 4-tuple."""
        emb, mask, emb2, mask2 = out
        return (
            emb[j : j + 1].cpu(),
            mask[j : j + 1].cpu(),
            emb2[j : j + 1].cpu(),
            mask2[j : j + 1].cpu(),
        )

    # ── te3 mask-pair packing (the two int64 masks share one disk slot) ──

    def _pack_masks(self, mask: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """Concatenate ``(mask [1,L1], mask2 [1,L2])`` → ``[1, L1+L2]`` int64."""
        return torch.cat([mask.to(torch.int64), mask2.to(torch.int64)], dim=1)

    def _unpack_masks(
        self, packed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a te3 tensor back into ``(mask, mask2)`` — the ByT5 mask is
        the FIXED-width tail (``te2.max_length``, default 256)."""
        te2_len = int(getattr(self.driver, "te2_max_length", 256))
        return packed[:, :-te2_len], packed[:, -te2_len:]

    # ── Text-embedding pre-cache (disk triple, te1-last commit marker) ────

    def _pre_cache_text_embeddings(self) -> None:
        """Warm the text cache (disk + memory) before the dual TE is offloaded.

        ``run_trainer`` runs this → ``_offload_text_encoders``; the base
        pre-cache is a no-op, so without this override the 7B Qwen2.5-VL is
        offloaded with an EMPTY cache and the first training step (and the
        sampler) would crash on the ``None`` encoder.

        Disk triple (LTX-2 precedent): te1 = Qwen embedding, te2 = ByT5 glyph
        embedding, te3 = the packed mask pair — keyed on the caption hash under
        ``{ds}/.cache/{model}/{ver}/embeddings/{te_quant}/{te1,te2,te3}/``.
        te1 is written LAST so its presence is the commit marker: a crash
        mid-write leaves te1 absent (clean miss next run) instead of poisoning
        the cache with a partial tuple. A te1 hit with te2/te3 missing is
        treated as a MISS (partial triple from an interrupted run).

        No-quote captions cache a ZERO te2 (+ zero tail in te3) — cheap, and
        round-tripping them through the same triple keeps the load path
        uniform (the zero-tensor path is pinned by a dedicated test).
        """
        if not self.config.get("cache_text_embeddings", True):
            return
        if self.driver.text_encoder is None:
            return

        from app.engine.components.text_embeddings import TextEmbeddingCache

        te_cache_dirs = self._resolve_te_cache_dirs()
        te_quant = self.config.get("te_quantization", "none")

        def _slot_dir(slot: str) -> str:
            return (
                os.path.join(te_cache_dirs[0], "embeddings", te_quant, slot)
                if te_cache_dirs
                else ""
            )

        te1_dir, te2_dir, te3_dir = _slot_dir("te1"), _slot_dir("te2"), _slot_dir("te3")

        dtype = self._resolve_loading_dtype()

        # ── Full ordered work set: training captions, expanded sample
        # prompts, then the CFG negative ("" = standard unconditional). ──
        work: list[tuple[str, str]] = []
        seen: set[str] = set()

        def _add(cap: str, hint: str) -> None:
            if cap in self.text_cache or cap in seen:
                return
            seen.add(cap)
            work.append((cap, hint))

        # NOTE: _build_caption_hints() already expands DETERMINISTIC sample
        # prompts; the explicit _sample_prompt_texts() loop additionally
        # catches RANDOM-wildcard expansions (re-rolled independently).
        for cap, hint in self._build_caption_hints().items():
            _add(cap, hint)
        sample_texts = self._sample_prompt_texts()
        for sp in sample_texts:
            _add(sp, "")
        if sample_texts:
            _add(str(self.config.get("sample_negative_prompt", "") or ""), "")

        # ── Phase 1: load the triple from disk (te1 presence gates the hit) ──
        disk_loaded = 0
        need_encode: list[tuple[str, str]] = []
        for cap, hint in work:
            if te1_dir:
                emb = TextEmbeddingCache.load(cap, te1_dir, hint)
                if emb is not None:
                    emb2 = (
                        TextEmbeddingCache.load(cap, te2_dir, hint) if te2_dir else None
                    )
                    packed = (
                        TextEmbeddingCache.load(cap, te3_dir, hint) if te3_dir else None
                    )
                    partial = (te2_dir and emb2 is None) or (te3_dir and packed is None)
                    if not partial:
                        mask, mask2 = self._unpack_masks(packed)
                        self.text_cache[cap] = (emb, mask, emb2, mask2)
                        disk_loaded += 1
                        continue
                    self.logger.warning(
                        "hv15_partial_triple_treated_as_miss",
                        caption_hash=hashlib.sha256(cap.encode("utf-8")).hexdigest()[
                            :16
                        ],
                        hint=hint,
                    )
            need_encode.append((cap, hint))

        if not need_encode:
            if getattr(self, "_log_writer", None):
                self._log_writer.status("TE Cache Loaded from Disk")
            self.logger.info(
                "hv15_text_cache_complete",
                cached=len(self.text_cache),
                from_disk=disk_loaded,
                source="disk",
            )
            return

        # ── Phase 2: encode the misses (batched) + persist the triple ──
        if getattr(self, "_log_writer", None):
            self._log_writer.status("Caching Text Embeddings (0%)")

        total = len(need_encode)
        batch_size = 4
        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch_items = need_encode[i : i + batch_size]
                chunk = [cap for cap, _ in batch_items]
                out = self.driver.encode_text(chunk, dtype)
                for j, (cap, hint) in enumerate(batch_items):
                    emb, mask, emb2, mask2 = self._slice_te_output(out, j)
                    self.text_cache[cap] = (emb, mask, emb2, mask2)
                    # Save order matters: te3/te2 first, te1 LAST (commit
                    # marker — see the docstring).
                    if te3_dir:
                        TextEmbeddingCache.save(
                            cap, self._pack_masks(mask, mask2), te3_dir, hint
                        )
                    if te2_dir:
                        TextEmbeddingCache.save(cap, emb2, te2_dir, hint)
                    if te1_dir:
                        TextEmbeddingCache.save(cap, emb, te1_dir, hint)
                if getattr(self, "_log_writer", None):
                    pct = round(min(i + batch_size, total) / total * 100)
                    self._log_writer.status(f"Caching Text Embeddings ({pct}%)")

        self.logger.info(
            "hv15_text_cache_complete",
            cached=len(self.text_cache),
            from_disk=disk_loaded,
            newly_encoded=total,
        )

    def _sample_prompt_texts(self) -> list[str]:
        """Expanded sample-prompt strings to pre-cache (shared expansion helper
        so the sampler's later cache key matches exactly)."""
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

    # ── I2V Siglip image-embedding aux cache (data path) ──────────────────

    @staticmethod
    def _siglip_cache_dir(video_cache_dir: str) -> str:
        """Siglip embeddings live in a ``siglip/`` sibling of the latent dir.

        A separate content-addressed file per item means enabling i2v never
        disturbs an existing latent cache (LTX-2 audio-cache precedent).
        """
        return os.path.join(video_cache_dir, "siglip")

    def _pre_cache_aux(self) -> None:
        """Pre-cache Siglip first-frame image embeddings for I2V runs.

        Runs right after ``_pre_cache_latents`` while the Siglip image encoder
        is resident. For each item the FIRST FRAME of its trim window (or the
        still image itself) is preprocessed through the checkpoint's
        ``SiglipImageProcessor`` and encoded to ``last_hidden_state``
        ``[729, 1152]`` — the ``image_embeds`` the transformer expects for
        I2V (upstream ``_get_image_embeds``). No-op for T2V runs.

        The image encoder is offloaded to CPU afterwards (it is unused during
        UNet training).
        """
        if not getattr(self.driver, "is_i2v", False):
            return
        if self.driver.image_encoder is None or self.driver.feature_extractor is None:
            return
        if not self.config.get("cache_latents", True):
            return

        from safetensors.torch import save_file

        from app.engine.core.pipeline.pipeline_data import video_trim_extra_key

        encoded = skipped = failed = 0
        for item in self.inventory:
            extra_key = video_trim_extra_key(item) if item.get("is_video") else ""
            sdir = self._siglip_cache_dir(item["cache_dir"])
            fname = self.latent_manager.latent_filename(
                item["id"], item["path"], extra_key
            )
            path = os.path.join(sdir, fname)
            if os.path.exists(path):
                skipped += 1
                continue

            pil = self._load_first_frame_pil(item)
            if pil is None:
                failed += 1
                continue

            try:
                with torch.no_grad():
                    inputs = self.driver.feature_extractor.preprocess(
                        images=pil,
                        do_resize=True,
                        return_tensors="pt",
                        do_convert_rgb=True,
                    )
                    ie_dtype = next(self.driver.image_encoder.parameters()).dtype
                    pixel_values = inputs["pixel_values"].to(self.device, dtype=ie_dtype)
                    emb = self.driver.image_encoder(
                        pixel_values=pixel_values
                    ).last_hidden_state  # [1, 729, 1152]
                os.makedirs(sdir, exist_ok=True)
                save_file({"image_embeds": emb[0].detach().cpu()}, path)
                encoded += 1
            except Exception as e:  # noqa: BLE001 — a bad item must not kill the run
                # Graceful path: leave this item uncached — build_batch_extra
                # will ZERO-FILL its image_embeds (image stream inactive). That
                # degradation must be VISIBLE, never silent.
                failed += 1
                self.logger.warning(
                    "hv15_siglip_encode_failed",
                    path=item.get("path"),
                    error=str(e),
                )

        if failed:
            # Surface the zero-fill exposure loudly — a run with N uncached
            # items trains those items with ZERO image_embeds (I2V conditioning
            # effectively off for them), which is otherwise silent.
            self.logger.warning(
                "hv15_siglip_precache_incomplete",
                failed=failed,
                encoded=encoded,
                skipped=skipped,
                hint="items without a cached Siglip embedding train with ZERO "
                     "image_embeds (image stream inactive for those items)",
            )
        self.logger.info(
            "hv15_siglip_precache_done",
            encoded=encoded,
            skipped=skipped,
            failed=failed,
        )

        if failed and not encoded:
            # TOTAL failure: every item that needed encoding failed and NONE
            # succeeded. Proceeding would train a 100% zero-image_embeds run —
            # the image stream is inactive for every item, i.e. a silently
            # mislabeled T2V run. Escalate loudly instead of degrading silently.
            raise RuntimeError(
                f"hv15 Siglip precache produced ZERO image embeddings: all "
                f"{failed} item(s) failed to encode "
                f"(hv15_siglip_precache_incomplete). An I2V run with no "
                f"image_embeds trains as T2V — refusing to proceed."
            )

        # Offload the image encoder — unused during UNet training.
        image_encoder = self.driver.image_encoder
        if image_encoder is not None and hasattr(image_encoder, "to"):
            image_encoder.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load_first_frame_pil(self, item: dict):
        """First frame of a clip's trim window (or the still itself) as PIL RGB.

        The Siglip ``feature_extractor`` does its own 384×384 resize from the
        raw frame — no smart-resize/crop here (upstream feeds the original
        PIL image).
        """
        from PIL import Image

        try:
            if not item.get("is_video"):
                return Image.open(item["path"]).convert("RGB")

            import av

            trim_start = float(item.get("trim_start_s") or 0.0)
            container = av.open(str(item["path"]))
            try:
                stream = container.streams.video[0]
                time_base = float(stream.time_base) if stream.time_base else None
                if time_base and trim_start > 0:
                    try:
                        container.seek(
                            int(trim_start / time_base),
                            stream=stream,
                            backward=True,
                            any_frame=False,
                        )
                    except (OSError, ValueError):
                        pass
                for frame in container.decode(stream):
                    t = (
                        float(frame.time)
                        if frame.time is not None
                        else (float(frame.pts) * time_base if frame.pts and time_base else 0.0)
                    )
                    if t >= trim_start - 1e-6:
                        return Image.fromarray(
                            frame.to_ndarray(format="rgb24"), mode="RGB"
                        )
                return None
            finally:
                try:
                    container.close()
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001 — a bad item must not kill the run
            self.logger.warning(
                "hv15_first_frame_load_failed", path=item.get("path"), error=str(e)
            )
            return None

    def build_batch_extra(self, items: list[dict]) -> dict[str, Any]:
        """Attach cached Siglip image embeddings to the I2V training batch.

        Returns ``{"hv15_image_embed": [B, 729, 1152]}``. Items without a
        cached embedding get ZEROS (the transformer's all-zero detection
        treats their image stream as inactive; the first-frame cond channels
        still condition). T2V runs return ``{}``.
        """
        if not getattr(self.driver, "is_i2v", False):
            return {}
        if not self.config.get("cache_latents", True):
            return {}

        from safetensors.torch import load_file

        from app.engine.core.pipeline.pipeline_data import video_trim_extra_key

        embeds: list[torch.Tensor | None] = []
        for item in items:
            emb: torch.Tensor | None = None
            extra_key = video_trim_extra_key(item) if item.get("is_video") else ""
            sdir = self._siglip_cache_dir(item["cache_dir"])
            fname = self.latent_manager.latent_filename(
                item["id"], item["path"], extra_key
            )
            path = os.path.join(sdir, fname)
            if os.path.exists(path):
                try:
                    emb = load_file(path)["image_embeds"]
                except (OSError, KeyError) as e:
                    self.logger.warning(
                        "hv15_siglip_cache_load_failed", path=path, error=str(e)
                    )
            embeds.append(emb)

        ref = next((e for e in embeds if e is not None), None)
        if ref is None:
            return {}  # nothing cached → driver falls back to zeros

        filled = [
            e if (e is not None and e.shape == ref.shape) else torch.zeros_like(ref)
            for e in embeds
        ]
        return {
            Hv15Driver.BATCH_IMAGE_EMBED: torch.stack(filled).to(self.device)
        }
