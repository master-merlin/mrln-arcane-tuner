"""Kandinsky 5.0 Trainer — family hooks for the generic training pipeline.

Implements the Kandinsky-specific behaviour on top of
:class:`GenericTrainingPipeline`:

- **Dual-TE disk/memory cache triple** (LTX-2 precedent): each unique caption
  caches ``(te1=Qwen sequence emb, te2=CLIP pooled, te3=cu_seqlens)`` — te1 is
  written LAST so its presence is the commit marker for the whole triple (a
  crash mid-write leaves a clean miss, never a poisoned partial hit).
- Per-caption te1 entries are trimmed to the caption's TRUE length (the driver
  returns pipeline-faithful ``padding="longest"`` batches); reassembly pads to
  the batch max and rebuilds ``cu_seqlens`` — byte-equivalent to encoding the
  batch fresh.
- **I2V gating**: per-step Bernoulli on ``first_frame_conditioning_probability``
  (default 1.0 — the I2V checkpoints condition every step at inference) when
  ``video_mode == "i2v"``.
- **I2V loss exclusion**: frame 0 is the clean conditioning frame (its
  velocity target is meaningless) — ``_compute_step_loss`` drops it, mirroring
  the upstream I2V pipeline's frame-0 scheduler skip (LTX-2's excluded-token
  precedent).
- **``add_noise`` auto-delegation** (W5.T10 — no trainer override needed):
  ``PipelineBaseMixin.add_noise`` auto-delegates to ``driver.add_noise``
  whenever the driver meaningfully overrides it
  (``core/hook_dispatch.py``/``pipeline_base.py:176-231``) — the I2V
  frame-0-clean pin reaches the REAL training loop's ``self.add_noise(...)``
  family-hook call structurally, with no per-family delegation method to
  keep in sync (see ``test_kandinsky5_addnoise_wiring.py``).
- The trainer-override trio (``_setup_family`` / ``_create_sampler`` /
  ``_update_primary_model``) — the seam-contract ``FamilySpec`` pins them.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from app.engine.core.text_encoding import TextEncoderOutput

from .driver import Kandinsky5Driver, build_cu_seqlens, resolve_negative_prompt
from .loader import Kandinsky5Loader

logger = structlog.get_logger(__name__)


class Kandinsky5Trainer(GenericTrainingPipeline):
    """Kandinsky 5.0 (T2V Lite / I2V Pro) LoRA trainer."""

    # ── Setup (override trio 1/3) ─────────────────────────────────────────

    def _setup_family(self) -> None:
        self.driver = Kandinsky5Driver(self.definition, self.device)
        self.loader = Kandinsky5Loader(self.device)

    def _create_sampler(self):
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import Kandinsky5Sampler

            return Kandinsky5Sampler(self)
        return None

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep transformer references in sync after PEFT/quant wrapping.

        The base method does NOT sync the driver — forgetting this strands the
        driver on the unwrapped graph (the historical krea2 bug class the
        seam-contract test pins).
        """
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

    # ── I2V per-step gate ─────────────────────────────────────────────────

    def _attach_conditioning(self, batch: dict, latents: torch.Tensor) -> None:
        """Set the driver's per-step i2v flag, then stash the first frame.

        I2V is active when ``video_mode == "i2v"`` AND a Bernoulli draw with
        ``first_frame_conditioning_probability`` succeeds. Default probability
        is 1.0 — the Kandinsky I2V checkpoints are always image-conditioned at
        inference (the mask channel signals conditioning), unlike the LTX
        mixed recipe. Stills (F == 1) disengage inside the driver.
        """
        active = False
        if str(self.config.get("video_mode", "t2v")).lower() == "i2v":
            p = float(self.config.get("first_frame_conditioning_probability", 1.0))
            active = random.random() < p
        self.driver._i2v_active = active
        if active:
            self.driver.attach_conditioning(batch, latents)

    # ── I2V frame-0 loss exclusion ────────────────────────────────────────

    def _compute_step_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        batch: dict[str, Any],
        grad_accum: int,
    ) -> torch.Tensor:
        """Plain flow-match MSE; engaged I2V drops frame 0 from BOTH sides.

        pred/target are channels-last ``[B, F, H, W, C]`` — frame 0 is the
        clean conditioning frame (never noised; the upstream I2V pipeline's
        scheduler step updates ``latents[:, 1:]`` only), so its "velocity" is
        not a training signal.
        """
        if self.driver._i2v_conditioning_engaged():
            pred = pred[:, 1:]
            target = target[:, 1:]
        return super()._compute_step_loss(
            pred, target, timesteps, batch, grad_accum
        )

    # ── Text-embedding cache (warm before the dual TE is offloaded) ──────

    def _pre_cache_text_embeddings(self) -> None:
        """Warm the text cache (disk + memory) before both TEs are offloaded.

        ``run_trainer`` runs this → ``_offload_text_encoders``; the base warm
        step is a no-op, so without this override the 7B Qwen2.5-VL + CLIP are
        offloaded with an EMPTY cache and the first step (and every sampling
        round) has no embeddings.

        Disk layout mirrors LTX-2's triple:
        ``{ds}/.cache/{model}/{ver}/embeddings/{te_quant}/{te1,te2,te3}/`` —
        te1 = Qwen sequence emb (trimmed to true length), te2 = CLIP pooled,
        te3 = cu_seqlens ``[0, L]`` int32. te1 is saved LAST (commit marker):
        a partial triple on disk is treated as a miss and re-encoded.
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
        # prompts, then the CFG negative (pipeline default when unset). ──
        work: list[tuple[str, str]] = []
        seen: set[str] = set()

        def _add(cap: str, hint: str) -> None:
            if cap in self.text_cache or cap in seen:
                return
            seen.add(cap)
            work.append((cap, hint))

        # NOTE: _build_caption_hints() already expands DETERMINISTIC sample
        # prompts; the explicit _sample_prompt_texts() loop additionally
        # catches RANDOM-wildcard expansions (dedup via `seen`).
        for cap, hint in self._build_caption_hints().items():
            _add(cap, hint)
        sample_texts = self._sample_prompt_texts()
        for sp in sample_texts:
            _add(sp, "")
        if sample_texts:
            # Kandinsky previews run true CFG (T2V default 5.0) and the
            # pipeline INJECTS a default negative when none is set — warm the
            # exact string the sampler will later request.
            _add(resolve_negative_prompt(self.config), "")

        # ── Phase 1: load the triple from disk (te1 presence gates the hit;
        # a partial triple — te2/te3 missing — is a MISS, see LTX-2) ──
        disk_loaded = 0
        need_encode: list[tuple[str, str]] = []
        for cap, hint in work:
            if te1_dir:
                emb = TextEmbeddingCache.load(cap, te1_dir, hint)
                if emb is not None:
                    pooled = (
                        TextEmbeddingCache.load(cap, te2_dir, hint) if te2_dir else None
                    )
                    cu = (
                        TextEmbeddingCache.load(cap, te3_dir, hint) if te3_dir else None
                    )
                    partial = (te2_dir and pooled is None) or (te3_dir and cu is None)
                    if not partial:
                        self.text_cache[cap] = (emb, pooled, cu)
                        disk_loaded += 1
                        continue
                    self.logger.warning(
                        "kandinsky5_partial_triple_treated_as_miss",
                        caption_hash=hashlib.sha256(
                            cap.encode("utf-8")
                        ).hexdigest()[:16],
                        hint=hint,
                    )
            need_encode.append((cap, hint))

        if not need_encode:
            if getattr(self, "_log_writer", None):
                self._log_writer.status("TE Cache Loaded from Disk")
            self.logger.info(
                "kandinsky5_text_cache_complete",
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
                    emb, pooled, cu = self._slice_te_triple(out, j)
                    self.text_cache[cap] = (emb, pooled, cu)
                    # Save order matters: te3/te2 first, te1 LAST (te1 is the
                    # Phase-1 disk-hit gate → the triple's commit marker).
                    if te3_dir and cu is not None:
                        TextEmbeddingCache.save(cap, cu, te3_dir, hint)
                    if te2_dir and pooled is not None:
                        TextEmbeddingCache.save(cap, pooled, te2_dir, hint)
                    if te1_dir:
                        TextEmbeddingCache.save(cap, emb, te1_dir, hint)
                if getattr(self, "_log_writer", None):
                    pct = round(min(i + batch_size, total) / total * 100)
                    self._log_writer.status(f"Caching Text Embeddings ({pct}%)")

        self.logger.info(
            "kandinsky5_text_cache_complete",
            cached=len(self.text_cache),
            from_disk=disk_loaded,
            newly_encoded=total,
        )

    def _sample_prompt_texts(self) -> list[str]:
        """Expanded sample-prompt strings to pre-cache (sampler cache keys)."""
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

    @staticmethod
    def _slice_te_triple(
        out: TextEncoderOutput, j: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Item ``j`` of a TE batch as CPU ``(emb, pooled, cu_seqlens)``.

        The driver returns pipeline-faithful ``padding="longest"`` batches, so
        item ``j``'s embedding is TRIMMED to its true length (from the batch
        cu_seqlens) before caching — a per-caption entry never carries another
        caption's padding. Its te3 is the caption's OWN ``[0, L]`` int32.
        """
        cu = out.attention_mask  # [B+1] int32 cumulative true lengths
        if cu is not None:
            length = int(cu.diff()[j].item())
            emb = out.embeddings[j : j + 1, :length].cpu()
            cu_j = build_cu_seqlens([length]).cpu()
        else:  # pragma: no cover - defensive (driver always returns cu)
            emb = out.embeddings[j : j + 1].cpu()
            cu_j = build_cu_seqlens([emb.shape[1]]).cpu()
        pooled = out.pooled
        return (
            emb,
            pooled[j : j + 1].cpu() if pooled is not None else None,
            cu_j,
        )

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None
    ) -> TextEncoderOutput:
        """Reassemble a batched :class:`TextEncoderOutput` from the warm cache.

        Cached per-caption embeddings are exact-length; the batch is padded to
        the max TRUE length (``padding="longest"`` semantics) and cu_seqlens is
        rebuilt from the true lengths — matching a fresh batch encode. Caching
        off → encode directly via the driver. A miss after TE offload is a
        hard error (the pre-cache must cover every caption).
        """
        if not self.config.get("cache_text_embeddings", True):
            return self.driver.encode_text(captions, dtype)

        embs: list[torch.Tensor] = []
        pooleds: list[torch.Tensor | None] = []
        lengths: list[int] = []
        for cap in captions:
            entry = self.text_cache.get(cap)
            if entry is None:
                te = self.driver.text_encoder
                if te is None:
                    raise RuntimeError(
                        "Text encoder offloaded and caption not pre-cached: "
                        f"{cap[:60]!r}"
                    )
                # Guard: a cache miss can hit with the TE merely CPU-resident
                # (unload_text_encoder: False only offloads — it doesn't null
                # driver.text_encoder), not fully unloaded. Bracket to GPU for
                # the encode, matching the canonical qwen_image miss path.
                te_was_offloaded = False
                if isinstance(te, torch.nn.Module):
                    te_was_offloaded = next(te.parameters()).device != self.device
                    if te_was_offloaded:
                        self.logger.warning(
                            "te_cache_miss_after_offload",
                            hint="pre-caching should have covered all captions",
                        )
                        te.to(self.device)
                out = self.driver.encode_text([cap], dtype)
                if te_was_offloaded:
                    te.to("cpu")
                    torch.cuda.empty_cache()
                entry = self._slice_te_triple(out, 0)
                self.text_cache[cap] = entry
            emb_c, pooled_c, cu_c = entry
            embs.append(emb_c)
            pooleds.append(pooled_c)
            lengths.append(
                int(cu_c.diff()[0].item()) if cu_c is not None else emb_c.shape[1]
            )

        max_len = max(lengths) if lengths else 0
        padded: list[torch.Tensor] = []
        for emb_c, length in zip(embs, lengths):
            e = emb_c.to(self.device, dtype=dtype)
            if length < max_len:
                e = torch.nn.functional.pad(e, (0, 0, 0, max_len - length))
            padded.append(e)
        embeddings = torch.cat(padded, dim=0)

        pooled = None
        if all(p is not None for p in pooleds):
            pooled = torch.cat(
                [p.to(self.device, dtype=dtype) for p in pooleds], dim=0
            )
        cu_seqlens = build_cu_seqlens(lengths).to(self.device)
        return TextEncoderOutput(
            embeddings=embeddings, attention_mask=cu_seqlens, pooled=pooled
        )
