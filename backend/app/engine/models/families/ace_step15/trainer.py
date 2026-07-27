"""ACE-Step 1.5 Trainer — family hooks for the generic training pipeline.

Implements the ACE-Step-specific behaviour on top of
:class:`GenericTrainingPipeline`:

- **Composite (caption, lyrics) TE cache**: unlike every image/video family,
  the conditioning sequence depends on TWO independent per-item text fields
  (caption AND the lyrics sidecar). Cache keys compose both
  (:meth:`_compose_key`) so two items sharing a caption but differing lyrics
  (or vice versa) never collide. Disk layout mirrors the LTX-2/Kandinsky5
  slot convention: ``{ds}/.cache/{model}/{ver}/embeddings/{te_quant}/{te1,te2}/``
  — te1 = ``encoder_hidden_states`` (the condition encoder's FINAL packed
  cross-attn sequence — cached post-condition-encoder, not raw TE output, so
  the pre-cache also warms/exercises the condition encoder before it's
  offloaded), te2 = ``encoder_attention_mask``. te1 saved LAST (commit marker,
  LTX-2 precedent).
- **Variable-length reassembly**: each item is encoded ALONE (batch=1) so its
  cached embedding is exactly its own true length; :meth:`encode_text` pads
  cached entries to the batch's max length when reassembling — byte-
  equivalent to encoding the batch fresh with the pipeline's own
  ``padding="longest"`` tokenization (Kandinsky5 precedent). Validated at
  ``train_batch_size: 1`` (the definition's shipped default, matching
  upstream's own LoRA preset — see recon report §2); batch_size > 1 pads on
  the right with zero rows since ``AceStepTransformer1DModel.forward`` has NO
  attention-mask parameter to exclude them (an upstream architecture trait,
  not something this driver introduces — see driver.py's condition-encoding
  docstring).
- The trainer-override trio (``_setup_family`` / ``_create_sampler`` /
  ``_update_primary_model``) — the seam-contract ``FamilySpec`` pins them.
"""

from __future__ import annotations

import os

import structlog
import torch
import torch.nn.functional as F

from app.engine.core.pipeline import GenericTrainingPipeline

from .driver import AceStep15Driver
from .loader import AceStep15Loader

logger = structlog.get_logger(__name__)

# Rare separator unlikely to appear in either a caption or a lyrics sidecar —
# composes the two-field cache key. U+241E (SYMBOL FOR RECORD SEPARATOR).
_KEY_SEP = "␞"


class AceStep15Trainer(GenericTrainingPipeline):
    """ACE-Step 1.5 (text2music) LoRA trainer."""

    # ── Setup (override trio 1/3) ─────────────────────────────────────────

    def _setup_family(self) -> None:
        self.driver = AceStep15Driver(self.definition, self.device)
        self.loader = AceStep15Loader(self.device)
        # genre_ratio lives in the run config, not architecture_params — the
        # driver has no `self.config` (only definition + device, per the
        # house IModelDriver contract), so the trainer wires it explicitly.
        self.driver.genre_ratio = float(self.config.get("genre_ratio", 0.15) or 0.0)

    def _create_sampler(self):
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import AceStep15Sampler

            return AceStep15Sampler(self)
        return None

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep transformer references in sync after PEFT/quant wrapping."""
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

    # ── Composite (caption, lyrics) key ────────────────────────────────────

    @staticmethod
    def _compose_key(caption: str, lyrics: str) -> str:
        return f"{caption}{_KEY_SEP}{lyrics}"

    # ── Text/condition-embedding cache (warm before TE+condition_encoder
    #    are offloaded) ──────────────────────────────────────────────────

    def _pre_cache_text_embeddings(self) -> None:
        """Warm the (caption, lyrics) -> condition cache before offload.

        ``run_trainer`` runs this -> ``_offload_text_encoders``; the base warm
        step is a no-op, so without this override the text_encoder +
        condition_encoder are offloaded with an EMPTY cache and the first
        training step (and every sampling round) has nothing cached.
        """
        if not self.config.get("cache_text_embeddings", True):
            return
        if self.driver.text_encoder is None:
            return

        from app.engine.components.text_embeddings import TextEmbeddingCache
        from app.engine.core.pipeline.caption_selection import select_training_caption
        from app.engine.core.sampling import expand_prompt_wildcards

        te_cache_dirs = self._resolve_te_cache_dirs()
        te_quant = self.config.get("te_quantization", "none")

        def _slot_dir(slot: str) -> str:
            return (
                os.path.join(te_cache_dirs[0], "embeddings", te_quant, slot)
                if te_cache_dirs
                else ""
            )

        te1_dir, te2_dir = _slot_dir("te1"), _slot_dir("te2")
        dtype = self._resolve_loading_dtype()
        _def_id = getattr(self.definition, "id", None)

        # ── Full ordered work set: training (caption, lyrics) pairs (incl. a
        # caption-dropout variant that KEEPS the item's own lyrics — dropout
        # only blanks the caption text), then sample prompts + their lyrics ──
        work: list[tuple[str, str, str]] = []
        seen: set[str] = set()

        def _add(cap: str, lyr: str, hint: str) -> None:
            key = self._compose_key(cap, lyr)
            if key in self.text_cache or key in seen:
                return
            seen.add(key)
            work.append((cap, lyr, hint))

        for item in self.inventory:
            cap = select_training_caption(item, _def_id)
            lyr = item.get("lyrics_content", "") or ""
            img_name = os.path.splitext(os.path.basename(item.get("path", "")))[0]
            _add(cap, lyr, img_name)
            _add("", lyr, f"{img_name}_dropout")

        for idx, sp in enumerate(self.config.get("sample_prompts", []) or []):
            prompt = (
                sp.get("prompt", "") if isinstance(sp, dict) else getattr(sp, "prompt", "")
            )
            lyrics = (
                sp.get("lyrics") if isinstance(sp, dict) else getattr(sp, "lyrics", None)
            ) or ""
            expanded = expand_prompt_wildcards(prompt, self.config) if prompt else ""
            if expanded or lyrics:
                _add(expanded, lyrics, f"sample_{idx}")

        # ── Phase 1: disk load (te1 presence gates the hit — LTX-2 precedent) ──
        disk_loaded = 0
        need_encode: list[tuple[str, str, str]] = []
        for cap, lyr, hint in work:
            key = self._compose_key(cap, lyr)
            if te1_dir:
                eh = TextEmbeddingCache.load(key, te1_dir, hint)
                if eh is not None:
                    em = TextEmbeddingCache.load(key, te2_dir, hint) if te2_dir else None
                    if not (te2_dir and em is None):
                        self.text_cache[key] = (eh, em)
                        disk_loaded += 1
                        continue
            need_encode.append((cap, lyr, hint))

        if not need_encode:
            if getattr(self, "_log_writer", None):
                self._log_writer.status("TE Cache Loaded from Disk")
            self.logger.info(
                "ace_step15_text_cache_complete",
                cached=len(self.text_cache),
                from_disk=disk_loaded,
                source="disk",
            )
            return

        # ── Phase 2: encode the misses (batch=1 each — see module docstring
        # for why per-item encoding, not a batched "longest" tokenize) ──
        if getattr(self, "_log_writer", None):
            self._log_writer.status("Caching Text Embeddings (0%)")

        total = len(need_encode)
        audio_duration = float(self.config.get("duration_s", 30.0) or 30.0)
        with torch.no_grad():
            for i, (cap, lyr, hint) in enumerate(need_encode):
                eh, em = self.driver.encode_condition(
                    [cap], [lyr], dtype, audio_duration=audio_duration
                )
                key = self._compose_key(cap, lyr)
                eh_cpu, em_cpu = eh.cpu(), em.cpu()
                self.text_cache[key] = (eh_cpu, em_cpu)
                # Save order matters: te2 first, te1 LAST (commit marker).
                if te2_dir:
                    TextEmbeddingCache.save(key, em_cpu, te2_dir, hint)
                if te1_dir:
                    TextEmbeddingCache.save(key, eh_cpu, te1_dir, hint)
                if getattr(self, "_log_writer", None) and (i + 1) % max(total // 20, 1) == 0:
                    pct = round((i + 1) / total * 100)
                    self._log_writer.status(f"Caching Text Embeddings ({pct}%)")

        self.logger.info(
            "ace_step15_text_cache_complete",
            cached=len(self.text_cache),
            from_disk=disk_loaded,
            newly_encoded=total,
        )

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reassemble a batched condition tuple from the warm (caption, lyrics)
        cache — pads cached per-item entries to the batch's max length
        (``padding="longest"`` semantics, matching a fresh batched encode).
        """
        lyrics = (batch or {}).get("lyrics") or [""] * len(captions)
        if not self.config.get("cache_text_embeddings", True):
            audio_duration = float(self.config.get("duration_s", 30.0) or 30.0)
            return self.driver.encode_condition(
                captions, lyrics, dtype, audio_duration=audio_duration
            )

        entries: list[tuple[torch.Tensor, torch.Tensor]] = []
        for cap, lyr in zip(captions, lyrics):
            key = self._compose_key(cap, lyr)
            entry = self.text_cache.get(key)
            if entry is None:
                if self.driver.text_encoder is None:
                    raise RuntimeError(
                        "Text/condition encoder offloaded and (caption, "
                        f"lyrics) not pre-cached: {cap[:60]!r}"
                    )
                audio_duration = float(self.config.get("duration_s", 30.0) or 30.0)
                eh, em = self.driver.encode_condition(
                    [cap], [lyr], dtype, audio_duration=audio_duration
                )
                entry = (eh.cpu(), em.cpu())
                self.text_cache[key] = entry
            entries.append(entry)

        lengths = [e[0].shape[1] for e in entries]
        max_len = max(lengths) if lengths else 0
        eh_padded: list[torch.Tensor] = []
        em_padded: list[torch.Tensor] = []
        for (eh, em), length in zip(entries, lengths):
            e = eh.to(self.device, dtype=dtype)
            m = em.to(self.device) if em is not None else None
            if length < max_len:
                e = F.pad(e, (0, 0, 0, max_len - length))
                if m is not None:
                    m = F.pad(m, (0, max_len - length), value=False)
            eh_padded.append(e)
            if m is not None:
                em_padded.append(m)

        encoder_hidden_states = torch.cat(eh_padded, dim=0)
        encoder_attention_mask = (
            torch.cat(em_padded, dim=0) if len(em_padded) == len(entries) else None
        )
        return encoder_hidden_states, encoder_attention_mask
