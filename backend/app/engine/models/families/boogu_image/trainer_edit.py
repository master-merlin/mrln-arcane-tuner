"""BooguImageEditTrainer — image-conditioned ("edit") variant (task A4 fix wave).

Subclasses :class:`BooguImageTrainer`. Unlike qwen_image_edit/flux1-kontext
(which patchify + sequence-concat the control latents in the TRAINER), the
Boogu transformer consumes the clean control latents natively via
``ref_image_hidden_states`` — that wiring lives in
``BooguImageDriver.forward_pass`` / ``_build_ref_image_hidden_states`` and
needs NO trainer-level forward override. What THIS subclass owns is the
OTHER half of the upstream TI2I conditioning (finding 1 of the A4 review):

**VL-side control-image encoding.** Upstream's
``_get_instruction_feature_embeds`` (pipeline_boogu.py:1181-1572) feeds the
reference image(s) into the Qwen3-VL chat template (image content entries
BEFORE the instruction text, TI2I system prompt) so the text encoder ATTENDS
the control pixels. Mirrored here via
``BooguImageDriver.encode_text_with_images`` (the vendored minimal
equivalent of upstream's with-image template branch + dataset-matching image
preprocessing). Because the embeddings now depend on **(caption, control
image)** jointly, the TE cache is keyed compositely
(:func:`composite_te_key` — same convention as
``qwen_image/trainer_edit.py``, helpers duplicated locally rather than
imported cross-family to keep family import graphs independent) so the same
instruction over different controls never collides. When the processor or
text encoder is unavailable the encode falls back to the text-only path
with a single loud warning (the qwen degraded-mode precedent) — the control
still conditions the transformer via ``ref_image_hidden_states``.

Disk pre-caching is skipped for edit runs (composite keys encode lazily at
first use; the base pre-cache keys by plain caption and would never be hit).
The TE therefore stays resident for the run — acceptable for the paired-edit
task, matching qwen_image_edit's documented tradeoff.
"""

from __future__ import annotations

import hashlib

import structlog
import torch

from .trainer import BooguImageTrainer

logger = structlog.get_logger(__name__)


# ── Pure helpers (qwen_image/trainer_edit.py convention, local copies) ─────


def composite_te_key(caption: str, control_hash: str) -> str:
    """TE cache key for an edit run — embeddings depend on the control too.

    Same caption + different control image → distinct key (prevents the
    silent bug where every edit shares one text embedding).
    """
    return f"{caption}||ctl:{control_hash}"


def control_files_hash(paths: list[str], memo: dict[str, bytes] | None = None) -> str:
    """Stable 16-hex digest of one item's control image(s).

    Hashes file bytes (so the key survives across runs); falls back to the
    path string if a file can't be read. ``memo`` caches per-path digests.
    """
    memo = memo if memo is not None else {}
    h = hashlib.sha256()
    for p in paths:
        digest = memo.get(p)
        if digest is None:
            try:
                with open(p, "rb") as f:
                    digest = hashlib.sha256(f.read()).digest()
            except OSError:
                digest = hashlib.sha256(p.encode("utf-8")).digest()
            memo[p] = digest
        h.update(digest)
    return h.hexdigest()[:16]


class BooguImageEditTrainer(BooguImageTrainer):
    """Boogu-Image Edit trainer — VL control-image encoding + composite TE cache.

    Transformer-side control conditioning (``ref_image_hidden_states``) is
    already handled family-wide by ``BooguImageDriver.forward_pass`` off
    ``batch["control_latents"]`` — no forward override here.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ctrl_hash_memo: dict[str, bytes] = {}
        self._warned_no_vl_processor = False

    def _create_sampler(self):
        """Edit-aware previews (A4 review finding 2)."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler_edit import BooguImageEditSampler  # noqa: PLC0415

            return BooguImageEditSampler(self)
        return None

    # ── Text encoding (composite cache key + VL image path) ──────────────

    def _pre_cache_text_embeddings(self) -> None:
        """Edit runs encode lazily per (caption, control) at first use.

        The base disk pre-cache keys by plain caption (text-only); those
        entries would never be hit by the composite-keyed lookups, so we
        skip it (mirrors ``QwenImageEditTrainer``). The TE stays resident
        for the run — it must see the control image.
        """
        self.logger.info(
            "boogu_edit_te_precache_skipped",
            reason="composite (caption, control) keys encode lazily",
        )

    def _offload_text_encoders(self) -> None:
        """Keep the VL text encoder resident — enforcement of the
        "TE stays resident" contract above (mirrors ``QwenImageEditTrainer``;
        GPU UAT 2026-07-14). The shared base offload would move the TE to CPU
        and pop it from ``self.components``, stranding the first composite-key
        cache miss mid-training on a CPU encoder with CUDA inputs."""
        self.logger.info(
            "te_offload_skipped_edit_lazy_encode",
            reason="edit runs encode (caption, control) composites lazily "
                   "all run — TE must stay resident",
        )

    def _ensure_te_on_device(self) -> None:
        """Move the (CPU-loaded) VL text encoder to the trainer device.

        With pre-caching skipped, the base pre-cache — the implicit TE→GPU
        mover for every other trainer — never runs, so the first lazy
        cache-miss encode would feed CUDA input ids to a CPU encoder
        (mirrors ``QwenImageEditTrainer``; GPU UAT 2026-07-14). Idempotent:
        after the first move the encoder stays resident.
        """
        te = getattr(getattr(self, "driver", None), "text_encoder", None)
        if te is None:
            return
        param = next(te.parameters(), None)
        if param is not None and param.device != self.device:
            te.to(self.device)

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions, keyed by (caption, control image).

        Falls back to the text-only base path when no control context is
        available (e.g. a sampler preview negative prompt, or a partial
        batch) — behavior then matches ``BooguImageTrainer`` exactly.
        """
        ctrl_paths = (batch or {}).get("control_paths")
        if not ctrl_paths:
            return super().encode_text(captions, dtype, batch)

        n_slots = len(ctrl_paths)
        embeds: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for i, caption in enumerate(captions):
            item_controls = [ctrl_paths[s][i] for s in range(n_slots)]
            key = composite_te_key(
                caption, control_files_hash(item_controls, self._ctrl_hash_memo),
            )
            if key not in self.text_cache:
                self._ensure_te_on_device()
                emb, mask = self._encode_text_with_control(
                    caption, item_controls, dtype,
                )
                # Trimmed to true (mask) length — the base trainer's ragged
                # cache convention (padding="longest", per-batch padding must
                # not leak into cache entries).
                self.text_cache[key] = self._trim_entry(
                    emb.squeeze(0).cpu(), mask.squeeze(0).cpu(),
                )
            cached_emb, cached_mask = self.text_cache[key]
            embeds.append(cached_emb.to(self.device, dtype=dtype))
            masks.append(cached_mask.to(self.device))

        return self._stack_padded(embeds, masks)

    @staticmethod
    def _stack_padded(
        embeds: list[torch.Tensor], masks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad per-item (embedding, mask) to a common length, then stack.

        Zero-padding with mask=0 is equivalent to the direct encode path's
        processor-side ``padding="longest"`` batch (the model derives
        per-sample lengths from ``mask.sum(dim=1)``) — same reassembly rule
        as the base trainer's ``_get_cached_text_embeddings``.
        """
        max_len = max(e.size(0) for e in embeds)
        emb = torch.stack([
            torch.cat([e, e.new_zeros(max_len - e.size(0), e.size(1))])
            for e in embeds
        ])
        msk = torch.stack([
            torch.cat([m, m.new_zeros(max_len - m.size(0))]) for m in masks
        ])
        return emb, msk

    def _encode_text_with_control(
        self, caption: str, control_paths: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one (caption, control image(s)) through Qwen3-VL.

        Uses the driver's ``encode_text_with_images`` (upstream TI2I
        chat-template branch — images attended by the VL encoder). If the
        with-image encode fails for an image-specific reason (unreadable
        control file, a processor build without image support, ...), fall
        back to the TEXT-ONLY encode with ONE loud warning (qwen
        degraded-mode precedent) — the control still conditions the
        transformer via ``ref_image_hidden_states``. A missing
        processor/text-encoder is NOT masked: boogu's text-only path needs
        the same processor, so the driver's fail-loud guard propagates from
        the fallback too (unlike qwen, whose text-only path has a separate
        tokenizer).
        """
        try:
            from PIL import Image  # noqa: PLC0415

            images = [Image.open(p).convert("RGB") for p in control_paths]
            out = self.driver.encode_text_with_images([caption], [images], dtype)
            return out.embeddings, out.attention_mask
        except RuntimeError:
            # Driver's assign_components fail-loud guard (no processor/TE):
            # the text-only path cannot work either — propagate, don't mask.
            raise
        except Exception as exc:  # noqa: BLE001 — degraded mode, warn once
            if not self._warned_no_vl_processor:
                self._warned_no_vl_processor = True
                self.logger.warning(
                    "boogu_edit_vl_image_encode_fallback_text_only",
                    reason=f"{type(exc).__name__}: {exc}",
                    hint="control image NOT attended by the VL text encoder "
                         "this run; control still conditions the transformer "
                         "via ref_image_hidden_states",
                )
            return self._encode_text_direct([caption], dtype)
